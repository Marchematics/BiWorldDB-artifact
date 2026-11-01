// panel_fmdi_sparse.cpp — timed worker for the 2D representation comparison:
// FMDI vs sparse dynamic 2D index vs dense dynamic 2D Fenwick (2026-11-02).  New file under
// results/fmdi_sparse_20261101/; the FMDI index itself is the existing fmd_index.hpp
// (unchanged).
//
// Modes:
//   fmd       FMDI: dim0 range tree (shared hierarchy) with per-node 1D dim1 index, per-group
//             base trees and support-sparse reversible deviations  -> O(N log C0) memory
//   sparse2d  sparse dynamic 2D index: per-group sparse 2D Fenwick-of-vectors over the
//             occupied supports only -> memory grows with the occupied support count
//   densefen  dynamic dense 2D Fenwick: G dense (C0+1)*(C1+1) i64 trees (the product-class
//             memory competitor) -> O(G*C0*C1) memory, O(log^2) updates/queries
//   oracle    independent brute force over the current rows (exactness only; use --max-v for
//             a version prefix on large domains)
//
// Same 2D contract as the earlier FMDI prototype: rows (c0,c1,amount,group), one edit per
// version (replace/insert/delete/move/factor), Q declared rectangles per state, answer =
// sum amount*fac(group) over the rectangle.  Phases: gen, build, execute, extract, cleanup.
// JSON adds representation bytes: static_index, product_class, dynamic, dense_need_bytes.
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <sys/resource.h>

#include "dense_fenwick_arm.hpp"
#include "fmd_index.hpp"
#include "sparse2d_arm.hpp"
// sha256.hpp expects the shipped bwh scalar typedefs in scope; inject just those.
namespace bwh {
using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
}  // namespace bwh
#include "sha256.hpp"

using namespace fmd;

static i64 ns_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

int main(int argc, char** argv) {
  std::string mode = "fmd", answers_path, json_path;
  Config cfg;
  i32 repeat = 0, max_v = -1;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](const char* k) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k); std::exit(2); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--mode")) mode = next(argv[i]);
    else if (!std::strcmp(argv[i], "--rows")) cfg.rows = std::atoll(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--c0")) cfg.c0 = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--c1")) cfg.c1 = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--groups")) cfg.groups = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--states")) cfg.states = std::atoll(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--q")) cfg.q = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--factor-period")) cfg.factor_period = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--topology")) cfg.topology = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--seed")) cfg.seed = std::atoll(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--repeat")) repeat = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--max-v")) max_v = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--answers")) answers_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--json")) json_path = next(argv[i]);
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  if (answers_path.empty() || json_path.empty()) {
    std::fprintf(stderr, "usage: panel_fmdi_sparse --mode fmd|sparse2d|densefen|oracle --rows N --c0 C0 "
                         "--c1 C1 --groups G --states V --q Q --factor-period P --topology T "
                         "--seed S --repeat R [--max-v M] --answers A --json J\n");
    return 2;
  }
  const i64 last_v = (max_v < 0 || max_v >= cfg.states - 1) ? cfg.states - 1 : max_v;
  const auto t0 = std::chrono::steady_clock::now();
  Workload w = make_workload(cfg);
  const auto t1 = std::chrono::steady_clock::now();

  std::unique_ptr<Arm> arm;
  if (mode == "fmd") arm.reset(new FmdArm(w));
  else if (mode == "sparse2d") arm.reset(new Sparse2DArm(w));
  else if (mode == "densefen") arm.reset(new DenseFenwickArm(w));
  else if (mode == "oracle") arm.reset(new fmd::OracleArm(w));
  else { std::fprintf(stderr, "unknown mode %s\n", mode.c_str()); return 3; }
  arm->build();
  const auto t2 = std::chrono::steady_clock::now();

  std::vector<i64> answers(size_t(last_v) * size_t(cfg.q), 0);
  std::vector<SavedRow> saved(size_t(cfg.states));
  std::unique_ptr<Arm> acq = std::move(arm);
  if (cfg.topology == 0) {
    for (i64 v = 1; v <= last_v; ++v) {
      acq->apply(w.edit[size_t(v)]);
      for (i32 k = 0; k < cfg.q; ++k)
        answers[size_t(v - 1) * size_t(cfg.q) + size_t(k)] = acq->query(i32(v), k);
    }
  } else {
    std::vector<std::vector<i32>> kids(static_cast<size_t>(cfg.states));
    for (i64 v = 1; v < cfg.states; ++v) kids[size_t(w.parent[size_t(v)])].push_back(i32(v));
    struct Ev { i32 v; u8 exit; };
    std::vector<Ev> st;
    st.push_back({0, 0});
    while (!st.empty()) {
      const Ev ev = st.back();
      st.pop_back();
      const i32 v = ev.v;
      if (ev.exit) {
        if (v > 0 && v <= last_v) acq->undo(w.edit[size_t(v)], saved[size_t(v)]);
        continue;
      }
      if (v > 0) {
        if (v > last_v) continue;
        saved[size_t(v)] = acq->snapshot(w.edit[size_t(v)]);
        acq->apply(w.edit[size_t(v)]);
        for (i32 k = 0; k < cfg.q; ++k)
          answers[size_t(v - 1) * size_t(cfg.q) + size_t(k)] = acq->query(v, k);
      }
      st.push_back({v, 1});
      const auto& ch = kids[size_t(v)];
      for (auto it = ch.rbegin(); it != ch.rend(); ++it) st.push_back({*it, 0});
    }
  }
  const auto t3 = std::chrono::steady_clock::now();

  u64 fnv = 1469598103934665603ULL;
  {
    const u8* p = reinterpret_cast<const u8*>(answers.data());
    for (size_t i = 0; i < answers.size() * sizeof(i64); ++i) fnv = (fnv ^ p[i]) * 1099511628211ULL;
  }
  const std::string sha = bwh::sha::of(answers.data(), answers.size() * sizeof(i64));
  if (!answers_path.empty()) {
    FILE* out = std::fopen(answers_path.c_str(), "wb");
    if (!out) { std::fprintf(stderr, "cannot write answers\n"); return 4; }
    const bool ok = std::fwrite(answers.data(), sizeof(i64), answers.size(), out) == answers.size();
    std::fclose(out);
    if (!ok) { std::fprintf(stderr, "short answer write\n"); return 4; }
  }
  const auto t4 = std::chrono::steady_clock::now();
  i64 static_bytes = 0, product_bytes = 0, work_bytes = 0;
  acq->stats(static_bytes, product_bytes, work_bytes);
  acq.reset();
  const auto t5 = std::chrono::steady_clock::now();

  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
  i64 peak_kb = 0;
  {
    FILE* s = std::fopen("/proc/self/status", "r");
    if (s) {
      char line[256];
      while (std::fgets(line, sizeof(line), s)) {
        long vv = 0;
        if (std::sscanf(line, "VmHWM: %ld kB", &vv) == 1) { peak_kb = vv; break; }
      }
      std::fclose(s);
    }
  }
  const i64 gen_ns = ns_between(t0, t1), build_ns = ns_between(t1, t2);
  const i64 exec_ns = ns_between(t2, t3), extract_ns = ns_between(t3, t4);
  const i64 cleanup_ns = ns_between(t4, t5);
  const i64 full_ns = gen_ns + build_ns + exec_ns + extract_ns + cleanup_ns;
  FILE* jf = std::fopen((json_path + ".tmp").c_str(), "w");
  if (!jf) { std::fprintf(stderr, "cannot write json\n"); return 4; }
  std::fprintf(jf, "{\n  \"mode\": \"%s\",\n", mode.c_str());
  std::fprintf(jf, "  \"rows\": %" PRId64 ", \"c0\": %d, \"c1\": %d, \"groups\": %d,\n",
               cfg.rows, cfg.c0, cfg.c1, cfg.groups);
  std::fprintf(jf, "  \"states\": %" PRId64 ", \"max_v\": %" PRId64 ", \"q\": %d, "
                   "\"factor_period\": %d, \"topology\": %d, \"seed\": %" PRId64 ",\n",
               cfg.states, last_v, cfg.q, cfg.factor_period, cfg.topology, cfg.seed);
  std::fprintf(jf, "  \"repeat\": %d,\n  \"answers_fn\": \"%016" PRIx64 "\",\n", repeat, fnv);
  std::fprintf(jf, "  \"answers_sha256\": \"%s\",\n  \"n_answers\": %zu,\n", sha.c_str(), answers.size());
  std::fprintf(jf, "  \"phases_ns\": {\"gen\": %" PRId64 ", \"build\": %" PRId64
                   ", \"execute\": %" PRId64 ", \"extract\": %" PRId64 ", \"cleanup\": %" PRId64 "},\n",
               gen_ns, build_ns, exec_ns, extract_ns, cleanup_ns);
  std::fprintf(jf, "  \"full_ns\": %" PRId64 ",\n  \"peak_rss_kb\": %" PRId64 ",\n", full_ns, peak_kb);
  std::fprintf(jf, "  \"bytes\": {\"static_index\": %" PRId64 ", \"product_class\": %" PRId64
                   ", \"dynamic\": %" PRId64 ", \"dense_need\": %" PRId64 "}\n}\n",
               static_bytes, product_bytes, work_bytes,
               i64(cfg.groups) * i64(cfg.c0 + 1) * i64(cfg.c1 + 1) * 8);
  std::fclose(jf);
  std::rename((json_path + ".tmp").c_str(), json_path.c_str());
  std::printf("mode=%s rows=%" PRId64 " c0=%d c1=%d G=%d V=%" PRId64 " mv=%" PRId64
              " full_ns=%" PRId64 " peak_kb=%" PRId64 " static=%" PRId64 " need=%" PRId64 "\n",
              mode.c_str(), cfg.rows, cfg.c0, cfg.c1, cfg.groups, cfg.states, last_v, full_ns,
              peak_kb, static_bytes,
              i64(cfg.groups) * i64(cfg.c0 + 1) * i64(cfg.c1 + 1) * 8);
  return 0;
}
