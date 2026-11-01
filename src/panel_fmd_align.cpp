// panel_fmd_align.cpp — one timed worker for the FMDI alignment experiment (2026-11-01).
// Copy of panel_fmd.cpp with one extra arm: `dyn2d` = dynamic dense per-group 2D Fenwick
// baseline (results/fmdi_align_20261101/fmd_align_index.hpp).  Same CLI, workload,
// phase accounting, answer layout and sha256 contract as panel_fmd.cpp.
// New file; shipped engine/panel/common are untouched.
//
// Phases (all charged to the full time, as in panel.cpp): gen (deterministic 2D workload
// construction), build (index construction), execute (version-tree traversal with edits and
// the declared rectangles), extract (answer checksum + sha256), cleanup (arm + fixture).
// Also reports peak RSS (VmHWM) and the representation sizes (static / product-class bytes).
//
// Arms: fmd (FMDI), full (full-state per-version 2D rebuild), naive (per-group dense 2D
// prefix), oracle (brute force; correctness only).
//
// Chain version trees are traversed sequentially (no undo); bushy/star use the reversible
// snapshot/undo path, which exercises the reversible row/factor corrections.
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/resource.h>

#include "fmd_index.hpp"
#include "fmd_align_index.hpp"
// sha256.hpp expects the shipped bwh scalar typedefs to be in scope; inject just those.
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
  std::string arm_name, answers_path, json_path;
  Config cfg;
  i32 repeat = 0;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](const char* k) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k); std::exit(2); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--arm")) arm_name = next(argv[i]);
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
    else if (!std::strcmp(argv[i], "--answers")) answers_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--json")) json_path = next(argv[i]);
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  if (arm_name.empty() || json_path.empty()) {
    std::fprintf(stderr, "usage: panel_fmd_align --arm fmd|dyn2d|full|naive|oracle --rows N --c0 C0 "
                         "--c1 C1 --groups G --states V --q Q --factor-period P --topology T "
                         "--seed S --repeat R [--answers A] --json J\n");
    return 2;
  }
  const auto t0 = std::chrono::steady_clock::now();
  Workload w = make_workload(cfg);
  const auto t1 = std::chrono::steady_clock::now();

  Arm* arm = make_arm(arm_name, w);
  if (!arm) arm = make_arm_align(arm_name, w);
  if (!arm) { std::fprintf(stderr, "unknown arm %s\n", arm_name.c_str()); return 3; }
  arm->build();
  const auto t2 = std::chrono::steady_clock::now();

  const i64 V = cfg.states;
  std::vector<i64> answers(size_t(V > 1 ? V - 1 : 0) * size_t(cfg.q), 0);
  std::vector<SavedRow> saved(static_cast<size_t>(V));
  if (cfg.topology == 0) {
    // chain: sequential application, no undo work
    for (i64 v = 1; v < V; ++v) {
      arm->apply(w.edit[size_t(v)]);
      for (i32 k = 0; k < cfg.q; ++k) answers[size_t(v - 1) * size_t(cfg.q) + size_t(k)] = arm->query(i32(v), k);
    }
  } else {
    std::vector<std::vector<i32>> kids(static_cast<size_t>(V));
    for (i64 v = 1; v < V; ++v) kids[size_t(w.parent[size_t(v)])].push_back(i32(v));
    struct Ev { i32 v; u8 exit; };
    std::vector<Ev> st;
    st.push_back({0, 0});
    while (!st.empty()) {
      const Ev ev = st.back();
      st.pop_back();
      if (ev.exit) {
        if (ev.v > 0) arm->undo(w.edit[size_t(ev.v)], saved[size_t(ev.v)]);
        continue;
      }
      if (ev.v > 0) {
        saved[size_t(ev.v)] = arm->snapshot(w.edit[size_t(ev.v)]);
        arm->apply(w.edit[size_t(ev.v)]);
        for (i32 k = 0; k < cfg.q; ++k)
          answers[size_t(ev.v - 1) * size_t(cfg.q) + size_t(k)] = arm->query(ev.v, k);
      }
      st.push_back({ev.v, 1});
      const auto& ch = kids[size_t(ev.v)];
      for (auto it = ch.rbegin(); it != ch.rend(); ++it) st.push_back({*it, 0});
    }
  }
  const auto t3 = std::chrono::steady_clock::now();

  const u64 cks = 1469598103934665603ULL;
  u64 fnv = cks;
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
  arm->stats(static_bytes, product_bytes, work_bytes);
  delete arm;
  const auto t5 = std::chrono::steady_clock::now();

  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
  i64 peak_kb = 0;
#if defined(__linux__)
  {
    FILE* s = std::fopen("/proc/self/status", "r");
    if (s) {
      char line[256];
      while (std::fgets(line, sizeof(line), s)) {
        long v = 0;
        if (std::sscanf(line, "VmHWM: %ld kB", &v) == 1) { peak_kb = v; break; }
      }
      std::fclose(s);
    }
  }
#else
  peak_kb = i64(ru.ru_maxrss) / 1024;
#endif
  const i64 gen_ns = ns_between(t0, t1), build_ns = ns_between(t1, t2);
  const i64 exec_ns = ns_between(t2, t3), extract_ns = ns_between(t3, t4);
  const i64 cleanup_ns = ns_between(t4, t5);
  const i64 full_ns = gen_ns + build_ns + exec_ns + extract_ns + cleanup_ns;
  FILE* jf = std::fopen((json_path + ".tmp").c_str(), "w");
  if (!jf) { std::fprintf(stderr, "cannot write json\n"); return 4; }
  std::fprintf(jf, "{\n  \"arm\": \"%s\",\n", arm_name.c_str());
  std::fprintf(jf, "  \"rows\": %" PRId64 ", \"c0\": %d, \"c1\": %d, \"groups\": %d,\n",
               cfg.rows, cfg.c0, cfg.c1, cfg.groups);
  std::fprintf(jf, "  \"states\": %" PRId64 ", \"q\": %d, \"factor_period\": %d, "
                   "\"topology\": %d, \"seed\": %" PRId64 ",\n",
               cfg.states, cfg.q, cfg.factor_period, cfg.topology, cfg.seed);
  std::fprintf(jf, "  \"repeat\": %d,\n  \"answers_fn\": \"%016" PRIx64 "\",\n", repeat, fnv);
  std::fprintf(jf, "  \"answers_sha256\": \"%s\",\n", sha.c_str());
  std::fprintf(jf, "  \"n_answers\": %zu,\n", answers.size());
  std::fprintf(jf, "  \"phases_ns\": {\"gen\": %" PRId64 ", \"build\": %" PRId64
                   ", \"execute\": %" PRId64 ", \"extract\": %" PRId64 ", \"cleanup\": %" PRId64 "},\n",
               gen_ns, build_ns, exec_ns, extract_ns, cleanup_ns);
  std::fprintf(jf, "  \"full_ns\": %" PRId64 ",\n", full_ns);
  std::fprintf(jf, "  \"peak_rss_kb\": %" PRId64 ",\n", peak_kb);
  std::fprintf(jf, "  \"bytes\": {\"static_index\": %" PRId64 ", \"product_class\": %" PRId64
                   ", \"dynamic\": %" PRId64 "}\n}\n",
               static_bytes, product_bytes, work_bytes);
  std::fclose(jf);
  if (std::rename((json_path + ".tmp").c_str(), json_path.c_str()) != 0) {
    std::fprintf(stderr, "cannot rename json\n");
    return 4;
  }
  std::printf("arm=%s rows=%" PRId64 " c0=%d c1=%d G=%d V=%" PRId64 " q=%d full_ns=%" PRId64
              " peak_kb=%" PRId64 " static=%" PRId64 " product=%" PRId64 "\n",
              arm_name.c_str(), cfg.rows, cfg.c0, cfg.c1, cfg.groups, cfg.states, cfg.q,
              full_ns, peak_kb, static_bytes, product_bytes);
  return 0;
}
