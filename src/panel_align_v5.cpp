// panel_align_v5.cpp — aligned PGD-G baseline with the SAME SHA-NI/OpenSSL extract path as
// panel_batch_v5.cpp, so the native-vs-aligned comparison charges an identical extract phase
// (BWH_SHA=sw selects the shipped software SHA-256 for both panels). New file; the original
// panel_pgdg_align.cpp and sha256.hpp are untouched.
//
// Sibling of panel.cpp with the same CLI, phase accounting, answer serialization, RSS
// measurement and JSON fields; new file (the shipped panel.cpp is untouched).  Arms:
//
//   pgdg_dense        compiled PGD-G, dense per-group base prefix rows
//   pgdg_dense_corr   compiled PGD-G + eager per-class correction prefix
//   pgdg_sparse       compiled PGD-G, sparse rank-LUT per-group base supports
//   pgdg_sparse_corr  sparse base + correction prefix
//   pgdg_auto         static regime policy (see pgdg_align_auto in pgdg_align.hpp)
//
// All arms share the offline declared-request preprocessing (make_workload_fast) and the
// preorder batch pass; the "mode" JSON field records the selected plan.
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <sys/resource.h>

#include "common.hpp"
#include "batch_engine.hpp"
#include "pgdg_align.hpp"
#include "sha256.hpp"
#include "sha_fast.hpp"

using namespace bwh;

static i64 ns_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

static bool write_all(FILE* fp, const void* p, size_t n) { return std::fwrite(p, 1, n, fp) == n; }

static const char* variant_name(int v) {
  switch (v) {
    case PgdAlignEngine::DENSE: return "pgdg_dense";
    case PgdAlignEngine::DENSE_CORR: return "pgdg_dense_corr";
    case PgdAlignEngine::SPARSE: return "pgdg_sparse";
    default: return "pgdg_sparse_corr";
  }
}

int main(int argc, char** argv) {
  std::string arm_name, fixture_path, base_id, answers_path, json_path;
  i32 topology = 0, seed = 7, factor_period = 4, q = 2, repeat = 0;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](const char* k) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k); std::exit(2); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--arm")) arm_name = next(argv[i]);
    else if (!std::strcmp(argv[i], "--fixture")) fixture_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--base-id")) base_id = next(argv[i]);
    else if (!std::strcmp(argv[i], "--topology")) topology = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--seed")) seed = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--factor-period")) factor_period = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--q")) q = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--repeat")) repeat = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--answers")) answers_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--json")) json_path = next(argv[i]);
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  if (fixture_path.empty() || answers_path.empty() || json_path.empty()) {
    std::fprintf(stderr, "usage: panel_pgdg_align --arm NAME --fixture PATH --base-id ID "
                         "--topology T --seed S --factor-period P --q Q --repeat R "
                         "--answers A --json J\n");
    return 2;
  }
  if (arm_name.empty()) arm_name = "pgdg_auto";
  if (q < 1 || q > kMaxQ) { std::fprintf(stderr, "unsupported q\n"); return 2; }

  const char* scope_env = std::getenv("BWH_RANGE_SCOPE");
  const bool per_state = scope_env && std::strcmp(scope_env, "per_state") == 0;
  const auto t0 = std::chrono::steady_clock::now();
  Fixture f;
  if (!load_fixture(fixture_path, f)) {
    std::fprintf(stderr, "cannot load fixture %s\n", fixture_path.c_str());
    return 3;
  }
  const auto t1 = std::chrono::steady_clock::now();

  // ---- setup: offline declared-workload preprocessing + compiled PGD-G build
  BatchWorkload bw = make_workload_fast(f, topology, seed, factor_period, q, per_state);
  int variant = PgdAlignEngine::DENSE;
  PgdAutoChoice auto_choice{};
  if (arm_name == "pgdg_auto") {
    auto_choice = pgdg_align_auto(f, bw);
    variant = auto_choice.variant;
  } else if (arm_name == "pgdg_dense") {
    variant = PgdAlignEngine::DENSE;
  } else if (arm_name == "pgdg_dense_corr") {
    variant = PgdAlignEngine::DENSE_CORR;
  } else if (arm_name == "pgdg_sparse") {
    variant = PgdAlignEngine::SPARSE;
  } else if (arm_name == "pgdg_sparse_corr") {
    variant = PgdAlignEngine::SPARSE_CORR;
  } else {
    std::fprintf(stderr, "unknown arm %s\n", arm_name.c_str());
    return 3;
  }
  std::unique_ptr<PgdAlignEngine> eng(new PgdAlignEngine(f, bw));
  if (!eng->build(static_cast<PgdAlignEngine::Variant>(variant))) {
    std::fprintf(stderr, "pgdg_align build unsupported (arm=%s)\n", arm_name.c_str());
    return 5;
  }
  const auto t2 = std::chrono::steady_clock::now();

  // ---- execute: one preorder batch pass over the declared version tree
  std::vector<i64> answers;
  eng->run(answers);
  const auto t3 = std::chrono::steady_clock::now();

  // ---- extract: complete answer vector + checksums (identical to panel.cpp)
  FILE* out = std::fopen(answers_path.c_str(), "wb");
  if (!out) { std::fprintf(stderr, "cannot write answers\n"); return 4; }
  const bool ok = write_all(out, answers.data(), answers.size() * sizeof(i64));
  std::fclose(out);
  if (!ok) { std::fprintf(stderr, "short answer write\n"); return 4; }
  const u64 cks = fnv1a(answers.data(), answers.size() * sizeof(i64));
  const std::string sha = answers_sha256(answers.data(), answers.size() * sizeof(i64));
  const auto t4 = std::chrono::steady_clock::now();

  // ---- cleanup: engine and workload destruction (records the RSS cost of the plan)
  const PgdAlignStats st = eng->stats;
  eng.reset();
  bw = BatchWorkload{};
  f.release_rows();
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

  const i64 load_ns = ns_between(t0, t1);
  const i64 setup_ns = ns_between(t1, t2);
  const i64 exec_ns = ns_between(t2, t3);
  const i64 extract_ns = ns_between(t3, t4);
  const i64 cleanup_ns = ns_between(t4, t5);
  const i64 full_ns = load_ns + setup_ns + exec_ns + extract_ns + cleanup_ns;

  FILE* jf = std::fopen((json_path + ".tmp").c_str(), "w");
  if (!jf) { std::fprintf(stderr, "cannot write json\n"); return 4; }
  std::fprintf(jf, "{\n");
  std::fprintf(jf, "  \"arm\": \"%s\",\n  \"base_id\": \"%s\",\n", arm_name.c_str(), base_id.c_str());
  std::fprintf(jf, "  \"mode\": \"%s\",\n  \"family\": \"pgdg_aligned\",\n", variant_name(variant));
  std::fprintf(jf, "  \"fixture\": \"%s\",\n  \"topology\": %d,\n  \"seed\": %d,\n  \"factor_period\": %d,\n",
               fixture_path.c_str(), topology, seed, factor_period);
  std::fprintf(jf, "  \"q\": %d,\n  \"repeat\": %d,\n  \"n_states\": %d, \"n_answers\": %" PRId64 ",\n",
               q, repeat, kStates, i64(answers.size()));
  std::fprintf(jf, "  \"range_scope\": \"%s\",\n", per_state ? "per_state" : "shared");
  std::fprintf(jf, "  \"answers_fn\": \"%016" PRIx64 "\",\n", cks);
  std::fprintf(jf, "  \"answers_sha256\": \"%s\",\n", sha.c_str());
  std::fprintf(jf, "  \"phases_ns\": {\"load\": %" PRId64 ", \"setup\": %" PRId64
                   ", \"execute\": %" PRId64 ", \"extract\": %" PRId64 ", \"cleanup\": %" PRId64 "},\n",
               load_ns, setup_ns, exec_ns, extract_ns, cleanup_ns);
  std::fprintf(jf, "  \"full_ns\": %" PRId64 ",\n  \"peak_rss_kb\": %" PRId64 ",\n", full_ns, peak_kb);
  std::fprintf(jf, "  \"stats\": {\"n_active\": %d, \"n_groups\": %d, \"classes\": %" PRId64
                   ", \"union_classes\": %" PRId64 ", \"dense_frame\": %s, \"sparse_base\": %s, "
                   "\"corr\": %s, \"n_factor_edits\": %" PRId64 ", \"n_ins\": %" PRId64
                   ", \"nnz\": %" PRId64 ", \"base_bytes\": %" PRId64
                   ", \"dense_base_bytes\": %" PRId64 ", \"plan_bytes\": %" PRId64
                   ", \"plan_us\": %" PRId64 ", \"base_us\": %" PRId64 "},\n",
               st.n_active, st.n_groups, st.classes, st.union_classes,
               st.dense_frame ? "true" : "false", st.sparse_base ? "true" : "false",
               st.corr ? "true" : "false", st.n_factor_edits, st.n_ins, st.nnz,
               st.base_bytes, st.dense_base_bytes, st.plan_bytes, st.plan_us, st.base_us);
  std::fprintf(jf, "  \"auto\": {\"classes\": %" PRId64 ", \"dense_bytes\": %" PRId64
                   ", \"est_nnz\": %" PRId64 ", \"sparse_bytes\": %" PRId64
                   ", \"occ\": %.6f}\n}\n",
               auto_choice.classes, auto_choice.dense_bytes, auto_choice.est_nnz,
               auto_choice.sparse_bytes, auto_choice.occ);
  std::fclose(jf);
  if (std::rename((json_path + ".tmp").c_str(), json_path.c_str()) != 0) {
    std::fprintf(stderr, "cannot rename json\n");
    return 4;
  }
  std::printf("arm=%s mode=%s base=%s t%d s%d f%d q%d r%d full_ns=%" PRId64 " peak_kb=%" PRId64 "\n",
              arm_name.c_str(), variant_name(variant), base_id.c_str(), topology, seed,
              factor_period, q, repeat, full_ns, peak_kb);
  return 0;
}
