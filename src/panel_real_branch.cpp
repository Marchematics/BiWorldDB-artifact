// panel_real_branch.cpp — one timed worker of the real branching-version-history benchmark
// (2026-10-31, real-history P1).  New file; shipped engine.hpp/panel.cpp are untouched.
//
// Same CLI contract, phase accounting, answer serialization, RSS measurement and JSON fields
// as panel.cpp, but the declared workload is the real branch spec built by
// real_branch_build.cpp (real chronological invoice/trip events) instead of a synthetic
// random workload.  Arms: the shipped hybrid, the rollback-Fenwick baseline, the genuine
// full-state projected Fenwick control, and the aligned PGD-G engine (auto / forced variants).
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
#include "engine.hpp"
#include "batch_engine.hpp"
#include "pgdg_align.hpp"
#include "arms_classic.hpp"
#include "real_branch_spec.hpp"
#include "sha256.hpp"

using namespace bwh;

static i64 ns_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}
static bool write_all(FILE* fp, const void* p, size_t n) { return std::fwrite(p, 1, n, fp) == n; }

int main(int argc, char** argv) {
  std::string arm_name, fixture_path, spec_path, answers_path, json_path;
  i32 repeat = 0;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](const char* k) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k); std::exit(2); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--arm")) arm_name = next(argv[i]);
    else if (!std::strcmp(argv[i], "--fixture")) fixture_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--spec")) spec_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--repeat")) repeat = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--answers")) answers_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--json")) json_path = next(argv[i]);
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  if (arm_name.empty() || fixture_path.empty() || spec_path.empty() || answers_path.empty() ||
      json_path.empty()) {
    std::fprintf(stderr, "usage: panel_real_history --arm NAME --fixture PATH --spec PATH "
                         "--repeat R --answers A --json J\n");
    return 2;
  }
  const auto t0 = std::chrono::steady_clock::now();
  Fixture f;
  if (!load_fixture(fixture_path, f)) { std::fprintf(stderr, "cannot load fixture\n"); return 3; }
  const auto t1 = std::chrono::steady_clock::now();

  // ---- setup: declared real history + engine build (same phase contract as panel.cpp)
  RealHistorySpec spec;
  if (!load_real_history_spec(spec_path, spec)) {
    std::fprintf(stderr, "cannot load real history spec\n");
    return 3;
  }
  if (spec.n_base != f.h.n) { std::fprintf(stderr, "spec/fixture base mismatch\n"); return 3; }
  const i32 q = spec.q;
  std::vector<i64> answers(size_t(kStates - 1) * size_t(q), 0);
  std::unique_ptr<PgdAlignEngine> pgdg;
  Arm* arm = nullptr;
  Workload w;
  bool arm_is_pgdg = false;
  int pgdg_variant = PgdAlignEngine::DENSE;
  if (arm_name == "pgdg_auto" || arm_name == "pgdg_dense" || arm_name == "pgdg_dense_corr" ||
      arm_name == "pgdg_sparse" || arm_name == "pgdg_sparse_corr") {
    BatchWorkload bw;
    fill_batch_workload(spec, bw);
    arm_is_pgdg = true;
    if (arm_name == "pgdg_auto") {
      PgdAutoChoice ch = pgdg_align_auto(f, bw);
      pgdg_variant = ch.variant;
    } else if (arm_name == "pgdg_dense") {
      pgdg_variant = PgdAlignEngine::DENSE;
    } else if (arm_name == "pgdg_dense_corr") {
      pgdg_variant = PgdAlignEngine::DENSE_CORR;
    } else if (arm_name == "pgdg_sparse") {
      pgdg_variant = PgdAlignEngine::SPARSE;
    } else {
      pgdg_variant = PgdAlignEngine::SPARSE_CORR;
    }
    pgdg.reset(new PgdAlignEngine(f, bw));
    if (!pgdg->build(static_cast<PgdAlignEngine::Variant>(pgdg_variant))) {
      std::fprintf(stderr, "pgdg build unsupported\n");
      return 5;
    }
  } else {
    fill_workload(spec, w);
    if (arm_name == "hybrid") arm = make_arm("hybrid", f, w);
    else if (arm_name == "full_projected") arm = make_arm("full_projected", f, w);
    else if (arm_name == "full_raw") arm = make_arm("full_raw", f, w);
    else if (arm_name == "rf") arm = make_classic_arm("rf", f, w);
    else { std::fprintf(stderr, "unknown arm %s\n", arm_name.c_str()); return 3; }
    if (!arm) { std::fprintf(stderr, "arm factory failed\n"); return 3; }
    arm->build();
  }
  const auto t2 = std::chrono::steady_clock::now();

  // ---- execute: the real version chain, in real chronological order
  if (arm_is_pgdg) {
    pgdg->run(answers);
  } else {
    // shipped reversible DFS (identical to panel.cpp); the real history is a chain, so no
    // undo work is performed inside the sweep
    std::vector<SavedRow> saved(static_cast<size_t>(kStates));
    std::vector<std::vector<i32>> kids(static_cast<size_t>(kStates));
    for (i32 v = 1; v < kStates; ++v) kids[size_t(w.parent[size_t(v)])].push_back(v);
    struct Ev { i32 v; u8 exit; };
    std::vector<Ev> st;
    st.reserve(size_t(kStates) + 8);
    st.push_back({0, 0});
    while (!st.empty()) {
      const Ev ev = st.back();
      st.pop_back();
      const i32 v = ev.v;
      if (ev.exit) {
        if (v > 0) arm->undo(w.edit[size_t(v)], saved[size_t(v)]);
        continue;
      }
      if (v > 0) {
        saved[size_t(v)] = arm->snapshot(w.edit[size_t(v)]);
        arm->apply(w.edit[size_t(v)]);
        arm->query_batch(v, &answers[size_t(v - 1) * size_t(q)], q);
      }
      st.push_back({v, 1});
      const auto& ch = kids[size_t(v)];
      for (auto it = ch.rbegin(); it != ch.rend(); ++it) st.push_back({*it, 0});
    }
  }
  const auto t3 = std::chrono::steady_clock::now();
  PgdAlignStats pgd_stats{};
  if (arm_is_pgdg) pgd_stats = pgdg->stats;

  // ---- extract
  FILE* out = std::fopen(answers_path.c_str(), "wb");
  if (!out) { std::fprintf(stderr, "cannot write answers\n"); return 4; }
  const bool ok = write_all(out, answers.data(), answers.size() * sizeof(i64));
  std::fclose(out);
  if (!ok) { std::fprintf(stderr, "short answer write\n"); return 4; }
  const u64 cks = fnv1a(answers.data(), answers.size() * sizeof(i64));
  const std::string sha = sha::of(answers.data(), answers.size() * sizeof(i64));
  const auto t4 = std::chrono::steady_clock::now();

  delete arm;
  pgdg.reset();
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
  const i64 load_ns = ns_between(t0, t1), setup_ns = ns_between(t1, t2),
            exec_ns = ns_between(t2, t3), extract_ns = ns_between(t3, t4),
            cleanup_ns = ns_between(t4, t5);
  const i64 full_ns = load_ns + setup_ns + exec_ns + extract_ns + cleanup_ns;

  const char* mode = arm_name.c_str();
  if (arm_is_pgdg)
    mode = (pgdg_variant == PgdAlignEngine::DENSE) ? "pgdg_dense"
           : (pgdg_variant == PgdAlignEngine::DENSE_CORR) ? "pgdg_dense_corr"
           : (pgdg_variant == PgdAlignEngine::SPARSE) ? "pgdg_sparse" : "pgdg_sparse_corr";

  FILE* jf = std::fopen((json_path + ".tmp").c_str(), "w");
  if (!jf) { std::fprintf(stderr, "cannot write json\n"); return 4; }
  std::fprintf(jf, "{\n  \"arm\": \"%s\",\n  \"mode\": \"%s\",\n  \"family\": \"real_branch\",\n",
               arm_name.c_str(), mode);
  std::fprintf(jf, "  \"fixture\": \"%s\",\n  \"spec\": \"%s\",\n", fixture_path.c_str(), spec_path.c_str());
  std::fprintf(jf, "  \"repeat\": %d,\n  \"n_states\": %d,\n  \"q\": %d,\n  \"n_answers\": %" PRId64 ",\n",
               repeat, kStates, q, i64(answers.size()));
  std::fprintf(jf, "  \"answers_fn\": \"%016" PRIx64 "\",\n  \"answers_sha256\": \"%s\",\n", cks, sha.c_str());
  std::fprintf(jf, "  \"phases_ns\": {\"load\": %" PRId64 ", \"setup\": %" PRId64
                   ", \"execute\": %" PRId64 ", \"extract\": %" PRId64 ", \"cleanup\": %" PRId64 "},\n",
               load_ns, setup_ns, exec_ns, extract_ns, cleanup_ns);
  std::fprintf(jf, "  \"full_ns\": %" PRId64 ",\n  \"peak_rss_kb\": %" PRId64 ",\n", full_ns, peak_kb);
  std::fprintf(jf, "  \"history\": {\"n_base\": %" PRId64 ", \"n_ins\": %" PRId64
                   ", \"inserts\": %" PRId64 ", \"deletes\": %" PRId64 ", \"credits\": %" PRId64
                   ", \"factor_edits\": %" PRId64 ", \"endpoints\": %" PRId64
                   ", \"period_lo\": %" PRId64 ", \"period_hi\": %" PRId64
                   ", \"bucket\": %" PRId64 "},\n",
               spec.n_base, spec.n_ins, spec.n_inserts, spec.n_deletes, spec.n_credits,
               spec.n_factor_edits, spec.n_ep, spec.period_lo, spec.period_hi, spec.bucket);
  if (arm_is_pgdg) {
    const PgdAlignStats& st = pgd_stats;
    std::fprintf(jf, "  \"stats\": {\"n_active\": %d, \"classes\": %" PRId64
                     ", \"union_classes\": %" PRId64 ", \"sparse_base\": %s, \"corr\": %s"
                     ", \"base_bytes\": %" PRId64 ", \"dense_base_bytes\": %" PRId64
                     ", \"plan_us\": %" PRId64 ", \"base_us\": %" PRId64 "}\n}\n",
                 st.n_active, st.classes, st.union_classes, st.sparse_base ? "true" : "false",
                 st.corr ? "true" : "false", st.base_bytes, st.dense_base_bytes, st.plan_us,
                 st.base_us);
  } else {
    std::fprintf(jf, "  \"stats\": {}\n}\n");
  }
  std::fclose(jf);
  if (std::rename((json_path + ".tmp").c_str(), json_path.c_str()) != 0) {
    std::fprintf(stderr, "cannot rename json\n");
    return 4;
  }
  std::printf("arm=%s mode=%s full_ns=%" PRId64 " peak_kb=%" PRId64 "\n", arm_name.c_str(), mode,
              full_ns, peak_kb);
  return 0;
}
