// panel_batch.cpp — one timed worker for the version-tree BATCH execution prototype.
//
// Sibling of panel.cpp (same CLI, phase accounting, answer serialization and JSON
// fields) with two factory routes:
//   * batch   : BatchEngine (batch_engine.hpp) — offline declared-workload preprocessing
//               plus a single preorder batch pass over the version tree.
//   * fallback: the shipped hybrid arm, used when the batch engine reports the frame as
//               outside its supported region (dense frame or correction table too large).
//               It reuses the fast declared-workload construction, so the setup phase is
//               still the batch preprocessing cost.
// The JSON records which route ran ("mode").
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/resource.h>

#include "common.hpp"
#include "engine.hpp"
#include "batch_engine.hpp"
#include "sha256.hpp"

using namespace bwh;

static i64 ns_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

static bool write_all(FILE* fp, const void* p, size_t n) { return std::fwrite(p, 1, n, fp) == n; }

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
    std::fprintf(stderr, "usage: panel_batch --fixture PATH --base-id ID --topology T --seed S "
                         "--factor-period P --q Q --repeat R --answers A --json J\n");
    return 2;
  }
  if (arm_name.empty()) arm_name = "batch";
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

  // ---- setup: fast declared-workload preprocessing + batch plan
  BatchWorkload bw = make_workload_fast(f, topology, seed, factor_period, q, per_state);
  BatchEngine eng(f, bw);
  const bool batch_ok = eng.build();
  bool used_batch = false;
  std::vector<i64> answers(size_t(kStates - 1) * size_t(q), 0);
  Arm* fallback_arm = nullptr;
  Workload fw;      // only materialized on the fallback route
  std::vector<SavedRow> saved(static_cast<size_t>(kStates));
  std::vector<std::vector<i32>> kids;
  std::vector<std::pair<i32, u8>> st;
  auto run_fallback_setup = [&]() {
    fw.queries.clear();
    fw.topology = bw.topology; fw.seed = bw.seed; fw.factor_period = bw.factor_period;
    fw.q = bw.q; fw.per_state = bw.per_state; fw.n_base = bw.n_base;
    fw.n_inserted = bw.n_inserted; fw.groups = bw.groups;
    fw.t_min = bw.t_min; fw.t_max = bw.t_max;
    fw.parent = bw.parent; fw.edit = bw.edit; fw.ins_t = bw.ins_t; fw.ins_g = bw.ins_g;
    fw.endpoints = bw.endpoints;
    if (bw.per_state) {
      fw.queries.resize(size_t(kStates) * size_t(kMaxQ));
      for (i32 v = 1; v < kStates; ++v)
        for (i32 k = 0; k < bw.q; ++k)
          fw.queries[size_t(v) * size_t(kMaxQ) + size_t(k)] = bw.query(v, k);
    } else {
      fw.queries.resize(size_t(kMaxQ));
      for (i32 k = 0; k < bw.q; ++k) fw.queries[size_t(k)] = bw.query(0, k);
    }
    fallback_arm = make_arm("hybrid", f, fw);
    if (!fallback_arm) { std::fprintf(stderr, "fallback arm failed\n"); std::exit(3); }
    fallback_arm->build();
    saved.assign(static_cast<size_t>(kStates), SavedRow{});
    kids.assign(static_cast<size_t>(kStates), {});
    for (i32 v = 1; v < kStates; ++v) kids[size_t(fw.parent[size_t(v)])].push_back(v);
  };
  if (!batch_ok) run_fallback_setup();
  const auto t2 = std::chrono::steady_clock::now();

  // ---- execute
  if (batch_ok) {
    used_batch = true;
    eng.run(answers);
  } else {
    // shipped reversible DFS with the batch-preprocessed workload
    st.reserve(static_cast<size_t>(kStates) + 8);
    st.push_back({0, 0});
    while (!st.empty()) {
      const auto ev = st.back();
      st.pop_back();
      const i32 v = ev.first;
      if (ev.second) {
        if (v > 0) fallback_arm->undo(fw.edit[size_t(v)], saved[size_t(v)]);
        continue;
      }
      if (v > 0) {
        saved[size_t(v)] = fallback_arm->snapshot(fw.edit[size_t(v)]);
        fallback_arm->apply(fw.edit[size_t(v)]);
        for (i32 qi = 0; qi < q; ++qi)
          answers[size_t(v - 1) * size_t(q) + size_t(qi)] = fallback_arm->query(v, qi);
      }
      st.push_back({v, 1});
      const auto& ch = kids[size_t(v)];
      for (auto it = ch.rbegin(); it != ch.rend(); ++it) st.push_back({*it, 0});
    }
  }
  const auto t3 = std::chrono::steady_clock::now();

  // ---- extract: complete answer vector + checksums (identical to panel.cpp)
  FILE* out = std::fopen(answers_path.c_str(), "wb");
  if (!out) { std::fprintf(stderr, "cannot write answers\n"); return 4; }
  const bool ok = write_all(out, answers.data(), answers.size() * sizeof(i64));
  std::fclose(out);
  if (!ok) { std::fprintf(stderr, "short answer write\n"); return 4; }
  const u64 cks = fnv1a(answers.data(), answers.size() * sizeof(i64));
  const std::string sha = sha::of(answers.data(), answers.size() * sizeof(i64));
  const auto t4 = std::chrono::steady_clock::now();

  delete fallback_arm;
  fw = Workload{};
  f.release_rows();
  const auto t5 = std::chrono::steady_clock::now();

  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
  i64 peak_kb = 0;
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

  const i64 load_ns = ns_between(t0, t1);
  const i64 setup_ns = ns_between(t1, t2);
  const i64 exec_ns = ns_between(t2, t3);
  const i64 extract_ns = ns_between(t3, t4);
  const i64 cleanup_ns = ns_between(t4, t5);
  const i64 full_ns = load_ns + setup_ns + exec_ns + extract_ns + cleanup_ns;

  FILE* jf = std::fopen((json_path + ".tmp").c_str(), "w");
  if (!jf) { std::fprintf(stderr, "cannot write json\n"); return 4; }
  std::fprintf(jf, "{\n");
  std::fprintf(jf, "  \"arm\": \"%s\",\n  \"base_id\": \"%s\",\n", used_batch ? "batch" : "batch_fallback", base_id.c_str());
  std::fprintf(jf, "  \"fixture\": \"%s\",\n  \"topology\": %d,\n  \"seed\": %d,\n  \"factor_period\": %d,\n",
               fixture_path.c_str(), topology, seed, factor_period);
  std::fprintf(jf, "  \"q\": %d,\n  \"repeat\": %d,\n  \"n_states\": %d,\n  \"n_answers\": %" PRId64 ",\n",
               q, repeat, kStates, i64(answers.size()));
  std::fprintf(jf, "  \"range_scope\": \"%s\",\n", per_state ? "per_state" : "shared");
  std::fprintf(jf, "  \"mode\": \"%s\",\n", used_batch ? "batch" : "fallback");
  std::fprintf(jf, "  \"answers_fn\": \"%016" PRIx64 "\",\n", cks);
  std::fprintf(jf, "  \"answers_sha256\": \"%s\",\n", sha.c_str());
  std::fprintf(jf, "  \"phases_ns\": {\"load\": %" PRId64 ", \"setup\": %" PRId64 ", \"execute\": %" PRId64
                   ", \"extract\": %" PRId64 ", \"cleanup\": %" PRId64 "},\n",
               load_ns, setup_ns, exec_ns, extract_ns, cleanup_ns);
  std::fprintf(jf, "  \"full_ns\": %" PRId64 ",\n  \"peak_rss_kb\": %" PRId64 "\n}\n", full_ns, peak_kb);
  std::fclose(jf);
  if (std::rename((json_path + ".tmp").c_str(), json_path.c_str()) != 0) {
    std::fprintf(stderr, "cannot rename json\n");
    return 4;
  }
  std::printf("arm=%s base=%s t%d s%d f%d q%d r%d full_ns=%" PRId64 " peak_kb=%" PRId64 "\n",
              used_batch ? "batch" : "batch_fallback", base_id.c_str(), topology, seed,
              factor_period, q, repeat, full_ns, peak_kb);
  return 0;
}
