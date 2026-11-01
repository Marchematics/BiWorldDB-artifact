// panel_real_branch_v3.cpp — timed worker for "batch v3" on the real branching version history
// (2026-10-31, real-history P1).  New file; batch_engine_v3.hpp / panel_batch_v3.cpp are used
// read-only as shipped artifacts of the deep-tree executor.
//
// Same CLI/phase/JSON contract as panel_real_branch.cpp.  The batch v2 engine is given the
// identical real declared workload; when its policy reports the frame unsupported it falls
// back to the shipped hybrid arm (mode=fallback), exactly like panel_batch_v3.cpp.
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
#include "batch_engine_v3.hpp"
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
  if (fixture_path.empty() || spec_path.empty() || answers_path.empty() || json_path.empty()) {
    std::fprintf(stderr, "usage: panel_real_history_v2 --fixture PATH --spec PATH --repeat R "
                         "--answers A --json J [--arm batchv3]\n");
    return 2;
  }
  if (arm_name.empty()) arm_name = "batchv3";
  const auto t0 = std::chrono::steady_clock::now();
  Fixture f;
  if (!load_fixture(fixture_path, f)) { std::fprintf(stderr, "cannot load fixture\n"); return 3; }
  const auto t1 = std::chrono::steady_clock::now();

  // ---- setup: declared real history + batch v2 build (fallback keeps the same fast spec load)
  RealHistorySpec spec;
  if (!load_real_history_spec(spec_path, spec)) {
    std::fprintf(stderr, "cannot load real history spec\n");
    return 3;
  }
  if (spec.n_base != f.h.n) { std::fprintf(stderr, "spec/fixture base mismatch\n"); return 3; }
  const i32 q = spec.q;
  BatchWorkloadV3 bw;
  fill_batch_workload(spec, bw);
  BatchEngineV3 eng(f, bw);
  const bool batch_ok = eng.build();
  std::vector<i64> answers(size_t(kStates - 1) * size_t(q), 0);
  bool used_batch = false;
  Workload fw;
  Arm* fallback_arm = nullptr;
  std::vector<SavedRow> saved;
  std::vector<std::vector<i32>> kids;
  std::vector<std::pair<i32, u8>> st;
  auto setup_fallback = [&]() {
    fw.queries.clear();
    fw.topology = bw.topology; fw.seed = bw.seed; fw.factor_period = bw.factor_period;
    fw.q = bw.q; fw.per_state = bw.per_state; fw.n_base = bw.n_base;
    fw.n_inserted = bw.n_inserted; fw.groups = bw.groups;
    fw.t_min = bw.t_min; fw.t_max = bw.t_max;
    fw.parent = bw.parent; fw.edit = bw.edit; fw.ins_t = bw.ins_t; fw.ins_g = bw.ins_g;
    fw.endpoints = bw.endpoints;
    fw.queries.assign(size_t(kStates) * size_t(kMaxQ), Query{0, 0});
    for (i32 v = 1; v < kStates; ++v)
      for (i32 k = 0; k < bw.q; ++k)
        fw.queries[size_t(v) * size_t(kMaxQ) + size_t(k)] = bw.query(v, k);
    fallback_arm = make_arm("hybrid", f, fw);
    if (!fallback_arm) { std::fprintf(stderr, "fallback arm failed\n"); std::exit(3); }
    fallback_arm->build();
    saved.assign(size_t(kStates), SavedRow{});
    kids.assign(size_t(kStates), {});
    for (i32 v = 1; v < kStates; ++v) kids[size_t(fw.parent[size_t(v)])].push_back(v);
  };
  if (!batch_ok) setup_fallback();
  const auto t2 = std::chrono::steady_clock::now();

  // ---- execute
  if (batch_ok) {
    used_batch = true;
    eng.run(answers);
  } else {
    st.reserve(size_t(kStates) + 8);
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
        fallback_arm->query_batch(v, &answers[size_t(v - 1) * size_t(q)], q);
      }
      st.push_back({v, 1});
      const auto& ch = kids[size_t(v)];
      for (auto it = ch.rbegin(); it != ch.rend(); ++it) st.push_back({*it, 0});
    }
  }
  const auto t3 = std::chrono::steady_clock::now();

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
  const i64 load_ns = ns_between(t0, t1), setup_ns = ns_between(t1, t2),
            exec_ns = ns_between(t2, t3), extract_ns = ns_between(t3, t4),
            cleanup_ns = ns_between(t4, t5);
  const i64 full_ns = load_ns + setup_ns + exec_ns + extract_ns + cleanup_ns;

  FILE* jf = std::fopen((json_path + ".tmp").c_str(), "w");
  if (!jf) { std::fprintf(stderr, "cannot write json\n"); return 4; }
  std::fprintf(jf, "{\n  \"arm\": \"batchv3\",\n  \"mode\": \"%s\",\n  \"family\": \"real_branch\",\n",
               used_batch ? "batch_v3" : "fallback");
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
  std::fprintf(jf, "  \"stats\": {}\n}\n");
  std::fclose(jf);
  if (std::rename((json_path + ".tmp").c_str(), json_path.c_str()) != 0) {
    std::fprintf(stderr, "cannot rename json\n");
    return 4;
  }
  std::printf("arm=batchv3 mode=%s full_ns=%" PRId64 " peak_kb=%" PRId64 "\n",
              used_batch ? "batch_v3" : "fallback", full_ns, peak_kb);
  return 0;
}
