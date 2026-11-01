// panel_planned.cpp — one timed worker of the matched panel for the classic
// direct-algorithm baselines (rollback Fenwick "rf", rollback segment tree "rs",
// persistent segment tree "pst") under the same offline final-vector contract as
// panel.cpp.  This is a byte-for-byte sibling of panel.cpp except for the arm
// factory: it first asks make_classic_arm() and then falls back to make_arm() for
// the existing arms (hybrid, full_raw, full_projected), so the CLI, phase JSON and
// answer serialization are unchanged.
//
// Phases (all charged): load, setup (declared-request parse + shared planning + arm
// build), execute (version traversal, edits, declared queries), extract (complete
// answer vector + checksum), cleanup (destruction).  Post-worker oracle validation
// is done outside the worker by ext/classic_probe.py.
#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/resource.h>

#include "common.hpp"
#include "engine.hpp"
#include "arms_classic.hpp"
#include "cross_family_weights.hpp"
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
  if (arm_name.empty() || fixture_path.empty() || answers_path.empty() || json_path.empty()) {
    std::fprintf(stderr, "usage: panel_planned --arm planned|hybrid|classic "
                         "--fixture PATH --base-id ID --topology T --seed S --factor-period P "
                         "--q Q --repeat R --answers A --json J\n");
    return 2;
  }
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

  Workload w = make_workload(f, topology, seed, factor_period, q, per_state);
  // The planned driver owns the production configuration: apply the declared optimizations as
  // in-process defaults (without overriding an explicit launcher setting) before any arm is
  // created, so Planned and Forced-Hybrid runs measure the same configuration as the Gate-1
  // production panels. Without this the hybrid arm silently ran unoptimized.
  ::setenv("BWH_CLASS_LUT", "1", 0);
  ::setenv("BWH_OVERLAY", "1", 0);
  ::setenv("BWH_TRANSPOSE_BASEP", "auto", 0);
  ::setenv("BWH_GROUP_PREFIX", "active", 0);
  ::setenv("BWH_AGG2", "auto", 0);
  const auto tp0 = std::chrono::steady_clock::now();
  i64 factor_edits = 0, row_edits = 0;
  for (i32 v = 1; v < kStates; ++v) {
    if (w.edit[size_t(v)].op == OP_FACTOR) ++factor_edits; else ++row_edits;
  }
  const double vq = double(kStates) * double(std::max<i32>(w.q, 1));

  const double log_n = std::log(double(std::max<i64>(f.h.n, 1)));
  const double ng = double(f.h.n) / double(std::max<i32>(f.h.groups, 1));
  const double min_gf = double(std::min<i64>(f.h.groups, std::max<i64>(factor_edits, 1)));
  const double hf[bwh::kCrossFamilyHybridDims] = {
      1.0,
      double(f.h.n) / 1e5,
      min_gf * double(std::max<size_t>(w.endpoints.size(), 1)) / 1e8,
      vq * min_gf / 1e8,
      vq * std::log(double(std::max<i64>(row_edits, 1)) + 1.0) / 1e7,
      vq / 1e5};
  const double cf[bwh::kCrossFamilyClassicDims] = {
      1.0,
      double(f.h.n) * log_n / 1e9,
      vq * log_n / 1e7,
      double(std::max<i64>(factor_edits, 1)) * ng * log_n / 1e7,
      vq / 1e5};
  double pred_h = 0.0, pred_c = 0.0;
  for (int i = 0; i < bwh::kCrossFamilyHybridDims; ++i)
    pred_h += bwh::kCrossFamilyHybrid[i] * (hf[i] - bwh::kCrossFamilyHybridMu[i]) /
              bwh::kCrossFamilyHybridSd[i];
  for (int i = 0; i < bwh::kCrossFamilyClassicDims; ++i)
    pred_c += bwh::kCrossFamilyClassic[i] * (cf[i] - bwh::kCrossFamilyClassicMu[i]) /
              bwh::kCrossFamilyClassicSd[i];
  pred_h = std::exp(pred_h); pred_c = std::exp(pred_c);
  // Conservative guard: route to the classic family only when the analytic model predicts it
  // at least 15% better. Measured sessions show hybrid-side fluctuations and two-state-wide
  // prediction errors; the margin removes catastrophic misroutes at a small cost in missed
  // classic wins (documented in the planner note).
  std::string chosen = (arm_name == "hybrid" || arm_name == "classic") ? arm_name
      : (pred_c < 0.85 * pred_h ? "classic" : "hybrid");
  const auto tp1 = std::chrono::steady_clock::now();
  const i64 plan_ns = ns_between(tp0, tp1);
  Arm* arm = chosen == "classic" ? make_classic_arm("rf", f, w) : make_arm("hybrid", f, w);
  if (!arm) { std::fprintf(stderr, "cannot create arm %s\n", chosen.c_str()); return 3; }
  arm->build();
  const auto t2 = std::chrono::steady_clock::now();

  // Depth-first traversal of the declared version tree; answers for states 1..kStates-1.
  std::vector<i64> answers(size_t(kStates - 1) * size_t(q), 0);
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
      for (i32 qi = 0; qi < q; ++qi)
        answers[size_t(v - 1) * size_t(q) + size_t(qi)] = arm->query(v, qi);
    }
    st.push_back({v, 1});
    const auto& ch = kids[size_t(v)];
    for (auto it = ch.rbegin(); it != ch.rend(); ++it) st.push_back({*it, 0});
  }
  const auto t3 = std::chrono::steady_clock::now();

  // extract: write the complete answer vector (post-worker validation is done outside).
  FILE* out = std::fopen(answers_path.c_str(), "wb");
  if (!out) { std::fprintf(stderr, "cannot write answers\n"); return 4; }
  const bool ok = write_all(out, answers.data(), answers.size() * sizeof(i64));
  std::fclose(out);
  if (!ok) { std::fprintf(stderr, "short answer write\n"); return 4; }
  const u64 cks = fnv1a(answers.data(), answers.size() * sizeof(i64));
  const std::string sha = sha::of(answers.data(), answers.size() * sizeof(i64));
  const auto t4 = std::chrono::steady_clock::now();

  delete arm;
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
  std::fprintf(jf, "  \"arm\": \"%s\",\n  \"base_id\": \"%s\",\n", chosen.c_str(), base_id.c_str());
  auto envs = [](const char* k) { const char* v = std::getenv(k); return v ? v : "unset"; };
  std::fprintf(jf, "  \"chosen_family\": \"%s\",\n  \"predicted_hybrid_ms\": %.4f,\n"
                   "  \"predicted_classic_ms\": %.4f,\n  \"plan_ns\": %" PRId64 ",\n"
                   "  \"env\": {\"BWH_CLASS_LUT\": \"%s\", \"BWH_OVERLAY\": \"%s\", "
                   "\"BWH_TRANSPOSE_BASEP\": \"%s\", \"BWH_GROUP_PREFIX\": \"%s\", "
                   "\"BWH_AGG2\": \"%s\", \"BWH_RANGE_SCOPE\": \"%s\"},\n",
               chosen.c_str(), pred_h, pred_c, plan_ns,
               envs("BWH_CLASS_LUT"), envs("BWH_OVERLAY"), envs("BWH_TRANSPOSE_BASEP"),
               envs("BWH_GROUP_PREFIX"), envs("BWH_AGG2"), envs("BWH_RANGE_SCOPE"));
  std::fprintf(jf, "  \"fixture\": \"%s\",\n  \"topology\": %d,\n  \"seed\": %d,\n  \"factor_period\": %d,\n",
               fixture_path.c_str(), topology, seed, factor_period);
  std::fprintf(jf, "  \"q\": %d,\n  \"repeat\": %d,\n  \"n_states\": %d,\n  \"n_answers\": %" PRId64 ",\n",
               q, repeat, kStates, i64(answers.size()));
  std::fprintf(jf, "  \"range_scope\": \"%s\",\n", per_state ? "per_state" : "shared");
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
              chosen.c_str(), base_id.c_str(), topology, seed, factor_period, q, repeat, full_ns, peak_kb);
  return 0;
}
