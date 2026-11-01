// panel_lazy.cpp — panel worker for the Factor-Lazy Correction engine (hybrid/engine_lazy.hpp).
// Same CLI / JSON / answer contract as hybrid/panel.cpp: the answer vector for states
// 1..kStates-1 x declared queries is written verbatim and the JSON record carries the same
// phase timings. Extra JSON keys report the requested/resolved factor mode and the planner
// cost model; they do not change the existing contract.
//
// Arms: --arm lazyhybrid (BWH_FACTOR_MODE=eager|lazy|auto|hybrid), plus the explicit
// aliases lazyhybrid_eager / lazyhybrid_lazy / lazyhybrid_hybrid / lazyhybrid_auto, and
// "hybrid" as a shipped-eager alias.
//
// --bench factor-update runs the single-group 100k-nonzero-deviation factor-update
// microbenchmark instead of the panel traversal (interleaved eager/lazy windows) and writes
// its measurements to --json.
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/resource.h>

#include <algorithm>

#include "common.hpp"
#include "engine_lazy.hpp"
#include "sha256.hpp"

using namespace bwh;

static i64 ns_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

static bool write_all(FILE* fp, const void* p, size_t n) { return std::fwrite(p, 1, n, fp) == n; }


// ---------------------------------------------------------------- factor microbench
// Single-group 100k-nonzero-deviation factor-update microbenchmark. The declared engine is
// built normally, then group 0's deviation profile is replaced by a synthetic dense one
// (coords 0..99999, deviation +1 each) and the per-group Fenwick / union / weighted indices
// are rebuilt consistently, so set_factor() sees exactly 100000 nonzero deviation entries.
// Two separate measurements are reported: (a) snapshot+set_factor+undo per operation and
// (b) one query with the group active. Exactness is checked against the closed form
//   answer = BASE0(I) + (fac-s0)*B_g(I) + fac*sum_{c in I} Delta
// which must hold for both the eager and the lazy weighting by construction.
struct BenchResult {
  bool ok = false;
  bool exact = false;
  double update_ns = 0;
  double query_ns = 0;
  std::string msg;
};

static double median_of(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static BenchResult bench_one(const Fixture& f, const Workload& w, const char* mode, int ops,
                             int nz) {
  BenchResult r;
  LazyHybridArm arm(f, w, mode);
  arm.build();
  if (!arm.dense) { r.msg = "bench fixture must build a dense frame"; return r; }
  if (arm.G < 1 || arm.ncoord < i64(nz) + 1024) { r.msg = "bench frame too small"; return r; }
  const i32 g0 = 0;
  // synthetic deviation profile: 100k distinct coordinates, one nonzero entry each
  arm.tcoord[size_t(g0)].resize(size_t(nz));
  arm.tdev[size_t(g0)].assign(size_t(nz), 1);
  for (i32 k = 0; k < nz; ++k) arm.tcoord[size_t(g0)][size_t(k)] = i64(k);
  {
    std::vector<i32> sz(size_t(arm.G));
    for (i32 g = 0; g < arm.G; ++g) sz[size_t(g)] = i32(arm.tcoord[size_t(g)].size());
    arm.tgf.init(sz);
    for (i32 g = 0; g < arm.G; ++g) {
      i64* lv = arm.tgf.leaves(g);
      const auto& dv = arm.tdev[size_t(g)];
      for (size_t k = 0; k < dv.size(); ++k) lv[k + 1] = dv[k];
    }
    arm.tgf.build_from_leaves();
  }
  {
    std::vector<i64> un = arm.gunion;
    un.reserve(un.size() + size_t(nz));
    for (i32 k = 0; k < nz; ++k) un.push_back(i64(k));
    std::sort(un.begin(), un.end());
    un.erase(std::unique(un.begin(), un.end()), un.end());
    arm.gunion.swap(un);
    arm.gfen.init(i32(arm.gunion.size()));
    arm.hfen.init(i32(arm.gunion.size()));
    const i64 curf = arm.fac[size_t(g0)];
    for (i32 k = 0; k < nz; ++k) {
      const i32 pos = arm.ucl(i64(k)) + 1;
      if (!arm.lazy_path) {
        arm.gfen.add(pos, curf);
      } else {
        arm.gfen.add(pos, arm.s0[size_t(g0)]);
        if (arm.fmode == LazyHybridArm::FM_HYBRID && arm.hflag[size_t(g0)])
          arm.hfen.add(pos, curf - arm.s0[size_t(g0)]);
      }
    }
  }
  const i64 i0 = 40000, j0 = 90000;  // declared range strictly inside the synthetic support
  arm.qlb[size_t(1) * kMaxQ + 0] = i0;
  arm.qhb[size_t(1) * kMaxQ + 0] = j0;
  const i32 r0 = arm.g2row[size_t(g0)];
  if (r0 < 0) { r.msg = "bench group lacks an immutable base-prefix row"; return r; }
  i64 B = 0;
  if (arm.tp_base) {
    B = arm.baseP[size_t(j0) * size_t(arm.n_active) + size_t(r0)] -
        arm.baseP[size_t(i0) * size_t(arm.n_active) + size_t(r0)];
  } else {
    const i64* row = &arm.baseP[size_t(r0) * size_t(arm.ncoord + 1)];
    B = row[j0] - row[i0];
  }
  const i64 base0d = arm.base0[size_t(j0)] - arm.base0[size_t(i0)];
  const i64 syn = j0 - i0;
  const i64 s0g = arm.s0[size_t(g0)];
  const i64 q_before = arm.query(1, 0);
  const bool ok_before = (q_before == base0d + s0g * syn);
  Edit e{};
  e.op = OP_FACTOR;
  e.g = g0;
  {
    volatile i64 sink = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < ops; ++it) {
      e.val = (it & 1) ? 950 : 1050;
      const SavedRow sv = arm.snapshot(e);
      arm.apply(e);
      arm.undo(e, sv);
      sink += sv.amount;
    }
    const auto t1 = std::chrono::steady_clock::now();
    (void)sink;
    r.update_ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) /
                  double(ops);
  }
  {
    e.val = 1050;
    const SavedRow sv = arm.snapshot(e);
    arm.apply(e);
    (void)sv;
    const i64 q_after = arm.query(1, 0);
    const i64 d = 1050 - s0g;
    r.exact = ok_before && (q_after == base0d + d * B + i64(1050) * syn);
    volatile i64 sink = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < ops; ++it) sink += arm.query(1, 0);
    const auto t1 = std::chrono::steady_clock::now();
    (void)sink;
    r.query_ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) /
                 double(ops);
  }
  r.ok = true;
  return r;
}

static int run_factor_bench(const Fixture& f, const Workload& w, const std::string& json_path) {
  const i32 nz = 100000;
  const int ops = 200;
  const int windows = 3;
  std::vector<double> eager_upd, lazy_upd, eager_q, lazy_q;
  bool eager_exact = true, lazy_exact = true;
  std::string msg;
  for (int win = 0; win < windows; ++win) {
    const BenchResult be = bench_one(f, w, "eager", ops, nz);
    const BenchResult bl = bench_one(f, w, "lazy", ops, nz);
    if (!be.ok) { msg = "eager: " + be.msg; break; }
    if (!bl.ok) { msg = "lazy: " + bl.msg; break; }
    eager_upd.push_back(be.update_ns);
    lazy_upd.push_back(bl.update_ns);
    eager_q.push_back(be.query_ns);
    lazy_q.push_back(bl.query_ns);
    eager_exact = eager_exact && be.exact;
    lazy_exact = lazy_exact && bl.exact;
  }
  if (!msg.empty()) { std::fprintf(stderr, "bench failed: %s\n", msg.c_str()); return 5; }
  const double eu = median_of(eager_upd), lu = median_of(lazy_upd);
  const double eq = median_of(eager_q), lq = median_of(lazy_q);
  FILE* jf = std::fopen(json_path.c_str(), "w");
  if (!jf) { std::fprintf(stderr, "cannot write json\n"); return 4; }
  std::fprintf(jf, "{\n  \"bench\": \"factor-update\",\n  \"nz_nonzero_deviations\": %d,\n", int(nz));
  std::fprintf(jf, "  \"group\": 0,\n  \"ops_per_window\": %d,\n  \"windows\": %d,\n", ops, windows);
  std::fprintf(jf, "  \"eager_update_ns_per_op\": %.3f,\n  \"lazy_update_ns_per_op\": %.3f,\n", eu, lu);
  std::fprintf(jf, "  \"update_speedup_lazy_vs_eager\": %.2f,\n", lu > 0 ? eu / lu : 0.0);
  std::fprintf(jf, "  \"eager_query_ns\": %.3f,\n  \"lazy_query_ns\": %.3f,\n", eq, lq);
  std::fprintf(jf, "  \"eager_exact\": %s,\n  \"lazy_exact\": %s\n}\n",
               eager_exact ? "true" : "false", lazy_exact ? "true" : "false");
  std::fclose(jf);
  std::printf("bench=factor-update nz=%d eager_update_ns=%.3f lazy_update_ns=%.3f speedup=%.2f "
              "eager_exact=%d lazy_exact=%d\n",
              int(nz), eu, lu, lu > 0 ? eu / lu : 0.0, int(eager_exact), int(lazy_exact));
  return (eager_exact && lazy_exact) ? 0 : 6;
}

int main(int argc, char** argv) {
  std::string arm_name, fixture_path, base_id, answers_path, json_path, bench;
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
    else if (!std::strcmp(argv[i], "--bench")) bench = next(argv[i]);
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  const bool bench_mode = !bench.empty();
  if (bench_mode) {
    if (fixture_path.empty() || json_path.empty()) {
      std::fprintf(stderr, "bench usage: panel_lazy --bench factor-update --fixture PATH "
                           "--base-id ID --topology T --seed S --factor-period P --q Q --json J\n");
      return 2;
    }
  } else if (arm_name.empty() || fixture_path.empty() || answers_path.empty() || json_path.empty()) {
    std::fprintf(stderr, "usage: panel --arm NAME --fixture PATH --base-id ID --topology T --seed S "
                         "--factor-period P --q Q --repeat R --answers A --json J\n");
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
  if (bench_mode) {
    if (bench != "factor-update") { std::fprintf(stderr, "unknown bench %s\n", bench.c_str()); return 2; }
    return run_factor_bench(f, w, json_path);
  }
  Arm* arm = make_arm(arm_name, f, w);
  if (!arm) { std::fprintf(stderr, "unknown arm %s\n", arm_name.c_str()); return 3; }
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

  // Snapshot the factor planner state *before* destroying the arm: the JSON record is
  // written after the cleanup phase, and dynamic_cast/reads through the freed pointer are
  // undefined behaviour (this was the original rc=139 root cause: use-after-free here).
  bool plan_ran = false;
  LazyHybridArm::PlanTrace pt{};
  const char* req_mode = "eager";
  const char* res_mode = "eager";
  if (auto* lh = dynamic_cast<LazyHybridArm*>(arm)) {
    req_mode = lh->ptrace.requested;
    res_mode = lh->resolved_mode();
    plan_ran = lh->ptrace.ran;
    pt = lh->ptrace;
  }
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
  std::fprintf(jf, "  \"arm\": \"%s\",\n  \"base_id\": \"%s\",\n", arm_name.c_str(), base_id.c_str());
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
  {
    std::fprintf(jf, "  \"factor_mode_requested\": \"%s\",\n  \"factor_mode\": \"%s\",\n",
                 req_mode, res_mode);
    if (plan_ran) {
      std::fprintf(jf,
                   "  \"factor_plan\": {\"ran\": true, \"chosen\": \"%s\", "
                   "\"eager_units\": %.0f, \"lazy_units\": %.0f, \"hybrid_units\": %.0f, "
                   "\"chosen_units\": %.0f, \"eager_dv_iters\": %" PRId64 ", "
                   "\"lazy_group_queries\": %" PRId64 ", \"n_factor_edits\": %" PRId64 ", "
                   "\"n_row_edits\": %" PRId64 ", \"factor_groups\": %d, "
                   "\"active_groups\": %d, \"heavy_groups\": %d, "
                   "\"log_m\": %.2f, \"queries\": %.0f, \"escan_units\": %" PRId64 ", "
                   "\"edv_units\": %" PRId64 ", \"rc_units\": %" PRId64 ", "
                   "\"lazy_raw\": %" PRId64 "},\n",
                   pt.chosen, pt.eager_units, pt.lazy_units, pt.hybrid_units, pt.chosen_units,
                   pt.eager_dv_iters, pt.lazy_group_queries, pt.n_factor_edits, pt.n_row_edits,
                   pt.factor_groups, pt.active_groups, pt.heavy_groups,
                   pt.log_m, pt.queries, pt.escan_units, pt.edv_units, pt.rc_units, pt.lazy_raw);
      if (std::getenv("BWH_PLANNER_TRACE")) {
        std::fprintf(stderr,
                     "PLAN requested=%s chosen=%s eager=%.0f lazy=%.0f hybrid=%.0f "
                     "chosen_units=%.0f factor_edits=%lld row_edits=%lld active_groups=%d "
                     "heavy_groups=%d dv_iters=%lld lazy_group_queries=%lld\n",
                     req_mode, pt.chosen, pt.eager_units, pt.lazy_units, pt.hybrid_units,
                     pt.chosen_units, (long long)pt.n_factor_edits, (long long)pt.n_row_edits,
                     pt.active_groups, pt.heavy_groups, (long long)pt.eager_dv_iters,
                     (long long)pt.lazy_group_queries);
      }
      if (std::getenv("BWH_PLAN_GROUPS")) {
        std::fprintf(jf, "  \"group_feats\": [");
        for (size_t k = 0; k < pt.gfeat.size(); ++k)
          std::fprintf(jf, "%s%" PRId64, (k % 5 == 0) ? (k ? "], [" : "[") : ",", pt.gfeat[k]);
        std::fprintf(jf, "]],\n");
      }
    } else {
      std::fprintf(jf, "  \"factor_plan\": {\"ran\": false},\n");
    }
  }
  std::fprintf(jf, "  \"full_ns\": %" PRId64 ",\n  \"peak_rss_kb\": %" PRId64 "\n}\n", full_ns, peak_kb);
  std::fclose(jf);
  if (std::rename((json_path + ".tmp").c_str(), json_path.c_str()) != 0) {
    std::fprintf(stderr, "cannot rename json\n");
    return 4;
  }
  std::printf("arm=%s base=%s t%d s%d f%d q%d r%d full_ns=%" PRId64 " peak_kb=%" PRId64 "\n",
              arm_name.c_str(), base_id.c_str(), topology, seed, factor_period, q, repeat, full_ns, peak_kb);
  return 0;
}
