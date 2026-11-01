// panel_adhoc.cpp — dual-path execution: declared-range hybrid path plus a lazily built
// exact fallback for ad-hoc ranges whose endpoints were never declared.
//
// The hybrid arm answers the declared workload exactly as usual. The first ad-hoc range (an
// arbitrary [lo,hi) with endpoints outside the declared set) triggers a one-time construction of
// a rollback-Fenwick fallback over the current-state relation; subsequent ad-hoc ranges are
// answered by that fallback without replanning and without touching the declared path. The panel
// records the fallback construction time, the first and subsequent ad-hoc query latencies, the
// declared-path timing, and brute-force exactness checks on a sample of ad-hoc queries.
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <sys/resource.h>

#include "arms_classic.hpp"
#include "common.hpp"
#include "engine.hpp"
#include "sha256.hpp"

using namespace bwh;

static i64 ns_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}
static bool write_all(FILE* fp, const void* p, size_t n) { return std::fwrite(p, 1, n, fp) == n; }

int main(int argc, char** argv) {
  std::string fixture_path, base_id, answers_path, json_path, fallback_mode = "rf";
  i32 topology = 0, seed = 7, factor_period = 37, q = 8, repeat = 0;
  i32 adhoc_per_state = 0, brute_states = 0;
  u64 adhoc_seed = 0x9e3779b97f4a7c15ull;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](const char* k) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing %s\n", k); std::exit(2); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--fixture")) fixture_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--base-id")) base_id = next(argv[i]);
    else if (!std::strcmp(argv[i], "--topology")) topology = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--seed")) seed = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--factor-period")) factor_period = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--q")) q = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--repeat")) repeat = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--answers")) answers_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--json")) json_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--adhoc-per-state")) adhoc_per_state = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--brute-states")) brute_states = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--fallback")) fallback_mode = next(argv[i]);
    else { std::fprintf(stderr, "unknown %s\n", argv[i]); return 2; }
  }
  if (fixture_path.empty() || answers_path.empty() || json_path.empty()) {
    std::fprintf(stderr, "usage: panel_adhoc --fixture F --base-id B --topology T --seed S "
                         "--factor-period P --q Q --answers A --json J "
                         "--adhoc-per-state K --brute-states N\n");
    return 2;
  }
  const char* scope_env = std::getenv("BWH_RANGE_SCOPE");
  const bool per_state = scope_env && !std::strcmp(scope_env, "per_state");
  const auto t0 = std::chrono::steady_clock::now();
  Fixture f;
  if (!load_fixture(fixture_path, f)) { std::fprintf(stderr, "cannot load fixture\n"); return 3; }
  const auto t1 = std::chrono::steady_clock::now();
  Workload w = make_workload(f, topology, seed, factor_period, q, per_state);
  Arm* hybrid = make_arm("hybrid", f, w);
  if (!hybrid) { std::fprintf(stderr, "cannot create hybrid arm\n"); return 3; }
  hybrid->build();
  const auto t2 = std::chrono::steady_clock::now();

  const i64 N = f.h.n;
  const i64 n_rows = N + i64(w.n_inserted);
  std::vector<i64> bt(size_t(n_rows), 0), ba(size_t(n_rows), 0);
  std::vector<i32> bg(size_t(n_rows), 0);
  std::vector<u8> be(size_t(n_rows), 0);
  std::vector<i64> bfac = f.base_factor;
  for (i64 i = 0; i < N; ++i) {
    bt[size_t(i)] = f.rows[size_t(i)].t;
    ba[size_t(i)] = f.rows[size_t(i)].amount;
    bg[size_t(i)] = f.rows[size_t(i)].g;
    be[size_t(i)] = 1;
  }
  // brute-force snapshots along the current path
  std::vector<i64> sb_t(static_cast<size_t>(kStates)), sb_a(static_cast<size_t>(kStates));
  std::vector<i32> sb_g(static_cast<size_t>(kStates));
  std::vector<u8> sb_e(static_cast<size_t>(kStates));
  auto brute_apply = [&](i32 v) {
    const Edit& e = w.edit[size_t(v)];
    if (e.op == OP_FACTOR) { sb_g[size_t(v)] = e.g; sb_a[size_t(v)] = bfac[size_t(e.g)];
      bfac[size_t(e.g)] = e.val; return; }
    sb_t[size_t(v)] = bt[size_t(e.id)]; sb_a[size_t(v)] = ba[size_t(e.id)];
    sb_g[size_t(v)] = bg[size_t(e.id)]; sb_e[size_t(v)] = be[size_t(e.id)];
    if (e.op == OP_REPLACE) { ba[size_t(e.id)] = e.val; be[size_t(e.id)] = 1; }
    else if (e.op == OP_DELETE) be[size_t(e.id)] = 0;
    else if (e.op == OP_INSERT) { bt[size_t(e.id)] = e.t; bg[size_t(e.id)] = e.g; ba[size_t(e.id)] = e.val; be[size_t(e.id)] = 1; }
    else if (e.op == OP_MOVE) { bt[size_t(e.id)] = e.t; bg[size_t(e.id)] = e.g; ba[size_t(e.id)] = e.val; be[size_t(e.id)] = 1; }
  };
  auto brute_undo = [&](i32 v) {
    const Edit& e = w.edit[size_t(v)];
    if (e.op == OP_FACTOR) { bfac[size_t(e.g)] = sb_a[size_t(v)]; return; }
    const i64 id = e.id;
    bt[size_t(id)] = sb_t[size_t(v)]; ba[size_t(id)] = sb_a[size_t(v)];
    bg[size_t(id)] = sb_g[size_t(v)]; be[size_t(id)] = sb_e[size_t(v)];
  };
  auto brute_sum = [&](i64 lo, i64 hi) {
    i64 s = 0;
    for (i64 id = 0; id < n_rows; ++id)
      if (be[size_t(id)] && bt[size_t(id)] >= lo && bt[size_t(id)] < hi)
        s += ba[size_t(id)] * bfac[size_t(bg[size_t(id)])];
    return s;
  };

  std::vector<i64> answers(size_t(kStates - 1) * size_t(q), 0);
  std::vector<SavedRow> saved_h(static_cast<size_t>(kStates)), saved_f(static_cast<size_t>(kStates));
  std::vector<std::vector<i32>> kids(static_cast<size_t>(kStates));
  for (i32 v = 1; v < kStates; ++v) kids[size_t(w.parent[size_t(v)])].push_back(v);
  std::vector<i32> path; path.reserve(size_t(kStates));
  ClassicRfArm* fb = nullptr;
  i64 fb_build_ns = 0, adhoc_ns = 0, first_adhoc_ns = -1, adhoc_count = 0, brute_checked = 0, brute_bad = 0;
  u64 adhoc_checksum = 1469598103934665603ull;
  std::mt19937_64 rng(adhoc_seed ^ (u64(repeat) * 0x9e3779b97f4a7c15ull));
  std::uniform_int_distribution<i64> pick(f.h.t_min, f.h.t_max);
  struct Ev { i32 v; u8 exit; };
  std::vector<Ev> st; st.reserve(size_t(kStates) + 8); st.push_back({0, 0});
  while (!st.empty()) {
    const Ev ev = st.back(); st.pop_back();
    const i32 v = ev.v;
    if (ev.exit) {
      if (v > 0) {
        const Edit& e = w.edit[size_t(v)];
        hybrid->undo(e, saved_h[size_t(v)]);
        if (fb) fb->undo(e, saved_f[size_t(v)]);
        brute_undo(v);
        path.pop_back();
      }
      continue;
    }
    if (v > 0) {
      const Edit& e = w.edit[size_t(v)];
      path.push_back(v);
      saved_h[size_t(v)] = hybrid->snapshot(e);
      hybrid->apply(e);
      brute_apply(v);
      if (fb) { saved_f[size_t(v)] = fb->snapshot(e); fb->apply(e); }
      for (i32 qi = 0; qi < q; ++qi)
        answers[size_t(v - 1) * size_t(q) + size_t(qi)] = hybrid->query(v, qi);
      if (adhoc_per_state > 0 && (v % 7 == 0 || v == kStates - 1)) {
        const auto ta = std::chrono::steady_clock::now();
        const bool use_rf = (fallback_mode == "rf" || fallback_mode == "auto");
        if (use_rf && !fb) {
          fb = new ClassicRfArm(f, w);
          fb->build();
          for (i32 idx : path) { saved_f[size_t(idx)] = fb->snapshot(w.edit[size_t(idx)]); fb->apply(w.edit[size_t(idx)]); }
          fb_build_ns = ns_between(ta, std::chrono::steady_clock::now());
        }
        for (i32 k = 0; k < adhoc_per_state; ++k) {
          i64 a = pick(rng), b = pick(rng);
          i64 lo = std::min(a, b), hi = std::max(a, b);
          if (lo == hi) hi = std::min<i64>(hi + 1, f.h.t_max);
          if (lo >= hi) continue;
          const auto tq0 = std::chrono::steady_clock::now();
          i64 ans;
          if (use_rf) ans = fb->range_sum(fb->rank_of(lo), fb->rank_of(hi));
          else ans = brute_sum(lo, hi);  // generic scan fallback: exact, no index build
          const i64 dt = ns_between(tq0, std::chrono::steady_clock::now());
          adhoc_ns += dt; ++adhoc_count;
          if (first_adhoc_ns < 0) first_adhoc_ns = dt;
          adhoc_checksum = (adhoc_checksum ^ u64(ans)) * 1099511628211ull;
          if (brute_states > 0 && (v % 512 == 0) && k < 2 && use_rf) {
            ++brute_checked;
            if (brute_sum(lo, hi) != ans) ++brute_bad;
          }
        }
      }
      st.push_back({v, 1});
    }
    for (auto it = kids[size_t(v)].rbegin(); it != kids[size_t(v)].rend(); ++it) st.push_back({*it, 0});
  }
  const auto t3 = std::chrono::steady_clock::now();

  FILE* out = std::fopen(answers_path.c_str(), "wb");
  if (!out) { std::fprintf(stderr, "cannot write answers\n"); return 4; }
  const bool ok = write_all(out, answers.data(), answers.size() * sizeof(i64));
  std::fclose(out);
  if (!ok) { std::fprintf(stderr, "short answer write\n"); return 4; }
  const auto t4 = std::chrono::steady_clock::now();
  const u64 cks = fnv1a(answers.data(), answers.size() * sizeof(i64));
  const std::string sha = sha::of(answers.data(), answers.size() * sizeof(i64));
  delete hybrid;
  delete fb;
  f.release_rows();
  const auto t5 = std::chrono::steady_clock::now();
  const i64 load_ns = ns_between(t0, t1), setup_ns = ns_between(t1, t2), exec_ns = ns_between(t2, t3);
  const i64 extract_ns = ns_between(t3, t4), cleanup_ns = ns_between(t4, t5);
  const i64 full_ns = load_ns + setup_ns + exec_ns + extract_ns + cleanup_ns;
  const i64 total_with_fallback_ns = full_ns + fb_build_ns;
  FILE* jf = std::fopen((json_path + ".tmp").c_str(), "w");
  if (!jf) { std::fprintf(stderr, "cannot write json\n"); return 4; }
  std::fprintf(jf, "{\n  \"base_id\": \"%s\", \"topology\": %d, \"seed\": %d, \"factor_period\": %d, \"q\": %d,\n",
               base_id.c_str(), topology, seed, factor_period, q);
  std::fprintf(jf, "  \"n_answers\": %" PRId64 ", \"answers_fn\": \"%016" PRIx64 "\", \"answers_sha256\": \"%s\",\n",
               i64(answers.size()), cks, sha.c_str());
  std::fprintf(jf, "  \"phases_ns\": {\"load\": %" PRId64 ", \"setup\": %" PRId64 ", \"execute\": %" PRId64
                   ", \"extract\": %" PRId64 ", \"cleanup\": %" PRId64 "},\n", load_ns, setup_ns, exec_ns, extract_ns, cleanup_ns);
  std::fprintf(jf, "  \"full_ns\": %" PRId64 ",\n", full_ns);
  std::fprintf(jf, "  \"fallback_mode\": \"%s\", \"adhoc_per_state\": %d, \"adhoc_count\": %" PRId64 ", \"adhoc_ns\": %" PRId64
                   ", \"adhoc_first_query_ns\": %" PRId64 ", \"fallback_build_ns\": %" PRId64
                   ", \"total_with_fallback_ns\": %" PRId64 ",\n",
               fallback_mode.c_str(), adhoc_per_state, adhoc_count, adhoc_ns, first_adhoc_ns, fb_build_ns, total_with_fallback_ns);
  std::fprintf(jf, "  \"adhoc_median_query_ns\": %" PRId64 ",\n", adhoc_count ? adhoc_ns / adhoc_count : 0);
  std::fprintf(jf, "  \"brute_checked\": %" PRId64 ", \"brute_mismatches\": %" PRId64 ",\n", brute_checked, brute_bad);
  std::fprintf(jf, "  \"adhoc_checksum\": \"%016" PRIx64 "\"\n}\n", adhoc_checksum);
  std::fclose(jf);
  if (std::rename((json_path + ".tmp").c_str(), json_path.c_str()) != 0) return 4;
  std::printf("base=%s t%d s%d f%d q%d adhoc=%" PRId64 " full_ns=%" PRId64 " fallback_build_ms=%.2f "
              "total_with_fallback_ms=%.2f first_adhoc_us=%.2f median_adhoc_us=%.2f brute=%" PRId64 "/%" PRId64 "\n",
              base_id.c_str(), topology, seed, factor_period, q, adhoc_count, full_ns,
              double(fb_build_ns) / 1e6, double(total_with_fallback_ns) / 1e6,
              double(first_adhoc_ns) / 1e3,
              adhoc_count ? double(adhoc_ns) / double(adhoc_count) / 1e3 : 0.0,
              brute_checked - brute_bad, brute_checked);
  return 0;
}
