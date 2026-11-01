// panel_adhoc_oracle.cpp — independent brute-force oracle for the mixed declared+ad-hoc load
// (new file 2026-11-01).  It reproduces the exact mixed query stream of
// `panel_deri --arm deri --adhoc-fraction F` (same draw constants/endpoint arithmetic) and answers
// every query by scanning the full current relation: no Fenwick, no class projection, no index.
// Used to certify the DERI raw-rank backend (and to prove that the scan arm's all-zero vectors are
// a capability limitation, not an answer).
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "common.hpp"
#include "batch_engine_v7c.hpp"
#include "sha256.hpp"
using namespace bwh;
static i64 nsb(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}
static Query declared_range(const BatchWorkloadV7& bw, i32 v, i32 k, bool per_state) {
  if (!bw.queries.empty()) return bw.query(v, k);
  const i64 tmin = bw.t_min, tmax = bw.t_max;
  const size_t s = per_state ? size_t(v - 1) : 0;
  const u8* d = &bw.qdraws[s * size_t(2 * bw.q)];
  i64 lo = tmin, hi = tmax + 1;
  const i64 span0 = hi - lo;
  for (i32 j = 0; j <= k; ++j) {
    i64 span = hi - lo;
    if (span <= 4096) { lo = tmin + span0 / 2; hi = lo + 2048; span = hi - lo; }
    lo += span * i64(3 + d[2 * j]) / 100;
    hi -= span * i64(3 + d[2 * j + 1]) / 100;
    if (hi - lo < 1024) hi = lo + 1024;
  }
  return Query{lo, hi};
}
int main(int argc, char** argv) {
  std::string fixture_path, base_id, answers_path, json_path;
  i32 topology = 0, seed = 7, factor_period = 37, q = 2, adhoc_pct = 0, repeat = 0;
  for (int i = 1; i < argc; ++i) {
    auto nx = [&](const char* k) -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing %s\n", k); std::exit(2); } return argv[++i]; };
    if (!std::strcmp(argv[i], "--fixture")) fixture_path = nx(argv[i]);
    else if (!std::strcmp(argv[i], "--base-id")) base_id = nx(argv[i]);
    else if (!std::strcmp(argv[i], "--topology")) topology = std::atoi(nx(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--seed")) seed = std::atoi(nx(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--factor-period")) factor_period = std::atoi(nx(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--q")) q = std::atoi(nx(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--adhoc-fraction")) adhoc_pct = std::atoi(nx(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--repeat")) repeat = std::atoi(nx(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--answers")) answers_path = nx(argv[i]);
    else if (!std::strcmp(argv[i], "--json")) json_path = nx(argv[i]);
    else { std::fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
  }
  const char* scope = std::getenv("BWH_RANGE_SCOPE");
  const bool per_state = scope && !std::strcmp(scope, "per_state");
  const auto t0 = std::chrono::steady_clock::now();
  Fixture f;
  if (!load_fixture(fixture_path, f)) { std::fprintf(stderr, "load failed\n"); return 3; }
  const auto t1 = std::chrono::steady_clock::now();
  BatchWorkloadV7 bw = make_workload_fast_v7(f, topology, seed, factor_period, q, per_state);
  const auto t2 = std::chrono::steady_clock::now();
  const i32 V = kStates - 1, qq = q;
  const i64 tmin = f.h.t_min, tmax = f.h.t_max;
  const u64 s0 = u64(seed) * 1000003ULL + u64(topology) * 7919ULL +
                 u64(factor_period) * 104729ULL + 20261025ULL;
  auto adhoc_of = [&](i32 v, i32 k) {
    return adhoc_pct > 0 && (u32(v7c_draw_at(s0, u64(0xAD0000u) + u64(v) * 8u + u64(k)) % 100u) < u32(adhoc_pct));
  };
  auto adhoc_range = [&](i32 v, i32 k, i64& lo, i64& hi) {
    const u64 a = v7c_draw_at(s0, u64(0xAD1000u) + u64(v) * 8u + u64(k));
    const u64 b = v7c_draw_at(s0, u64(0xAD2000u) + u64(v) * 8u + u64(k));
    const u64 span = u64(tmax + 1 - tmin);
    lo = tmin + i64(a % span);
    hi = lo + 1 + i64(b % (span ? span : 1));
    if (hi > tmax + 1) hi = tmax + 1;
    if (hi <= lo) hi = std::min<i64>(tmax + 1, lo + 1);
  };
  // full current relation (row states), replaying the same absolute edits
  i64 n = f.h.n;
  std::vector<i64> ct(static_cast<size_t>(n)), ca(static_cast<size_t>(n));
  std::vector<i32> cg(static_cast<size_t>(n)); std::vector<u8> cx(static_cast<size_t>(n));
  std::vector<i64> fac = f.base_factor;
  for (i64 i = 0; i < n; ++i) {
    ct[size_t(i)] = f.rows[size_t(i)].t; ca[size_t(i)] = f.rows[size_t(i)].amount;
    cg[size_t(i)] = f.rows[size_t(i)].g; cx[size_t(i)] = 1;
  }
  auto brute = [&](i64 lo, i64 hi) {
    i64 s = 0;
    for (size_t i = 0; i < ct.size(); ++i)
      if (cx[i] && ca[i] && ct[i] >= lo && ct[i] < hi) s += ca[i] * fac[size_t(cg[i])];
    return s;
  };
  std::vector<i64> answers(size_t(V) * size_t(qq), 0);
  const auto x0 = std::chrono::steady_clock::now();
  i64 n_adhoc = 0, n_decl = 0;
  for (i32 v = 1; v < kStates; ++v) {
    const Edit& e = bw.edit[size_t(v)];
    switch (e.op) {
      case OP_REPLACE: ca[size_t(e.id)] = e.val; cx[size_t(e.id)] = 1; break;
      case OP_DELETE: cx[size_t(e.id)] = 0; ca[size_t(e.id)] = 0; break;
      case OP_MOVE: ct[size_t(e.id)] = e.t; cg[size_t(e.id)] = e.g; ca[size_t(e.id)] = e.val; cx[size_t(e.id)] = 1; break;
      case OP_INSERT: { const size_t id = size_t(e.id); if (id >= ct.size()) { ct.resize(id + 1); ca.resize(id + 1); cg.resize(id + 1); cx.resize(id + 1); }
        ct[id] = e.t; cg[id] = e.g; ca[id] = e.val; cx[id] = 1; break; }
      case OP_FACTOR: fac[size_t(e.g)] = e.val; break;
      default: break;
    }
    for (i32 k = 0; k < qq; ++k) {
      i64 lo, hi;
      if (adhoc_of(v, k)) { adhoc_range(v, k, lo, hi); ++n_adhoc; }
      else { const Query qq2 = declared_range(bw, v, k, per_state); lo = qq2.lo; hi = qq2.hi; ++n_decl; }
      answers[size_t(v - 1) * size_t(qq) + size_t(k)] = brute(lo, hi);
    }
  }
  const auto x1 = std::chrono::steady_clock::now();
  const std::string sha = bwh::sha::of(answers.data(), answers.size() * sizeof(i64));
  FILE* out = std::fopen(answers_path.c_str(), "wb");
  if (!out) { std::fprintf(stderr, "write failed\n"); return 4; }
  std::fwrite(answers.data(), sizeof(i64), answers.size(), out);
  std::fclose(out);
  const auto t3 = std::chrono::steady_clock::now();
  f.release_rows();
  const auto t4 = std::chrono::steady_clock::now();
  i64 peak_kb = 0;
  { FILE* s = std::fopen("/proc/self/status", "r"); if (s) { char line[256]; while (std::fgets(line, sizeof(line), s)) { long x = 0; if (std::sscanf(line, "VmHWM: %ld kB", &x) == 1) { peak_kb = x; break; } } std::fclose(s); } }
  FILE* jf = std::fopen((json_path + ".tmp").c_str(), "w");
  if (!jf) return 4;
  std::fprintf(jf, "{\n  \"arm\": \"brute_force_oracle\",\n  \"base_id\": \"%s\",\n  \"topology\": %d,\n  \"seed\": %d,\n"
                   "  \"factor_period\": %d,\n  \"q\": %d,\n  \"repeat\": %d,\n  \"adhoc_fraction\": %d,\n"
                   "  \"n_states\": %d,\n  \"n_answers\": %zu,\n  \"n_adhoc_queries\": %" PRId64 ", \"n_declared_queries\": %" PRId64 ",\n"
                   "  \"range_scope\": \"%s\",\n  \"answers_sha256\": \"%s\",\n",
               base_id.c_str(), topology, seed, factor_period, qq, repeat, adhoc_pct, kStates,
               answers.size(), n_adhoc, n_decl, per_state ? "per_state" : "shared", sha.c_str());
  std::fprintf(jf, "  \"phases_ns\": {\"load\": %" PRId64 ", \"gen\": %" PRId64 ", \"execute\": %" PRId64
                   ", \"extract\": %" PRId64 ", \"cleanup\": %" PRId64 "},\n",
               nsb(t0, t1), nsb(t1, t2), nsb(x0, x1), nsb(t2, t3), nsb(t3, t4));
  std::fprintf(jf, "  \"full_ns\": %" PRId64 ",\n  \"peak_rss_kb\": %" PRId64 "\n}\n",
               nsb(t0, t4), peak_kb);
  std::fclose(jf);
  std::rename((json_path + ".tmp").c_str(), json_path.c_str());
  std::printf("brute_oracle base=%s adhoc=%d%% n=%lld full_ms=%.1f sha=%s\n", base_id.c_str(), adhoc_pct,
              (long long)n, double(nsb(t0, t4)) / 1e6, sha.c_str());
  return 0;
}
