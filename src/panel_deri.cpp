// panel_deri.cpp — mixed declared + ad-hoc endpoint workload driver for DERI (new file
// 2026-11-01).  The declared fast path (engine.hpp/panel.cpp) is not modified; DERI is an
// opt-in backend that answers arbitrary (declared or ad-hoc) endpoints exactly.
//
// CLI: same fixture/dimension flags as panel.cpp, plus
//   --arm deri|scan|rf|declared    (deri = this index; scan = brute-force fallback per ad-hoc
//                                  query; rf = eager rollback-Fenwick fallback; declared = no DERI)
//   --adhoc-fraction F             percent of queries answered with ad-hoc endpoints (0..100)
//   --rep R                        same repeat dimension as panel.cpp
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.hpp"
#include "batch_engine_v7c.hpp"
#include "deri_index.hpp"

using namespace bwh;

static i64 ns_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

// ---- eager rollback-Fenwick fallback (only for comparison): one global Fenwick over the raw
// endpoint domain with the current weighted masses; factor edits eagerly walk the group's rows.
struct RfFallback {
  const Fixture& f;
  DeriIndex di;                    // reuse the domain / char state / group lists
  std::vector<i64> fen;            // global weighted Fenwick over domain ranks
  std::vector<std::vector<i64>> rows_of_g;   // current contributing rows per group
  std::vector<i32> pos_of_row;               // row -> index in rows_of_g[g]
  i32 M = 0;
  explicit RfFallback(const Fixture& fi) : f(fi), di(fi) {}
  void build(const std::vector<Edit>& edits) {
    di.build(edits);
    M = i32(di.domain.size());
    fen.assign(size_t(M) + 1, 0);
    rows_of_g.assign(size_t(di.G), {});
    pos_of_row.assign(size_t(di.n), -1);
    for (i64 i = 0; i < di.n; ++i) {
      const BaseRec& r = f.rows[size_t(i)];
      fen_add(di.cur_rank[size_t(i)], r.amount * di.fac[size_t(r.g)]);
      pos_of_row[size_t(i)] = i32(rows_of_g[size_t(r.g)].size());
      rows_of_g[size_t(r.g)].push_back(i);
    }
  }
  inline void fen_add(i32 rank, i64 v) { for (i32 i = rank + 1; i <= M; i += i & (-i)) fen[size_t(i)] += v; }
  inline i64 fen_pref(i32 rank) const { i64 s = 0; for (i32 i = rank; i > 0; i -= i & (-i)) s += fen[size_t(i)]; return s; }
  void remove_row(i64 id) {
    const i32 g = di.cur_g[size_t(id)];
    if (!di.cur_ex[size_t(id)] || di.cur_amt[size_t(id)] == 0) return;
    fen_add(di.cur_rank[size_t(id)], -di.cur_amt[size_t(id)] * di.fac[size_t(g)]);
  }
  void add_row(i64 id) {
    const i32 g = di.cur_g[size_t(id)];
    if (di.cur_ex[size_t(id)] && di.cur_amt[size_t(id)] != 0)
      fen_add(di.cur_rank[size_t(id)], di.cur_amt[size_t(id)] * di.fac[size_t(g)]);
  }
  void apply(const Edit& e) {
    switch (e.op) {
      case OP_REPLACE: case OP_DELETE: case OP_MOVE:
        remove_row(e.id);
        if (e.op == OP_REPLACE) di.cur_amt[size_t(e.id)] = e.val;
        else if (e.op == OP_DELETE) { di.cur_ex[size_t(e.id)] = 0; di.cur_amt[size_t(e.id)] = 0; }
        else {
          const i32 go = di.cur_g[size_t(e.id)];
          auto& v = rows_of_g[size_t(go)];
          const i32 p = pos_of_row[size_t(e.id)];
          if (p >= 0) {
            const i64 last = v.back();
            v[size_t(p)] = last; pos_of_row[size_t(last)] = p; v.pop_back();
            pos_of_row[size_t(e.id)] = -1;
          }
          di.cur_t[size_t(e.id)] = e.t; di.cur_g[size_t(e.id)] = e.g;
          di.cur_amt[size_t(e.id)] = e.val; di.cur_rank[size_t(e.id)] = di.rank_of(e.t);
          pos_of_row[size_t(e.id)] = i32(rows_of_g[size_t(e.g)].size());
          rows_of_g[size_t(e.g)].push_back(e.id);
        }
        add_row(e.id);
        break;
      case OP_INSERT: {
        const size_t id = size_t(e.id);
        if (id >= di.cur_t.size()) {
          di.cur_t.resize(id + 1); di.cur_amt.resize(id + 1); di.cur_g.resize(id + 1);
          di.cur_rank.resize(id + 1); di.cur_ex.resize(id + 1); di.base_amt.resize(id + 1, 0);
          pos_of_row.resize(id + 1, -1);
        }
        di.cur_t[id] = e.t; di.cur_g[id] = e.g; di.cur_amt[id] = e.val; di.cur_ex[id] = 1;
        di.cur_rank[id] = di.rank_of(e.t);
        pos_of_row[id] = i32(rows_of_g[size_t(e.g)].size());
        rows_of_g[size_t(e.g)].push_back(i64(id));
        add_row(e.id);
        break;
      }
      case OP_FACTOR: {
        const i64 old = di.fac[size_t(e.g)];
        const i64 d = e.val - old;
        if (d != 0) {
          for (i64 id : rows_of_g[size_t(e.g)]) {
            const i64 amt = di.cur_amt[size_t(id)];
            if (amt != 0) fen_add(di.cur_rank[size_t(id)], d * amt);
          }
          di.fac[size_t(e.g)] = e.val;
        }
        break;
      }
      default: break;
    }
  }
  inline i64 query(i64 lo, i64 hi) const {
    if (hi <= lo) return 0;
    return fen_pref(di.rank_of(hi)) - fen_pref(di.rank_of(lo));
  }
};

int main(int argc, char** argv) {
  std::string arm_name = "deri", fixture_path, base_id, answers_path, json_path;
  i32 topology = 0, seed = 7, factor_period = 37, q = 2, repeat = 0, adhoc_pct = 0, pool_k = 4096;
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
    else if (!std::strcmp(argv[i], "--adhoc-fraction")) adhoc_pct = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--pool")) pool_k = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--answers")) answers_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--json")) json_path = next(argv[i]);
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  if (fixture_path.empty() || answers_path.empty() || json_path.empty()) {
    std::fprintf(stderr, "usage: panel_deri --arm deri|scan|rf|declared --fixture PATH --base-id ID "
                         "--topology T --seed S --factor-period P --q Q --repeat R "
                         "--adhoc-fraction F --answers A --json J\n");
    return 2;
  }
  if (topology != 0) { std::fprintf(stderr, "panel_deri currently supports chain topology only\n"); return 2; }
  bool adapt_unsupported = false;
  if (arm_name == "adapt") {
    // adaptive policy: declared-only -> union/declared batch; repeated limited ad-hoc endpoints
    // (pool <= 2048) -> union fast path; unique/large ad-hoc -> not supportable in this prototype
    // (scan is O(n)/query, RF build is ~2x the declared path: documented crossover).
    if (adhoc_pct == 0 || pool_k <= 2048) arm_name = "union";
    else { adapt_unsupported = true; arm_name = "union"; }
  }
  const char* scope_env = std::getenv("BWH_RANGE_SCOPE");
  const bool per_state = scope_env && std::strcmp(scope_env, "per_state") == 0;

  const auto t0 = std::chrono::steady_clock::now();
  Fixture f;
  if (!load_fixture(fixture_path, f)) { std::fprintf(stderr, "cannot load fixture\n"); return 3; }
  const auto t1 = std::chrono::steady_clock::now();
  BatchWorkloadV7 bw = make_workload_fast_v7(f, topology, seed, factor_period, q, per_state);
  const auto t2 = std::chrono::steady_clock::now();

  const i32 V = kStates - 1;
  const i64 tmin = f.h.t_min, tmax = f.h.t_max;
  const u64 s0 = u64(seed) * 1000003ULL + u64(topology) * 7919ULL + u64(factor_period) * 104729ULL +
                 20261025ULL;
  // deterministic ad-hoc decision + endpoints per (state, k)
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

  auto declared_range = [&](i32 v, i32 k, i64& lo, i64& hi) {
    const size_t s = per_state ? size_t(v - 1) : 0;
    const u8* d = &bw.qdraws[s * size_t(2 * q)];
    lo = tmin; hi = tmax + 1;
    const i64 span0 = hi - lo;
    for (i32 j = 0; j <= k; ++j) {
      i64 span = hi - lo;
      if (span <= 4096) { lo = tmin + span0 / 2; hi = lo + 2048; span = hi - lo; }
      lo += span * i64(3 + d[2 * j]) / 100;
      hi -= span * i64(3 + d[2 * j + 1]) / 100;
      if (hi - lo < 1024) hi = lo + 1024;
    }
  };
  std::vector<i64> answers(size_t(V) * size_t(q), 0);
  i64 build_ns = 0, execute_ns = 0, first_query_ns = -1, adhoc_query_ns = 0, adhoc_query_cnt = 0;
  i64 union_mismatch = 0, union_ncls = 0, union_nend = 0;

  if (arm_name == "deri" || arm_name == "declared") {
    DeriIndex di(f);
    auto b0 = std::chrono::steady_clock::now();
    di.build(bw.edit);
    auto b1 = std::chrono::steady_clock::now();
    build_ns = ns_between(b0, b1);
    auto x0 = std::chrono::steady_clock::now();
    for (i32 v = 1; v < kStates; ++v) {
      di.apply(bw.edit[size_t(v)]);
      for (i32 k = 0; k < q; ++k) {
        i64 lo, hi;
        if (adhoc_of(v, k)) {
          adhoc_range(v, k, lo, hi);
          const auto q0 = std::chrono::steady_clock::now();
          answers[size_t(v - 1) * size_t(q) + size_t(k)] = di.query(lo, hi);
          const auto q1 = std::chrono::steady_clock::now();
          if (first_query_ns < 0) first_query_ns = ns_between(b1, q1);
          adhoc_query_ns += ns_between(q0, q1); ++adhoc_query_cnt;
        } else {
          declared_range(v, k, lo, hi);
          answers[size_t(v - 1) * size_t(q) + size_t(k)] = di.query(lo, hi);
        }
      }
    }
    auto x1 = std::chrono::steady_clock::now();
    execute_ns = ns_between(x0, x1);
  } else if (arm_name == "union") {
    // adaptive fast path: the mixed query set is known up front, so fold the ad-hoc endpoints
    // into the declared endpoint set and run the declared batch engine on the union.  Exact
    // (class projection is exact when every query endpoint is a class boundary) as long as the
    // union class count stays inside the batch gates; otherwise the engine reports unsupported
    // and the adaptive backend must fall back to scan/RF.
    const u64 s0u = u64(seed) * 1000003ULL + u64(topology) * 7919ULL + u64(factor_period) * 104729ULL + 20261025ULL;
    std::vector<i64> pool(size_t(std::max(2, pool_k)));
    for (size_t i = 0; i < pool.size(); ++i)
      pool[i] = tmin + i64(v7c_draw_at(s0u, u64(0xAB0000u) + u64(i)) % u64(std::max<i64>(1, tmax + 1 - tmin)));
    std::sort(pool.begin(), pool.end());
    std::vector<Query> qs(size_t(kStates) * size_t(q));
    for (i32 v = 1; v < kStates; ++v)
      for (i32 k = 0; k < q; ++k) {
        i64 lo, hi;
        if (adhoc_of(v, k)) {
          const u64 a = v7c_draw_at(s0u, u64(0xAC0000u) + u64(v) * 8u + u64(k));
          const u64 b = v7c_draw_at(s0u, u64(0xAD0000u) + u64(v) * 8u + u64(k));
          const i64 i0 = i64(a % u64(pool.size()));
          const i64 i1 = i64(b % u64(pool.size()));
          lo = std::min(pool[size_t(i0)], pool[size_t(i1)]);
          hi = std::max(pool[size_t(i0)], pool[size_t(i1)]);
          if (hi <= lo) hi = lo + 1;
        } else {
          declared_range(v, k, lo, hi);
        }
        qs[size_t(v) * size_t(q) + size_t(k)] = Query{lo, hi};
      }
    bw.qdraws.clear();
    bw.queries = std::move(qs);
    {
      std::vector<i64> ue;
      ue.reserve(size_t(kStates) * size_t(q) * 2);
      for (const Query& qq : bw.queries) { ue.push_back(qq.lo); ue.push_back(qq.hi); }
      std::sort(ue.begin(), ue.end());
      ue.erase(std::unique(ue.begin(), ue.end()), ue.end());
      bw.endpoints = std::move(ue);
    }
    BatchEngineV7 eng(f, bw);
    auto b0 = std::chrono::steady_clock::now();
    const bool ok = eng.build();
    auto b1 = std::chrono::steady_clock::now();
    build_ns = ns_between(b0, b1);
    if (!ok) {
      std::fprintf(stderr, "union fast path unsupported (class-count gate): C=%d\n", eng.ncls);
    } else {
      auto x0 = std::chrono::steady_clock::now();
      eng.run(answers);
      execute_ns = ns_between(x0, std::chrono::steady_clock::now());
    }
    first_query_ns = -1;
    // brute-force verification on a sample of states (exact mixed answers, not just declared)
    i64 mismatch = 0;
    const bool verify = std::getenv("BWH_VERIFY") != nullptr;
    if (ok && verify) {
      std::vector<i64> ct(size_t(f.h.n)), ca(size_t(f.h.n));
      std::vector<i32> cg(size_t(f.h.n)); std::vector<u8> cx(size_t(f.h.n));
      std::vector<i64> fac = f.base_factor;
      for (i64 i = 0; i < f.h.n; ++i) {
        ct[size_t(i)] = f.rows[size_t(i)].t; ca[size_t(i)] = f.rows[size_t(i)].amount;
        cg[size_t(i)] = f.rows[size_t(i)].g; cx[size_t(i)] = 1;
      }
      auto brute = [&](i64 lo, i64 hi) {
        i64 s = 0;
        for (size_t i = 0; i < ct.size(); ++i)
          if (cx[i] && ca[i] && ct[i] >= lo && ct[i] < hi) s += ca[i] * fac[size_t(cg[i])];
        return s;
      };
      for (i32 v = 1; v < kStates; ++v) {
        const Edit& e = bw.edit[size_t(v)];
        switch (e.op) {
          case OP_REPLACE: ca[size_t(e.id)] = e.val; cx[size_t(e.id)] = 1; break;
          case OP_DELETE: cx[size_t(e.id)] = 0; ca[size_t(e.id)] = 0; break;
          case OP_MOVE: ct[size_t(e.id)] = e.t; cg[size_t(e.id)] = e.g; ca[size_t(e.id)] = e.val; cx[size_t(e.id)] = 1; break;
          case OP_INSERT: { const size_t id = size_t(e.id); if (id >= ct.size()) { ct.resize(id+1); ca.resize(id+1); cg.resize(id+1); cx.resize(id+1); }
                            ct[id] = e.t; cg[id] = e.g; ca[id] = e.val; cx[id] = 1; break; }
          case OP_FACTOR: fac[size_t(e.g)] = e.val; break;
          default: break;
        }
        if (v % 4096 == 0 || v == kStates - 1) {
          for (i32 k = 0; k < q; ++k) {
            const Query& qq = bw.queries[size_t(v) * size_t(q) + size_t(k)];
            if (brute(qq.lo, qq.hi) != answers[size_t(v - 1) * size_t(q) + size_t(k)]) ++mismatch;
          }
        }
      }
    }
    union_mismatch = mismatch;
    union_ncls = ok ? eng.ncls : 0;
    union_nend = i64(bw.endpoints.size());
  } else if (arm_name == "scan") {
    // brute-force fallback: no index; maintain row states, scan all rows per ad-hoc query
    DeriIndex di(f);
    auto b0 = std::chrono::steady_clock::now();
    std::vector<i64> cur_t(size_t(f.h.n)), cur_amt(size_t(f.h.n));
    std::vector<i32> cur_g(size_t(f.h.n)); std::vector<u8> cur_ex(size_t(f.h.n));
    for (i64 i = 0; i < f.h.n; ++i) {
      cur_t[size_t(i)] = f.rows[size_t(i)].t; cur_amt[size_t(i)] = f.rows[size_t(i)].amount;
      cur_g[size_t(i)] = f.rows[size_t(i)].g; cur_ex[size_t(i)] = 1;
    }
    std::vector<i64> fac = f.base_factor;
    const auto b1 = std::chrono::steady_clock::now();
    build_ns = ns_between(b0, b1);
    auto x0 = std::chrono::steady_clock::now();
    auto scan_q = [&](i64 lo, i64 hi) {
      i64 s = 0;
      const i64 n = i64(cur_t.size());
      for (i64 i = 0; i < n; ++i)
        if (cur_ex[size_t(i)] && cur_amt[size_t(i)] != 0 && cur_t[size_t(i)] >= lo && cur_t[size_t(i)] < hi)
          s += cur_amt[size_t(i)] * fac[size_t(cur_g[size_t(i)])];
      return s;
    };
    for (i32 v = 1; v < kStates; ++v) {
      const Edit& e = bw.edit[size_t(v)];
      switch (e.op) {
        case OP_REPLACE: cur_amt[size_t(e.id)] = e.val; break;
        case OP_DELETE: cur_ex[size_t(e.id)] = 0; cur_amt[size_t(e.id)] = 0; break;
        case OP_MOVE: cur_t[size_t(e.id)] = e.t; cur_g[size_t(e.id)] = e.g; cur_amt[size_t(e.id)] = e.val; break;
        case OP_INSERT: {
          const size_t id = size_t(e.id);
          if (id >= cur_t.size()) { cur_t.resize(id + 1); cur_amt.resize(id + 1); cur_g.resize(id + 1); cur_ex.resize(id + 1); }
          cur_t[id] = e.t; cur_g[id] = e.g; cur_amt[id] = e.val; cur_ex[id] = 1;
          break;
        }
        case OP_FACTOR: fac[size_t(e.g)] = e.val; break;
        default: break;
      }
      for (i32 k = 0; k < q; ++k) {
        i64 lo, hi;
        if (adhoc_of(v, k)) {
          adhoc_range(v, k, lo, hi);
          const auto q0 = std::chrono::steady_clock::now();
          answers[size_t(v - 1) * size_t(q) + size_t(k)] = scan_q(lo, hi);
          const auto q1 = std::chrono::steady_clock::now();
          if (first_query_ns < 0) first_query_ns = ns_between(b1, q1);
          adhoc_query_ns += ns_between(q0, q1); ++adhoc_query_cnt;
          if (adhoc_query_cnt >= 3) { v = kStates; break; }  // brute force: sample a few queries only
        } else {
          declared_range(v, k, lo, hi);
          answers[size_t(v - 1) * size_t(q) + size_t(k)] = scan_q(lo, hi);
        }
      }
      if (adhoc_query_cnt >= 3) break;
    }
    execute_ns = ns_between(x0, std::chrono::steady_clock::now());
    (void)di;
  } else if (arm_name == "rf") {
    RfFallback rf(f);
    auto b0 = std::chrono::steady_clock::now();
    rf.build(bw.edit);   // domain + base index; the eager factor replay is in the execute loop
    auto b1 = std::chrono::steady_clock::now();
    build_ns = ns_between(b0, b1);
    auto x0 = std::chrono::steady_clock::now();
    for (i32 v = 1; v < kStates; ++v) {
      rf.apply(bw.edit[size_t(v)]);
      for (i32 k = 0; k < q; ++k) {
        i64 lo, hi;
        if (adhoc_of(v, k)) { adhoc_range(v, k, lo, hi); }
        else { declared_range(v, k, lo, hi); }
        const auto q0 = std::chrono::steady_clock::now();
        answers[size_t(v - 1) * size_t(q) + size_t(k)] = rf.query(lo, hi);
        const auto q1 = std::chrono::steady_clock::now();
        if (adhoc_of(v, k)) {
          if (first_query_ns < 0) first_query_ns = ns_between(b1, q1);
          adhoc_query_ns += ns_between(q0, q1); ++adhoc_query_cnt;
        }
      }
    }
    execute_ns = ns_between(x0, std::chrono::steady_clock::now());
  } else {
    std::fprintf(stderr, "unknown arm %s\n", arm_name.c_str());
    return 2;
  }

  const auto t3 = std::chrono::steady_clock::now();
  FILE* out = std::fopen(answers_path.c_str(), "wb");
  if (!out) { std::fprintf(stderr, "cannot write answers\n"); return 4; }
  std::fwrite(answers.data(), sizeof(i64), answers.size(), out);
  std::fclose(out);
  const auto t4 = std::chrono::steady_clock::now();
  f.release_rows();
  const auto t5 = std::chrono::steady_clock::now();
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
  const i64 gen_ns = ns_between(t1, t2);
  const i64 setup_ns = ns_between(t1, t3) - execute_ns;
  const i64 extract_ns = ns_between(t3, t4);
  const i64 cleanup_ns = ns_between(t4, t5);
  const i64 full_ns = ns_between(t0, t5);
  FILE* jf = std::fopen((json_path + ".tmp").c_str(), "w");
  if (!jf) { std::fprintf(stderr, "cannot write json\n"); return 4; }
  std::fprintf(jf, "{\n  \"arm\": \"%s\",\n  \"base_id\": \"%s\",\n  \"topology\": %d,\n  \"seed\": %d,\n"
                   "  \"factor_period\": %d,\n  \"q\": %d,\n  \"repeat\": %d,\n  \"adhoc_fraction\": %d,\n"
                   "  \"n_states\": %d,\n  \"answers_n\": %zu,\n  \"range_scope\": \"%s\",\n",
               arm_name.c_str(), base_id.c_str(), topology, seed, factor_period, q, repeat, adhoc_pct,
               kStates, answers.size(), per_state ? "per_state" : "shared");
  std::fprintf(jf, "  \"phases_ns\": {\"load\": %" PRId64 ", \"setup\": %" PRId64 ", \"execute\": %" PRId64
                   ", \"extract\": %" PRId64 ", \"cleanup\": %" PRId64 "},\n",
               load_ns, setup_ns, execute_ns, extract_ns, cleanup_ns);
  std::fprintf(jf, "  \"deri\": {\"build_ns\": %" PRId64 ", \"first_query_ns\": %" PRId64
                   ", \"adhoc_query_total_ns\": %" PRId64 ", \"adhoc_query_count\": %" PRId64
                   ", \"per_adhoc_query_ns\": %" PRId64 ", \"gen_ns\": %" PRId64 "},\n",
               build_ns, first_query_ns, adhoc_query_ns, adhoc_query_cnt,
               adhoc_query_cnt ? adhoc_query_ns / adhoc_query_cnt : 0, gen_ns);
  std::fprintf(jf, "  \"pool\": %d,\n  \"adapt_unsupported\": %d,\n  \"union\": {\"ncls\": %" PRId64 ", \"n_endpoints\": %" PRId64 ", \"brute_mismatch\": %" PRId64 "},\n"
                   "  \"full_ns\": %" PRId64 ",\n  \"peak_rss_kb\": %" PRId64 "\n}\n",
               pool_k, int(adapt_unsupported), union_ncls, union_nend, union_mismatch, full_ns, peak_kb);
  std::fclose(jf);
  if (std::rename((json_path + ".tmp").c_str(), json_path.c_str()) != 0) return 4;
  std::printf("arm=%s base=%s adhoc=%d%% build_ms=%.2f first_query_ms=%.2f per_q_us=%.2f full_ns=%" PRId64
              " peak_kb=%" PRId64 "\n",
              arm_name.c_str(), base_id.c_str(), adhoc_pct, double(build_ns) / 1e6,
              first_query_ns < 0 ? -1.0 : double(first_query_ns) / 1e6,
              adhoc_query_cnt ? double(adhoc_query_ns) / 1e3 / double(adhoc_query_cnt) : 0.0,
              full_ns, peak_kb);
  return 0;
}
