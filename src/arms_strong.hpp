// arms_strong.hpp — stronger classic baselines for the Best-Paper evaluation.
//
// pgd: per-group delta index with *lazy* factor correction.  The immutable base masses are
// stored as class-major per-group prefix rows for the groups that a declared factor edit can
// touch; row edits maintain a sparse per-group delta map (current - base) keyed by endpoint
// class; factors are only scalars (no eager re-materialization).  A query is
//
//   base0[j]-base0[i] + sum_{g: fac!=s0} (fac[g]-s0[g]) * BasePrefix_g(i,j)
//                      + sum_{g: delta!=0} fac[g] * Delta_g(i,j)
//
// i.e. the same offline information as BiWorldDB, the same lazy-factor optimization, but a
// classic per-group index instead of the shared gfen/prefix-row decomposition.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "common.hpp"
#include "engine.hpp"

namespace bwh {

struct PgdArm : Arm {
  const Fixture& f;
  const Workload& w;
  i32 G = 0, C = 0, Ga = 0, qn = 0;
  bool dense = false;
  std::vector<i64> endpoints;
  std::vector<i64> s0, fac;
  std::vector<i64> base0;     // prefix over classes, size C+1
  std::vector<i64> pref;      // class-major [c*Ga + r], size C*Ga
  std::vector<i32> gpref;     // group -> pref row, -1 if never factor-touched
  std::vector<std::map<i32, i64>> dmap;  // per-group delta (current-base) by class
  std::vector<std::vector<i64>> fen;     // per-group Fenwick over classes (lazy allocation)
  std::vector<u8> dirty;
  std::vector<i32> dirty_list, dirty_pos;
  std::vector<u8> fact_on;
  std::vector<i32> fact_list, fact_pos;
  std::vector<i64> cur_t, cur_amt;
  std::vector<i32> cur_g, cur_cls;
  std::vector<u8> cur_ex;
  std::vector<i32> ql, qh;
  bool global_index = false;   // pgdg: global weighted-deviation Fenwick (eager on factor edits)
  std::vector<i64> gfw;        // global Fenwick over classes (indexed c+1)
  // class lookup (sparse): bucketed LUT over the declared endpoint set
  std::vector<i32> lut;
  i64 lut_tmin = 0, lut_width = 1;
  i64 lut_buckets = 0;
  i32 lut_shift = 0;

  PgdArm(const Fixture& fi, const Workload& wo, bool global) : f(fi), w(wo), global_index(global) {}
  const char* name() const override { return "pgd"; }

  inline i32 cls_of(i64 t) const {
    if (dense) return i32(t - f.h.t_min);
    if (lut.empty()) {
      return i32(std::upper_bound(endpoints.begin(), endpoints.end(), t) - endpoints.begin());
    }
    i64 j = (t - lut_tmin) >> lut_shift;
    if (j < 0) j = 0;
    if (j >= lut_buckets) j = lut_buckets - 1;
    i32 lo = lut[size_t(j)], hi = lut[size_t(j) + 1];
    while (lo < hi && endpoints[size_t(lo)] <= t) ++lo;
    return lo;
  }

  // dirty-set helpers
  inline void mark_dirty(i32 g, bool on) {
    if (on && !dirty[size_t(g)]) {
      dirty[size_t(g)] = 1;
      dirty_pos[size_t(g)] = i32(dirty_list.size());
      dirty_list.push_back(g);
    } else if (!on && dirty[size_t(g)]) {
      dirty[size_t(g)] = 0;
      const i32 p = dirty_pos[size_t(g)];
      const i32 last = dirty_list.back();
      dirty_list[size_t(p)] = last;
      dirty_pos[size_t(last)] = p;
      dirty_list.pop_back();
    }
  }
  inline void mark_fact(i32 g, bool on) {
    if (on && !fact_on[size_t(g)]) {
      fact_on[size_t(g)] = 1;
      fact_pos[size_t(g)] = i32(fact_list.size());
      fact_list.push_back(g);
    } else if (!on && fact_on[size_t(g)]) {
      fact_on[size_t(g)] = 0;
      const i32 p = fact_pos[size_t(g)];
      const i32 last = fact_list.back();
      fact_list[size_t(p)] = last;
      fact_pos[size_t(last)] = p;
      fact_list.pop_back();
    }
  }

  void build() override {
    G = f.h.groups;
    qn = w.q > 0 ? w.q : 1;
    const i64 N = f.h.n;
    s0 = f.base_factor;
    fac = f.base_factor;
    endpoints = w.endpoints;
    i64 U = 0;
    for (i32 v = 1; v < kStates; ++v) {
      const u8 op = w.edit[size_t(v)].op;
      if (op == OP_REPLACE || op == OP_INSERT || op == OP_DELETE || op == OP_MOVE) ++U;
    }
    const i64 span = f.h.t_max - f.h.t_min + 1;
    dense = Plan::predicate_dense(N, U, G, span);
    C = i32(dense ? span : i64(endpoints.size()) + 1);
    if (!dense && endpoints.size() >= 1024) {
      i64 want = std::max<i64>(4096, std::min<i64>(i64(1) << 20, 4 * i64(endpoints.size()) + 1));
      lut_buckets = 1;
      while (lut_buckets < want) lut_buckets <<= 1;
      lut_tmin = f.h.t_min;
      {
        const i64 target = lut_buckets;
        i64 width = std::max<i64>(1, (span + target - 1) / target);
        lut_shift = 0;
        while ((i64(1) << lut_shift) < width) ++lut_shift;
        lut_width = i64(1) << lut_shift;
        lut_buckets = (span + lut_width - 1) / lut_width;
        if (lut_buckets < 1) lut_buckets = 1;
      }
      lut.assign(size_t(lut_buckets) + 1, 0);
      size_t ep = 0;
      for (i64 j = 0; j <= lut_buckets; ++j) {
        const i64 val = lut_tmin + j * lut_width;
        while (ep < endpoints.size() && endpoints[size_t(ep)] <= val) ++ep;
        lut[size_t(j)] = i32(ep);
      }
    }
    // groups that a declared factor edit can touch need a static base prefix row
    gpref.assign(size_t(G), -1);
    for (i32 v = 1; v < kStates; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if (e.op == OP_FACTOR && gpref[size_t(e.g)] < 0) gpref[size_t(e.g)] = Ga++;
    }
    pref.assign(size_t(C + 1) * size_t(Ga), 0);
    base0.assign(size_t(C) + 1, 0);
    for (i64 i = 0; i < N; ++i) {
      const BaseRec& br = f.rows[size_t(i)];
      const i32 c = cls_of(br.t);
      base0[size_t(c) + 1] += br.amount * s0[size_t(br.g)];
      const i32 r = gpref[size_t(br.g)];
      if (r >= 0) pref[size_t(c + 1) * size_t(Ga) + size_t(r)] += br.amount;
    }
    for (i32 c = 1; c <= C; ++c) {
      i64* row = &pref[size_t(c) * size_t(Ga)];
      const i64* prev = &pref[size_t(c - 1) * size_t(Ga)];
      for (i32 r = 0; r < Ga; ++r) row[size_t(r)] += prev[size_t(r)];
    }
    for (i32 c = 1; c <= C; ++c) base0[size_t(c)] += base0[size_t(c - 1)];
    // row state
    const i64 n_rows = N + i64(w.n_inserted);
    cur_t.assign(size_t(n_rows), 0);
    cur_amt.assign(size_t(n_rows), 0);
    cur_g.assign(size_t(n_rows), 0);
    cur_cls.assign(size_t(n_rows), 0);
    cur_ex.assign(size_t(n_rows), 0);
    for (i64 i = 0; i < N; ++i) {
      const BaseRec& br = f.rows[size_t(i)];
      cur_t[size_t(i)] = br.t;
      cur_amt[size_t(i)] = br.amount;
      cur_g[size_t(i)] = br.g;
      cur_cls[size_t(i)] = cls_of(br.t);
      cur_ex[size_t(i)] = 1;
    }
    dmap.assign(size_t(G), {});
    fen.assign(size_t(G), {});
    if (global_index) gfw.assign(size_t(C) + 1, 0);
    dirty.assign(size_t(G), 0);
    dirty_pos.assign(size_t(G), -1);
    fact_on.assign(size_t(G), 0);
    fact_pos.assign(size_t(G), -1);
    // match the shipped engine's query-frame sizing: only the declared q frames are queried
    ql.assign(size_t(kStates) * size_t(qn), 0);
    qh.assign(size_t(kStates) * size_t(qn), 0);
    for (i32 v = 1; v < kStates; ++v) {
      for (i32 k = 0; k < qn; ++k) {
        const Query& qq = query_for(w, v, k);
        const size_t qidx = size_t(v) * size_t(qn) + size_t(k);
        ql[qidx] = cls_of(qq.lo);
        qh[qidx] = cls_of(qq.hi);
      }
    }
  }

  inline void delta_add(i32 g, i32 c, i64 v) {
    if (v == 0) return;
    auto& m = dmap[size_t(g)];
    auto it = m.find(c);
    if (it == m.end()) m.emplace(c, v);
    else { it->second += v; if (it->second == 0) m.erase(it); }
    auto& fv = fen[size_t(g)];
    if (fv.empty()) fv.assign(size_t(C) + 1, 0);
    for (i32 x = c + 1; x <= C; x += x & (-x)) fv[size_t(x)] += v;
    if (global_index) {
      const i64 wv = v * fac[size_t(g)];
      for (i32 x = c + 1; x <= C; x += x & (-x)) gfw[size_t(x)] += wv;
    }
    mark_dirty(g, !m.empty());
  }
  inline i64 fen_range(i32 g, i32 l, i32 r) const {
    const auto& fv = fen[size_t(g)];
    i64 s = 0;
    for (i32 x = r; x > 0; x -= x & (-x)) s += fv[size_t(x)];
    for (i32 x = l; x > 0; x -= x & (-x)) s -= fv[size_t(x)];
    return s;
  }

  void set_row(i64 id, i64 t, i32 g, i64 amount, u8 ex) {
    if (cur_ex[size_t(id)] && cur_amt[size_t(id)] != 0) {
      delta_add(cur_g[size_t(id)], cur_cls[size_t(id)], -cur_amt[size_t(id)]);
    }
    cur_t[size_t(id)] = t;
    cur_g[size_t(id)] = g;
    cur_amt[size_t(id)] = amount;
    cur_ex[size_t(id)] = ex;
    cur_cls[size_t(id)] = cls_of(t);
    if (ex && amount != 0) delta_add(g, cur_cls[size_t(id)], amount);
  }
  void set_factor(i32 g, i64 nf) {
    if (fac[size_t(g)] == nf) return;
    if (global_index) {
      const i64 ds = nf - fac[size_t(g)];
      for (const auto& kv : dmap[size_t(g)]) {
        const i64 wv = ds * kv.second;
        for (i32 x = kv.first + 1; x <= C; x += x & (-x)) gfw[size_t(x)] += wv;
      }
    }
    fac[size_t(g)] = nf;
    mark_fact(g, nf != s0[size_t(g)]);
  }

  SavedRow snapshot(const Edit& e) const override {
    SavedRow s{};
    if (e.op == OP_FACTOR) {
      s.kind = 1;
      s.g = e.g;
      s.amount = fac[size_t(e.g)];
    } else {
      s.kind = 0;
      s.id = e.id;
      s.t = cur_t[size_t(e.id)];
      s.g = cur_g[size_t(e.id)];
      s.amount = cur_amt[size_t(e.id)];
      s.ex = cur_ex[size_t(e.id)];
    }
    return s;
  }
  void apply(const Edit& e) override {
    switch (e.op) {
      case OP_REPLACE: set_row(e.id, cur_t[size_t(e.id)], cur_g[size_t(e.id)], e.val, 1); break;
      case OP_DELETE: set_row(e.id, cur_t[size_t(e.id)], cur_g[size_t(e.id)], cur_amt[size_t(e.id)], 0); break;
      case OP_INSERT: set_row(e.id, e.t, e.g, e.val, 1); break;
      case OP_MOVE: set_row(e.id, e.t, e.g, e.val, 1); break;
      case OP_FACTOR: set_factor(e.g, e.val); break;
      default: break;
    }
  }
  void undo(const Edit& e, const SavedRow& s) override {
    (void)e;
    if (s.kind == 1) set_factor(s.g, s.amount);
    else set_row(s.id, s.t, s.g, s.amount, s.ex);
  }

  i64 query(i32 state, i32 k) const override {
    const size_t qidx = size_t(state) * size_t(qn) + size_t(k);
    const i32 i = ql[qidx], j = qh[qidx];
    i64 ans = base0[size_t(j)] - base0[size_t(i)];
    for (i32 g : fact_list) {
      const i32 r = gpref[size_t(g)];
      ans += (fac[size_t(g)] - s0[size_t(g)]) *
             (pref[size_t(j) * size_t(Ga) + size_t(r)] - pref[size_t(i) * size_t(Ga) + size_t(r)]);
    }
    if (global_index) {
      i64 s = 0;
      for (i32 x = j; x > 0; x -= x & (-x)) s += gfw[size_t(x)];
      for (i32 x = i; x > 0; x -= x & (-x)) s -= gfw[size_t(x)];
      ans += s;
    } else {
      for (i32 g : dirty_list) {
        ans += fac[size_t(g)] * fen_range(g, i, j);
      }
    }
    return ans;
  }
};

inline Arm* make_strong_arm(const std::string& name, const Fixture& f, const Workload& w) {
  if (name == "pgd") return new PgdArm(f, w, false);
  if (name == "pgdg") return new PgdArm(f, w, true);
  return nullptr;
}

}  // namespace bwh
