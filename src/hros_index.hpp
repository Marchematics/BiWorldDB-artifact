// hros_index.hpp — exact holistic order-statistics backend (HROS), prototype for BiWorldDB.
//
// New file (2026-11-01); the shipped engine/panel/oracle are untouched.
//
// Semantics (documented in results/hros_20261101/NOTE.md):
//   The relation is a set of active rows, each carrying a raw integer amount (the measure).
//   A version is a state of the relation reached from its parent by one edit (row replace /
//   insert / delete / move, or a group-factor update).  A query is a 1D half-open range
//   [lo,hi) over the MEASURE axis; HROS returns, for the current version:
//       cnt       number of active rows whose amount lies in [lo,hi)
//       sum       sum of those amounts (i64)
//       median    lower median of those amounts (0 when cnt==0)
//       distinct  number of distinct amounts in that range
//   This is the axis the shipped declared-prefix backend cannot answer at all (it answers
//   additive sums over TIME ranges); holistic statistics cannot be obtained by subtracting
//   prefix aggregates.
//
// Structure:
//   * value universe: all base amounts plus every absolute amount written by a row edit,
//     coordinate-compressed to ranks; base rows are sorted by rank.
//   * reversible treap keyed by value rank, one node per distinct rank, with multiplicity,
//     subtree count and subtree sum; insert/delete are O(log sigma) expected (deterministic
//     priorities derived from the rank, so runs are reproducible).
//   * reversible presence Fenwick over ranks (active_count[rank] > 0) for exact DISTINCT
//     counts in a rank interval: distinct(lo,hi) = prefix(hi-1) - prefix(lo-1), O(log sigma).
//   * version traversal is an iterative DFS with one saved row state per version (each
//     version carries exactly one edit), so apply/undo are O(log sigma) and exact; factor
//     updates touch no order-statistic state because the measure is the raw amount (the
//     factor-weighted holistic variant would need per-group value remapping; out of scope
//     for this prototype and documented as a limitation).
//
// No randomisation at run time; priorities are a fixed hash of the value rank, so answer
// vectors and timings are reproducible.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

#include "common.hpp"
#include "sha256.hpp"

namespace bwh {
namespace hros {

struct Treap {
  struct Node {
    i32 l = 0, r = 0;      // child node indices (0 = null)
    i32 key = 0;           // value rank
    i32 mult = 0;          // active rows with this rank
    i32 cnt = 0;           // subtree row count
    i64 sum = 0;           // subtree amount sum
    u32 pri = 0;
  };
  std::vector<Node> nd;
  std::vector<i32> free_ids;   // recycled node ids (keeps the pool at O(sigma) nodes)
  i32 root = 0;
  u32 next_id = 1;

  static inline u32 prio(i32 key) {
    u32 x = u32(key) + 0x9E3779B9u;
    x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
    return x;
  }
  inline i32 cnt(i32 t) const { return t ? nd[size_t(t)].cnt : 0; }
  inline i64 sum(i32 t) const { return t ? nd[size_t(t)].sum : 0; }
  inline void pull(i32 t) {
    if (!t) return;
    Node& n = nd[size_t(t)];
    n.cnt = cnt(n.l) + cnt(n.r) + n.mult;
    n.sum = sum(n.l) + sum(n.r) + i64(n.mult) * i64(value_of_rank(n.key));
  }
  // the rank->amount table is injected at build time
  const std::vector<i64>* vals = nullptr;
  inline i64 value_of_rank(i32 r) const { return (*vals)[size_t(r)]; }

  void init(i32 sigma) { nd.assign(size_t(sigma) + 1, Node{}); root = 0; next_id = 1; }
  i32 make(i32 key) {
    i32 id;
    if (!free_ids.empty()) { id = free_ids.back(); free_ids.pop_back(); }
    else {
      if (size_t(next_id) >= nd.size()) nd.push_back(Node{});
      id = i32(next_id++);
    }
    Node& n = nd[size_t(id)];
    n = Node{}; n.key = key; n.pri = prio(key);
    return id;
  }
  void split(i32 t, i32 key, i32& a, i32& b) {
    // a: keys < key, b: keys >= key
    if (!t) { a = b = 0; return; }
    if (nd[size_t(t)].key < key) {
      split(nd[size_t(t)].r, key, nd[size_t(t)].r, b);
      a = t; pull(a);
    } else {
      split(nd[size_t(t)].l, key, a, nd[size_t(t)].l);
      b = t; pull(b);
    }
  }
  i32 merge(i32 a, i32 b) {
    if (!a) return b;
    if (!b) return a;
    if (nd[size_t(a)].pri > nd[size_t(b)].pri) {
      nd[size_t(a)].r = merge(nd[size_t(a)].r, b);
      pull(a); return a;
    }
    nd[size_t(b)].l = merge(a, nd[size_t(b)].l);
    pull(b); return b;
  }
  // add delta active rows with this rank (delta may be negative); creates/removes nodes
  void add(i32 rank, i32 delta) {
    if (delta == 0) return;
    i32 a, b, c;
    split(root, rank, a, b);
    i32 d;
    split(b, rank + 1, d, c);
    if (d) {
      nd[size_t(d)].mult += delta;
      if (nd[size_t(d)].mult <= 0) { free_ids.push_back(d); d = 0; }
      else pull(d);
    } else if (delta > 0) {
      d = make(rank);
      nd[size_t(d)].mult = delta;
      pull(d);
    }
    root = merge(merge(a, d), c);
  }
  i32 count_less(i32 rank) const {          // rows with key < rank
    i32 t = root, s = 0;
    while (t) {
      if (rank <= nd[size_t(t)].key) t = nd[size_t(t)].l;
      else { s += cnt(nd[size_t(t)].l) + nd[size_t(t)].mult; t = nd[size_t(t)].r; }
    }
    return s;
  }
  i64 sum_less(i32 rank) const {
    i32 t = root; i64 s = 0;
    while (t) {
      if (rank <= nd[size_t(t)].key) t = nd[size_t(t)].l;
      else { s += sum(nd[size_t(t)].l) + i64(nd[size_t(t)].mult) * (*vals)[size_t(nd[size_t(t)].key)]; t = nd[size_t(t)].r; }
    }
    return s;
  }
  // 1-based k-th smallest active value rank; 0 if out of range
  i32 select(i32 k) const {
    i32 t = root;
    while (t && k > 0) {
      i32 lc = cnt(nd[size_t(t)].l);
      if (k <= lc) t = nd[size_t(t)].l;
      else if (k <= lc + nd[size_t(t)].mult) return nd[size_t(t)].key;
      else { k -= lc + nd[size_t(t)].mult; t = nd[size_t(t)].r; }
    }
    return 0;
  }
};

struct PresenceFenwick {                    // reversible distinct support over ranks
  std::vector<i32> bit;
  void init(i32 n) { bit.assign(size_t(n) + 1, 0); }
  void add(i32 rank, i32 delta) {           // rank is 0-based
    for (i32 i = rank + 1; i < i32(bit.size()); i += i & -i) bit[size_t(i)] += delta;
  }
  i32 prefix(i32 rank_exclusive) const {    // count of present ranks < rank_exclusive
    i32 s = 0;
    for (i32 i = rank_exclusive; i > 0; i -= i & -i) s += bit[size_t(i)];
    return s;
  }
  i32 range(i32 lo, i32 hi) const { return prefix(hi) - prefix(lo); }
};

struct HrosIndex {
  const Fixture& f;
  const Workload& w;
  i32 sigma = 0;
  std::vector<i64> value_of_rank;
  Treap tr;
  PresenceFenwick pres;
  std::vector<i32> active_cnt;              // active rows per rank
  // row state: index = base row id; inserted rows are appended at build (ids >= n)
  i32 n_rows = 0;
  std::vector<u8> row_active;
  std::vector<i32> row_rank;
  std::vector<i64> row_value;
  // saved state for undo (per version)
  std::vector<u8> saved_active;
  std::vector<i32> saved_rank;
  std::vector<i64> saved_value;
  i64 updates = 0, queries = 0;

  HrosIndex(const Fixture& fi, const Workload& wo) : f(fi), w(wo) {}

  i32 rank_of(i64 value) const {
    return i32(std::lower_bound(value_of_rank.begin(), value_of_rank.end(), value) - value_of_rank.begin());
  }
  void build() {
    // 1. value universe: base amounts + every absolute amount written by a row edit
    std::vector<i64> universe;
    universe.reserve(size_t(f.h.n) + size_t(w.n_inserted) + size_t(kStates));
    for (i64 i = 0; i < f.h.n; ++i) universe.push_back(f.rows[size_t(i)].amount);
    for (i32 v = 1; v < kStates; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if (e.op == OP_REPLACE || e.op == OP_INSERT || e.op == OP_MOVE) universe.push_back(e.val);
    }
    std::sort(universe.begin(), universe.end());
    universe.erase(std::unique(universe.begin(), universe.end()), universe.end());
    value_of_rank = universe;
    sigma = i32(value_of_rank.size());
    // 2. rows: base rows plus all inserted row ids
    i32 max_id = i32(f.h.n);
    for (i32 v = 1; v < kStates; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if ((e.op == OP_REPLACE || e.op == OP_DELETE || e.op == OP_MOVE) && e.id + 1 > i64(max_id)) max_id = i32(e.id) + 1;
      if (e.op == OP_INSERT && e.id + 1 > i64(max_id)) max_id = i32(e.id) + 1;
    }
    n_rows = max_id + 1;
    row_active.assign(size_t(n_rows), 0);
    row_rank.assign(size_t(n_rows), 0);
    row_value.assign(size_t(n_rows), 0);
    saved_active.assign(size_t(kStates), 0);
    saved_rank.assign(size_t(kStates), 0);
    saved_value.assign(size_t(kStates), 0);
    active_cnt.assign(size_t(sigma), 0);
    pres.init(sigma);
    // 3. base rows active
    for (i64 i = 0; i < f.h.n; ++i) {
      i32 r = rank_of(f.rows[size_t(i)].amount);
      row_active[size_t(i)] = 1; row_rank[size_t(i)] = r; row_value[size_t(i)] = f.rows[size_t(i)].amount;
      if (active_cnt[size_t(r)]++ == 0) pres.add(r, 1);
    }
    tr.init(sigma);
    tr.vals = &value_of_rank;
    // multiplicity insert by rank: build via one pass over ranks with count>0
    for (i32 r = 0; r < sigma; ++r)
      if (active_cnt[size_t(r)]) tr.add(r, active_cnt[size_t(r)]);
  }
  void snapshot(i32 v) {
    // exactly one edited row per version (row edit) or none (factor update)
    const Edit& e = w.edit[size_t(v)];
    if (e.op == OP_FACTOR) { saved_active[size_t(v)] = 2; return; }
    const size_t id = size_t(e.id);
    saved_active[size_t(v)] = row_active[id];
    saved_rank[size_t(v)] = row_rank[id];
    saved_value[size_t(v)] = row_value[id];
  }
  void set_row(i32 id, bool act, i32 rank, i64 value) {
    const size_t i = size_t(id);
    if (row_active[i]) {
      i32 old = row_rank[i];
      if (--active_cnt[size_t(old)] == 0) pres.add(old, -1);
      tr.add(old, -1);
    }
    if (act && rank >= 0 && rank < sigma) {
      row_active[i] = 1; row_rank[i] = rank; row_value[i] = value;
      if (active_cnt[size_t(rank)]++ == 0) pres.add(rank, 1);
      tr.add(rank, +1);
    } else {
      row_active[i] = 0;
    }
    ++updates;
  }
  void apply(i32 v) {
    snapshot(v);
    const Edit& e = w.edit[size_t(v)];
    if (e.op == OP_FACTOR) return;
    ++updates;                            // count the edit even if it is a no-op change
    if (e.op == OP_DELETE) { set_row(i32(e.id), false, -1, 0); return; }
    set_row(i32(e.id), true, rank_of(e.val), e.val);
  }
  void undo(i32 v) {
    const Edit& e = w.edit[size_t(v)];
    if (e.op == OP_FACTOR) return;
    const size_t i = size_t(e.id);
    // deactivate current, restore saved
    if (row_active[i]) {
      i32 r = row_rank[i];
      if (--active_cnt[size_t(r)] == 0) pres.add(r, -1);
      tr.add(r, -1);
    }
    if (saved_active[size_t(v)] == 1) {
      i32 r = saved_rank[size_t(v)];
      row_active[i] = 1; row_rank[i] = r; row_value[i] = saved_value[size_t(v)];
      if (active_cnt[size_t(r)]++ == 0) pres.add(r, 1);
      tr.add(r, +1);
    } else {
      row_active[i] = 0;
    }
    ++updates;
  }
  // query over value range [lo,hi); answer layout: cnt, sum, median, distinct
  void query(i64 lo, i64 hi, i64* out) const {
    i32 rlo = i32(std::lower_bound(value_of_rank.begin(), value_of_rank.end(), lo) - value_of_rank.begin());
    i32 rhi = i32(std::lower_bound(value_of_rank.begin(), value_of_rank.end(), hi) - value_of_rank.begin());
    if (rhi < rlo) rhi = rlo;
    i32 cnt = tr.cnt(tr.root);
    i32 below = tr.count_less(rlo);
    i32 n = tr.count_less(rhi) - below;
    i64 s = tr.sum_less(rhi) - tr.sum_less(rlo);
    i64 med = 0;
    if (n > 0) {
      i32 key = tr.select(below + (n - 1) / 2 + 1);
      med = value_of_rank[size_t(key)];
    }
    i32 dis = pres.range(rlo, rhi);
    out[0] = n; out[1] = s; out[2] = med; out[3] = dis;
    (void)cnt;
    ++const_cast<HrosIndex*>(this)->queries;
  }
};

// Deterministic nested 1D ranges over a value domain (same generator shape as common.hpp
// gen_nested but over the measure axis; q ranges per state, fresh RNG per state).
inline void gen_value_ranges(Rng& r, i64 vmin, i64 vmax, i32 q, i64* out_lo, i64* out_hi) {
  i64 lo = vmin, hi = vmax + 1;
  const i64 span0 = hi - lo;
  for (i32 k = 0; k < q; ++k) {
    i64 span = hi - lo;
    if (span <= 4096) { lo = vmin + span0 / 2; hi = lo + 2048; span = hi - lo; }
    i64 a = span * i64(3 + r.bounded(16)) / 100;
    i64 b = span * i64(3 + r.bounded(16)) / 100;
    lo += a; hi -= b;
    if (hi - lo < 1024) hi = lo + 1024;
    out_lo[k] = lo; out_hi[k] = hi;
  }
}

}  // namespace hros
}  // namespace bwh
