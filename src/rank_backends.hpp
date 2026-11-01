// rank_backends.hpp — two fair, same-dynamic-semantics baselines for the HROS holistic
// order-statistics contract (2026-11-01, P0 HROS alignment).
//
// New file, lives under results/hros_align_20261101/.  Shipped engine/paper untouched.
//
// Both backends implement exactly the HROS contract on the same value-universe/row-state and
// the same version-tree apply/undo semantics (snapshot per version, one edit per version,
// reversible):
//
//   FenwickBackend  "Fenwick-only": value-rank coordinate compression + three Fenwick trees
//                   over ranks -- i64 counts, i64 sums and an i32 presence tree.  count/sum are
//                   two prefix reads; distinct is two presence-prefix reads; the lower median
//                   is the k-th active row found by Fenwick binary lifting, O(log sigma).
//                   No balanced tree is used anywhere.
//
//   AvlBackend      "augmented balanced tree": an independent AVL tree (own insertion /
//                   deletion / rotations, NOT a treap and not std::map) keyed by value rank,
//                   one node per distinct rank with multiplicity.  Each node is augmented with
//                   subtree row count, subtree amount sum and subtree distinct-node count, so
//                   count / sum / distinct are one root-to-leaf descent and the lower median is
//                   a subtree-count descent, all O(log sigma).  Distinct needs no presence
//                   Fenwick because the distinct-node-count augmentation answers it directly.
//
// Value universe (base amounts + every absolute amount written by a replace/insert/move edit)
// and the row state/restore logic are shared with HROS (RankTable below), so the three
// backends are compared on identical semantics and identical edits/queries.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

#include "common.hpp"

namespace bwh {
namespace hros {

// ---------------------------------------------------------------- shared value/row table
struct RankTable {
  const Fixture& f;
  const Workload& w;
  i32 sigma = 0;
  std::vector<i64> value_of_rank;
  i32 n_rows = 0;
  std::vector<u8> row_active;
  std::vector<i32> row_rank;
  std::vector<i64> row_value;
  std::vector<u8> saved_active;
  std::vector<i32> saved_rank;
  std::vector<i64> saved_value;

  RankTable(const Fixture& fi, const Workload& wo) : f(fi), w(wo) {}

  i32 rank_of(i64 value) const {
    return i32(std::lower_bound(value_of_rank.begin(), value_of_rank.end(), value) -
               value_of_rank.begin());
  }
  void build() {
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
    i32 max_id = i32(f.h.n);
    for (i32 v = 1; v < kStates; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if (e.op == OP_REPLACE || e.op == OP_DELETE || e.op == OP_MOVE || e.op == OP_INSERT) {
        if (e.id + 1 > i64(max_id)) max_id = i32(e.id) + 1;
      }
    }
    n_rows = max_id + 1;
    row_active.assign(size_t(n_rows), 0);
    row_rank.assign(size_t(n_rows), 0);
    row_value.assign(size_t(n_rows), 0);
    saved_active.assign(size_t(kStates), 0);
    saved_rank.assign(size_t(kStates), 0);
    saved_value.assign(size_t(kStates), 0);
    for (i64 i = 0; i < f.h.n; ++i) {
      row_active[size_t(i)] = 1;
      row_rank[size_t(i)] = rank_of(f.rows[size_t(i)].amount);
      row_value[size_t(i)] = f.rows[size_t(i)].amount;
    }
  }
  // per-version saved state: one edit per version; factor edits carry no row state
  void snapshot(i32 v) {
    const Edit& e = w.edit[size_t(v)];
    if (e.op == OP_FACTOR) { saved_active[size_t(v)] = 2; return; }
    const size_t id = size_t(e.id);
    saved_active[size_t(v)] = row_active[id];
    saved_rank[size_t(v)] = row_rank[id];
    saved_value[size_t(v)] = row_value[id];
  }
};

// ---------------------------------------------------------------- Fenwick-only backend
struct FenwickBackend {
  RankTable rt;
  i32 sigma = 0;
  i32 n_rows = 0;
  std::vector<i64> cnt_t, sum_t;   // Fenwick over ranks: active-row counts / amounts
  std::vector<i32> pres_t;         // Fenwick over ranks: 1 iff active_cnt[rank] > 0
  std::vector<i32> active_cnt;
  i64 updates = 0, queries = 0;

  FenwickBackend(const Fixture& f, const Workload& w) : rt(f, w) {}

  static inline void fadd(std::vector<i64>& t, i32 n, i32 r, i64 d) {
    for (i32 i = r + 1; i <= n; i += i & (-i)) t[size_t(i)] += d;
  }
  static inline i64 fpref(const std::vector<i64>& t, i32 n, i32 r) {  // sum over ranks < r
    i64 s = 0;
    for (i32 i = r; i > 0; i -= i & (-i)) s += t[size_t(i)];
    (void)n;
    return s;
  }
  static inline void padd(std::vector<i32>& t, i32 n, i32 r, i32 d) {
    for (i32 i = r + 1; i <= n; i += i & (-i)) t[size_t(i)] += d;
  }
  static inline i32 ppref(const std::vector<i32>& t, i32 n, i32 r) {
    i32 s = 0;
    for (i32 i = r; i > 0; i -= i & (-i)) s += t[size_t(i)];
    (void)n;
    return s;
  }
  void build() {
    rt.build();
    sigma = rt.sigma;
    n_rows = rt.n_rows;
    active_cnt.assign(size_t(sigma), 0);
    for (i64 i = 0; i < rt.f.h.n; ++i) ++active_cnt[size_t(rt.row_rank[size_t(i)])];
    cnt_t.assign(size_t(sigma) + 1, 0);
    sum_t.assign(size_t(sigma) + 1, 0);
    pres_t.assign(size_t(sigma) + 1, 0);
    // Fenwick's natural linear build from the base frequency arrays
    for (i32 r = 0; r < sigma; ++r) {
      cnt_t[size_t(r) + 1] = active_cnt[size_t(r)];
      sum_t[size_t(r) + 1] = i64(active_cnt[size_t(r)]) * rt.value_of_rank[size_t(r)];
      pres_t[size_t(r) + 1] = active_cnt[size_t(r)] > 0 ? 1 : 0;
    }
    for (i32 i = 1; i <= sigma; ++i) {
      const i32 j = i + (i & (-i));
      if (j <= sigma) {
        cnt_t[size_t(j)] += cnt_t[size_t(i)];
        sum_t[size_t(j)] += sum_t[size_t(i)];
        pres_t[size_t(j)] += pres_t[size_t(i)];
      }
    }
  }
  inline i32 cnt_less(i32 rank) const { return i32(fpref(cnt_t, sigma, rank)); }
  inline i64 sum_less(i32 rank) const { return fpref(sum_t, sigma, rank); }
  inline i32 distinct_less(i32 rank) const { return ppref(pres_t, sigma, rank); }
  inline i32 kth(i64 k) const {  // 1-based k-th active row -> 0-based value rank
    i32 idx = 0;
    i32 bit = 1;
    while ((bit << 1) <= sigma) bit <<= 1;
    for (; bit; bit >>= 1) {
      const i32 nxt = idx + bit;
      if (nxt <= sigma && cnt_t[size_t(nxt)] < k) {
        idx = nxt;
        k -= cnt_t[size_t(nxt)];
      }
    }
    return idx;
  }
  void set_row(i32 id, bool act, i32 rank, i64 value) {
    const size_t i = size_t(id);
    if (rt.row_active[i]) {
      const i32 old = rt.row_rank[i];
      if (--active_cnt[size_t(old)] == 0) padd(pres_t, sigma, old, -1);
      fadd(cnt_t, sigma, old, -1);
      fadd(sum_t, sigma, old, -rt.value_of_rank[size_t(old)]);
    }
    if (act && rank >= 0 && rank < sigma) {
      rt.row_active[i] = 1;
      rt.row_rank[i] = rank;
      rt.row_value[i] = value;
      if (active_cnt[size_t(rank)]++ == 0) padd(pres_t, sigma, rank, 1);
      fadd(cnt_t, sigma, rank, 1);
      fadd(sum_t, sigma, rank, rt.value_of_rank[size_t(rank)]);
    } else {
      rt.row_active[i] = 0;
    }
    ++updates;
  }
  void apply(i32 v) {
    rt.snapshot(v);
    const Edit& e = rt.w.edit[size_t(v)];
    if (e.op == OP_FACTOR) return;
    if (e.op == OP_DELETE) { set_row(i32(e.id), false, -1, 0); return; }
    set_row(i32(e.id), true, rt.rank_of(e.val), e.val);
  }
  void undo(i32 v) {
    const Edit& e = rt.w.edit[size_t(v)];
    if (e.op == OP_FACTOR) return;
    const size_t i = size_t(e.id);
    if (rt.row_active[i]) {
      const i32 r = rt.row_rank[i];
      if (--active_cnt[size_t(r)] == 0) padd(pres_t, sigma, r, -1);
      fadd(cnt_t, sigma, r, -1);
      fadd(sum_t, sigma, r, -rt.value_of_rank[size_t(r)]);
    }
    if (rt.saved_active[size_t(v)] == 1) {
      const i32 r = rt.saved_rank[size_t(v)];
      rt.row_active[i] = 1;
      rt.row_rank[i] = r;
      rt.row_value[i] = rt.saved_value[size_t(v)];
      if (active_cnt[size_t(r)]++ == 0) padd(pres_t, sigma, r, 1);
      fadd(cnt_t, sigma, r, 1);
      fadd(sum_t, sigma, r, rt.value_of_rank[size_t(r)]);
    } else {
      rt.row_active[i] = 0;
    }
    ++updates;
  }
  // out = {count, sum, lower_median, distinct}
  void query(i64 lo, i64 hi, i64* out) {
    const i32 rlo = i32(std::lower_bound(rt.value_of_rank.begin(), rt.value_of_rank.end(), lo) -
                        rt.value_of_rank.begin());
    const i32 rhi = i32(std::lower_bound(rt.value_of_rank.begin(), rt.value_of_rank.end(), hi) -
                        rt.value_of_rank.begin());
    const i32 n = cnt_less(rhi) - cnt_less(rlo);
    out[0] = n;
    out[1] = sum_less(rhi) - sum_less(rlo);
    out[2] = 0;
    if (n > 0) {
      const i64 k = i64(cnt_less(rlo)) + i64((n - 1) / 2) + 1;  // 1-based global k
      out[2] = rt.value_of_rank[size_t(kth(k))];
    }
    out[3] = distinct_less(rhi) - distinct_less(rlo);
    ++queries;
  }
};

// ---------------------------------------------------------------- AVL augmented-tree backend
struct AvlBackend {
  RankTable rt;
  i32 sigma = 0, n_rows = 0;
  struct Node {
    i32 l = 0, r = 0;
    i32 key = 0;      // value rank
    i32 mult = 0;     // active rows at this rank
    i32 cnt = 0;      // subtree active rows
    i32 sub = 0;      // subtree distinct ranks (nodes)
    i32 h = 0;        // subtree height
    i64 sum = 0;      // subtree amount sum
  };
  std::vector<Node> nd;
  std::vector<i32> free_ids;
  i32 root = 0, next_id = 1;
  std::vector<i32> active_cnt;
  i64 updates = 0, queries = 0;

  AvlBackend(const Fixture& f, const Workload& w) : rt(f, w) {}

  inline i32 cnt(i32 t) const { return t ? nd[size_t(t)].cnt : 0; }
  inline i32 sub(i32 t) const { return t ? nd[size_t(t)].sub : 0; }
  inline i32 hgt(i32 t) const { return t ? nd[size_t(t)].h : 0; }
  inline i64 isum(i32 t) const { return t ? nd[size_t(t)].sum : 0; }
  inline i64 val_of(i32 t) const { return rt.value_of_rank[size_t(nd[size_t(t)].key)]; }
  void pull(i32 t) {
    if (!t) return;
    Node& n = nd[size_t(t)];
    n.cnt = cnt(n.l) + cnt(n.r) + n.mult;
    n.sum = isum(n.l) + isum(n.r) + i64(n.mult) * rt.value_of_rank[size_t(n.key)];
    n.sub = sub(n.l) + sub(n.r) + 1;
    n.h = 1 + std::max(hgt(n.l), hgt(n.r));
  }
  i32 rotR(i32 t) {
    i32 x = nd[size_t(t)].l;
    nd[size_t(t)].l = nd[size_t(x)].r;
    nd[size_t(x)].r = t;
    pull(t);
    pull(x);
    return x;
  }
  i32 rotL(i32 t) {
    i32 x = nd[size_t(t)].r;
    nd[size_t(t)].r = nd[size_t(x)].l;
    nd[size_t(x)].l = t;
    pull(t);
    pull(x);
    return x;
  }
  i32 balance(i32 t) {
    pull(t);
    const i32 b = hgt(nd[size_t(t)].l) - hgt(nd[size_t(t)].r);
    if (b > 1) {
      if (hgt(nd[size_t(nd[size_t(t)].l)].l) < hgt(nd[size_t(nd[size_t(t)].l)].r)) {
        nd[size_t(t)].l = rotL(nd[size_t(t)].l);
      }
      return rotR(t);
    }
    if (b < -1) {
      if (hgt(nd[size_t(nd[size_t(t)].r)].r) < hgt(nd[size_t(nd[size_t(t)].r)].l)) {
        nd[size_t(t)].r = rotR(nd[size_t(t)].r);
      }
      return rotL(t);
    }
    return t;
  }
  i32 make(i32 key, i32 mult) {
    i32 id;
    if (!free_ids.empty()) { id = free_ids.back(); free_ids.pop_back(); }
    else {
      if (size_t(next_id) >= nd.size()) nd.push_back(Node{});
      id = i32(next_id++);
    }
    Node& n = nd[size_t(id)];
    n = Node{};
    n.key = key;
    n.mult = mult;
    n.h = 1;
    pull(id);
    return id;
  }
  void recycle(i32 id) { free_ids.push_back(id); }
  i32 ins(i32 t, i32 key) {
    if (!t) return make(key, 1);
    if (key == nd[size_t(t)].key) {
      ++nd[size_t(t)].mult;
    } else if (key < nd[size_t(t)].key) {
      nd[size_t(t)].l = ins(nd[size_t(t)].l, key);
    } else {
      nd[size_t(t)].r = ins(nd[size_t(t)].r, key);
    }
    return balance(t);
  }
  // remove the whole node with this key (not one multiplicity)
  i32 remove_full(i32 t, i32 key) {
    if (!t) return 0;
    if (key < nd[size_t(t)].key) {
      nd[size_t(t)].l = remove_full(nd[size_t(t)].l, key);
    } else if (key > nd[size_t(t)].key) {
      nd[size_t(t)].r = remove_full(nd[size_t(t)].r, key);
    } else {
      if (!nd[size_t(t)].l || !nd[size_t(t)].r) {
        const i32 child = nd[size_t(t)].l ? nd[size_t(t)].l : nd[size_t(t)].r;
        recycle(t);
        return child;
      }
      i32 s = nd[size_t(t)].r;
      while (nd[size_t(s)].l) s = nd[size_t(s)].l;
      nd[size_t(t)].key = nd[size_t(s)].key;
      nd[size_t(t)].mult = nd[size_t(s)].mult;
      nd[size_t(t)].r = remove_full(nd[size_t(t)].r, nd[size_t(s)].key);
    }
    return balance(t);
  }
  i32 ers(i32 t, i32 key) {
    if (!t) return 0;
    if (key == nd[size_t(t)].key) {
      if (--nd[size_t(t)].mult > 0) return balance(t);
      return remove_full(t, key);
    }
    if (key < nd[size_t(t)].key) nd[size_t(t)].l = ers(nd[size_t(t)].l, key);
    else nd[size_t(t)].r = ers(nd[size_t(t)].r, key);
    return balance(t);
  }
  i32 build_balanced(const std::vector<i32>& ranks, const std::vector<i32>& mults, i32 lo, i32 hi) {
    if (lo > hi) return 0;
    const i32 mid = lo + (hi - lo) / 2;
    const i32 id = make(ranks[size_t(mid)], mults[size_t(mid)]);
    nd[size_t(id)].l = build_balanced(ranks, mults, lo, mid - 1);
    nd[size_t(id)].r = build_balanced(ranks, mults, mid + 1, hi);
    pull(id);
    return id;
  }
  void build() {
    rt.build();
    sigma = rt.sigma;
    n_rows = rt.n_rows;
    nd.assign(size_t(sigma) + 1, Node{});
    active_cnt.assign(size_t(sigma), 0);
    for (i64 i = 0; i < rt.f.h.n; ++i) ++active_cnt[size_t(rt.row_rank[size_t(i)])];
    std::vector<i32> ranks, mults;
    for (i32 r = 0; r < sigma; ++r) {
      if (active_cnt[size_t(r)]) { ranks.push_back(r); mults.push_back(active_cnt[size_t(r)]); }
    }
    root = build_balanced(ranks, mults, 0, i32(ranks.size()) - 1);
  }
  i32 count_less(i32 key) const {
    i32 t = root;
    i32 s = 0;
    while (t) {
      if (key <= nd[size_t(t)].key) t = nd[size_t(t)].l;
      else { s += cnt(nd[size_t(t)].l) + nd[size_t(t)].mult; t = nd[size_t(t)].r; }
    }
    return s;
  }
  i64 sum_less(i32 key) const {
    i32 t = root;
    i64 s = 0;
    while (t) {
      if (key <= nd[size_t(t)].key) t = nd[size_t(t)].l;
      else {
        s += isum(nd[size_t(t)].l) + i64(nd[size_t(t)].mult) * rt.value_of_rank[size_t(nd[size_t(t)].key)];
        t = nd[size_t(t)].r;
      }
    }
    return s;
  }
  i32 distinct_less(i32 key) const {  // number of distinct ranks (nodes) < key
    i32 t = root;
    i32 s = 0;
    while (t) {
      if (key <= nd[size_t(t)].key) t = nd[size_t(t)].l;
      else { s += sub(nd[size_t(t)].l) + 1; t = nd[size_t(t)].r; }
    }
    return s;
  }
  i32 select(i64 k) const {  // 1-based k-th active row -> value rank
    i32 t = root;
    while (t && k > 0) {
      const i32 lc = cnt(nd[size_t(t)].l);
      if (k <= lc) t = nd[size_t(t)].l;
      else if (k <= i64(lc) + nd[size_t(t)].mult) return nd[size_t(t)].key;
      else { k -= i64(lc) + nd[size_t(t)].mult; t = nd[size_t(t)].r; }
    }
    return 0;
  }
  void set_row(i32 id, bool act, i32 rank, i64 value) {
    const size_t i = size_t(id);
    if (rt.row_active[i]) {
      const i32 old = rt.row_rank[i];
      root = ers(root, old);
      --active_cnt[size_t(old)];
    }
    if (act && rank >= 0 && rank < sigma) {
      rt.row_active[i] = 1;
      rt.row_rank[i] = rank;
      rt.row_value[i] = value;
      root = ins(root, rank);
      ++active_cnt[size_t(rank)];
    } else {
      rt.row_active[i] = 0;
    }
    ++updates;
  }
  void apply(i32 v) {
    rt.snapshot(v);
    const Edit& e = rt.w.edit[size_t(v)];
    if (e.op == OP_FACTOR) return;
    if (e.op == OP_DELETE) { set_row(i32(e.id), false, -1, 0); return; }
    set_row(i32(e.id), true, rt.rank_of(e.val), e.val);
  }
  void undo(i32 v) {
    const Edit& e = rt.w.edit[size_t(v)];
    if (e.op == OP_FACTOR) return;
    const size_t i = size_t(e.id);
    if (rt.row_active[i]) {
      const i32 r = rt.row_rank[i];
      root = ers(root, r);
      --active_cnt[size_t(r)];
    }
    if (rt.saved_active[size_t(v)] == 1) {
      const i32 r = rt.saved_rank[size_t(v)];
      rt.row_active[i] = 1;
      rt.row_rank[i] = r;
      rt.row_value[i] = rt.saved_value[size_t(v)];
      root = ins(root, r);
      ++active_cnt[size_t(r)];
    } else {
      rt.row_active[i] = 0;
    }
    ++updates;
  }
  void query(i64 lo, i64 hi, i64* out) {
    const i32 rlo = i32(std::lower_bound(rt.value_of_rank.begin(), rt.value_of_rank.end(), lo) -
                        rt.value_of_rank.begin());
    const i32 rhi = i32(std::lower_bound(rt.value_of_rank.begin(), rt.value_of_rank.end(), hi) -
                        rt.value_of_rank.begin());
    const i32 n = count_less(rhi) - count_less(rlo);
    out[0] = n;
    out[1] = sum_less(rhi) - sum_less(rlo);
    out[2] = 0;
    if (n > 0) {
      const i64 k = i64(count_less(rlo)) + i64((n - 1) / 2) + 1;
      out[2] = rt.value_of_rank[size_t(select(k))];
    }
    out[3] = distinct_less(rhi) - distinct_less(rlo);
    ++queries;
  }
};

}  // namespace hros
}  // namespace bwh
