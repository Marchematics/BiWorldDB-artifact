// arms_classic.hpp — classic direct-index baselines for the BiWorldDB offline panel.
//
// The three arms answer exactly the same declared workload as the hybrid and the
// full-state controls in engine.hpp: for every edited state v, sum over the current
// rows of amount(row) * factor(group(row)) restricted to the declared half-open
// timestamp range [lo, hi).  Edits are absolute and are applied/undone along a
// reversible depth-first traversal of the 2,049-state version tree.
//
//   rf   rollback Fenwick over raw timestamp ranks.  One global Fenwick tree stores
//        the current per-row contribution amount*factor at the rank of the row's
//        current timestamp.  A factor change for group g walks the *current* row
//        list of g and applies one point update per contributing row, i.e.
//        O(n_g log N) eager materialization; row edits are O(log N), queries
//        O(log N), undo negates the same point updates.
//   rs   identical semantics, but the accumulator is an iterative point-add /
//        range-sum segment tree (also O(log N) per point and per range).
//   pst  persistent segment tree over the same raw ranks; one root per state.  A
//        factor change materializes one persistent point update per current row of
//        the group, so undo is just restoring the parent root (no inverse
//        materialization).  This arm is memory/CPU bound by construction; the node
//        pool size is estimated with an exact dry replay of the workload before the
//        pool is allocated, and the estimate is reported through the optional JSON
//        run (see panel_classic.cpp).
//
// The coordinate domain ("raw timestamp ranks") is the sorted distinct set of all
// raw timestamps that any row can occupy: every base-row timestamp plus every
// declared insert/move destination.  Queries are mapped to rank bounds by binary
// search over that domain, so there is no endpoint projection or class compaction:
// this is the honest direct-index baseline, not the hybrid's projected variant.
//
// All arms keep the full current row state (t, group, amount, existence, rank) and
// a per-group list of currently contributing rows (existence=1, amount!=0), which
// is what makes the eager factor materialization safe under row moves, inserts,
// deletes and replacements.

#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.hpp"
#include "engine.hpp"

namespace bwh {

// ---------------------------------------------------------------- accumulators
// Both accumulators use the prefix convention range(l, r) = sum over ranks [l, r).
struct FenwickAcc {
  std::vector<i64> t;
  i32 n = 0;
  void init(i32 m) {
    n = m;
    t.assign(size_t(m) + 1, 0);
  }
  // Direct leaf write used only during construction; finalize() then builds the
  // tree in O(n) (the standard bottom-up Fenwick build), instead of n point adds.
  inline void leaf(i32 rank, i64 v) { t[size_t(rank) + 1] += v; }
  void finalize() {
    for (i32 i = 1; i <= n; ++i) {
      const i32 p = i + (i & (-i));
      if (p <= n) t[size_t(p)] += t[size_t(i)];
    }
  }
  inline void add(i32 rank, i64 v) {
    for (i32 i = rank + 1; i <= n; i += i & (-i)) t[size_t(i)] += v;
  }
  inline i64 prefix(i32 rank) const {
    i64 s = 0;
    for (i32 i = rank; i > 0; i -= i & (-i)) s += t[size_t(i)];
    return s;
  }
  inline i64 range(i32 l, i32 r) const { return prefix(r) - prefix(l); }
};

struct SegtreeAcc {
  std::vector<i64> t;
  i32 n = 0;
  i32 sz = 1;
  void init(i32 m) {
    n = m;
    sz = 1;
    while (sz < m) sz <<= 1;
    t.assign(size_t(2) * size_t(sz), 0);
  }
  inline void leaf(i32 rank, i64 v) { t[size_t(sz) + size_t(rank)] += v; }
  void finalize() {
    for (i32 i = sz - 1; i >= 1; --i) t[size_t(i)] = t[size_t(2) * size_t(i)] + t[size_t(2) * size_t(i) + 1];
  }
  inline void add(i32 rank, i64 v) {
    size_t i = size_t(sz) + size_t(rank);
    t[i] += v;
    for (i >>= 1; i; i >>= 1) t[i] = t[2 * i] + t[2 * i + 1];
  }
  inline i64 prefix(i32 rank) const {
    i64 s = 0;
    i32 l = sz, r = sz + rank;
    while (l < r) {
      if (l & 1) s += t[size_t(l++)];
      if (r & 1) s += t[size_t(--r)];
      l >>= 1;
      r >>= 1;
    }
    return s;
  }
  inline i64 range(i32 l, i32 r) const { return prefix(r) - prefix(l); }
};

// ---------------------------------------------------------------- shared state
// CRTP base holding everything except the accumulator: raw-rank compression, the
// full current row state, the per-group current-row lists (for eager factor
// materialization), the mutable per-group factors and the declared query bounds.
template <class Derived>
struct ClassicState : Arm {
  const Fixture& f;
  const Workload& w;
  const char* nm;
  i32 G = 0;
  i64 n_rows = 0;
  i32 M = 0;  // number of raw timestamp ranks
  std::vector<i64> coords;      // sorted distinct raw timestamps (absolute values)
  std::vector<i64> fac;         // current per-group multiplier
  std::vector<i64> cur_t;       // current timestamp of every base + inserted row
  std::vector<i64> cur_amt;     // current amount
  std::vector<i32> cur_g;       // current group
  std::vector<i32> cur_rank;    // rank of cur_t
  std::vector<u8> cur_ex;       // current existence
  std::vector<i32> gpos;        // position in grow[cur_g], -1 if not contributing
  std::vector<std::vector<i64>> grow;  // current contributing row ids per group
  std::vector<i32> ql;          // [state*kMaxQ + k] rank bound of query lo
  std::vector<i32> qh;          // [state*kMaxQ + k] rank bound of query hi

  ClassicState(const Fixture& fi, const Workload& wo, const char* name)
      : f(fi), w(wo), nm(name) {}
  const char* name() const override { return nm; }

  Derived& d() { return *static_cast<Derived*>(this); }
  const Derived& d() const { return *static_cast<const Derived*>(this); }

  inline i32 rank_of(i64 t) const {
    return i32(std::lower_bound(coords.begin(), coords.end(), t) - coords.begin());
  }

  // ---- per-group current-row list maintenance (O(1) swap-remove)
  inline void list_add(i64 id, i32 g) {
    gpos[size_t(id)] = i32(grow[size_t(g)].size());
    grow[size_t(g)].push_back(id);
  }
  inline void list_remove(i64 id) {
    const i32 p = gpos[size_t(id)];
    if (p < 0) return;
    const i32 g = cur_g[size_t(id)];
    auto& v = grow[size_t(g)];
    const i64 last = v.back();
    v[size_t(p)] = last;
    gpos[size_t(last)] = p;
    v.pop_back();
    gpos[size_t(id)] = -1;
  }

  // ---- reversible row transition.  tree=false updates only the row bookkeeping
  // (used by the persistent arm on undo, where the tree is restored by root).
  void set_row(i64 id, i64 t, i32 g, i64 amount, u8 ex, bool tree = true) {
    if (cur_ex[size_t(id)] && cur_amt[size_t(id)] != 0) {
      if (tree)
        d().point_add(cur_rank[size_t(id)],
                      -(cur_amt[size_t(id)] * fac[size_t(cur_g[size_t(id)])]));
      list_remove(id);
    }
    cur_t[size_t(id)] = t;
    cur_g[size_t(id)] = g;
    cur_amt[size_t(id)] = amount;
    cur_ex[size_t(id)] = ex;
    cur_rank[size_t(id)] = rank_of(t);
    if (ex && amount != 0) {
      if (tree) d().point_add(cur_rank[size_t(id)], amount * fac[size_t(g)]);
      list_add(id, g);
    } else {
      gpos[size_t(id)] = -1;
    }
  }

  // ---- eager factor materialization: one point update per current row of g.
  void set_factor(i32 g, i64 nf, bool tree = true) {
    const i64 old = fac[size_t(g)];
    if (old == nf) return;
    if (tree) {
      const i64 delta = nf - old;
      for (i64 id : grow[size_t(g)])
        d().point_add(cur_rank[size_t(id)], delta * cur_amt[size_t(id)]);
    }
    fac[size_t(g)] = nf;
  }

  // ---- Arm interface
  void build() override {
    G = f.h.groups;
    n_rows = i64(f.h.n) + i64(w.n_inserted);
    fac = f.base_factor;

    // ---- raw timestamp rank domain: base timestamps + declared write destinations
    size_t n_dest = 0;
    for (i32 v = 1; v < kStates; ++v) {
      const u8 op = w.edit[size_t(v)].op;
      if (op == OP_INSERT || op == OP_MOVE) ++n_dest;
    }
    const i64 tmin = f.h.t_min;
    const size_t K = size_t(f.h.n) + n_dest;
    std::vector<u64> keys(K);
    std::vector<u32> vals(K);
    for (i64 i = 0; i < f.h.n; ++i) {
      keys[size_t(i)] = u64(f.rows[size_t(i)].t - tmin);
      vals[size_t(i)] = u32(i);
    }
    {
      size_t k = size_t(f.h.n);
      for (i32 v = 1; v < kStates; ++v) {
        const Edit& e = w.edit[size_t(v)];
        if (e.op == OP_INSERT || e.op == OP_MOVE) {
          keys[k] = u64(e.t - tmin);
          vals[k] = u32(k);
          ++k;
        }
      }
    }
    RadixSorter rsort;
    rsort.sort(keys, vals);
    coords.reserve(K);
    std::vector<i32> base_rank(size_t(f.h.n), -1);
    for (size_t i = 0; i < K; ++i) {
      if (i == 0 || keys[i] != keys[i - 1]) coords.push_back(i64(keys[i]) + tmin);
      const u32 id = vals[i];
      if (i64(id) < f.h.n) base_rank[size_t(id)] = i32(coords.size()) - 1;
    }
    M = i32(coords.size());

    // ---- declared query frame bounds (rank bounds) for every state
    ql.assign(size_t(kStates) * size_t(kMaxQ), 0);
    qh.assign(size_t(kStates) * size_t(kMaxQ), 0);
    for (i32 v = 1; v < kStates; ++v) {
      for (i32 k = 0; k < kMaxQ; ++k) {
        const Query& q = query_for(w, v, k);
        const size_t qi = size_t(v) * size_t(kMaxQ) + size_t(k);
        ql[qi] = rank_of(q.lo);
        qh[qi] = rank_of(q.hi);
      }
    }

    // ---- current row state (full copy, like the full-state controls)
    cur_t.assign(size_t(n_rows), 0);
    cur_amt.assign(size_t(n_rows), 0);
    cur_g.assign(size_t(n_rows), 0);
    cur_rank.assign(size_t(n_rows), 0);
    cur_ex.assign(size_t(n_rows), 0);
    gpos.assign(size_t(n_rows), -1);
    grow.assign(size_t(G), {});
    for (i64 i = 0; i < f.h.n; ++i) {
      const BaseRec& r = f.rows[size_t(i)];
      cur_t[size_t(i)] = r.t;
      cur_amt[size_t(i)] = r.amount;
      cur_g[size_t(i)] = r.g;
      cur_rank[size_t(i)] = base_rank[size_t(i)];
      cur_ex[size_t(i)] = 1;
    }
    for (i32 k = 0; k < w.n_inserted; ++k) {
      const i64 id = i64(f.h.n) + k;
      cur_t[size_t(id)] = w.ins_t[size_t(k)];
      cur_amt[size_t(id)] = 0;
      cur_g[size_t(id)] = w.ins_g[size_t(k)];
      cur_rank[size_t(id)] = rank_of(w.ins_t[size_t(k)]);
      cur_ex[size_t(id)] = 0;
    }
    for (i64 id = 0; id < n_rows; ++id)
      if (cur_ex[size_t(id)] && cur_amt[size_t(id)] != 0) list_add(id, cur_g[size_t(id)]);

    d().build_index();
  }

  SavedRow snapshot(const Edit& e) const override {
    SavedRow s;
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

  void apply_edit(const Edit& e, bool tree = true) {
    switch (e.op) {
      case OP_REPLACE: set_row(e.id, cur_t[size_t(e.id)], cur_g[size_t(e.id)], e.val, 1, tree); break;
      case OP_INSERT: set_row(e.id, e.t, e.g, e.val, 1, tree); break;
      case OP_DELETE:
        set_row(e.id, cur_t[size_t(e.id)], cur_g[size_t(e.id)], cur_amt[size_t(e.id)], 0, tree);
        break;
      case OP_MOVE: set_row(e.id, e.t, e.g, e.val, 1, tree); break;
      case OP_FACTOR: set_factor(e.g, e.val, tree); break;
    }
  }

  void revert_state(const SavedRow& s, bool tree = true) {
    if (s.kind == 1) set_factor(s.g, s.amount, tree);
    else set_row(s.id, s.t, s.g, s.amount, s.ex, tree);
  }

  void apply(const Edit& e) override { apply_edit(e, true); }
  void undo(const Edit& /*e*/, const SavedRow& s) override { revert_state(s, true); }

  i64 query(i32 state, i32 k) const override {
    const size_t qi = size_t(state) * size_t(kMaxQ) + size_t(k);
    return d().range_sum(ql[qi], qh[qi]);
  }
};

// ---------------------------------------------------------------- rollback Fenwick
struct ClassicRfArm : ClassicState<ClassicRfArm> {
  FenwickAcc acc;
  ClassicRfArm(const Fixture& fi, const Workload& wo) : ClassicState<ClassicRfArm>(fi, wo, "rf") {}
  inline void point_add(i32 rank, i64 v) { acc.add(rank, v); }
  inline i64 range_sum(i32 l, i32 r) const { return acc.range(l, r); }
  void build_index() {
    acc.init(this->M);
    for (i64 i = 0; i < this->f.h.n; ++i) {
      const BaseRec& r = this->f.rows[size_t(i)];
      acc.leaf(this->cur_rank[size_t(i)], r.amount * this->fac[size_t(r.g)]);
    }
    acc.finalize();
  }
};

// ---------------------------------------------------------------- rollback segment tree
struct ClassicRsArm : ClassicState<ClassicRsArm> {
  SegtreeAcc acc;
  ClassicRsArm(const Fixture& fi, const Workload& wo) : ClassicState<ClassicRsArm>(fi, wo, "rs") {}
  inline void point_add(i32 rank, i64 v) { acc.add(rank, v); }
  inline i64 range_sum(i32 l, i32 r) const { return acc.range(l, r); }
  void build_index() {
    acc.init(this->M);
    for (i64 i = 0; i < this->f.h.n; ++i) {
      const BaseRec& r = this->f.rows[size_t(i)];
      acc.leaf(this->cur_rank[size_t(i)], r.amount * this->fac[size_t(r.g)]);
    }
    acc.finalize();
  }
};

// ---------------------------------------------------------------- persistent segment tree
// Iterative DFS already avoids native recursion for the version tree; the persistent
// tree recursion depth is only log2(|ranks|) (~22 on the large fixtures), so plain
// recursion is safe here.
struct ClassicPstArm : ClassicState<ClassicPstArm> {
  std::vector<i32> lc, rc;  // node pool, index 0 is the null node
  std::vector<i64> sm;
  std::vector<i32> root_hist;  // roots pushed as edits are applied (DFS nesting)
  i32 root_init = 0;
  i32 cur_root = 0;
  i32 max_depth = 1;
  i64 est_updates = 0;

  ClassicPstArm(const Fixture& fi, const Workload& wo)
      : ClassicState<ClassicPstArm>(fi, wo, "pst") {}

  inline i32 new_node(i32 l, i32 r, i64 s) {
    lc.push_back(l);
    rc.push_back(r);
    sm.push_back(s);
    return i32(lc.size()) - 1;
  }

  i32 build_rec(const i64* leaf, i32 l, i32 r) {
    if (r - l == 1) return new_node(0, 0, leaf[l]);
    const i32 mid = (l + r) >> 1;
    const i32 a = build_rec(leaf, l, mid);
    const i32 b = build_rec(leaf, mid, r);
    return new_node(a, b, sm[size_t(a)] + sm[size_t(b)]);
  }

  i32 upd(i32 node, i32 l, i32 r, i32 pos, i64 delta) {
    const i32 nn = new_node(lc[size_t(node)], rc[size_t(node)], sm[size_t(node)] + delta);
    if (r - l == 1) return nn;
    const i32 mid = (l + r) >> 1;
    if (pos < mid) lc[size_t(nn)] = upd(lc[size_t(node)], l, mid, pos, delta);
    else rc[size_t(nn)] = upd(rc[size_t(node)], mid, r, pos, delta);
    return nn;
  }

  i64 qrec(i32 node, i32 l, i32 r, i32 a, i32 b) const {
    if (!node || b <= l || r <= a) return 0;
    if (a <= l && r <= b) return sm[size_t(node)];
    const i32 mid = (l + r) >> 1;
    return qrec(lc[size_t(node)], l, mid, a, b) + qrec(rc[size_t(node)], mid, r, a, b);
  }

  inline void point_add(i32 rank, i64 v) { cur_root = upd(cur_root, 0, this->M, rank, v); }
  inline i64 range_sum(i32 l, i32 r) const {
    if (l >= r) return 0;
    return qrec(cur_root, 0, this->M, l, r);
  }

  // Dry replay of the declared workload counting the persistent point updates the
  // arm will perform, so the node pool can be reserved up front (and so infeasibility
  // can be reported instead of discovered as an out-of-memory kill).
  i64 estimate_updates() {
    const i64 N = i64(this->f.h.n);
    std::vector<i64> amt(this->cur_amt), t(this->cur_t);
    std::vector<i32> g(this->cur_g);
    std::vector<u8> ex(this->cur_ex);
    std::vector<std::vector<i64>> lists(this->grow);
    std::vector<i32> pos(this->gpos);
    auto remove_list = [&](i64 id) {
      const i32 p = pos[size_t(id)];
      if (p < 0) return;
      auto& v = lists[size_t(g[size_t(id)])];
      const i64 last = v.back();
      v[size_t(p)] = last;
      pos[size_t(last)] = p;
      v.pop_back();
      pos[size_t(id)] = -1;
    };
    auto add_list = [&](i64 id) {
      pos[size_t(id)] = i32(lists[size_t(g[size_t(id)])].size());
      lists[size_t(g[size_t(id)])].push_back(id);
    };
    auto set_row = [&](i64 id, i64 nt, i32 ng, i64 na, u8 nex) -> i64 {
      i64 u = 0;
      if (ex[size_t(id)] && amt[size_t(id)] != 0) {
        remove_list(id);
        ++u;
      }
      t[size_t(id)] = nt;
      g[size_t(id)] = ng;
      amt[size_t(id)] = na;
      ex[size_t(id)] = nex;
      if (nex && na != 0) {
        add_list(id);
        ++u;
      }
      return u;
    };
    // Deterministic replay of the reversible DFS over a copy of the row bookkeeping.
    // Each applied edit counts exactly the persistent point updates the real run
    // performs; undo restores the row state only (the real run restores a root), so
    // it adds no nodes.  Factor edits are counted from the maintained current-row
    // lists of the touched group.
    std::vector<std::vector<i32>> kids{size_t(kStates)};
    for (i32 v = 1; v < kStates; ++v) kids[size_t(w.parent[size_t(v)])].push_back(v);
    std::vector<SavedRow> saved{size_t(kStates)};
    struct Ev {
      i32 v;
      u8 exit;
    };
    std::vector<Ev> st;
    st.reserve(size_t(kStates) + 8);
    st.push_back({0, 0});
    i64 count = 0;
    while (!st.empty()) {
      const Ev ev = st.back();
      st.pop_back();
      const i32 v = ev.v;
      if (ev.exit) {
        if (v > 0) {
          const Edit& e = w.edit[size_t(v)];
          if (e.op != OP_FACTOR) {
            const SavedRow& s = saved[size_t(v)];
            // restore the row state; persistent node creation is not counted on undo
            (void)set_row(s.id, s.t, s.g, s.amount, s.ex);
          }
        }
        continue;
      }
      if (v > 0) {
        const Edit& e = w.edit[size_t(v)];
        if (e.op == OP_FACTOR) {
          count += i64(lists[size_t(e.g)].size());
        } else {
          SavedRow s;
          s.kind = 0;
          s.id = e.id;
          s.t = t[size_t(e.id)];
          s.g = g[size_t(e.id)];
          s.amount = amt[size_t(e.id)];
          s.ex = ex[size_t(e.id)];
          saved[size_t(v)] = s;
          count += set_row(e.id, e.op == OP_INSERT ? e.t : (e.op == OP_MOVE ? e.t : t[size_t(e.id)]),
                           e.op == OP_INSERT || e.op == OP_MOVE ? e.g : g[size_t(e.id)],
                           e.op == OP_DELETE ? amt[size_t(e.id)] : e.val,
                           e.op == OP_DELETE ? u8(0) : u8(1));
        }
      }
      st.push_back({v, 1});
      const auto& ch = kids[size_t(v)];
      for (auto it = ch.rbegin(); it != ch.rend(); ++it) st.push_back({*it, 0});
    }
    return count;
  }

  void build_index() {
    // exact dry-replay count of persistent point updates
    est_updates = estimate_updates();
    i32 span = this->M;
    max_depth = 1;
    while (span > 1) {
      span = (span + 1) >> 1;
      ++max_depth;
    }
    // node pool: initial tree (2M-1 nodes) + one root-to-leaf path per update
    const i64 init_nodes = 2 * i64(this->M) - 1;
    const i64 est_nodes = init_nodes + est_updates * i64(max_depth) + 16;
    if (est_nodes >= (i64(1) << 30)) {
      std::fprintf(stderr,
                   "pst: node estimate %lld exceeds i32 node index capacity; refusing\n",
                   (long long)est_nodes);
      std::exit(8);
    }
    const char* cap_env = std::getenv("BWC_PST_MAX_GB");
    const double cap_gb = cap_env ? std::atof(cap_env) : 0.0;
    const double est_gb = double(est_nodes) * double(sizeof(i32) * 2 + sizeof(i64)) / 1e9;
    if (std::getenv("BWC_PST_INFO")) {
      std::fprintf(stderr,
                   "pst_estimate: ranks=%d max_depth=%d updates=%lld init_nodes=%lld "
                   "est_nodes=%lld est_GB=%.3f\n",
                   this->M, max_depth, (long long)est_updates, (long long)init_nodes,
                   (long long)est_nodes, est_gb);
    }
    if (cap_gb > 0.0 && est_gb > cap_gb) {
      std::fprintf(stderr,
                   "pst: estimated %.2f GB > BWC_PST_MAX_GB=%.2f; refusing to allocate\n",
                   est_gb, cap_gb);
      std::exit(8);
    }
    lc.reserve(size_t(est_nodes));
    rc.reserve(size_t(est_nodes));
    sm.reserve(size_t(est_nodes));
    lc.push_back(0);
    rc.push_back(0);
    sm.push_back(0);

    // initial leaf sums: amount * initial factor for every base row
    std::vector<i64> leaf(size_t(this->M), 0);
    for (i64 i = 0; i < this->f.h.n; ++i) {
      const BaseRec& r = this->f.rows[size_t(i)];
      leaf[size_t(this->cur_rank[size_t(i)])] += r.amount * this->fac[size_t(r.g)];
    }
    root_init = build_rec(leaf.data(), 0, this->M);
    cur_root = root_init;
    root_hist.reserve(size_t(kStates) + 8);
  }

  void apply(const Edit& e) override {
    root_hist.push_back(cur_root);
    this->apply_edit(e, true);
  }
  void undo(const Edit& /*e*/, const SavedRow& s) override {
    if (root_hist.empty()) return;
    cur_root = root_hist.back();
    root_hist.pop_back();
    this->revert_state(s, false);
  }
};

// ---------------------------------------------------------------- factory
inline Arm* make_classic_arm(const std::string& name, const Fixture& f, const Workload& w) {
  if (name == "rf") return new ClassicRfArm(f, w);
  if (name == "rs") return new ClassicRsArm(f, w);
  if (name == "pst") return new ClassicPstArm(f, w);
  return nullptr;
}

}  // namespace bwh
