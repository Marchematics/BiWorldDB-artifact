// fmd_index.hpp — FMDI: a 2D factorized structural backend for the BiWorldDB version-tree
// range-aggregate contract (2026-11-01, sparse-csr / FMDI prototype).
//
// New file; the shipped engine.hpp / panel.cpp / common.hpp / arms_*.hpp are untouched.
//
// Contract (2D generalisation of the 1D version-tree panel):
//   base rows  (c0, c1, amount, group), c0 in [0,C0), c1 in [0,C1), group in [0,G);
//   each version v has current group factors fac_v(g) (initial s0(g)=1000) and a declared
//   per-state list of rectangles [lo0,hi0) x [lo1,hi1) (ad-hoc rectangles are equally
//   supported: the index answers any rectangle, nothing is precomputed per query);
//   row edits replace/insert/delete/move (move changes c0,c1,group); factor edits set fac(g);
//   answer_v(R) = sum over current rows in R of amount * fac_v(group).
//
// FMDI representation (no C0*C1 product anywhere):
//   * static "hierarchical shared base over dim0": one range tree over the dim0 class domain
//     (shared node boundaries); every node carries a 1D index over dim1 --- sorted distinct
//     c1 classes plus a cumulative amount prefix.  base0 weights the base rows by s0; for each
//     factor-active group a per-group range tree stores the group's base amounts.  A
//     rectangle query decomposes dim0 into O(log C0) canonical nodes and does two dim1
//     lower_bounds per node: O(log C0 * log) time, memory O((N+N_active) log C0).
//   * reversible row/factor corrections: current-state deviations live in per-group
//     2D Fenwick-of-vectors over the compressed supports the declared edits can touch
//     (dim0 Fenwick, dim1 sorted vector + inner Fenwick), point update / rectangle query
//     O(log C0 log).  A factor edit is O(1) (a scalar), because the query applies fac(g)
//     to the full per-group sum Base_g(R)+Dev_g(R):
//         answer = base0(R) + sum_g (fac_g-s0_g) Base_g(R) + sum_g fac_g Dev_g(R)
//     (exact: for an edited row the static base term is +fac_gb*base and the deviation is
//     +fac_gcur*cur - fac_gb*base, cancelling the static term).
//   * undo/redo stack (snapshot/rollback) makes the corrections reversible for a DFS over
//     the version tree.
//
// Baselines for evaluation (also new code here):
//   * FullArm   "full-state per-version 2D rebuild": every state rebuilds a dense current
//               mass grid C0*C1 from all rows and its 2D prefix.  Cost N + C0*C1 per state,
//               memory C0*C1 (the product class).
//   * NaiveArm  "naive 2D prefix": per-group dense C0*C1 amount grids + per-group 2D
//               prefixes rebuilt for every group an edit touches; memory G*C0*C1.
//   * OracleArm brute force over the current rows (correctness only).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace fmd {

using i8 = int8_t;
using u8 = uint8_t;
using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;

// ------------------------------------------------------------------ RNG
struct Rng {
  u64 s;
  explicit Rng(u64 seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
  u64 next() {
    u64 z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  u32 bounded(u32 n) { return n ? u32(next() % n) : 0; }
  i64 range(i64 lo, i64 hi) { return hi <= lo ? lo : lo + i64(next() % u64(hi - lo)); }
};

// ------------------------------------------------------------------ workload
enum : u8 { OP_REPLACE = 0, OP_INSERT = 1, OP_DELETE = 2, OP_MOVE = 3, OP_FACTOR = 4 };

struct Edit {
  u8 op;
  u8 pad[3];
  i32 g;
  i32 c0, c1;
  i64 id;
  i64 val;
};
static_assert(sizeof(Edit) == 32, "edit size");

struct Query2 {
  i32 lo0, hi0, lo1, hi1;  // half-open rectangle over class ids
};

struct Config {
  i64 rows = 1000000;
  i32 c0 = 64, c1 = 64;
  i32 groups = 8;
  i64 states = 300000;
  i32 q = 2;
  i32 factor_period = 37;
  i32 topology = 0;  // 0 chain, 1 bushy heap, 2 star
  i64 seed = 7;
};

struct Workload {
  Config cfg;
  std::vector<i32> parent;   // states
  std::vector<Edit> edit;    // states (edit[0] unused)
  std::vector<Query2> query; // states * q
  std::vector<i32> b_c0, b_c1, b_g;
  std::vector<i64> b_amt, s0;
  i32 n_ins = 0;
  std::vector<i32> ins_c0, ins_c1, ins_g;
  std::vector<i64> ins_amt;
};

inline i32 version_parent(i32 topology, i32 v) {
  if (v <= 0) return -1;
  switch (topology) {
    case 0: return v - 1;
    case 1: return (v - 1) / 2;
    default: return 0;
  }
}

inline Workload make_workload(const Config& cfg) {
  Workload w;
  w.cfg = cfg;
  const i64 N = cfg.rows, V = cfg.states;
  Rng rng(u64(cfg.seed) * 1000003ULL + u64(cfg.c0) * 7919ULL + u64(cfg.c1) * 104729ULL + 20261101ULL);
  w.b_c0.resize(size_t(N));
  w.b_c1.resize(size_t(N));
  w.b_g.resize(size_t(N));
  w.b_amt.resize(size_t(N));
  w.s0.assign(size_t(cfg.groups), 1000);
  for (i64 i = 0; i < N; ++i) {
    w.b_c0[size_t(i)] = i32(rng.bounded(u32(cfg.c0)));
    w.b_c1[size_t(i)] = i32(rng.bounded(u32(cfg.c1)));
    w.b_g[size_t(i)] = i32(rng.bounded(u32(cfg.groups)));
    // positive, skewed amounts
    w.b_amt[size_t(i)] = i64(std::exp(5.0 + 1.2 * (double(rng.bounded(10000)) / 10000.0) * 1.7)) + 1;
  }
  w.parent.resize(size_t(V));
  w.edit.resize(size_t(V));
  for (i32 v = 0; v < V; ++v) w.parent[size_t(v)] = version_parent(cfg.topology, v);
  std::vector<i64> gfac(size_t(cfg.groups), 1000);
  i32 next_group = 0;
  i32 next_ins = 0;
  Rng er(u64(cfg.seed) * 1000033ULL + u64(cfg.c0) * 65537ULL + 17ULL);
  for (i64 v = 1; v < V; ++v) {
    Edit e{};
    const bool factor_step = cfg.factor_period > 0 && (v % cfg.factor_period == 0);
    if (factor_step) {
      i32 g = next_group++ % cfg.groups;
      i64 cur = gfac[size_t(g)];
      i64 nf = (cur == 950) ? 1000 : (cur == 1000 ? 1050 : 950);
      gfac[size_t(g)] = nf;
      e.op = OP_FACTOR;
      e.g = g;
      e.val = nf;
    } else {
      u32 roll = er.bounded(100);
      if (roll < 55) {
        e.op = OP_REPLACE;
        e.id = er.range(0, N);
        e.val = w.b_amt[size_t(e.id)] + er.range(-100000, 100000);
      } else if (roll < 75) {
        e.op = OP_INSERT;
        e.id = N + next_ins++;
        e.c0 = i32(er.bounded(u32(cfg.c0)));
        e.c1 = i32(er.bounded(u32(cfg.c1)));
        e.g = i32(er.bounded(u32(cfg.groups)));
        e.val = w.b_amt[size_t(er.range(0, N))] + er.range(-100000, 100000);
        w.ins_c0.push_back(e.c0);
        w.ins_c1.push_back(e.c1);
        w.ins_g.push_back(e.g);
        w.ins_amt.push_back(e.val);
      } else if (roll < 90) {
        e.op = OP_DELETE;
        e.id = er.range(0, N);
      } else {
        e.op = OP_MOVE;
        e.id = er.range(0, N);
        e.c0 = i32(er.bounded(u32(cfg.c0)));
        e.c1 = i32(er.bounded(u32(cfg.c1)));
        e.g = i32(er.bounded(u32(cfg.groups)));
        e.val = w.b_amt[size_t(e.id)] + er.range(-100000, 100000);
      }
    }
    w.edit[size_t(v)] = e;
  }
  w.n_ins = next_ins;
  // declared per-state rectangles (also valid as ad-hoc rectangles: any rectangle is exact)
  w.query.resize(size_t(V) * size_t(cfg.q));
  Rng qr(u64(cfg.seed) * 1000000007ULL + u64(cfg.topology) * 65537ULL + 20261101ULL);
  for (i64 v = 1; v < V; ++v) {
    for (i32 k = 0; k < cfg.q; ++k) {
      Query2& qq = w.query[size_t(v) * size_t(cfg.q) + size_t(k)];
      i32 a = i32(qr.bounded(u32(cfg.c0)));
      i32 b = i32(qr.bounded(u32(cfg.c0)));
      if (a > b) std::swap(a, b);
      if (a == b) b = std::min(cfg.c0, b + 1);
      i32 c = i32(qr.bounded(u32(cfg.c1)));
      i32 d = i32(qr.bounded(u32(cfg.c1)));
      if (c > d) std::swap(c, d);
      if (c == d) d = std::min(cfg.c1, d + 1);
      qq = {a, b, c, d};
    }
  }
  return w;
}

// ------------------------------------------------------------------ static factorized base
// One range tree over the dim0 class domain; each canonical node holds a 1D index over dim1:
// sorted distinct c1 classes + a cumulative amount prefix.  A rectangle query visits
// O(log C0) nodes and does two dim1 lower_bounds per node.
struct FlatDim1Tree {
  i32 size = 0;              // pow2 >= C0
  i32 nodes = 0;             // 2*size
  std::vector<u32> nstart;   // nodes+1: entry offset of each node
  std::vector<u32> ncount;   // nodes: number of distinct c1 entries
  std::vector<u32> c1v;      // flattened sorted distinct dim1 classes
  std::vector<i64> prefv;    // flattened per-node cumulative amount

  struct Pt { i32 c0, c1; i64 amt; };

  void build(i32 C0, const std::vector<Pt>& pts) {
    size = 1;
    while (size < C0) size <<= 1;
    nodes = 2 * size;
    std::vector<u32> cnt(size_t(nodes) + 1, 0);
    for (const Pt& p : pts) {
      for (i32 n = size + p.c0; n >= 1; n >>= 1) ++cnt[size_t(n)];
    }
    nstart.assign(size_t(nodes) + 1, 0);
    for (i32 n = 1; n < nodes; ++n) nstart[size_t(n) + 1] = nstart[size_t(n)] + cnt[size_t(n)];
    const size_t total = size_t(nstart[size_t(nodes)]);
    c1v.assign(total, 0);
    std::vector<i64> amtv(total, 0);
    std::vector<u32> pos(nstart.begin(), nstart.begin() + nodes);
    for (const Pt& p : pts) {
      for (i32 n = size + p.c0; n >= 1; n >>= 1) {
        const u32 k = pos[size_t(n)]++;
        c1v[size_t(k)] = u32(p.c1);
        amtv[size_t(k)] = p.amt;
      }
    }
    ncount.assign(size_t(nodes), 0);
    prefv.assign(total, 0);
    std::vector<std::pair<u32, i64>> tmp;
    for (i32 n = 1; n < nodes; ++n) {
      const u32 b = nstart[size_t(n)], c = cnt[size_t(n)];
      if (c == 0) { ncount[size_t(n)] = 0; continue; }
      tmp.clear();
      tmp.reserve(c);
      for (u32 k = b; k < b + c; ++k) tmp.emplace_back(c1v[size_t(k)], amtv[size_t(k)]);
      std::sort(tmp.begin(), tmp.end(),
                [](const auto& x, const auto& y) { return x.first < y.first; });
      u32 m = 0;
      i64 run = 0;
      for (size_t k = 0; k < tmp.size(); ++k) {
        run += tmp[k].second;
        if (k + 1 == tmp.size() || tmp[k + 1].first != tmp[k].first) {
          c1v[size_t(b + m)] = tmp[k].first;
          prefv[size_t(b + m)] = run;
          ++m;
        }
      }
      ncount[size_t(n)] = m;
      // prefv entries below the node's range are unused after compression
    }
    // rebuild the dense per-node offsets after compression
    u32 acc = 0;
    std::vector<u32> ns(size_t(nodes) + 1, 0);
    for (i32 n = 1; n < nodes; ++n) {
      ns[size_t(n)] = acc;
      acc += ncount[size_t(n)];
    }
    ns[size_t(nodes)] = acc;
    // compact the entries (they were written at the uncompressed offsets)
    std::vector<u32> c1c(acc);
    std::vector<i64> pfc(acc);
    for (i32 n = 1; n < nodes; ++n) {
      const u32 b = nstart[size_t(n)];
      for (u32 k = 0; k < ncount[size_t(n)]; ++k) {
        c1c[size_t(ns[size_t(n)] + k)] = c1v[size_t(b + k)];
        pfc[size_t(ns[size_t(n)] + k)] = prefv[size_t(b + k)];
      }
    }
    c1v.swap(c1c);
    prefv.swap(pfc);
    nstart.swap(ns);
  }

  inline i64 node_range(i32 n, i32 lo1, i32 hi1) const {
    const u32 b = nstart[size_t(n)], c = ncount[size_t(n)];
    if (c == 0) return 0;
    const u32* a = c1v.data() + b;
    const u32 lo = u32(std::lower_bound(a, a + c, u32(lo1)) - a);
    const u32 hi = u32(std::lower_bound(a, a + c, u32(hi1)) - a);
    if (hi <= lo) return 0;
    const i64 s_hi = prefv[size_t(b + hi - 1)];
    const i64 s_lo = lo == 0 ? 0 : prefv[size_t(b + lo - 1)];
    return s_hi - s_lo;
  }
  i64 query(i32 lo0, i32 hi0, i32 lo1, i32 hi1) const {
    i64 s = 0;
    i32 l = lo0 + size, r = hi0 + size;
    while (l < r) {
      if (l & 1) s += node_range(l++, lo1, hi1);
      if (r & 1) s += node_range(--r, lo1, hi1);
      l >>= 1;
      r >>= 1;
    }
    return s;
  }
  size_t bytes() const {
    return nstart.size() * 4 + ncount.size() * 4 + c1v.size() * 4 + prefv.size() * 8;
  }
};

// ------------------------------------------------------------------ dynamic deviations
// Per-group 2D Fenwick: outer Fenwick over dim0 (C0 classes), each outer node holds the sorted
// distinct dim1 classes of the compressed supports in it plus an inner i64 Fenwick.
struct Bit2DGroup {
  i32 C0 = 0;
  std::vector<u32> nstart;  // C0+1
  std::vector<u32> ncount;  // C0
  std::vector<u32> c1v;
  std::vector<u32> bstart;  // C0: offset of each inner Fenwick (size count+1)
  std::vector<i64> bit;
  std::vector<u8> touched;

  void build(i32 c0, const std::vector<std::pair<i32, i32>>& supports) {
    C0 = c0;
    std::vector<u32> cnt(size_t(C0) + 2, 0);
    for (const auto& s : supports) {
      for (i32 n = s.first + 1; n <= C0; n += n & (-n)) ++cnt[size_t(n)];
    }
    nstart.assign(size_t(C0) + 1, 0);
    for (i32 n = 1; n <= C0; ++n) nstart[size_t(n)] = nstart[size_t(n - 1)] + cnt[size_t(n)];
    c1v.assign(size_t(nstart[size_t(C0)]), 0);
    std::vector<u32> pos(nstart.begin(), nstart.begin() + C0);
    for (const auto& s : supports) {
      for (i32 n = s.first + 1; n <= C0; n += n & (-n)) c1v[size_t(pos[size_t(n - 1)]++)] = u32(s.second);
    }
    ncount.assign(size_t(C0), 0);
    bstart.assign(size_t(C0), 0);
    u32 acc = 0;
    for (i32 n = 1; n <= C0; ++n) {
      const u32 b = nstart[size_t(n - 1)], c = cnt[size_t(n)];
      if (c) {
        std::sort(c1v.begin() + b, c1v.begin() + b + c);
        const u32 same = u32(std::unique(c1v.begin() + b, c1v.begin() + b + c) - (c1v.begin() + b));
        ncount[size_t(n - 1)] = same;
      }
      bstart[size_t(n - 1)] = acc;
      acc += ncount[size_t(n - 1)] + 1;
    }
    bit.assign(size_t(acc), 0);
    // compact c1v after unique
    std::vector<u32> cc;
    cc.reserve(c1v.size());
    for (i32 n = 1; n <= C0; ++n) {
      const u32 b = nstart[size_t(n - 1)];
      for (u32 k = 0; k < ncount[size_t(n - 1)]; ++k) cc.push_back(c1v[size_t(b + k)]);
    }
    c1v.swap(cc);
    // rebuild nstart as compact offsets
    u32 a2 = 0;
    for (i32 n = 1; n <= C0; ++n) {
      nstart[size_t(n - 1)] = a2;
      a2 += ncount[size_t(n - 1)];
    }
    nstart[size_t(C0)] = a2;
    touched.assign(size_t(C0), 0);
  }
  inline void add(i32 c0, i32 c1, i64 v) {
    if (v == 0) return;
    for (i32 n = c0 + 1; n <= C0; n += n & (-n)) {
      const i32 ni = n - 1;
      const u32 b = nstart[size_t(ni)], c = ncount[size_t(ni)];
      const u32* a = c1v.data() + b;
      const u32 p = u32(std::lower_bound(a, a + c, u32(c1)) - a);
      if (p < c && a[p] == u32(c1)) {
        i64* inner = bit.data() + bstart[size_t(ni)];
        for (u32 x = p + 1; x <= c; x += x & (-x)) inner[x] += v;
      }
    }
  }
  inline i64 prefix(i32 x0, i32 y1) const {
    if (C0 == 0) return 0;
    i64 s = 0;
    for (i32 n = x0; n > 0; n -= n & (-n)) {
      const i32 ni = n - 1;
      const u32 b = nstart[size_t(ni)], c = ncount[size_t(ni)];
      if (!c) continue;
      const u32* a = c1v.data() + b;
      const u32 hi = u32(std::lower_bound(a, a + c, u32(y1)) - a);
      const i64* inner = bit.data() + bstart[size_t(ni)];
      for (u32 x = hi; x > 0; x -= x & (-x)) s += inner[x];
    }
    return s;
  }
  inline i64 query(i32 lo0, i32 hi0, i32 lo1, i32 hi1) const {
    return prefix(hi0, hi1) - prefix(lo0, hi1) - prefix(hi0, lo1) + prefix(lo0, lo1);
  }
  size_t bytes() const {
    return nstart.size() * 4 + ncount.size() * 4 + c1v.size() * 4 + bstart.size() * 4 +
           bit.size() * 8 + touched.size();
  }
};

// ------------------------------------------------------------------ arm interface
struct SavedRow {
  u8 kind;  // 0 row, 1 factor
  u8 prev_edited;
  i64 id;
  i32 c0, c1, g;
  i64 amount;
  u8 ex;
  i64 fac;
};

struct Arm {
  virtual ~Arm() = default;
  virtual const char* name() const = 0;
  virtual void build() = 0;
  virtual SavedRow snapshot(const Edit& e) const = 0;
  virtual void apply(const Edit& e) = 0;
  virtual void undo(const Edit& e, const SavedRow& s) = 0;
  virtual i64 query(i32 state, i32 k) = 0;
  virtual void stats(i64& static_bytes, i64& product_bytes, i64& work_bytes) const {
    static_bytes = product_bytes = work_bytes = 0;
  }
};

// shared mutable row + factor state (current values; base values are immutable)
struct RowState {
  i64 n_base = 0, n_tot = 0;
  std::vector<i32> c_c0, c_c1, c_g;
  std::vector<i64> c_amt;
  std::vector<u8> c_ex;
  std::vector<i64> fac, s0;
  void init(const Workload& w) {
    n_tot = w.cfg.rows + w.n_ins;
    c_c0.assign(size_t(n_tot), 0);
    c_c1.assign(size_t(n_tot), 0);
    c_g.assign(size_t(n_tot), 0);
    c_amt.assign(size_t(n_tot), 0);
    c_ex.assign(size_t(n_tot), 0);
    for (i64 i = 0; i < w.cfg.rows; ++i) {
      c_c0[size_t(i)] = w.b_c0[size_t(i)];
      c_c1[size_t(i)] = w.b_c1[size_t(i)];
      c_g[size_t(i)] = w.b_g[size_t(i)];
      c_amt[size_t(i)] = w.b_amt[size_t(i)];
      c_ex[size_t(i)] = 1;
    }
    for (i32 k = 0; k < w.n_ins; ++k) {
      const i64 id = w.cfg.rows + k;
      c_c0[size_t(id)] = w.ins_c0[size_t(k)];
      c_c1[size_t(id)] = w.ins_c1[size_t(k)];
      c_g[size_t(id)] = w.ins_g[size_t(k)];
      c_amt[size_t(id)] = 0;
      c_ex[size_t(id)] = 0;
    }
    s0 = w.s0;
    fac = w.s0;
  }
};

// ------------------------------------------------------------------ FMDI arm
struct FmdArm : Arm {
  const Workload& w;
  Config cfg;
  i64 n_base, n_tot;
  FlatDim1Tree base0;                     // base rows weighted by s0
  std::vector<FlatDim1Tree> gt;            // active groups: base amounts by group
  std::vector<Bit2DGroup> dev;             // per-group deviation 2D Fenwick (compressed)
  std::vector<u8> active, dev_ever;
  RowState rs;
  std::vector<u8> edited;
  std::vector<i32> dev_count;
  i64 static_bytes = 0, work_bytes = 0, product_bytes = 0;
  // move destinations per row, used to compress deviation supports
  std::vector<std::vector<std::pair<i32, i32>>> moved_c0c1;
  std::vector<std::vector<i32>> moved_g;

  explicit FmdArm(const Workload& wo) : w(wo), cfg(wo.cfg) {}
  const char* name() const override { return "fmd"; }

  void build() override {
    n_base = cfg.rows;
    n_tot = cfg.rows + w.n_ins;
    rs.init(w);
    edited.assign(size_t(n_tot), 0);
    active.assign(size_t(cfg.groups), 0);
    dev_ever.assign(size_t(cfg.groups), 0);
    dev_count.assign(size_t(cfg.groups), 0);
    product_bytes = i64(cfg.c0) * i64(cfg.c1) * 8;
    // which groups ever see a factor edit? which rows move / are inserted?
    for (i64 v = 1; v < cfg.states; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if (e.op == OP_FACTOR) active[size_t(e.g)] = 1;
    }
    gt.resize(size_t(cfg.groups));
    dev.resize(size_t(cfg.groups));
    moved_c0c1.resize(size_t(n_base));
    moved_g.resize(size_t(n_base));
    for (i64 v = 1; v < cfg.states; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if (e.op == OP_MOVE && e.id < n_base) {
        moved_c0c1[size_t(e.id)].emplace_back(e.c0, e.c1);
        moved_g[size_t(e.id)].push_back(e.g);
      }
    }
    // ---- static trees
    {
      std::vector<FlatDim1Tree::Pt> pts;
      pts.reserve(size_t(n_base));
      for (i64 i = 0; i < n_base; ++i)
        pts.push_back({w.b_c0[size_t(i)], w.b_c1[size_t(i)], w.b_amt[size_t(i)] * w.s0[size_t(w.b_g[size_t(i)])]});
      base0.build(cfg.c0, pts);
      static_bytes += i64(base0.bytes());
    }
    for (i32 g = 0; g < cfg.groups; ++g) {
      if (!active[size_t(g)]) continue;
      std::vector<FlatDim1Tree::Pt> pts;
      for (i64 i = 0; i < n_base; ++i)
        if (w.b_g[size_t(i)] == g)
          pts.push_back({w.b_c0[size_t(i)], w.b_c1[size_t(i)], w.b_amt[size_t(i)]});
      gt[size_t(g)].build(cfg.c0, pts);
      static_bytes += i64(gt[size_t(g)].bytes());
    }
    // ---- deviation supports per group (compressed)
    std::vector<std::vector<std::pair<i32, i32>>> sup(size_t(cfg.groups));
    auto add_sup = [&](i32 g, i32 c0, i32 c1) {
      if (g >= 0 && g < cfg.groups) sup[size_t(g)].emplace_back(c0, c1);
    };
    // rows edited by any REPLACE/DELETE/MOVE (only these create base-row deviations)
    std::vector<u8> row_edited(size_t(n_base), 0);
    for (i64 v = 1; v < cfg.states; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if ((e.op == OP_REPLACE || e.op == OP_DELETE || e.op == OP_MOVE) && e.id < n_base) {
        row_edited[size_t(e.id)] = 1;
        add_sup(w.b_g[size_t(e.id)], w.b_c0[size_t(e.id)], w.b_c1[size_t(e.id)]);
      }
    }
    for (i64 i = 0; i < n_base; ++i) {
      if (!row_edited[size_t(i)]) continue;
      for (size_t k = 0; k < moved_c0c1[size_t(i)].size(); ++k)
        add_sup(moved_g[size_t(i)][k], moved_c0c1[size_t(i)][k].first, moved_c0c1[size_t(i)][k].second);
    }
    for (i32 k = 0; k < w.n_ins; ++k) add_sup(w.ins_g[size_t(k)], w.ins_c0[size_t(k)], w.ins_c1[size_t(k)]);
    for (i32 g = 0; g < cfg.groups; ++g) {
      std::sort(sup[size_t(g)].begin(), sup[size_t(g)].end());
      sup[size_t(g)].erase(std::unique(sup[size_t(g)].begin(), sup[size_t(g)].end()), sup[size_t(g)].end());
      if (!sup[size_t(g)].empty()) {
        dev[size_t(g)].build(cfg.c0, sup[size_t(g)]);
        work_bytes += i64(dev[size_t(g)].bytes());
      }
    }
    static_bytes += work_bytes;
  }

  // ---- deviation updates
  inline void dev_add(i32 g, i32 c0, i32 c1, i64 v) {
    if (v == 0 || g < 0) return;
    dev_ever[size_t(g)] = 1;
    dev[size_t(g)].add(c0, c1, v);
  }
  inline i64 cur_contrib(i64 id) const { return rs.c_ex[size_t(id)] ? rs.c_amt[size_t(id)] : 0; }

  void set_row(i64 id, i32 c0, i32 c1, i32 g, i64 amount, u8 ex) {
    if (edited[size_t(id)]) {
      dev_add(rs.c_g[size_t(id)], rs.c_c0[size_t(id)], rs.c_c1[size_t(id)], -cur_contrib(id));
    } else if (id < n_base) {
      dev_add(w.b_g[size_t(id)], w.b_c0[size_t(id)], w.b_c1[size_t(id)], -w.b_amt[size_t(id)]);
    }
    rs.c_c0[size_t(id)] = c0;
    rs.c_c1[size_t(id)] = c1;
    rs.c_g[size_t(id)] = g;
    rs.c_amt[size_t(id)] = amount;
    rs.c_ex[size_t(id)] = ex;
    edited[size_t(id)] = 1;
    if (ex) dev_add(g, c0, c1, amount);
  }
  void set_factor(i32 g, i64 nf) { rs.fac[size_t(g)] = nf; }

  SavedRow snapshot(const Edit& e) const override {
    SavedRow s{};
    if (e.op == OP_FACTOR) {
      s.kind = 1;
      s.g = e.g;
      s.fac = rs.fac[size_t(e.g)];
    } else {
      s.kind = 0;
      s.id = e.id;
      s.prev_edited = edited[size_t(e.id)];
      s.c0 = rs.c_c0[size_t(e.id)];
      s.c1 = rs.c_c1[size_t(e.id)];
      s.g = rs.c_g[size_t(e.id)];
      s.amount = rs.c_amt[size_t(e.id)];
      s.ex = rs.c_ex[size_t(e.id)];
    }
    return s;
  }
  void apply(const Edit& e) override {
    switch (e.op) {
      case OP_REPLACE: set_row(e.id, rs.c_c0[size_t(e.id)], rs.c_c1[size_t(e.id)], rs.c_g[size_t(e.id)], e.val, 1); break;
      case OP_INSERT: set_row(e.id, e.c0, e.c1, e.g, e.val, 1); break;
      case OP_DELETE: set_row(e.id, rs.c_c0[size_t(e.id)], rs.c_c1[size_t(e.id)], rs.c_g[size_t(e.id)], rs.c_amt[size_t(e.id)], 0); break;
      case OP_MOVE: set_row(e.id, e.c0, e.c1, e.g, e.val, 1); break;
      case OP_FACTOR: set_factor(e.g, e.val); break;
    }
  }
  void undo(const Edit& e, const SavedRow& s) override {
    if (s.kind == 1) { set_factor(s.g, s.fac); return; }
    const i64 id = s.id;
    if (edited[size_t(id)]) {
      dev_add(rs.c_g[size_t(id)], rs.c_c0[size_t(id)], rs.c_c1[size_t(id)], -cur_contrib(id));
    }
    if (s.prev_edited) {
      rs.c_c0[size_t(id)] = s.c0;
      rs.c_c1[size_t(id)] = s.c1;
      rs.c_g[size_t(id)] = s.g;
      rs.c_amt[size_t(id)] = s.amount;
      rs.c_ex[size_t(id)] = s.ex;
      edited[size_t(id)] = 1;
      if (s.ex) dev_add(s.g, s.c0, s.c1, s.amount);
    } else {
      rs.c_c0[size_t(id)] = s.c0;
      rs.c_c1[size_t(id)] = s.c1;
      rs.c_g[size_t(id)] = s.g;
      rs.c_amt[size_t(id)] = s.amount;
      rs.c_ex[size_t(id)] = s.ex;
      edited[size_t(id)] = 0;
      if (id < n_base) dev_add(w.b_g[size_t(id)], w.b_c0[size_t(id)], w.b_c1[size_t(id)], w.b_amt[size_t(id)]);
    }
  }

  i64 query(i32 state, i32 k) override {
    const Query2& qq = w.query[size_t(state) * size_t(cfg.q) + size_t(k)];
    i64 ans = base0.query(qq.lo0, qq.hi0, qq.lo1, qq.hi1);
    for (i32 g = 0; g < cfg.groups; ++g) {
      const i64 d = rs.fac[size_t(g)] - w.s0[size_t(g)];
      if (d && active[size_t(g)]) ans += d * gt[size_t(g)].query(qq.lo0, qq.hi0, qq.lo1, qq.hi1);
      if (dev_ever[size_t(g)]) ans += rs.fac[size_t(g)] * dev[size_t(g)].query(qq.lo0, qq.hi0, qq.lo1, qq.hi1);
    }
    return ans;
  }
  void stats(i64& sb, i64& pb, i64& wb) const override { sb = static_bytes; pb = product_bytes; wb = work_bytes; }
};

// ------------------------------------------------------------------ full-state baseline
struct FullArm : Arm {
  const Workload& w;
  Config cfg;
  RowState rs;
  i64 last_state = -1;
  std::vector<i64> grid, pref;  // grid C0*C1, pref (C0+1)*(C1+1)
  explicit FullArm(const Workload& wo) : w(wo), cfg(wo.cfg) {}
  const char* name() const override { return "full"; }
  void build() override {
    rs.init(w);
    grid.assign(size_t(cfg.c0) * size_t(cfg.c1), 0);
    pref.assign(size_t(cfg.c0 + 1) * size_t(cfg.c1 + 1), 0);
  }
  SavedRow snapshot(const Edit& e) const override {
    SavedRow s{};
    if (e.op == OP_FACTOR) { s.kind = 1; s.g = e.g; s.fac = rs.fac[size_t(e.g)]; }
    else {
      s.kind = 0; s.id = e.id; s.prev_edited = 0;
      s.c0 = rs.c_c0[size_t(e.id)]; s.c1 = rs.c_c1[size_t(e.id)]; s.g = rs.c_g[size_t(e.id)];
      s.amount = rs.c_amt[size_t(e.id)]; s.ex = rs.c_ex[size_t(e.id)];
    }
    return s;
  }
  void set_row(i64 id, i32 c0, i32 c1, i32 g, i64 amount, u8 ex) {
    rs.c_c0[size_t(id)] = c0; rs.c_c1[size_t(id)] = c1; rs.c_g[size_t(id)] = g;
    rs.c_amt[size_t(id)] = amount; rs.c_ex[size_t(id)] = ex;
  }
  void apply(const Edit& e) override {
    switch (e.op) {
      case OP_REPLACE: set_row(e.id, rs.c_c0[size_t(e.id)], rs.c_c1[size_t(e.id)], rs.c_g[size_t(e.id)], e.val, 1); break;
      case OP_INSERT: set_row(e.id, e.c0, e.c1, e.g, e.val, 1); break;
      case OP_DELETE: set_row(e.id, rs.c_c0[size_t(e.id)], rs.c_c1[size_t(e.id)], rs.c_g[size_t(e.id)], rs.c_amt[size_t(e.id)], 0); break;
      case OP_MOVE: set_row(e.id, e.c0, e.c1, e.g, e.val, 1); break;
      case OP_FACTOR: rs.fac[size_t(e.g)] = e.val; break;
    }
  }
  void undo(const Edit& e, const SavedRow& s) override {
    if (s.kind == 1) rs.fac[size_t(s.g)] = s.fac;
    else set_row(s.id, s.c0, s.c1, s.g, s.amount, s.ex);
  }
  void rebuild() {
    std::fill(grid.begin(), grid.end(), 0);
    for (i64 id = 0; id < rs.n_tot; ++id) {
      if (!rs.c_ex[size_t(id)]) continue;
      grid[size_t(rs.c_c0[size_t(id)]) * size_t(cfg.c1) + size_t(rs.c_c1[size_t(id)])] +=
          rs.c_amt[size_t(id)] * rs.fac[size_t(rs.c_g[size_t(id)])];
    }
    const i32 W = cfg.c0 + 1;
    std::fill(pref.begin(), pref.end(), 0);
    for (i32 i = 1; i <= cfg.c0; ++i) {
      i64 rowsum = 0;
      for (i32 j = 1; j <= cfg.c1; ++j) {
        rowsum += grid[size_t(i - 1) * size_t(cfg.c1) + size_t(j - 1)];
        pref[size_t(i) * size_t(W) + size_t(j)] = pref[size_t(i - 1) * size_t(W) + size_t(j)] + rowsum;
      }
    }
  }
  i64 query(i32 state, i32 k) override {
    if (last_state != state) { rebuild(); last_state = state; }
    const Query2& q = w.query[size_t(state) * size_t(cfg.q) + size_t(k)];
    const i32 W = cfg.c0 + 1;
    auto S = [&](i32 x, i32 y) -> i64 {
      if (x <= 0 || y <= 0) return 0;
      return pref[size_t(x) * size_t(W) + size_t(y)];
    };
    return S(q.hi0, q.hi1) - S(q.lo0, q.hi1) - S(q.hi0, q.lo1) + S(q.lo0, q.lo1);
  }
  void stats(i64& sb, i64& pb, i64& wb) const override {
    sb = 0;
    pb = i64(cfg.c0) * i64(cfg.c1) * 8 + i64(cfg.c0 + 1) * i64(cfg.c1 + 1) * 8;
    wb = pb;
  }
};

// ------------------------------------------------------------------ naive per-group 2D prefix
struct NaiveArm : Arm {
  const Workload& w;
  Config cfg;
  RowState rs;
  std::vector<i64> ggrid;  // G * C0 * C1 amounts (unfactored)
  std::vector<i64> gpref;  // G * (C0+1)*(C1+1)
  std::vector<u8> dirty;
  std::vector<u8> dev_used;
  explicit NaiveArm(const Workload& wo) : w(wo), cfg(wo.cfg) {}
  const char* name() const override { return "naive"; }
  void build() override {
    rs.init(w);
    ggrid.assign(size_t(cfg.groups) * size_t(cfg.c0) * size_t(cfg.c1), 0);
    gpref.assign(size_t(cfg.groups) * size_t(cfg.c0 + 1) * size_t(cfg.c1 + 1), 0);
    dirty.assign(size_t(cfg.groups), 1);
    for (i64 id = 0; id < rs.n_tot; ++id) {
      if (!rs.c_ex[size_t(id)]) continue;
      ggrid[size_t(rs.c_g[size_t(id)]) * size_t(cfg.c0) * size_t(cfg.c1) +
            size_t(rs.c_c0[size_t(id)]) * size_t(cfg.c1) + size_t(rs.c_c1[size_t(id)])] += rs.c_amt[size_t(id)];
    }
  }
  inline void grid_add(i32 g, i32 c0, i32 c1, i64 v) {
    if (v == 0) return;
    ggrid[size_t(g) * size_t(cfg.c0) * size_t(cfg.c1) + size_t(c0) * size_t(cfg.c1) + size_t(c1)] += v;
    dirty[size_t(g)] = 1;
  }
  void set_row(i64 id, i32 c0, i32 c1, i32 g, i64 amount, u8 ex) {
    if (rs.c_ex[size_t(id)]) grid_add(rs.c_g[size_t(id)], rs.c_c0[size_t(id)], rs.c_c1[size_t(id)], -rs.c_amt[size_t(id)]);
    rs.c_c0[size_t(id)] = c0; rs.c_c1[size_t(id)] = c1; rs.c_g[size_t(id)] = g;
    rs.c_amt[size_t(id)] = amount; rs.c_ex[size_t(id)] = ex;
    if (ex) grid_add(g, c0, c1, amount);
  }
  SavedRow snapshot(const Edit& e) const override {
    SavedRow s{};
    if (e.op == OP_FACTOR) { s.kind = 1; s.g = e.g; s.fac = rs.fac[size_t(e.g)]; }
    else {
      s.kind = 0; s.id = e.id; s.prev_edited = 0;
      s.c0 = rs.c_c0[size_t(e.id)]; s.c1 = rs.c_c1[size_t(e.id)]; s.g = rs.c_g[size_t(e.id)];
      s.amount = rs.c_amt[size_t(e.id)]; s.ex = rs.c_ex[size_t(e.id)];
    }
    return s;
  }
  void apply(const Edit& e) override {
    switch (e.op) {
      case OP_REPLACE: set_row(e.id, rs.c_c0[size_t(e.id)], rs.c_c1[size_t(e.id)], rs.c_g[size_t(e.id)], e.val, 1); break;
      case OP_INSERT: set_row(e.id, e.c0, e.c1, e.g, e.val, 1); break;
      case OP_DELETE: set_row(e.id, rs.c_c0[size_t(e.id)], rs.c_c1[size_t(e.id)], rs.c_g[size_t(e.id)], rs.c_amt[size_t(e.id)], 0); break;
      case OP_MOVE: set_row(e.id, e.c0, e.c1, e.g, e.val, 1); break;
      case OP_FACTOR: rs.fac[size_t(e.g)] = e.val; break;
    }
  }
  void undo(const Edit& e, const SavedRow& s) override {
    if (s.kind == 1) rs.fac[size_t(s.g)] = s.fac;
    else set_row(s.id, s.c0, s.c1, s.g, s.amount, s.ex);
  }
  void rebuild_group(i32 g) {
    const i64 plane = i64(cfg.c0) * i64(cfg.c1);
    const i32 W = cfg.c0 + 1;
    i64* pre = gpref.data() + size_t(g) * size_t(cfg.c0 + 1) * size_t(cfg.c1 + 1);
    std::fill(pre, pre + size_t(cfg.c0 + 1) * size_t(cfg.c1 + 1), 0);
    const i64* gr = ggrid.data() + size_t(g) * size_t(plane);
    for (i32 i = 1; i <= cfg.c0; ++i) {
      i64 rowsum = 0;
      for (i32 j = 1; j <= cfg.c1; ++j) {
        rowsum += gr[size_t(i - 1) * size_t(cfg.c1) + size_t(j - 1)];
        pre[size_t(i) * size_t(W) + size_t(j)] = pre[size_t(i - 1) * size_t(W) + size_t(j)] + rowsum;
      }
    }
  }
  i64 query(i32 state, i32 k) override {
    for (i32 g = 0; g < cfg.groups; ++g)
      if (dirty[size_t(g)]) { rebuild_group(g); dirty[size_t(g)] = 0; }
    const Query2& q = w.query[size_t(state) * size_t(cfg.q) + size_t(k)];
    const i32 W = cfg.c0 + 1;
    i64 ans = 0;
    for (i32 g = 0; g < cfg.groups; ++g) {
      const i64 f = rs.fac[size_t(g)];
      if (f == 0) continue;
      const i64* pre = gpref.data() + size_t(g) * size_t(W) * size_t(cfg.c1 + 1);
      auto S = [&](i32 x, i32 y) -> i64 {
        if (x <= 0 || y <= 0) return 0;
        return pre[size_t(x) * size_t(W) + size_t(y)];
      };
      const i64 v = S(q.hi0, q.hi1) - S(q.lo0, q.hi1) - S(q.hi0, q.lo1) + S(q.lo0, q.lo1);
      ans += f * v;
    }
    return ans;
  }
  void stats(i64& sb, i64& pb, i64& wb) const override {
    sb = 0;
    pb = i64(cfg.groups) * i64(cfg.c0) * i64(cfg.c1) * 8 +
         i64(cfg.groups) * i64(cfg.c0 + 1) * i64(cfg.c1 + 1) * 8;
    wb = pb;
  }
};

// ------------------------------------------------------------------ brute-force oracle
struct OracleArm : Arm {
  const Workload& w;
  Config cfg;
  RowState rs;
  explicit OracleArm(const Workload& wo) : w(wo), cfg(wo.cfg) {}
  const char* name() const override { return "oracle"; }
  void build() override { rs.init(w); }
  SavedRow snapshot(const Edit& e) const override {
    SavedRow s{};
    if (e.op == OP_FACTOR) { s.kind = 1; s.g = e.g; s.fac = rs.fac[size_t(e.g)]; }
    else {
      s.kind = 0; s.id = e.id; s.prev_edited = 0;
      s.c0 = rs.c_c0[size_t(e.id)]; s.c1 = rs.c_c1[size_t(e.id)]; s.g = rs.c_g[size_t(e.id)];
      s.amount = rs.c_amt[size_t(e.id)]; s.ex = rs.c_ex[size_t(e.id)];
    }
    return s;
  }
  void undo(const Edit& e, const SavedRow& s) override {
    (void)e;
    if (s.kind == 1) rs.fac[size_t(s.g)] = s.fac;
    else {
      rs.c_c0[size_t(s.id)] = s.c0; rs.c_c1[size_t(s.id)] = s.c1; rs.c_g[size_t(s.id)] = s.g;
      rs.c_amt[size_t(s.id)] = s.amount; rs.c_ex[size_t(s.id)] = s.ex;
    }
  }
  void apply(const Edit& e) override {
    switch (e.op) {
      case OP_REPLACE: rs.c_amt[size_t(e.id)] = e.val; rs.c_ex[size_t(e.id)] = 1; break;
      case OP_INSERT: rs.c_c0[size_t(e.id)] = e.c0; rs.c_c1[size_t(e.id)] = e.c1; rs.c_g[size_t(e.id)] = e.g;
        rs.c_amt[size_t(e.id)] = e.val; rs.c_ex[size_t(e.id)] = 1; break;
      case OP_DELETE: rs.c_ex[size_t(e.id)] = 0; break;
      case OP_MOVE: rs.c_c0[size_t(e.id)] = e.c0; rs.c_c1[size_t(e.id)] = e.c1; rs.c_g[size_t(e.id)] = e.g;
        rs.c_amt[size_t(e.id)] = e.val; rs.c_ex[size_t(e.id)] = 1; break;
      case OP_FACTOR: rs.fac[size_t(e.g)] = e.val; break;
    }
  }
  i64 query(i32 state, i32 k) override {
    const Query2& q = w.query[size_t(state) * size_t(cfg.q) + size_t(k)];
    i64 ans = 0;
    for (i64 id = 0; id < rs.n_tot; ++id) {
      if (!rs.c_ex[size_t(id)]) continue;
      if (rs.c_c0[size_t(id)] < q.lo0 || rs.c_c0[size_t(id)] >= q.hi0) continue;
      if (rs.c_c1[size_t(id)] < q.lo1 || rs.c_c1[size_t(id)] >= q.hi1) continue;
      ans += rs.c_amt[size_t(id)] * rs.fac[size_t(rs.c_g[size_t(id)])];
    }
    return ans;
  }
};

inline Arm* make_arm(const std::string& name, const Workload& w) {
  if (name == "fmd") return new FmdArm(w);
  if (name == "full") return new FullArm(w);
  if (name == "naive") return new NaiveArm(w);
  if (name == "oracle") return new OracleArm(w);
  return nullptr;
}

}  // namespace fmd
