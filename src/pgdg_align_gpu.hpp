// pgdg_align.hpp — strongest same-information PGD-G baseline with BiWorldDB's execution
// capabilities (2026-10-31, pgdg-align).  New file; the shipped engine.hpp / panel.cpp /
// arms_strong.hpp / common.hpp are untouched.
//
// The shipped PGD-G baseline (arms_strong.hpp, PgdArm) keeps the same per-group immutable
// base prefixes and lazy factor scalars as the hybrid decomposition but is executed like a
// generic baseline: per-state virtual calls, a reversible DFS with snapshot/undo records,
// per-group std::map delta overlays, and a per-query loop over changed-factor groups.  This
// file gives the same algebra the execution capabilities of the shipped BiWorldDB engines so
// that the remaining difference is algorithmic, not overhead:
//
//   PGD-G algebra (unchanged):
//     Q_v(I) = BASE0(I)
//            + sum_{g: fac_v(g) != s0(g)} (fac_v(g)-s0(g)) * B_g(I)
//            + sum_g fac_v(g) * Delta_{v,g}(I)
//   with BASE0 the initial-factor-weighted immutable base prefix, B_g the immutable per-group
//   base mass and Delta_{v,g} = current-base mass of edited rows.  PGD-G keeps B_g as a
//   per-group base prefix and the factor correction as a per-query loop over groups; it does
//   not fold the factor correction into an eager per-class prefix.
//
//   (1) COMPILED / OFFLINE EXECUTION (analogous to batch_engine.hpp).
//       The declared edit list and query bag are known offline, so every row edit's previous
//       state (existence, group, endpoint class, amount) and every factor edit's previous
//       value are resolved statically with a row-id radix sort plus a preorder ancestor
//       stack; the timed traversal never touches a hash overlay or a snapshot record.  One
//       preorder batch pass applies the precomputed records and undoes only when it leaves a
//       subtree (chain trees do zero undo work).  Dynamic deviations are kept in a contiguous
//       per-group support row plus one global Fenwick over the *union of classes reachable by
//       row edits*, so a query pays two Fenwick prefixes rather than a per-dirty-group loop.
//   (2) SPARSE EXECUTION (rank-LUT, analogous to sparse_engine.hpp).
//       B_g is stored as a counting-sort CSR of its nonzero (class, amount) supports with a
//       C-bit class bitmap and a 64-class cumulative rank LUT; B_g(I) = pre[rank(j)] - pre[rank(i)]
//       with no per-query bisection.  Delta supports are likewise per-group sparse.
//   (3) CORRECTION-PREFIX EXECUTION (the same precomputation class the shipped hybrid reaches
//       with BWH_CORR_PREFIX=auto).  When the class domain and the declared factor-edit count
//       make the O(classes)-per-factor-edit maintenance cheaper than the per-query loop, the
//       factor correction is kept as one eager per-class prefix, so a query reads it with two
//       array reads.  This is an execution plan over the PGD-G algebra, not a change to the
//       information the baseline is allowed to use.
//
// Exactness: all four arms evaluate the PGD-G decomposition with integer prefix arrays; the
// answer vector is bit-identical to the shipped hybrid arm and to the committed oracles.  The
// build agrees with sparse_engine.hpp's rank convention (rank(c) = # supports with class < c).
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.hpp"
#include "batch_engine.hpp"  // BatchWorkload / make_workload_fast / radix sorts / DenseFenwick
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace bwh {

inline bool& pgdg_omp_plan_par_flag() {
  static bool v = true;
  return v;
}

inline int pgdg_omp_max_threads() {
#if defined(_OPENMP)
  return omp_get_max_threads();
#else
  return 1;
#endif
}

// Parallel Fenwick rebuild from leaves.  Produces exactly the tree that
// DenseFenwick::rebuild_from_leaves builds (tree[i] = sum of leaves[i-lsb(i)..i]) but uses a
// block prefix scan (block prefixes in parallel, block offsets sequential over ~n/B blocks,
// per-block local prefix parallel) plus a parallel tree pass, so the O(classes) rebuild after
// a factor edit can use the whole team.  Integer additions are exact, so the tree is identical.
inline void pgdg_omp_rebuild_fenwick(DenseFenwick& gf, const std::vector<i64>& leaf) {
#if defined(_OPENMP)
  const i32 n = gf.n;
  if (n > 0 && omp_get_max_threads() > 1) {
    const i32 B = 256;
    const i32 nb = (n + B - 1) / B;
    std::vector<i64> pref(size_t(n) + 1, 0);
    std::vector<i64> bs(size_t(nb), 0);
    // pass 1: per-block running sums and local prefixes
#pragma omp parallel for schedule(static)
    for (i32 b = 0; b < nb; ++b) {
      const i32 e0 = b * B + 1, e1 = std::min<i32>(n, (b + 1) * B);
      i64 run = 0;
      for (i32 i = e0; i <= e1; ++i) { run += leaf[size_t(i)]; pref[size_t(i)] = run; }
      bs[size_t(b)] = run;
    }
    // block offsets (tiny sequential exclusive scan)
    std::vector<i64> boff(size_t(nb) + 1, 0);
    for (i32 b = 0; b < nb; ++b) boff[size_t(b) + 1] = boff[size_t(b)] + bs[size_t(b)];
#pragma omp parallel for schedule(static)
    for (i32 b = 0; b < nb; ++b) {
      const i32 e0 = b * B + 1, e1 = std::min<i32>(n, (b + 1) * B);
      const i64 off = boff[size_t(b)];
      for (i32 i = e0; i <= e1; ++i) pref[size_t(i)] += off;
    }
#pragma omp parallel for schedule(static)
    for (i32 i = 1; i <= n; ++i)
      gf.tree[size_t(i)] = pref[size_t(i)] - pref[size_t(i - (i & (-i)))];
    return;
  }
#endif
  gf.rebuild_from_leaves(leaf);
}

// ---------------------------------------------------------------- sparse base (rank LUT)
// Same shape as sparse_engine.hpp's CsrBase: one u64 class bitmap per active group, a
// 64-class cumulative-support rank LUT, and the cumulative support-amount prefix indexed by
// rank, so rank(c) needs one LUT read, one bitmap word and one popcount.
struct PgdCsrBase {
  std::vector<u32> ptr;   // n_active+1 support offsets
  std::vector<u64> bits;  // n_active * wpr class bitmap (1 = support at that class)
  std::vector<u32> rlut;  // n_active * (wpr+1) cumulative support counts per 64-class word
  std::vector<i64> pre;   // nnz cumulative base amounts inside every group slice
  std::vector<u32> cls;   // nnz support classes, ascending inside each slice (corr updates)
  i32 wpr = 0;
  i64 nnz = 0;

  inline i32 rank(i32 r, i64 c) const {
    if (c <= 0) return 0;
    const i64 b = c >> 6;
    if (b >= i64(wpr)) return i32(i64(ptr[size_t(r) + 1]) - i64(ptr[size_t(r)]));
    const u32 base = rlut[size_t(r) * size_t(wpr + 1) + size_t(b)];
    const i32 rem = i32(c) & 63;
    if (rem == 0) return i32(base);
    const u64 w = bits[size_t(r) * size_t(wpr) + size_t(b)];
    return i32(base) + i32(__builtin_popcountll(w & ((u64(1) << rem) - 1)));
  }
  inline i64 range(i32 r, i64 i, i64 j) const {
    const i32 a = rank(r, i);
    const i32 b = rank(r, j);
    if (a == b) return 0;
    const i64 base = i64(ptr[size_t(r)]);
    const i64 s_hi = b == 0 ? 0 : pre[size_t(base + b - 1)];
    const i64 s_lo = a == 0 ? 0 : pre[size_t(base + a - 1)];
    return s_hi - s_lo;
  }
  inline size_t bytes() const {
    return ptr.size() * sizeof(u32) + bits.size() * sizeof(u64) +
           rlut.size() * sizeof(u32) + pre.size() * sizeof(i64) + cls.size() * sizeof(u32);
  }
};

struct PgdAlignStats {
  i32 n_active = 0;
  i32 n_groups = 0;
  i64 classes = 0;         // class-domain size
  i64 union_classes = 0;   // classes reachable by row edits (dynamic Fenwick size)
  i64 span = 0;
  bool dense_frame = false;
  bool sparse_base = false;
  bool corr = false;
  i64 n_factor_edits = 0;
  i64 n_ins = 0;
  i64 nnz = 0;
  i64 base_bytes = 0;       // persistent base representation
  i64 dense_base_bytes = 0; // what the dense per-group table would have cost
  i64 plan_bytes = 0;
  i64 build_us = 0;
  i64 plan_us = 0;
  i64 base_us = 0;
};

// ---------------------------------------------------------------- GPU-exported aligned plan
// Plain (CUDA-free) view of the aligned PGD-G plan consumed by the GPU block-parallel
// executor.  The algebra is the unified weighted-mass decomposition
//   V_v(c) = sum_g fac_v(g) * (baseLeaf_g(c) + dev_v(g,c)),
// which is exactly the aligned PGD-G identity; per block the GPU replays the same delta events
// as the CPU baseline, so the answer vector is bit-identical.
struct PgdGpuExport {
  i32 kstates = 0, qn = 0, G = 0, n_active = 0, nclasses = 0, Cpfx = 0;
  std::vector<u8> kind;              // per version: 1 row, 2 factor
  std::vector<i32> go, gn, co, cn, tio, tin;
  std::vector<i64> ea, eb;           // row: amt_old/amt_new; factor: fac_val/fac_prev
  std::vector<i32> g2row;            // G
  std::vector<i32> qlb, qhb;         // kstates * qn (class frame bounds)
  std::vector<u32> ms_off;           // n_active+1 merged-support offsets
  std::vector<i64> ms_cls, ms_base;  // merged support: class, immutable base amount
  std::vector<i32> ms_dev;           // merged support: deviation index or -1
  std::vector<u32> tcoord_size;      // n_active: deviation support size (devbuf region)
  std::vector<i32> g_of_row;         // n_active: active row -> group id
  std::vector<i64> vbase;            // Cpfx leaves of V at block-0 start (index c+1)
};

// ---------------------------------------------------------------- compiled PGD-G engine
class PgdAlignEngine {
 public:
  enum Variant { DENSE = 0, DENSE_CORR = 1, SPARSE = 2, SPARSE_CORR = 3 };

  const Fixture& f;
  const BatchWorkload& w;
  i32 G = 0, qn = 0;
  i64 N = 0;
  bool dense_frame = false;
  bool sparse = false;  // sparse base representation
  bool corr = false;    // eager per-class correction prefix
  i32 nclasses = 0;     // class index range [0, nclasses-1] for rows; queries in [0,nclasses]
  i32 NB = 0;           // prefix-array length = nclasses+1
  i32 n_active = 0;

  // class lookup (bucketed LUT over the declared endpoint domain; shift instead of division)
  std::vector<i32> cl_lut;
  i64 lut_tmin = 0;
  i32 lut_shift = 0, lut_buckets = 0;

  // static base
  std::vector<i64> base0;          // NB: initial-factor weighted base prefix
  bool base_class_major = true;    // dense layout: class-major [k*n_active+r] or row-major [r*NB+k]
  std::vector<i64> baseP;          // dense base prefix table in the layout above
  PgdCsrBase csr;                  // sparse base
  std::vector<i64> corr_pref;      // NB when corr
  std::vector<i32> g2row;          // group -> active row, -1 when not factor-touched
  std::vector<i64> s0, fac;        // per-group initial / current factors
  std::vector<i64> row_fac, row_s0;  // per active row copies (factor loop)

  // dynamic deviation index over the union of classes reachable by row edits
  std::vector<i64> ucls;           // sorted union classes
  std::vector<i32> cl2fen;         // size nclasses+1: cl2fen[c] = # union classes < c
  std::vector<std::vector<i32>> tcoord;  // per group: sorted union ranks touched by its edits
  std::vector<std::vector<i64>> tdev;    // per group: current deviation at those ranks
  std::vector<i64> W;              // Fenwick leaves, W[rank+1]
  DenseFenwick gfen;

  // offline plan
  struct NodePlan {
    u8 kind = 0;  // 1 = row edit, 2 = factor edit
    u8 ex_old = 0, ex_new = 0;
    i32 g_old = -1, c_old = -1, r_old = -1, ti_old = -1;
    i32 g_new = -1, c_new = -1, r_new = -1, ti_new = -1;
    i64 amt_old = 0, amt_new = 0;
    i32 fg = -1, frow = -1;
    i64 fac_prev = 0, fac_val = 0;
  };
  std::vector<NodePlan> plan;  // kStates
  std::vector<i32> parent, order;
  std::vector<i32> qlb, qhb;   // class bounds per (state*q + k)

  PgdAlignStats stats;
  i32 omp_threads_ = 1;
  i64 exec_par_min_ = 65536;  // work threshold before spawn-level OpenMP is worth it

  PgdAlignEngine(const Fixture& fi, const BatchWorkload& wo) : f(fi), w(wo) {}

  // Row coordinates are clamped into the class range: a row lives at a coordinate in
  // [0, nclasses-1] in the dense frame.  Declared query bounds are *frame boundaries*, not row
  // coordinates: the upper bound may be t_max+1, which maps to the boundary coordinate nclasses
  // (the prefix arrays have length nclasses+1 and every row lies below it).  Clamping a query
  // bound to nclasses-1 would drop the rows at t = t_max, so query bounds use frame_bound().
  inline i64 class_of(i64 t) const {
    if (dense_frame) {
      i64 c = t - f.h.t_min;
      if (c < 0) c = 0;
      if (c >= i64(nclasses)) c = nclasses - 1;
      return c;
    }
    if (cl_lut.empty())
      return i64(std::upper_bound(w.endpoints.begin(), w.endpoints.end(), t) - w.endpoints.begin());
    i64 j = (t - lut_tmin) >> lut_shift;
    if (j < 0) j = 0;
    if (j >= i64(lut_buckets)) j = i64(lut_buckets) - 1;
    i32 lo = cl_lut[size_t(j)], hi = cl_lut[size_t(j) + 1];
    if (hi - lo > 32)
      return i64(std::upper_bound(w.endpoints.begin() + lo, w.endpoints.begin() + hi, t) -
                 w.endpoints.begin());
    while (lo < hi && w.endpoints[size_t(lo)] <= t) ++lo;
    return i64(lo);
  }

  // Declared-range frame bound.  Dense frame: unclamped coordinate t - t_min clamped to the
  // closed boundary range [0, nclasses]; sparse frame: the class index of a declared endpoint
  // (every declared bound is in the endpoint domain, so this is exact).
  inline i32 frame_bound(i64 t) const {
    if (dense_frame) {
      i64 c = t - f.h.t_min;
      if (c < 0) c = 0;
      if (c > i64(nclasses)) c = nclasses;
      return i32(c);
    }
    return i32(class_of(t));
  }

  bool build(Variant variant) {
    const auto t0 = std::chrono::steady_clock::now();
    auto us = [](std::chrono::steady_clock::time_point a) {
      return std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::steady_clock::now() - a)
          .count();
    };
    sparse = (variant == SPARSE || variant == SPARSE_CORR);
    corr = (variant == DENSE_CORR || variant == SPARSE_CORR);
    G = f.h.groups;
    N = f.h.n;
    qn = w.q;
    s0 = f.base_factor;
    fac = f.base_factor;
    omp_threads_ = i32(pgdg_omp_max_threads());
    if (std::getenv("BWH_PGDG_OMP_ENGINE_OFF")) omp_threads_ = 1;
    pgdg_omp_plan_par_flag() = (std::getenv("BWH_PGDG_OMP_PLAN_OFF") == nullptr);
    if (const char* em = std::getenv("BWH_PGDG_OMP_EXEC_MIN"))
      exec_par_min_ = std::atoll(em);
    // same density predicate as the shipped plan
    i64 Urow = 0;
    for (i32 v = 1; v < kStates; ++v) {
      const u8 op = w.edit[size_t(v)].op;
      if (op != OP_FACTOR) ++Urow;
    }
    const i64 span = f.h.t_max - f.h.t_min + 1;
    dense_frame = (__int128(G) * __int128(span) <= 2 * (__int128(N) + __int128(Urow))) &&
                  span <= (i64(1) << 30);
    if (dense_frame) {
      nclasses = i32(span);
    } else {
      nclasses = i32(w.endpoints.size()) + 1;
    }
    NB = nclasses + 1;
    stats.span = span;
    stats.dense_frame = dense_frame;
    stats.n_groups = G;
    stats.classes = nclasses;
    stats.n_ins = w.n_inserted;
    // ---- class lookup LUT (same construction as the shipped/batch engines)
    if (!dense_frame && w.endpoints.size() >= 1024) {
      const i32 want = std::max<i32>(4096, std::min<i32>(1 << 20,
                                     i32(4 * w.endpoints.size() + 1)));
      i32 buckets = 1;
      while (buckets < want) buckets <<= 1;
      i64 width = std::max<i64>(1, (span + buckets - 1) / buckets);
      lut_shift = 0;
      while ((i64(1) << lut_shift) < width) ++lut_shift;
      const i64 wd = i64(1) << lut_shift;
      i32 nb = i32((span + wd - 1) / wd);
      if (nb < 1) nb = 1;
      if (nb > (1 << 20)) nb = 1 << 20;
      lut_buckets = nb;
      lut_tmin = f.h.t_min;
      cl_lut.assign(size_t(nb) + 1, 0);
      size_t ep = 0;
      for (i32 j = 0; j <= nb; ++j) {
        const i64 val = lut_tmin + i64(j) * wd;
        while (ep < w.endpoints.size() && w.endpoints[size_t(ep)] <= val) ++ep;
        cl_lut[size_t(j)] = i32(ep);
      }
    }
    // ---- active groups: exactly those a declared factor edit can touch
    g2row.assign(size_t(G), -1);
    {
      std::vector<u8> touched(size_t(G), 0);
      for (i32 v = 1; v < kStates; ++v) {
        const Edit& e = w.edit[size_t(v)];
        if (e.op == OP_FACTOR) {
          ++stats.n_factor_edits;
          touched[size_t(e.g)] = 1;
        }
      }
      n_active = 0;
      for (i32 g = 0; g < G; ++g)
        if (touched[size_t(g)]) g2row[size_t(g)] = n_active++;
    }
    stats.n_active = n_active;
    row_fac.assign(size_t(n_active), 0);
    row_s0.assign(size_t(n_active), 0);
    for (i32 g = 0; g < G; ++g) {
      const i32 r = g2row[size_t(g)];
      if (r >= 0) { row_fac[size_t(r)] = fac[size_t(g)]; row_s0[size_t(r)] = s0[size_t(g)]; }
    }
    if (!build_plan()) return false;
    stats.plan_us = us(t0);
    const auto t1 = std::chrono::steady_clock::now();
    if (sparse) {
      if (!build_base_sparse()) return false;
    } else {
      if (!build_base_dense()) return false;
    }
    stats.base_us = us(t1);
    // ---- query frame bounds
    qlb.assign(size_t(kStates) * size_t(qn), 0);
    qhb.assign(size_t(kStates) * size_t(qn), 0);
#pragma omp parallel for schedule(static) if (omp_threads_ > 1 && pgdg_omp_plan_par_flag() && std::getenv("BWH_PGDG_OMP_FRAMES_OFF") == nullptr && kStates >= 8192)
    for (i32 v = 1; v < kStates; ++v)
      for (i32 k = 0; k < qn; ++k) {
        const Query& qq = w.query(v, k);
        const size_t idx = size_t(v) * size_t(qn) + size_t(k);
        qlb[idx] = frame_bound(qq.lo);
        qhb[idx] = frame_bound(qq.hi);
      }
    // ---- dynamic index
    const i64 U = i64(ucls.size());
    stats.union_classes = U;
    if (corr) corr_pref.assign(size_t(NB), 0);
    W.assign(size_t(U) + 1, 0);
    gfen.init(i32(U));
    tdev.assign(size_t(G), {});
    for (i32 g = 0; g < G; ++g) tdev[size_t(g)].assign(tcoord[size_t(g)].size(), 0);
    stats.sparse_base = sparse;
    stats.corr = corr;
    stats.plan_bytes = i64(plan.size()) * i64(sizeof(NodePlan)) +
                       i64(qlb.size()) * 4 * 2 + i64(cl2fen.size()) * 4;
    stats.build_us = stats.plan_us + stats.base_us;
    return true;
  }

  void run(std::vector<i64>& answers) {
    answers.assign(size_t(kStates - 1) * size_t(qn), 0);
    std::vector<i32> path;
    path.reserve(size_t(kStates) + 4);
    i64 n_apply = 0, n_undo = 0, n_query = 0;
    const bool prof = std::getenv("BWH_PGDG_PROF") != nullptr;
    auto now = []() { return std::chrono::steady_clock::now(); };
    auto el = [](std::chrono::steady_clock::time_point a) {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(
                 std::chrono::steady_clock::now() - a)
          .count();
    };
    for (size_t oi = 0; oi < order.size(); ++oi) {
      const i32 v = order[oi];
      if (oi > 0) {
        const i32 pv = parent[size_t(v)];
        while (!path.empty() && path.back() != pv) {
          auto ta = now();
          undo_node(path.back());
          n_undo += el(ta);
          path.pop_back();
        }
      }
      path.push_back(v);
      if (v > 0) {
        auto ta = now();
        apply_node(v);
        n_apply += el(ta);
        auto tq = now();
        answer_state(v, answers.data() + size_t(v - 1) * size_t(qn));
        n_query += el(tq);
      }
    }
    if (prof) {
      std::fprintf(stderr, "PGDGPROF arm=%s nclasses=%d n_active=%d U=%zu nnz=%lld\n",
                   corr ? (sparse ? "sparse_corr" : "dense_corr")
                        : (sparse ? "sparse" : "dense"),
                   nclasses, n_active, ucls.size(), (long long)csr.nnz);
      std::fprintf(stderr, "PGDGPROF apply_ms=%.1f undo_ms=%.1f query_ms=%.1f n_apply=%lld n_undo=%lld\n",
                   double(n_apply) / 1e6, double(n_undo) / 1e6, double(n_query) / 1e6,
                   (long long)n_apply, (long long)n_undo);
    }
  }

 private:
  static inline bool has_amt(u8 ex, i64 amt) { return ex && amt != 0; }

  inline i32 find_ti(i32 g, i32 rank) const {
    const auto& v = tcoord[size_t(g)];
    const auto it = std::lower_bound(v.begin(), v.end(), rank);
    return (it == v.end() || *it != rank) ? -1 : i32(it - v.begin());
  }

  // ---- offline plan: static previous row/factor state + preorder (no overlay/snapshot)
  bool build_plan() {
    const bool p2 = std::getenv("BWH_PGDG_PROF2") != nullptr;
    auto pn = []() { return std::chrono::steady_clock::now(); };
    auto pms = [](std::chrono::steady_clock::time_point a) {
      return std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::steady_clock::now() - a).count() / 1000.0;
    };
    auto t_plan0 = pn();
    parent = w.parent;
    plan.assign(size_t(kStates), NodePlan{});
    {
      // Chain fast path: for a path tree the preorder is the node order itself and every edit's
      // previous state is simply the last state of that row/group on the path to its left.
      // This removes the preorder CSR, tin/tout, row-id radix sort and ancestor stacks, and lets
      // the remaining plan work run per group (parallel).  Falls back to the general plan for a
      // non-chain declared tree, so the answer contract is unchanged.
      bool chain = parent[size_t(0)] == -1;
      for (i32 v = 1; v < kStates && chain; ++v)
        if (parent[size_t(v)] != v - 1) chain = false;
      if (chain) return build_plan_chain();
    }
    std::vector<i32> tin(static_cast<size_t>(kStates)), tout(static_cast<size_t>(kStates));
    order.clear();
    order.reserve(size_t(kStates));
    {
      std::vector<i32> cnt(size_t(kStates) + 1, 0);
      for (i32 v = 1; v < kStates; ++v) ++cnt[size_t(parent[size_t(v)]) + 1];
      for (i32 v = 0; v < kStates; ++v) cnt[size_t(v) + 1] += cnt[size_t(v)];
      std::vector<i32> kids(static_cast<size_t>(kStates - 1));
      std::vector<i32> off(cnt.begin(), cnt.begin() + kStates);
      for (i32 v = 1; v < kStates; ++v) kids[size_t(off[size_t(parent[size_t(v)])]++)] = v;
      std::vector<i32> cend(cnt.begin(), cnt.end());
      std::vector<std::pair<i32, i32>> st;
      st.reserve(size_t(kStates));
      st.push_back({0, 0});
      i32 clock = 0;
      tin[0] = clock++;
      order.push_back(0);
      while (!st.empty()) {
        auto& top = st.back();
        const i32 v = top.first;
        const i32 p = top.second;
        if (p < cend[size_t(v) + 1]) {
          const i32 ch = kids[size_t(p)];
          ++top.second;
          tin[ch] = clock++;
          order.push_back(ch);
          st.push_back({ch, cnt[size_t(ch)]});
        } else {
          st.pop_back();
          tout[v] = clock;
        }
      }
    }
    const double ms_pre = pms(t_plan0);
    auto t_rows = pn();
    // ---- row edits grouped by row id (radix sort)
    std::vector<u32> keys, vals;
    keys.reserve(size_t(kStates));
    for (i32 v = 1; v < kStates; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if (e.op != OP_FACTOR) { keys.push_back(u32(e.id)); vals.push_back(u32(v)); }
    }
    radix_sort_u32(keys, vals);
    auto base_of = [&](i64 id, u8& ex, i32& g, i32& c, i64& amt) {
      if (id < N) {
        const BaseRec& br = f.rows[size_t(id)];
        ex = 1; g = br.g; c = i32(class_of(br.t)); amt = br.amount;
      } else {
        const size_t k = size_t(id - N);
        ex = 0; g = w.ins_g[k]; c = i32(class_of(w.ins_t[k])); amt = 0;
      }
    };
    struct Ent { i32 node; u8 ex; i32 g, c; i64 amt; };
    std::vector<Ent> stack;
    // union of classes reachable by row edits (old and new states, nonzero amounts):
    // mark a class bitmap and enumerate in order -- no per-edit pair array and no sort.
    std::vector<u8> umark(size_t(nclasses), 0);
    size_t i = 0;
    while (i < keys.size()) {
      size_t j = i;
      while (j < keys.size() && keys[j] == keys[i]) ++j;
      const i64 id = i64(keys[i]);
      u8 ex; i32 g, c; i64 amt;
      base_of(id, ex, g, c, amt);
      for (size_t a = i + 1; a < j; ++a) {
        const u32 vn = vals[a];
        const i32 ti = tin[size_t(vn)];
        size_t b = a;
        while (b > i && tin[size_t(vals[b - 1])] > ti) { vals[b] = vals[b - 1]; --b; }
        vals[b] = vn;
      }
      stack.clear();
      for (size_t a = i; a < j; ++a) {
        const i32 v = i32(vals[a]);
        while (!stack.empty() && tout[size_t(stack.back().node)] <= tin[size_t(v)]) stack.pop_back();
        if (!stack.empty()) {
          const Ent& t = stack.back();
          ex = t.ex; g = t.g; c = t.c; amt = t.amt;
        } else {
          base_of(id, ex, g, c, amt);
        }
        NodePlan& p = plan[size_t(v)];
        p.kind = 1;
        p.ex_old = ex; p.g_old = g; p.c_old = c; p.amt_old = amt;
        const Edit& e = w.edit[size_t(v)];
        u8 nex = 1;
        i32 ng = g, nc = c;
        i64 namt = amt;
        switch (e.op) {
          case OP_REPLACE: nex = 1; namt = e.val; break;
          case OP_DELETE: nex = 0; break;
          case OP_INSERT: nex = 1; ng = e.g; nc = i32(class_of(e.t)); namt = e.val; break;
          case OP_MOVE: nex = 1; ng = e.g; nc = i32(class_of(e.t)); namt = e.val; break;
          default: break;
        }
        p.ex_new = nex; p.g_new = ng; p.c_new = nc; p.amt_new = namt;
        if (has_amt(ex, amt)) umark[size_t(c)] = 1;
        if (has_amt(nex, namt)) umark[size_t(nc)] = 1;
        stack.push_back({v, nex, ng, nc, namt});
      }
      i = j;
    }
    ucls.clear();
    for (i32 c = 0; c < nclasses; ++c)
      if (umark[size_t(c)]) ucls.push_back(i64(c));
    const i64 U = i64(ucls.size());
    stats.union_classes = U;
    // class -> union rank (number of union classes < c), valid for c in [0, nclasses]
    cl2fen.assign(size_t(nclasses) + 1, 0);
    {
      i64 idx = 0;
      for (i32 c = 0; c <= nclasses; ++c) {
        while (idx < U && ucls[size_t(idx)] < i64(c)) ++idx;
        cl2fen[size_t(c)] = i32(idx);
      }
    }
    const double ms_rows = pms(t_rows);
    auto t_sup = pn();
    // ---- per-group dynamic supports (union ranks) and static ti indices
    // One pass resolves each support's union rank; supports are grouped by group in flat
    // arrays, sorted/deduplicated per group, and every plan entry's index into its group's
    // support row is written back through a per-group rank->slot table (no second bisection
    // pass over the 300k-node plan).
    tcoord.assign(static_cast<size_t>(G), {});
    {
      auto t_sup1 = pn();
      std::vector<u32> ecnt(static_cast<size_t>(G), 0);
      for (i32 v = 1; v < kStates; ++v) {
        const NodePlan& p = plan[size_t(v)];
        if (p.kind != 1) continue;
        if (has_amt(p.ex_old, p.amt_old)) ++ecnt[size_t(p.g_old)];
        if (has_amt(p.ex_new, p.amt_new)) ++ecnt[size_t(p.g_new)];
      }
      std::vector<u32> eoff(static_cast<size_t>(G) + 1, 0);
      for (i32 g = 0; g < G; ++g) eoff[size_t(g) + 1] = eoff[size_t(g)] + ecnt[size_t(g)];
      const size_t n_ent = eoff[size_t(G)];
      std::vector<i32> ent_rr(n_ent, 0);
      std::vector<u32> ent_node(n_ent, 0);
      std::vector<u8> ent_side(n_ent, 0);
      std::vector<u32> epos(eoff.begin(), eoff.end() - 1);
      for (i32 v = 1; v < kStates; ++v) {
        const NodePlan& p = plan[size_t(v)];
        if (p.kind != 1) continue;
        if (has_amt(p.ex_old, p.amt_old)) {
          const u32 k = epos[size_t(p.g_old)]++;
          ent_rr[size_t(k)] = i32(std::lower_bound(ucls.begin(), ucls.end(), i64(p.c_old)) -
                                  ucls.begin());
          ent_node[size_t(k)] = u32(v);
          ent_side[size_t(k)] = 0;
        }
        if (has_amt(p.ex_new, p.amt_new)) {
          const u32 k = epos[size_t(p.g_new)]++;
          ent_rr[size_t(k)] = i32(std::lower_bound(ucls.begin(), ucls.end(), i64(p.c_new)) -
                                  ucls.begin());
          ent_node[size_t(k)] = u32(v);
          ent_side[size_t(k)] = 1;
        }
      }
      const double ms_sup1 = pms(t_sup1);
      auto t_sup2 = pn();
      std::vector<i32> rank2slot(size_t(U), -1);
      bool ok = true;
      for (i32 g = 0; g < G; ++g) {
        auto& tc = tcoord[size_t(g)];
        const u32 e0 = eoff[size_t(g)], e1 = eoff[size_t(g) + 1];
        tc.reserve(size_t(e1 - e0));
        for (u32 k = e0; k < e1; ++k) tc.push_back(ent_rr[size_t(k)]);
        std::sort(tc.begin(), tc.end());
        tc.erase(std::unique(tc.begin(), tc.end()), tc.end());
        for (size_t ti = 0; ti < tc.size(); ++ti) rank2slot[size_t(tc[ti])] = i32(ti);
        for (u32 k = e0; k < e1; ++k) {
          const i32 rr = ent_rr[size_t(k)];
          const i32 ti = rank2slot[size_t(rr)];
          NodePlan& p = plan[size_t(ent_node[size_t(k)])];
          if (ent_side[size_t(k)] == 0) {
            p.r_old = rr;
            p.ti_old = ti;
            if (ti < 0) ok = false;
          } else {
            p.r_new = rr;
            p.ti_new = ti;
            if (ti < 0) ok = false;
          }
        }
      }
      if (!ok) {
        if (std::getenv("BWH_PGDG_DEBUG")) std::fprintf(stderr, "pgdg_align: support slot missing\n");
        return false;
      }
      const double ms_sup2 = pms(t_sup2);
      if (p2)
        std::fprintf(stderr, "PGDGPROF2 support_build=%.2f sort_slot=%.2f n_ent=%zu\n",
                     ms_sup1, ms_sup2, n_ent);
    }
    const double ms_sup = pms(t_sup);
    auto t_fac = pn();
    // ---- factor plan: previous value per factor edit node (ancestor stack per group)
    {
      std::vector<std::vector<i32>> fg(static_cast<size_t>(G));
      for (i32 v = 1; v < kStates; ++v)
        if (w.edit[size_t(v)].op == OP_FACTOR) fg[size_t(w.edit[size_t(v)].g)].push_back(v);
      struct FEnt { i32 node; i64 val; };
      std::vector<FEnt> st2;
      for (i32 g = 0; g < G; ++g) {
        auto& lst = fg[size_t(g)];
        if (lst.empty()) continue;
        for (size_t a = 1; a < lst.size(); ++a) {
          const i32 vn = lst[a];
          const i32 ti = tin[size_t(vn)];
          size_t b = a;
          while (b > 0 && tin[size_t(lst[b - 1])] > ti) { lst[b] = lst[b - 1]; --b; }
          lst[b] = vn;
        }
        st2.clear();
        for (i32 v : lst) {
          while (!st2.empty() && tout[size_t(st2.back().node)] <= tin[size_t(v)]) st2.pop_back();
          NodePlan& p = plan[size_t(v)];
          p.kind = 2;
          p.fg = g;
          p.frow = g2row[size_t(g)];
          p.fac_prev = st2.empty() ? s0[size_t(g)] : st2.back().val;
          p.fac_val = w.edit[size_t(v)].val;
          st2.push_back({v, p.fac_val});
        }
      }
    }
    const double ms_fac = pms(t_fac);
    if (p2)
      std::fprintf(stderr,
                   "PGDGPROF2 pre_ms=%.2f rows_union_ms=%.2f support_ti_ms=%.2f factor_ms=%.2f "
                   "U=%zu plan_bytes=%zu\n",
                   ms_pre, ms_rows, ms_sup, ms_fac, ucls.size(), sizeof(NodePlan) * plan.size());
    return true;
  }

  // ---- chain-specialised offline plan (preorder = identity, previous state = last on path)
  bool build_plan_chain() {
    order.resize(static_cast<size_t>(kStates));
#pragma omp parallel for schedule(static) if (omp_threads_ > 1)
    for (i32 v = 0; v < kStates; ++v) order[size_t(v)] = v;
    const i64 n_rows = N + i64(w.n_inserted);
    // compact state slots only for row ids that are edited (dense id space, sparse slots)
    std::vector<u8> marked(static_cast<size_t>(n_rows), 0);
    i64 n_slots = 0;
    for (i32 v = 1; v < kStates; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if (e.op == OP_FACTOR) continue;
      if (e.id < 0 || e.id >= n_rows) continue;
      if (!marked[size_t(e.id)]) { marked[size_t(e.id)] = 1; ++n_slots; }
    }
    std::vector<i32> slot(static_cast<size_t>(n_rows), -1);
    {
      i64 s = 0;
      for (i64 id = 0; id < n_rows; ++id)
        if (marked[size_t(id)]) slot[size_t(id)] = i32(s++);
    }
    std::vector<u8> lv(static_cast<size_t>(n_slots), 0);
    std::vector<u8> lex(static_cast<size_t>(n_slots), 0);
    std::vector<i32> lg(static_cast<size_t>(n_slots), 0), lc(static_cast<size_t>(n_slots), 0);
    std::vector<i64> lamt(static_cast<size_t>(n_slots), 0);
    auto base_of = [&](i64 id, u8& ex, i32& g, i32& c, i64& amt) {
      if (id < N) {
        const BaseRec& br = f.rows[size_t(id)];
        ex = 1; g = br.g; c = i32(class_of(br.t)); amt = br.amount;
      } else {
        const size_t k = size_t(id - N);
        ex = 0; g = w.ins_g[k]; c = i32(class_of(w.ins_t[k])); amt = 0;
      }
    };
    std::vector<u8> umark(static_cast<size_t>(nclasses), 0);
    // pass 1: resolve previous row states (single left-to-right sweep, no sorting/stack)
    for (i32 v = 1; v < kStates; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if (e.op == OP_FACTOR) continue;
      const i64 id = e.id;
      const i32 sl = slot[size_t(id)];
      u8 ex; i32 g, c; i64 amt;
      if (!lv[size_t(sl)]) base_of(id, ex, g, c, amt);
      else { ex = lex[size_t(sl)]; g = lg[size_t(sl)]; c = lc[size_t(sl)]; amt = lamt[size_t(sl)]; }
      NodePlan& p = plan[size_t(v)];
      p.kind = 1;
      p.ex_old = ex; p.g_old = g; p.c_old = c; p.amt_old = amt;
      u8 nex = 1; i32 ng = g, nc = c; i64 namt = amt;
      switch (e.op) {
        case OP_REPLACE: nex = 1; namt = e.val; break;
        case OP_DELETE: nex = 0; break;
        case OP_INSERT: nex = 1; ng = e.g; nc = i32(class_of(e.t)); namt = e.val; break;
        case OP_MOVE: nex = 1; ng = e.g; nc = i32(class_of(e.t)); namt = e.val; break;
        default: break;
      }
      p.ex_new = nex; p.g_new = ng; p.c_new = nc; p.amt_new = namt;
      if (has_amt(ex, amt)) umark[size_t(c)] = 1;
      if (has_amt(nex, namt)) umark[size_t(nc)] = 1;
      lv[size_t(sl)] = 1; lex[size_t(sl)] = nex; lg[size_t(sl)] = ng;
      lc[size_t(sl)] = nc; lamt[size_t(sl)] = namt;
    }
    ucls.clear();
    for (i32 c = 0; c < nclasses; ++c)
      if (umark[size_t(c)]) ucls.push_back(i64(c));
    const i64 U = i64(ucls.size());
    stats.union_classes = U;
    cl2fen.assign(static_cast<size_t>(nclasses) + 1, 0);
#pragma omp parallel for schedule(static) if (omp_threads_ > 1 && pgdg_omp_plan_par_flag() && std::getenv("BWH_PGDG_OMP_CL2FEN_OFF") == nullptr && nclasses >= 4096)
    for (i32 c = 0; c <= nclasses; ++c)
      cl2fen[size_t(c)] = i32(std::lower_bound(ucls.begin(), ucls.end(), i64(c)) - ucls.begin());
    // factor plan: previous factor = last declared value on the path
    {
      std::vector<i64> fac_prev(s0);
      for (i32 v = 1; v < kStates; ++v) {
        const Edit& e = w.edit[size_t(v)];
        if (e.op != OP_FACTOR) continue;
        NodePlan& p = plan[size_t(v)];
        p.kind = 2;
        p.fg = e.g;
        p.frow = g2row[size_t(e.g)];
        p.fac_prev = fac_prev[size_t(e.g)];
        p.fac_val = e.val;
        fac_prev[size_t(e.g)] = e.val;
      }
    }
    // per-group dynamic supports: flat entries grouped by group, then per-group sort/dedup in
    // parallel; the support->slot table is per-thread so the ti assignment is race free.
    tcoord.assign(static_cast<size_t>(G), {});
    {
      std::vector<u32> ecnt(static_cast<size_t>(G), 0);
      for (i32 v = 1; v < kStates; ++v) {
        const NodePlan& p = plan[size_t(v)];
        if (p.kind != 1) continue;
        if (has_amt(p.ex_old, p.amt_old)) ++ecnt[size_t(p.g_old)];
        if (has_amt(p.ex_new, p.amt_new)) ++ecnt[size_t(p.g_new)];
      }
      std::vector<u32> eoff(static_cast<size_t>(G) + 1, 0);
      for (i32 g = 0; g < G; ++g) eoff[size_t(g) + 1] = eoff[size_t(g)] + ecnt[size_t(g)];
      const size_t n_ent = eoff[size_t(G)];
      std::vector<i32> ent_rr(n_ent, 0);
      std::vector<u32> ent_node(n_ent, 0);
      std::vector<u8> ent_side(n_ent, 0);
      std::vector<u32> epos(eoff.begin(), eoff.end() - 1);
      for (i32 v = 1; v < kStates; ++v) {
        const NodePlan& p = plan[size_t(v)];
        if (p.kind != 1) continue;
        if (has_amt(p.ex_old, p.amt_old)) {
          const u32 k = epos[size_t(p.g_old)]++;
          ent_rr[size_t(k)] = cl2fen[size_t(p.c_old)];
          ent_node[size_t(k)] = u32(v);
          ent_side[size_t(k)] = 0;
        }
        if (has_amt(p.ex_new, p.amt_new)) {
          const u32 k = epos[size_t(p.g_new)]++;
          ent_rr[size_t(k)] = cl2fen[size_t(p.c_new)];
          ent_node[size_t(k)] = u32(v);
          ent_side[size_t(k)] = 1;
        }
      }
      // parallel phase: per-group support sort + dedup only (each group owns its tcoord slice;
      // no plan-node writes happen concurrently).  The support->slot index and the plan writes
      // are then filled sequentially, which is cheap (one pass over the entries).
#pragma omp parallel for schedule(dynamic, 8) if (omp_threads_ > 1 && pgdg_omp_plan_par_flag() && \
                                                  std::getenv("BWH_PGDG_OMP_SUPPORT_OFF") == nullptr && G >= 32)
      for (i32 g = 0; g < G; ++g) {
        auto& tc = tcoord[size_t(g)];
        const u32 e0 = eoff[size_t(g)], e1 = eoff[size_t(g) + 1];
        tc.reserve(size_t(e1 - e0));
        for (u32 k = e0; k < e1; ++k) tc.push_back(ent_rr[size_t(k)]);
        std::sort(tc.begin(), tc.end());
        tc.erase(std::unique(tc.begin(), tc.end()), tc.end());
      }
      bool ok = true;
      std::vector<i32> rank2slot(static_cast<size_t>(U), -1);
      for (i32 g = 0; g < G; ++g) {
        auto& tc = tcoord[size_t(g)];
        const u32 e0 = eoff[size_t(g)], e1 = eoff[size_t(g) + 1];
        for (size_t ti = 0; ti < tc.size(); ++ti) rank2slot[size_t(tc[ti])] = i32(ti);
        for (u32 k = e0; k < e1; ++k) {
          const i32 rr = ent_rr[size_t(k)];
          const i32 ti = rank2slot[size_t(rr)];
          NodePlan& p = plan[size_t(ent_node[size_t(k)])];
          if (ent_side[size_t(k)] == 0) { p.r_old = rr; p.ti_old = ti; if (ti < 0) ok = false; }
          else { p.r_new = rr; p.ti_new = ti; if (ti < 0) ok = false; }
        }
      }
      if (!ok) {
        if (std::getenv("BWH_PGDG_DEBUG"))
          std::fprintf(stderr, "pgdg_align_omp: support slot missing (chain plan)\n");
        return false;
      }
    }
    return true;
  }

  // ---- dense per-group base prefix (class-major), PGD-G's pref rows
  bool build_base_dense() {
    // Layout choice mirrors the shipped engine's split: the per-query group correction loop
    // wants class-major rows (one contiguous run of active groups per query endpoint), while
    // the eager correction prefix wants per-group row-major rows (one contiguous run of
    // classes per factor edit).  A correction-prefix arm never runs the group loop, so it
    // builds only the row-major table.
    const bool class_major = !corr;
    base_class_major = class_major;
    base0.assign(size_t(NB), 0);
    baseP.assign(class_major ? size_t(NB) * size_t(n_active)
                             : size_t(n_active) * size_t(NB), 0);
    const BaseRec* rows = f.rows.data();
    const i32 NT = (omp_threads_ > 1 && N >= 200000 && std::getenv("BWH_PGDG_OMP_BASE_OFF") == nullptr)
                       ? omp_threads_ : 1;
    if (NT <= 1) {
      for (i64 i = 0; i < N; ++i) {
        const BaseRec& br = rows[size_t(i)];
        const i32 c = i32(class_of(br.t));
        base0[size_t(c) + 1] += br.amount * s0[size_t(br.g)];
        const i32 r = g2row[size_t(br.g)];
        if (r >= 0) {
          if (class_major) baseP[size_t(c + 1) * size_t(n_active) + size_t(r)] += br.amount;
          else baseP[size_t(r) * size_t(NB) + size_t(c + 1)] += br.amount;
        }
      }
    } else {
      // Parallel fixture scan: one private base0 prefix per thread (reduced exactly, integer
      // adds are associative) plus atomic scatter into the per-group base table.  The class
      // domain and the active-group table stay shared and read-only apart from the scatter, so
      // the sums are bit-identical to the sequential scan.
      std::vector<std::vector<i64>> part(static_cast<size_t>(NT),
                                         std::vector<i64>(static_cast<size_t>(NB), 0));
#pragma omp parallel num_threads(NT)
      {
        const i32 tid = omp_get_thread_num();
        i64* p0 = part[static_cast<size_t>(tid)].data();
#pragma omp for schedule(static)
        for (i64 i = 0; i < N; ++i) {
          const BaseRec& br = rows[size_t(i)];
          const i32 c = i32(class_of(br.t));
          p0[size_t(c) + 1] += br.amount * s0[size_t(br.g)];
          const i32 r = g2row[size_t(br.g)];
          if (r >= 0) {
            const i64 v = br.amount;
            if (class_major) {
              i64* dst = &baseP[size_t(c + 1) * size_t(n_active) + size_t(r)];
#pragma omp atomic update
              *dst += v;
            } else {
              i64* dst = &baseP[size_t(r) * size_t(NB) + size_t(c + 1)];
#pragma omp atomic update
              *dst += v;
            }
          }
        }
      }
#pragma omp parallel for schedule(static) num_threads(NT)
      for (i32 k = 0; k < nclasses + 1; ++k) {
        i64 v = 0;
        for (i32 t = 0; t < NT; ++t) v += part[static_cast<size_t>(t)][static_cast<size_t>(k)];
        base0[size_t(k)] = v;
      }
    }
    for (i32 k = 1; k <= nclasses; ++k) base0[size_t(k)] += base0[size_t(k) - 1];
    if (class_major) {
      for (i32 k = 1; k <= nclasses; ++k) {
        i64* row = &baseP[size_t(k) * size_t(n_active)];
        const i64* prev = &baseP[size_t(k - 1) * size_t(n_active)];
        for (i32 r = 0; r < n_active; ++r) row[size_t(r)] += prev[size_t(r)];
      }
    } else {
      // row-major per-group rows: every group's prefix is independent -> parallel over groups
#pragma omp parallel for schedule(static) if (NT > 1 && n_active >= 8)
      for (i32 r = 0; r < n_active; ++r) {
        i64* row = &baseP[size_t(r) * size_t(NB)];
        for (i32 k = 1; k <= nclasses; ++k) row[size_t(k)] += row[size_t(k) - 1];
      }
    }
    stats.base_bytes = i64(baseP.size()) * 8 + i64(base0.size()) * 8;
    stats.dense_base_bytes = stats.base_bytes;
    if (corr) stats.base_bytes += i64(NB) * 8;
    return true;
  }

  // ---- sparse per-group base supports: bitmap + 64-class rank LUT + amount prefix
  bool build_base_sparse() {
    const i32 wpr = (nclasses + 63) / 64;
    csr.wpr = wpr;
    csr.ptr.assign(size_t(n_active) + 1, 0);
    csr.bits.assign(size_t(n_active) * size_t(wpr), 0);
    base0.assign(size_t(NB), 0);
    std::vector<u32> cnt(size_t(NB), 0);  // rows per class (for the class-ordered sweep)
    const BaseRec* rows = f.rows.data();
    i64 nact = 0;
    // pass 1: class lookup once per row, base0 scatter, per-group class bitmap, class counts
    for (i64 i = 0; i < N; ++i) {
      const BaseRec& br = rows[size_t(i)];
      const i32 c = i32(class_of(br.t));
      base0[size_t(c) + 1] += br.amount * s0[size_t(br.g)];
      const i32 r = g2row[size_t(br.g)];
      if (r >= 0) {
        csr.bits[size_t(r) * size_t(wpr) + size_t(c >> 6)] |= (u64(1) << (c & 63));
        ++cnt[size_t(c) + 1];
        ++nact;
      }
    }
    // support offsets + rank LUT from the bitmap popcounts
    {
      i64 nnz = 0;
      csr.rlut.assign(size_t(n_active) * size_t(wpr + 1), 0);
      for (i32 r = 0; r < n_active; ++r) {
        csr.ptr[size_t(r)] = u32(nnz);
        u32 run = 0;
        for (i32 b = 0; b < wpr; ++b) {
          csr.rlut[size_t(r) * size_t(wpr + 1) + size_t(b)] = run;
          run += u32(__builtin_popcountll(csr.bits[size_t(r) * size_t(wpr) + size_t(b)]));
        }
        csr.rlut[size_t(r) * size_t(wpr + 1) + size_t(wpr)] = run;
        nnz += i64(run);
      }
      csr.ptr[size_t(n_active)] = u32(nnz);
      csr.nnz = nnz;
      csr.pre.assign(size_t(nnz), 0);
      csr.cls.assign(size_t(nnz), 0);
    }
    // class-ordered offsets then a second row scan that places (amount, group) per class
    std::vector<u32> cl_off(size_t(NB), 0);
    {
      u32 run = 0;
      for (i32 c = 0; c <= nclasses; ++c) {  // cnt[c+1] entries; offsets indexed by class
        cl_off[size_t(c)] = run;
        if (c < nclasses) run += cnt[size_t(c) + 1];
      }
    }
    std::vector<i64> sorted_amt(static_cast<size_t>(nact), 0);
    std::vector<u32> sorted_g(static_cast<size_t>(nact), 0);
    {
      std::vector<u32> pos(cl_off.begin(), cl_off.end());
      for (i64 i = 0; i < N; ++i) {
        const BaseRec& br = rows[size_t(i)];
        const i32 r = g2row[size_t(br.g)];
        if (r < 0) continue;
        const i32 c = i32(class_of(br.t));
        const u32 p = pos[size_t(c)]++;
        sorted_amt[size_t(p)] = br.amount;
        sorted_g[size_t(p)] = u32(br.g);
      }
    }
    // ascending-class sweep: one support per (class, group) pair; cumulative amount at every
    // support rank (== CsrBase.pre), and the support class list for correction updates.
    {
      std::vector<i64> run(size_t(n_active), 0);
      std::vector<i64> acc(size_t(n_active), 0);
      std::vector<u32> wr(size_t(n_active), 0);
      std::vector<u32> stamp(size_t(n_active), 0);
      std::vector<u32> touched_rows;
      for (i32 c = 0; c < nclasses; ++c) {
        const u32 e0 = cl_off[size_t(c)], e1 = cl_off[size_t(c) + 1];
        if (e0 == e1) continue;
        touched_rows.clear();
        for (u32 p = e0; p < e1; ++p) {
          const i32 r = g2row[size_t(sorted_g[size_t(p)])];
          if (stamp[size_t(r)] != u32(c) + 1) {
            stamp[size_t(r)] = u32(c) + 1;
            touched_rows.push_back(u32(r));
          }
          acc[size_t(r)] += sorted_amt[size_t(p)];
        }
        for (u32 r : touched_rows) {
          run[size_t(r)] += acc[size_t(r)];
          const i64 slot = i64(csr.ptr[size_t(r)]) + i64(wr[size_t(r)]);
          csr.pre[size_t(slot)] = run[size_t(r)];
          csr.cls[size_t(slot)] = u32(c);
          ++wr[size_t(r)];
          acc[size_t(r)] = 0;
        }
      }
    }
    for (i32 k = 1; k <= nclasses; ++k) base0[size_t(k)] += base0[size_t(k) - 1];
    stats.nnz = csr.nnz;
    stats.base_bytes = i64(csr.bytes()) + i64(base0.size()) * 8;
    stats.dense_base_bytes = (i64(n_active) * i64(NB) + i64(NB)) * 8;
    return true;
  }

  // ---- dynamic updates
  inline void row_delta(i32 g, i32 ti, i32 rr, i64 delta) {
    if (ti < 0 || rr < 0) return;
    tdev[size_t(g)][size_t(ti)] += delta;
    const i64 wv = fac[size_t(g)] * delta;
    W[size_t(rr) + 1] += wv;
    gfen.add(rr + 1, wv);
  }

  inline void row_apply(const NodePlan& p, i64 sgn) {
    if (sgn > 0) {
      if (has_amt(p.ex_old, p.amt_old)) row_delta(p.g_old, p.ti_old, p.r_old, -p.amt_old);
      if (has_amt(p.ex_new, p.amt_new)) row_delta(p.g_new, p.ti_new, p.r_new, +p.amt_new);
    } else {
      if (has_amt(p.ex_new, p.amt_new)) row_delta(p.g_new, p.ti_new, p.r_new, -p.amt_new);
      if (has_amt(p.ex_old, p.amt_old)) row_delta(p.g_old, p.ti_old, p.r_old, +p.amt_old);
    }
  }

  // corr_pref[k] += ds * B_g(classes < k), O(nclasses) per factor edit (interval fill)
  inline void corr_update(i32 frow, i64 ds) {
    if (!corr || frow < 0 || ds == 0) return;
    if (!sparse) {
      // row-major per-group prefix row: one contiguous sweep per factor edit.  The sweep has
      // no loop-carried dependence, so it is parallel over the class domain when the per-edit
      // work is large enough to amortise a fork/join (never on the 300k small-class chains).
      i64* cp = corr_pref.data();
      const i64* row = &baseP[size_t(frow) * size_t(NB)];
      const i32 C = nclasses;
#pragma omp parallel for schedule(static) if (omp_threads_ > 1 && (i64)C + 1 >= exec_par_min_)
      for (i32 k = 0; k <= C; ++k) cp[size_t(k)] += ds * row[size_t(k)];
      return;
    }
    const i64 b0 = i64(csr.ptr[size_t(frow)]);
    const i64 b1 = i64(csr.ptr[size_t(frow) + 1]);
    if (b0 == b1) return;
    i64* cp = corr_pref.data();
    i64 cur = 0;
    i32 kpos = 0;
    for (i64 idx = b0; idx < b1; ++idx) {
      const i32 c = i32(csr.cls[size_t(idx)]);
      const i64 cum = csr.pre[size_t(idx)];
      const i64 amt = (idx == b0) ? cum : cum - csr.pre[size_t(idx) - 1];
      if (cur != 0) {
        for (i32 k = kpos; k <= c; ++k) cp[size_t(k)] += ds * cur;
      }
      kpos = c + 1;
      cur += amt;
    }
    if (cur != 0) {
      for (i32 k = kpos; k <= nclasses; ++k) cp[size_t(k)] += ds * cur;
    }
  }

  inline void factor_apply(const NodePlan& p, i64 ds) {
    if (ds == 0) return;
    const i32 g = p.fg;
    auto& tc = tcoord[size_t(g)];
    auto& dv = tdev[size_t(g)];
    i64 nchanged = 0;
#pragma omp parallel for schedule(static) reduction(+ : nchanged) \
    if (omp_threads_ > 1 && (i64)tc.size() >= exec_par_min_)
    for (size_t k = 0; k < tc.size(); ++k) {
      const i64 d = dv[k];
      if (d == 0) continue;
      W[size_t(tc[k]) + 1] += ds * d;
      ++nchanged;
    }
    if (nchanged) {
      if (omp_threads_ > 1 && (i64)gfen.n >= exec_par_min_) pgdg_omp_rebuild_fenwick(gfen, W);
      else gfen.rebuild_from_leaves(W);
    }
    corr_update(p.frow, ds);
    fac[size_t(g)] += ds;
    if (p.frow >= 0) row_fac[size_t(p.frow)] = fac[size_t(g)];
  }

  inline void apply_node(i32 v) {
    NodePlan& p = plan[size_t(v)];
    if (p.kind == 1) {
      row_apply(p, +1);
    } else if (p.kind == 2) {
      factor_apply(p, p.fac_val - p.fac_prev);
    }
  }

  inline void undo_node(i32 v) {
    NodePlan& p = plan[size_t(v)];
    if (p.kind == 1) {
      row_apply(p, -1);
    } else if (p.kind == 2) {
      factor_apply(p, p.fac_prev - p.fac_val);
    }
  }

 public:
  // Export the static aligned plan for the GPU block-parallel executor (see PgdGpuExport).
  // Requires the dense correction-prefix layout (the 300k chain default), which is the layout
  // whose base masses are stored as row-major per-group prefix rows.
  bool export_gpu(PgdGpuExport& out) const {
    if (sparse || !corr || base_class_major) return false;
    out.kstates = kStates;
    out.qn = qn;
    out.G = G;
    out.n_active = n_active;
    out.nclasses = nclasses;
    out.Cpfx = NB;
    out.kind.assign(static_cast<size_t>(kStates), 0);
    out.go.assign(static_cast<size_t>(kStates), -1);
    out.gn.assign(static_cast<size_t>(kStates), -1);
    out.co.assign(static_cast<size_t>(kStates), -1);
    out.cn.assign(static_cast<size_t>(kStates), -1);
    out.tio.assign(static_cast<size_t>(kStates), -1);
    out.tin.assign(static_cast<size_t>(kStates), -1);
    out.ea.assign(static_cast<size_t>(kStates), 0);
    out.eb.assign(static_cast<size_t>(kStates), 0);
    for (i32 v = 1; v < kStates; ++v) {
      const NodePlan& p = plan[static_cast<size_t>(v)];
      const size_t i = static_cast<size_t>(v);
      out.kind[i] = static_cast<u8>(p.kind);
      if (p.kind == 1) {
        out.go[i] = p.g_old; out.gn[i] = p.g_new;
        out.co[i] = p.c_old; out.cn[i] = p.c_new;
        out.tio[i] = p.ti_old; out.tin[i] = p.ti_new;
        out.ea[i] = p.amt_old; out.eb[i] = p.amt_new;
      } else {
        // block replay uses g_old as the factor group (same convention as the CPU node plan)
        out.go[i] = p.fg; out.gn[i] = -1;
        out.ea[i] = p.fac_val; out.eb[i] = p.fac_prev;
      }
    }
    out.g2row.assign(g2row.begin(), g2row.end());
    out.qlb.assign(qlb.begin(), qlb.end());
    out.qhb.assign(qhb.begin(), qhb.end());
    out.tcoord_size.assign(static_cast<size_t>(n_active), 0);
    out.g_of_row.assign(static_cast<size_t>(n_active), -1);
    for (i32 g = 0; g < G; ++g) {
      const i32 r = g2row[static_cast<size_t>(g)];
      if (r >= 0) {
        out.g_of_row[static_cast<size_t>(r)] = g;
        out.tcoord_size[static_cast<size_t>(r)] =
            static_cast<u32>(tcoord[static_cast<size_t>(g)].size());
      }
    }
    // merged support per active group: union of base-mass classes and deviation classes
    out.ms_off.assign(static_cast<size_t>(n_active) + 1, 0);
    out.ms_cls.clear();
    out.ms_base.clear();
    out.ms_dev.clear();
    std::vector<i32> devti(static_cast<size_t>(nclasses), -1);
    for (i32 r = 0; r < n_active; ++r) {
      const i32 g = out.g_of_row[static_cast<size_t>(r)];
      const auto& tc = tcoord[static_cast<size_t>(g)];
      for (size_t ti = 0; ti < tc.size(); ++ti) {
        const i64 cl = ucls[static_cast<size_t>(tc[ti])];
        devti[static_cast<size_t>(cl)] = static_cast<i32>(ti);
      }
      out.ms_off[static_cast<size_t>(r)] = static_cast<u32>(out.ms_cls.size());
      const i64* row = &baseP[static_cast<size_t>(r) * static_cast<size_t>(NB)];
      for (i32 c = 0; c < nclasses; ++c) {
        const i64 amt = row[static_cast<size_t>(c) + 1] - row[static_cast<size_t>(c)];
        const i32 di = devti[static_cast<size_t>(c)];
        if (amt == 0 && di < 0) continue;
        out.ms_cls.push_back(static_cast<i64>(c));
        out.ms_base.push_back(amt);
        out.ms_dev.push_back(di);
      }
      for (size_t ti = 0; ti < tc.size(); ++ti) devti[static_cast<size_t>(ucls[static_cast<size_t>(tc[ti])])] = -1;
    }
    out.ms_off[static_cast<size_t>(n_active)] = static_cast<u32>(out.ms_cls.size());
    // V at block-0 start: sum_g s0(g) * baseLeaf_g(c)
    out.vbase.assign(static_cast<size_t>(NB), 0);
    for (i32 r = 0; r < n_active; ++r) {
      const i32 g = out.g_of_row[static_cast<size_t>(r)];
      const i64 f0 = s0[static_cast<size_t>(g)];
      const i64* row = &baseP[static_cast<size_t>(r) * static_cast<size_t>(NB)];
      for (i32 c = 0; c < nclasses; ++c) {
        const i64 amt = row[static_cast<size_t>(c) + 1] - row[static_cast<size_t>(c)];
        if (amt != 0) out.vbase[static_cast<size_t>(c) + 1] += f0 * amt;
      }
    }
    return true;
  }

 private:
  // ---- per-state answer batch
  inline void answer_state(i32 v, i64* out) const {
    const size_t base = size_t(v) * size_t(qn);
    if (corr) {
      for (i32 k = 0; k < qn; ++k) {
        const i32 i = qlb[base + size_t(k)], j = qhb[base + size_t(k)];
        out[k] = (base0[size_t(j)] - base0[size_t(i)]) +
                 (gfen.prefix(cl2fen[size_t(j)]) - gfen.prefix(cl2fen[size_t(i)])) +
                 (corr_pref[size_t(j)] - corr_pref[size_t(i)]);
      }
      return;
    }
    // dynamic term once per query (two Fenwick prefixes over the union class index)
    for (i32 k = 0; k < qn; ++k) {
      const i32 i = qlb[base + size_t(k)], j = qhb[base + size_t(k)];
      out[k] = (base0[size_t(j)] - base0[size_t(i)]) +
               (gfen.prefix(cl2fen[size_t(j)]) - gfen.prefix(cl2fen[size_t(i)]));
    }
    // factor-correction term: the PGD-G per-group base-prefix loop (PGD-G algebra)
    if (sparse) {
      for (i32 r = 0; r < n_active; ++r) {
        const i64 dfac = row_fac[size_t(r)] - row_s0[size_t(r)];
        if (dfac == 0) continue;
        const i64 df = dfac;
        for (i32 k = 0; k < qn; ++k) {
          const i32 i = qlb[base + size_t(k)], j = qhb[base + size_t(k)];
          out[k] += df * csr.range(r, i, j);
        }
      }
    } else {
      const i32* qi = qlb.data() + base;
      const i32* qj = qhb.data() + base;
      for (i32 k = 0; k < qn; ++k) {
        const i64* rowj = &baseP[size_t(qj[k]) * size_t(n_active)];
        const i64* rowi = &baseP[size_t(qi[k]) * size_t(n_active)];
        i64 acc = 0;
#pragma omp parallel for schedule(static) reduction(+ : acc) \
        if (omp_threads_ > 1 && i64(qn) * i64(n_active) >= exec_par_min_)
        for (i32 r = 0; r < n_active; ++r) {
          const i64 dfac = row_fac[size_t(r)] - row_s0[size_t(r)];
          if (dfac != 0) acc += dfac * (rowj[size_t(r)] - rowi[size_t(r)]);
        }
        out[k] += acc;
      }
    }
  }
};

// ---------------------------------------------------------------- static auto policy
// Picks an execution plan from declared-workload structure alone (no timing, no oracle).
// Two decisions, both with a structural cost model:
//
//   * sparse base when the dense per-group base table is large and the rank-LUT CSR is much
//     smaller (C >= 16,384, dense table >= 24 MiB, predicted sparse bytes <= a quarter of the
//     dense table).  The rank LUT removes the per-query bisection, so the selection is
//     memory-driven rather than occupancy-driven.
//   * correction prefix when the O(classes)-per-factor-edit maintenance is small relative to
//     the per-query changed-group loop it removes.  The loop cost is modelled as the estimated
//     number of groups whose factor currently differs from s0 (~2/3 of the factor-touched
//     groups) times the declared query count; the per-group query read is cheaper in the dense
//     base (2 prefix reads) than in the sparse rank-LUT base (bitmap + LUT + prefix reads), so
//     the sparse arm is allowed a 4x larger correction budget.
// This is the deployable "strongest PGD-G" route: a frozen regime policy, not an oracle over
// the measured runs (the measured per-variant results are reported separately).
struct PgdAutoChoice {
  int variant = PgdAlignEngine::DENSE;
  bool sparse = false;
  bool corr = false;
  i64 classes = 0;
  i64 dense_bytes = 0;
  i64 est_nnz = 0;
  i64 sparse_bytes = 0;
  double occ = 0;
  i64 changed_groups_est = 0;
  i64 query_work = 0;
  i64 corr_work = 0;
};

inline PgdAutoChoice pgdg_align_auto(const Fixture& f, const BatchWorkload& w) {
  PgdAutoChoice c;
  const i32 G = f.h.groups;
  const i64 N = f.h.n;
  i64 Urow = 0, nf = 0;
  std::vector<u8> touched(size_t(G), 0);
  for (i32 v = 1; v < kStates; ++v) {
    const u8 op = w.edit[size_t(v)].op;
    if (op != OP_FACTOR) ++Urow;
    else { ++nf; touched[size_t(w.edit[size_t(v)].g)] = 1; }
  }
  i32 n_active = 0;
  for (i32 g = 0; g < G; ++g) n_active += touched[size_t(g)];
  const i64 span = f.h.t_max - f.h.t_min + 1;
  const bool dense_frame = (__int128(G) * __int128(span) <= 2 * (__int128(N) + __int128(Urow))) &&
                           span <= (i64(1) << 30);
  const i64 nclasses = dense_frame ? span : i64(w.endpoints.size()) + 1;
  c.classes = nclasses;
  c.dense_bytes = i64(n_active) * (nclasses + 1) * 8 + (nclasses + 1) * 8;
  const double rpg = double(N) / double(G > 0 ? G : 1);
  c.occ = 1.0 - std::exp(-rpg / double(nclasses));
  c.est_nnz = i64(double(n_active) * double(nclasses) * c.occ);
  const i64 wpr = (nclasses + 63) / 64;
  c.sparse_bytes = c.est_nnz * 12 + i64(n_active) * nclasses / 8 + i64(n_active) * (wpr + 1) * 4 +
                   8 * (nclasses + 1);
  const bool sparse_ok = !dense_frame && nclasses >= 16384 &&
                         c.dense_bytes >= i64(24) * 1024 * 1024 &&
                         c.sparse_bytes * 4 <= c.dense_bytes;
  c.sparse = sparse_ok;
  c.changed_groups_est = std::max<i64>(1, (2 * std::min<i64>(G, nf)) / 3);
  c.query_work = i64(kStates) * i64(w.q) * c.changed_groups_est;
  c.corr_work = nf * (nclasses + 1);
  const i64 budget = (sparse_ok ? 16 : 4) * c.query_work;
  c.corr = nf > 0 && c.corr_work <= budget && c.corr_work <= 200000000LL;
  c.variant = (c.sparse ? (c.corr ? PgdAlignEngine::SPARSE_CORR : PgdAlignEngine::SPARSE)
                        : (c.corr ? PgdAlignEngine::DENSE_CORR : PgdAlignEngine::DENSE));
  return c;
}

}  // namespace bwh
