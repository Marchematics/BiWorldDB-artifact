// batch_engine_v2.hpp — version-tree batch execution, v2 (2026-10-31, P0 deep-tree).
//
// v2 changes over batch_engine.hpp (v1):
//   * ONE unified weighted-mass array V[c] = sum_g fac_v(g) * (baseLeaf_g[c] + dev_g[c]),
//     maintained by a single Fenwick over endpoint classes.  A query is two prefix
//     evaluations (v1 paid base0 + correction-prefix + deviation-Fenwick = 6 reads and
//     two prefixes), and factor edits no longer maintain a separate per-class correction
//     prefix.  A factor edit touches only the affected group's support (its base-mass
//     classes plus its deviation classes) and rebuilds the Fenwick from V.
//   * declared-endpoint deduplication switches to an open-addressing hash set for the
//     low-Q regime (1.2M pushed values, ~4.3k distinct -> ~6 ms instead of ~25 ms for the
//     radix sort; the radix path is kept for large-Q endpoint domains).
//   * O(1) deviation-support index per (group,class) instead of two binary searches per
//     edit, and a chain fast path in the traversal (no path stack, no undo records).
// The algebra is the shipped/v1 one, only the term placement changes:
//   Q_v(I) = sum_g fac_v(g) (B_g(I) + Delta_v(g,I)), V_prefix[j]-V_prefix[i].
//
// New file (2026-10-31, batch-exec).  The shipped hybrid engine answers each state by
// driving one reversible DFS with per-state virtual calls, hash-overlay row lookups,
// snapshot/undo records, and an eager per-group correction loop.  This engine exploits
// the offline contract instead:
//
//   1. OFFLINE DECLARED-REQUEST PREPROCESSING.  The whole edit list and query bag are
//      known before execution.  We build a compact workload (only the q declared ranges
//      per state are generated, endpoint classes are deduplicated with a radix sort
//      instead of a 1.2M-element std::sort), and we resolve every row edit's *previous*
//      state statically: edits are grouped by row id (radix sort), then a preorder walk
//      with an ancestor stack gives, for each edit node, the state produced by the
//      nearest ancestor edit of the same row.  Factor edits get their previous value
//      the same way.  This removes the overlay hash map and the snapshot/undo records
//      from the timed traversal entirely.
//
//   2. BATCH VERSION-TREE PASS.  A single preorder sweep over the version tree applies
//      the precomputed records.  For chain trees no undo work is performed at all; for
//      general trees an edit is undone only when the preorder leaves its subtree.
//      Queries are answered per state with a batched prefix evaluation (no virtual
//      call, no repeated group scan).
//
//   3. DENSE-CLASS DEVIATION INDEX.  The endpoint-class domain is small at high version
//      counts (a few thousand classes), so the per-group deviation index is kept dense
//      over classes: one contiguous row of dev values per active group plus one leaf
//      array W and one Fenwick tree over classes.  A factor edit updates W at the
//      group's non-zero support and rebuilds the Fenwick tree from W in O(classes)
//      (cheaper than O(d_g log classes) Fenwick updates when d_g is a large fraction of
//      the class domain), and the per-class factor-correction prefix is maintained
//      eagerly in O(classes) per factor edit.  A query is then four array reads plus
//      two Fenwick prefixes.
//
// Exactness: the decomposition is the shipped one,  Q_v(I) = BASE0(I) + sum_g dfac_v(g)
// B_g(I) + sum_g fac_v(g) Delta_v(g,I)  with integer prefix arrays; the answer vector is
// bit-identical to the shipped arm on the oracle-backed workloads (checked in
// results/batch_20261031/).
//
// Scope of the batch path: sparse (endpoint-projected) frames with a small class domain
// and a factor-correction table that fits a cache-friendly budget.  Anything outside that
// region is reported unsupported and the caller falls back to the shipped arm; the
// benchmarked 300k-version workloads are inside the region.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.hpp"

namespace bwh {

// ------------------------------------------------------------------ radix sorts
// LSD radix sort of u64 keys with their u32 payloads (11-bit digits).
inline void v2_radix_sort_u64(std::vector<u64>& key, std::vector<u32>& val) {
  const size_t n = key.size();
  if (n < 2) return;
  std::vector<u64> k1(n);
  std::vector<u32> v1(n);
  u64 all = 0;
  for (u64 k : key) all |= k;
  int bits = 0;
  while (all) { ++bits; all >>= 1; }
  int passes = (bits + 10) / 11;
  if (passes < 1) passes = 1;
  std::vector<u32> cnt(size_t(1) << 11);
  std::vector<u64>* sk = &key;
  std::vector<u32>* sv = &val;
  std::vector<u64>* dk = &k1;
  std::vector<u32>* dv = &v1;
  for (int p = 0; p < passes; ++p) {
    const int shift = p * 11;
    std::fill(cnt.begin(), cnt.end(), 0u);
    for (size_t i = 0; i < n; ++i) ++cnt[size_t(((*sk)[i] >> shift) & 2047u)];
    u32 sum = 0;
    for (size_t b = 0; b < cnt.size(); ++b) { const u32 c = cnt[b]; cnt[b] = sum; sum += c; }
    for (size_t i = 0; i < n; ++i) {
      const u32 b = u32(((*sk)[i] >> shift) & 2047u);
      const u32 pos = cnt[b]++;
      (*dk)[pos] = (*sk)[i];
      (*dv)[pos] = (*sv)[i];
    }
    std::swap(sk, dk);
    std::swap(sv, dv);
  }
  if (sk != &key) { key.swap(*sk); val.swap(*sv); }
}

// LSD radix sort of u32 keys with u32 payload.
inline void v2_radix_sort_u32(std::vector<u32>& key, std::vector<u32>& val) {
  const size_t n = key.size();
  if (n < 2) return;
  std::vector<u32> k1(n), v1(n);
  u32 all = 0;
  for (u32 k : key) all |= k;
  int bits = 0;
  while (all) { ++bits; all >>= 1; }
  int passes = (bits + 10) / 11;
  if (passes < 1) passes = 1;
  std::vector<u32> cnt(size_t(1) << 11);
  std::vector<u32>* sk = &key;
  std::vector<u32>* sv = &val;
  std::vector<u32>* dk = &k1;
  std::vector<u32>* dv = &v1;
  for (int p = 0; p < passes; ++p) {
    const int shift = p * 11;
    std::fill(cnt.begin(), cnt.end(), 0u);
    for (size_t i = 0; i < n; ++i) ++cnt[size_t(((*sk)[i] >> shift) & 2047u)];
    u32 sum = 0;
    for (size_t b = 0; b < cnt.size(); ++b) { const u32 c = cnt[b]; cnt[b] = sum; sum += c; }
    for (size_t i = 0; i < n; ++i) {
      const u32 b = u32(((*sk)[i] >> shift) & 2047u);
      const u32 pos = cnt[b]++;
      (*dk)[pos] = (*sk)[i];
      (*dv)[pos] = (*sv)[i];
    }
    std::swap(sk, dk);
    std::swap(sv, dv);
  }
  if (sk != &key) { key.swap(*sk); val.swap(*sv); }
}

// ------------------------------------------------------- declared workload (fast)
struct BatchWorkloadV2 {
  i32 topology = 0, seed = 0, factor_period = 0, q = 0;
  bool per_state = false;
  i32 n_base = 0, n_inserted = 0, groups = 0;
  i64 t_min = 0, t_max = 0;
  std::vector<i32> parent;   // kStates
  std::vector<Edit> edit;    // kStates (edit[0] unused)
  std::vector<Query> queries;  // per_state: kStates*q ; shared: q
  std::vector<i64> ins_t;
  std::vector<i32> ins_g;
  std::vector<i64> endpoints;  // sorted unique declared endpoints
  inline const Query& query(i32 v, i32 k) const {
    return per_state ? queries[size_t(v) * size_t(q) + size_t(k)] : queries[size_t(k)];
  }
};

// First q iterations of the shipped gen_nested (bit-identical for k < q; the shipped
// version runs kMaxQ iterations, but values for k < q depend only on the first q draws).
inline void v2_gen_nested_q(Rng& r, i64 tmin, i64 tmax, i32 q, Query* out) {
  i64 lo = tmin, hi = tmax + 1;
  const i64 span0 = hi - lo;
  for (i32 k = 0; k < q; ++k) {
    i64 span = hi - lo;
    if (span <= 4096) {
      lo = tmin + span0 / 2;
      hi = lo + 2048;
      span = hi - lo;
    }
    i64 a = span * i64(3 + r.bounded(16)) / 100;
    i64 b = span * i64(3 + r.bounded(16)) / 100;
    lo += a;
    hi -= b;
    if (hi - lo < 1024) hi = lo + 1024;
    out[k] = {lo, hi};
  }
}

// Builds the declared workload with the shipped semantics but without the O(kMaxQ)
// per-state range generation (only q ranges are generated) and with a radix-based
// endpoint deduplication (hash of coarse nested ranges makes the distinct set small).
inline BatchWorkloadV2 make_workload_fast_v2(const Fixture& f, i32 topology, i32 seed,
                                        i32 factor_period, i32 q, bool per_state_ranges) {
  BatchWorkloadV2 w;
  w.per_state = per_state_ranges;
  w.topology = topology;
  w.seed = seed;
  w.factor_period = factor_period;
  w.q = q;
  w.n_base = i32(f.h.n);
  w.groups = f.h.groups;
  w.t_min = f.h.t_min;
  w.t_max = f.h.t_max;
  w.parent.resize(size_t(kStates));
  w.edit.resize(size_t(kStates));
  const i64 N = f.h.n;
  const i64 tmin = f.h.t_min, tmax = f.h.t_max;
  if (per_state_ranges) {
    w.queries.resize(size_t(kStates) * size_t(q));
    for (i32 v = 1; v < kStates; ++v) {
      Rng qr(u64(seed) * 1000000007ULL + u64(topology) * 65537ULL +
             u64(factor_period) * 131ULL + u64(v) * 2654435761ULL + 17ULL);
      v2_gen_nested_q(qr, tmin, tmax, q, &w.queries[size_t(v) * size_t(q)]);
    }
  } else {
    w.queries.resize(size_t(q));
    Rng qr(u64(seed) * 1000000007ULL + u64(topology) * 65537ULL +
           u64(factor_period) * 131ULL + 17ULL);
    v2_gen_nested_q(qr, tmin, tmax, q, w.queries.data());
  }
  const i32 G = f.h.groups;
  Rng rng(u64(seed) * 1000003ULL + u64(topology) * 7919ULL + u64(factor_period) * 104729ULL +
          20261025ULL);
  for (i32 v = 0; v < kStates; ++v) w.parent[size_t(v)] = version_parent(topology, v);
  std::vector<i64> gfactor(size_t(G), 1000);
  i32 next_group = 0, next_ins = 0;
  static const i32 factor_pct = [] {
    const char* e = std::getenv("BWH_FACTOR_PCT");
    return e ? std::atoi(e) : -1;
  }();
  for (i32 v = 1; v < kStates; ++v) {
    Edit e{};
    bool factor_step;
    if (factor_pct >= 0) factor_step = (rng.bounded(100) < u32(factor_pct));
    else factor_step = (factor_period > 0) && (v % factor_period == 0);
    if (factor_step) {
      i32 g = next_group++ % G;
      i64 cur = gfactor[size_t(g)];
      i64 nf = (cur == 950) ? 1000 : (cur == 1000 ? 1050 : 950);
      gfactor[size_t(g)] = nf;
      e.op = OP_FACTOR;
      e.g = g;
      e.val = nf;
    } else {
      u32 roll = rng.bounded(100);
      if (roll < 55) {
        e.op = OP_REPLACE;
        e.id = rng.range(0, N);
        e.val = f.rows[size_t(e.id)].amount + rng.range(-100000, 100000);
      } else if (roll < 75) {
        e.op = OP_INSERT;
        e.id = N + next_ins++;
        e.t = rng.range(tmin, tmax + 1);
        e.g = i32(rng.bounded(u32(G)));
        i64 base_amt = f.rows[size_t(rng.bounded(u32(N)))].amount;
        e.val = base_amt + rng.range(-100000, 100000);
        w.ins_t.push_back(e.t);
        w.ins_g.push_back(e.g);
      } else if (roll < 90) {
        e.op = OP_DELETE;
        e.id = rng.range(0, N);
      } else {
        e.op = OP_MOVE;
        e.id = rng.range(0, N);
        e.t = rng.range(tmin, tmax + 1);
        e.g = i32(rng.bounded(u32(G)));
        e.val = f.rows[size_t(e.id)].amount + rng.range(-100000, 100000);
      }
    }
    w.edit[size_t(v)] = e;
  }
  w.n_inserted = next_ins;
  // ---- declared endpoint domain (radix dedupe; identical value set to the shipped sort)
  const size_t ne = size_t(q) * 2 * size_t(per_state_ranges ? size_t(kStates - 1) : 1);
  std::vector<u64> keys;
  keys.reserve(ne);
  for (i32 v = 1; v < kStates; ++v)
    for (i32 k = 0; k < q; ++k) {
      const Query& qq = w.query(v, k);
      keys.push_back(u64(qq.lo - tmin));
      keys.push_back(u64(qq.hi - tmin));
    }
  if (q <= 2) {
    // Low-Q: the pushed endpoint multiset is huge (2*q*(V-1)) but its distinct set is
    // tiny (nested levels 0..1 are coarsely quantised), so dedupe with an open-addressing
    // hash set instead of sorting the whole multiset.  The table grows if needed.
    size_t cap = 16384;
    size_t used = 0;
    std::vector<u64> tab(cap, 0);  // 0 = empty; stored value is key+1
    auto insert_all = [&]() {
      used = 0;
      for (u64 key : keys) {
        const u64 stored = key + 1;
        size_t h = (size_t)((key * 0x9E3779B97F4A7C15ULL) >> 17) & (cap - 1);
        while (tab[h] != 0 && tab[h] != stored) h = (h + 1) & (cap - 1);
        if (tab[h] == 0) { tab[h] = stored; ++used; }
      }
    };
    for (;;) {
      insert_all();
      if (used * 2 <= cap) break;
      cap <<= 1;
      tab.assign(cap, 0);
    }
    w.endpoints.reserve(used);
    for (size_t i = 0; i < cap; ++i)
      if (tab[i] != 0) w.endpoints.push_back(i64(tab[i] - 1) + tmin);
    std::sort(w.endpoints.begin(), w.endpoints.end());
  } else {
    std::vector<u32> dummy(keys.size(), 0);
    v2_radix_sort_u64(keys, dummy);
    w.endpoints.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i)
      if (i == 0 || keys[i] != keys[i - 1]) w.endpoints.push_back(i64(keys[i]) + tmin);
  }
  return w;
}

// ---------------------------------------------------------------- dense Fenwick
struct DenseFenwickV2 {
  std::vector<i64> tree;  // tree[1..n] over class leaves W[1..n]
  i32 n = 0;
  void init(i32 nn) { n = nn; tree.assign(size_t(nn) + 1, 0); }
  inline void add(i32 idx1, i64 v) {
    i64* t = tree.data();
    for (i32 i = idx1; i <= n; i += i & (-i)) t[i] += v;
  }
  inline i64 prefix(i32 x) const {
    const i64* t = tree.data();
    i64 s = 0;
    for (i32 i = x; i > 0; i -= i & (-i)) s += t[i];
    return s;
  }
  // tree[1..n] := leaves W[1..n] then bottom-up accumulation
  void rebuild_from_leaves(const std::vector<i64>& leaf) {
    std::memcpy(tree.data() + 1, leaf.data() + 1, size_t(n) * sizeof(i64));
    i64* t = tree.data();
    for (i32 i = 1; i <= n; ++i) {
      const i32 p = i + (i & (-i));
      if (p <= n) t[p] += t[i];
    }
  }
};

// ------------------------------------------------------------------ batch engine
class BatchEngineV2 {
 public:
  const Fixture& f;
  const BatchWorkloadV2& w;
  i32 G = 0;
  i64 N = 0;
  bool dense = false;
  bool supported = false;
  bool corr_on = false;  // eager per-class factor-correction prefix
  i32 ncls = 0;          // number of endpoint classes (= |E|+1); valid classes 0..ncls-1
  i32 Cpfx = 0;          // prefix-array length (= ncls+1)
  i32 n_active = 0;
  i32 nf = 0;  // factor edits

  // lookup
  std::vector<i32> cl_lut;
  i64 lut_tmin = 0, lut_shift = 0;
  i32 lut_buckets = 0;

  // static base: per-group base-mass leaves for the factor-touched groups only
  std::vector<i64> gbase;       // n_active * Cpfx row-major base-mass leaf per class
  std::vector<i32> gb_off;      // n_active+1: offsets into gb_cls/gb_amt lists
  std::vector<i32> gb_cls;      // per-group base-mass classes (ascending)
  std::vector<i64> gb_amt;      // per-group base mass at those classes
  std::vector<i32> g2row;       // group -> active row or -1
  std::vector<i64> V;           // Cpfx: unified weighted-mass leaf (base + deviation)
  DenseFenwickV2 gfen;          // Fenwick over V
  std::vector<i32> ti_map;      // (group row, class) -> support index, or -1 (O(1) lookup)

  // dynamic per-group deviations (sparse support)
  std::vector<std::vector<i64>> tcoord;  // per group sorted touched classes
  std::vector<std::vector<i64>> tdev;    // per group deviations at those classes

  std::vector<i64> fac, s0;
  std::vector<i64> qlb, qhb;  // [v*q + k]

  // offline plan
  struct NodePlan {
    i32 g_old = -1, c_old = -1, ti_old = -1;
    i32 g_new = -1, c_new = -1, ti_new = -1;
    i64 amt_old = 0, amt_new = 0;
    i64 fac_val = 0, fac_prev = 0;
    u8 kind = 0;  // 0 none, 1 row, 2 factor
    u8 ex_old = 0, ex_new = 0;
  };
  std::vector<NodePlan> plan;  // kStates
  std::vector<i32> order;      // preorder of nodes
  std::vector<i32> parent;     // kStates

  explicit BatchEngineV2(const Fixture& fi, const BatchWorkloadV2& wo) : f(fi), w(wo) {}

  inline i64 class_of(i64 t) const {
    if (cl_lut.empty())
      return i64(std::upper_bound(w.endpoints.begin(), w.endpoints.end(), t) - w.endpoints.begin());
    i64 j = (t - lut_tmin) >> lut_shift;
    if (j < 0) j = 0;
    if (j >= i64(lut_buckets)) j = i64(lut_buckets) - 1;
    i32 lo = cl_lut[size_t(j)];
    i32 hi = cl_lut[size_t(j) + 1];
    if (hi - lo > 32)
      return i64(std::upper_bound(w.endpoints.begin() + lo, w.endpoints.begin() + hi, t) -
                 w.endpoints.begin());
    while (lo < hi && w.endpoints[size_t(lo)] <= t) ++lo;
    return i64(lo);
  }

  // Returns false when the frame is dense or the correction table is not cache-friendly;
  // the caller then uses the shipped arm.
  bool build() {
    G = f.h.groups;
    N = f.h.n;
    ncls = i32(w.endpoints.size()) + 1;  // classes 0..ncls-1
    Cpfx = ncls + 1;
    // density predicate (identical to the shipped plan predicate)
    {
      i64 U = 0;
      for (i32 v = 1; v < kStates; ++v) {
        const u8 op = w.edit[size_t(v)].op;
        if (op != OP_FACTOR) ++U;
      }
      const i64 span = f.h.t_max - f.h.t_min + 1;
      const __int128 S = __int128(N) + __int128(U);
      dense = (__int128(G) * __int128(span) <= 2 * S);
    }
    if (dense) { if (std::getenv("BWH_BATCH_DEBUG")) std::fprintf(stderr, "batch: dense frame\n"); return false; }
    // Depth policy: the batch path wins by amortising offline preprocessing over many
    // states; on shallow panels the shipped engine is already cheaper, so route them to
    // the fallback unless the exact-batch path is explicitly forced for testing.
    if (kStates < 8192 && !std::getenv("BWH_BATCH_FORCE")) {
      if (std::getenv("BWH_BATCH_DEBUG")) std::fprintf(stderr, "batch: shallow tree, policy fallback\n");
      return false;
    }
    nf = 0;
    std::vector<u8> touched(size_t(G), 0);
    for (i32 v = 1; v < kStates; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if (e.op == OP_FACTOR) { ++nf; touched[size_t(e.g)] = 1; }
    }
    g2row.assign(size_t(G), -1);
    n_active = 0;
    for (i32 g = 0; g < G; ++g) if (touched[size_t(g)]) g2row[size_t(g)] = n_active++;
    // correction placement: eager per-class correction prefix only when the class domain
    // and the per-group base table are cache-friendly (this is the high-version regime)
    corr_on = (nf > 0);
    if (corr_on) {
      const __int128 work = __int128(nf) * __int128(Cpfx);
      const __int128 mem = __int128(n_active) * __int128(Cpfx);
      if (!(work <= 60000000 && mem <= 2500000)) { if (std::getenv("BWH_BATCH_DEBUG")) std::fprintf(stderr, "batch: corr gate work=%lld mem=%lld Cpfx=%d nf=%d\n", (long long)work, (long long)mem, Cpfx, nf); return false; }
    }
    // ---- class lookup LUT (same construction as the shipped engine)
    if (w.endpoints.size() >= 1024) {
      const i32 want = std::max<i32>(4096, std::min<i32>(1 << 20,
                                     i32(4 * w.endpoints.size() + 1)));
      i32 buckets = 1;
      while (buckets < want) buckets <<= 1;
      const i64 sspan = f.h.t_max - f.h.t_min + 1;
      i64 width = std::max<i64>(1, (sspan + buckets - 1) / buckets);
      lut_shift = 0;
      while ((i64(1) << lut_shift) < width) ++lut_shift;
      const i64 wd = i64(1) << lut_shift;
      i32 nb = i32((sspan + wd - 1) / wd);
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
    s0 = f.base_factor;
    fac = f.base_factor;
    gfen.init(ncls);
    // ---- offline plan
    auto pt0 = std::chrono::steady_clock::now();
    if (!build_plan()) { if (std::getenv("BWH_BATCH_DEBUG")) std::fprintf(stderr, "batch: plan unsupported\n"); return false; }
    if (std::getenv("BWH_BATCH_DEBUG")) std::fprintf(stderr, "batch: plan ok order=%zu\n", order.size());
    (void)pt0;
    // ---- static base arrays: bucketed by class (sequential accumulation)
    if (!build_base()) { if (std::getenv("BWH_BATCH_DEBUG")) std::fprintf(stderr, "batch: base unsupported\n"); return false; }
    if (std::getenv("BWH_BATCH_DEBUG")) std::fprintf(stderr, "batch: base ok\n");
    // ---- query frame bounds
    qlb.assign(size_t(kStates) * size_t(w.q), 0);
    qhb.assign(size_t(kStates) * size_t(w.q), 0);
    for (i32 v = 1; v < kStates; ++v)
      for (i32 k = 0; k < w.q; ++k) {
        const Query& qq = w.query(v, k);
        qlb[size_t(v) * size_t(w.q) + size_t(k)] = class_of(qq.lo);
        qhb[size_t(v) * size_t(w.q) + size_t(k)] = class_of(qq.hi);
      }
    tdev.assign(static_cast<size_t>(G), {});
    for (i32 g = 0; g < G; ++g) tdev[size_t(g)].assign(tcoord[size_t(g)].size(), 0);
    supported = true;
    return true;
  }

  void run(std::vector<i64>& answers) {
    answers.assign(size_t(kStates - 1) * size_t(w.q), 0);
    // Chain topology: the version tree is a path, so there is nothing to undo and the
    // preorder is the node order itself.
    if (w.topology == 0) {
      const i64* qlbp = qlb.data();
      const i64* qhbp = qhb.data();
      for (i32 v = 1; v < kStates; ++v) {
        apply_node(v);
        i64* out = answers.data() + size_t(v - 1) * size_t(w.q);
        const size_t base = size_t(v) * size_t(w.q);
        for (i32 k = 0; k < w.q; ++k) {
          const i32 i = i32(qlbp[base + size_t(k)]), j = i32(qhbp[base + size_t(k)]);
          out[k] = gfen.prefix(j) - gfen.prefix(i);
        }
      }
      return;
    }
    const bool prof = std::getenv("BWH_BATCH_PROF") != nullptr;
    i64 n_apply = 0, n_undo = 0, n_query = 0;
    auto now = []() { return std::chrono::steady_clock::now(); };
    auto el = [](std::chrono::steady_clock::time_point a) {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(
                 std::chrono::steady_clock::now() - a)
          .count();
    };
    std::vector<i32> path;
    path.reserve(size_t(kStates) + 4);
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
        answer_state(v, answers.data() + size_t(v - 1) * size_t(w.q));
        n_query += el(tq);
      }
    }
    if (prof) {
      std::fprintf(stderr, "BATCHPROF order=%zu ncls=%d Cpfx=%d\n", order.size(), ncls, Cpfx);
      std::fprintf(stderr, "BATCHPROF apply_ms=%.1f undo_ms=%.1f query_ms=%.1f n_apply=%d n_undo=%d\n",
                   double(n_apply) / 1e6, double(n_undo) / 1e6, double(n_query) / 1e6,
                   int(n_apply), int(n_undo));
    }
  }

 private:
  inline i32 find_ti(i32 g, i32 c) const {
    const auto& v = tcoord[size_t(g)];
    const auto it = std::lower_bound(v.begin(), v.end(), i64(c));
    return (it == v.end() || *it != c) ? -1 : i32(it - v.begin());
  }

  // ---- plan: previous states of row/factor edits, per-group support, preorder
  bool build_plan() {
    parent = w.parent;
    plan.assign(size_t(kStates), NodePlan{});
    // preorder + tin/tout
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
      std::vector<i32> cend(cnt.begin(), cnt.end());  // immutable child-range ends
      std::vector<std::pair<i32, i32>> st;  // (node, child cursor)
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
          tout[v] = clock;  // exclusive
        }
      }
    }
    // ---- row edits grouped by row id
    std::vector<u32> keys, vals;
    keys.reserve(size_t(kStates));
    for (i32 v = 1; v < kStates; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if (e.op != OP_FACTOR) { keys.push_back(u32(e.id)); vals.push_back(u32(v)); }
    }
    v2_radix_sort_u32(keys, vals);
    // base state for a row id
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
    size_t i = 0;
    while (i < keys.size()) {
      size_t j = i;
      while (j < keys.size() && keys[j] == keys[i]) ++j;
      const i64 id = i64(keys[i]);
      u8 ex; i32 g, c; i64 amt;
      base_of(id, ex, g, c, amt);
      // sort this row's edit nodes by tin (m is tiny; insertion sort)
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
        if (!stack.empty()) { const Ent& t = stack.back(); ex = t.ex; g = t.g; c = t.c; amt = t.amt; }
        else base_of(id, ex, g, c, amt);
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
        stack.push_back({v, nex, ng, nc, namt});
      }
      i = j;
    }
    // ---- per-group support from the row plan
    tcoord.assign(static_cast<size_t>(G), {});
    for (i32 v = 1; v < kStates; ++v) {
      const NodePlan& p = plan[size_t(v)];
      if (p.kind != 1) continue;
      if (p.ex_old && p.amt_old != 0) tcoord[size_t(p.g_old)].push_back(i64(p.c_old));
      if (p.ex_new && p.amt_new != 0) tcoord[size_t(p.g_new)].push_back(i64(p.c_new));
    }
    for (i32 g = 0; g < G; ++g) {
      auto& v = tcoord[size_t(g)];
      std::sort(v.begin(), v.end());
      v.erase(std::unique(v.begin(), v.end()), v.end());
    }
    // O(1) (group,class) -> support index map
    ti_map.assign(size_t(G) * size_t(ncls), -1);
    for (i32 g = 0; g < G; ++g) {
      const auto& tc = tcoord[size_t(g)];
      if (tc.empty()) continue;
      i32* row = &ti_map[size_t(g) * size_t(ncls)];
      for (size_t k = 0; k < tc.size(); ++k) row[size_t(tc[k])] = i32(k);
    }
    for (i32 v = 1; v < kStates; ++v) {
      NodePlan& p = plan[size_t(v)];
      if (p.kind != 1) continue;
      if (p.ex_old && p.amt_old != 0) {
        p.ti_old = ti_map[size_t(p.g_old) * size_t(ncls) + size_t(p.c_old)];
        if (p.ti_old < 0) { if (std::getenv("BWH_BATCH_DEBUG")) std::fprintf(stderr, "batch: ti_old missing v=%d g=%d c=%d\n", v, p.g_old, p.c_old); return false; }
      }
      if (p.ex_new && p.amt_new != 0) {
        p.ti_new = ti_map[size_t(p.g_new) * size_t(ncls) + size_t(p.c_new)];
        if (p.ti_new < 0) { if (std::getenv("BWH_BATCH_DEBUG")) std::fprintf(stderr, "batch: ti_new missing v=%d g=%d c=%d\n", v, p.g_new, p.c_new); return false; }
      }
    }
    // ---- factor plan: previous factor value per factor edit node
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
          p.g_old = g;
          p.fac_prev = st2.empty() ? s0[size_t(g)] : st2.back().val;
          p.fac_val = w.edit[size_t(v)].val;
          st2.push_back({v, p.fac_val});
        }
      }
    }
    return true;
  }

  bool build_base() {
    // One sequential scan of the mapped relation writes the per-group base-mass leaves
    // (factor-touched groups only) and the global initial-factor weighted leaf V.
    if (corr_on) gbase.assign(size_t(n_active) * size_t(Cpfx), 0);
    gb_cls.clear();
    gb_amt.clear();
    gb_off.assign(static_cast<size_t>(n_active) + 1, 0);
    {
      std::vector<std::vector<i32>> cls(static_cast<size_t>(n_active));
      std::vector<std::vector<i64>> amt(static_cast<size_t>(n_active));
      const BaseRec* rows = f.rows.data();
      // V leaf starts as the initial-factor weighted base mass.
      std::vector<i64> vleaf(static_cast<size_t>(Cpfx), 0);
      for (i64 i = 0; i < N; ++i) {
        const BaseRec& br = rows[size_t(i)];
        const i32 c = i32(class_of(br.t));
        vleaf[size_t(c) + 1] += br.amount * s0[size_t(br.g)];
        const i32 r = g2row[size_t(br.g)];
        if (r >= 0) {
          i64* row = &gbase[size_t(r) * size_t(Cpfx)];
          if (row[size_t(c) + 1] == 0) { cls[size_t(r)].push_back(c); }
          row[size_t(c) + 1] += br.amount;
        }
      }
      // compact per-group base-class lists (class ascending is not required for the
      // factor update, but ascending order keeps the writes sequential in V)
      gb_off[0] = 0;
      for (i32 r = 0; r < n_active; ++r) {
        auto& cv = cls[size_t(r)];
        auto& av = amt[size_t(r)];
        std::sort(cv.begin(), cv.end());
        // a class can be pushed more than once when a running group mass at that class
        // returns to exactly zero (negative base amounts, e.g. taxi refunds): dedupe.
        cv.erase(std::unique(cv.begin(), cv.end()), cv.end());
        av.resize(cv.size());
        const i64* row = &gbase[size_t(r) * size_t(Cpfx)];
        for (size_t k = 0; k < cv.size(); ++k) av[k] = row[size_t(cv[k]) + 1];
        gb_off[size_t(r) + 1] = gb_off[size_t(r)] + i32(cv.size());
        gb_cls.insert(gb_cls.end(), cv.begin(), cv.end());
        gb_amt.insert(gb_amt.end(), av.begin(), av.end());
      }
      V = std::move(vleaf);
      gfen.rebuild_from_leaves(V);
    }
    return true;
  }

  inline void row_apply(const NodePlan& p, i64 sgn) {
    // sgn = +1 applies the edit, -1 reverses it.  Row edits change only the deviation
    // part of V, so the base-mass leaves are untouched.
    if (sgn > 0) {
      if (p.ex_old && p.amt_old != 0) {
        const i64 wv = fac[size_t(p.g_old)] * p.amt_old;
        tdev[size_t(p.g_old)][size_t(p.ti_old)] -= p.amt_old;
        V[size_t(p.c_old) + 1] -= wv;
        gfen.add(p.c_old + 1, -wv);
      }
      if (p.ex_new && p.amt_new != 0) {
        const i64 wv = fac[size_t(p.g_new)] * p.amt_new;
        tdev[size_t(p.g_new)][size_t(p.ti_new)] += p.amt_new;
        V[size_t(p.c_new) + 1] += wv;
        gfen.add(p.c_new + 1, wv);
      }
    } else {
      if (p.ex_new && p.amt_new != 0) {
        const i64 wv = fac[size_t(p.g_new)] * p.amt_new;
        tdev[size_t(p.g_new)][size_t(p.ti_new)] -= p.amt_new;
        V[size_t(p.c_new) + 1] -= wv;
        gfen.add(p.c_new + 1, -wv);
      }
      if (p.ex_old && p.amt_old != 0) {
        const i64 wv = fac[size_t(p.g_old)] * p.amt_old;
        tdev[size_t(p.g_old)][size_t(p.ti_old)] += p.amt_old;
        V[size_t(p.c_old) + 1] += wv;
        gfen.add(p.c_old + 1, wv);
      }
    }
  }

  inline void factor_apply(const NodePlan& p, i64 ds) {
    if (ds == 0) return;
    const i32 gg = p.g_old;  // factor nodes keep their group in the g_old slot
    // base-mass classes of the group
    const i32 r = g2row[size_t(gg)];
    bool touched_any = false;
    if (r >= 0) {
      const i32 b = gb_off[size_t(r)], e = gb_off[size_t(r) + 1];
      for (i32 k = b; k < e; ++k) {
        const i64 a = gb_amt[size_t(k)];
        if (a == 0) continue;
        V[size_t(gb_cls[size_t(k)]) + 1] += ds * a;
        touched_any = true;
      }
    }
    // deviation classes of the group
    auto& tc = tcoord[size_t(gg)];
    auto& dv = tdev[size_t(gg)];
    for (size_t k = 0; k < tc.size(); ++k) {
      const i64 d = dv[k];
      if (d == 0) continue;
      V[size_t(tc[k]) + 1] += ds * d;
      touched_any = true;
    }
    if (touched_any) gfen.rebuild_from_leaves(V);
  }

  inline void apply_node(i32 v) {
    NodePlan& p = plan[size_t(v)];
    if (p.kind == 1) {
      row_apply(p, +1);
    } else if (p.kind == 2) {
      const i32 gg = p.g_old;
      const i64 ds = p.fac_val - p.fac_prev;
      factor_apply(p, ds);
      fac[size_t(gg)] = p.fac_val;
    }
  }

  inline void undo_node(i32 v) {
    NodePlan& p = plan[size_t(v)];
    if (p.kind == 1) {
      row_apply(p, -1);
    } else if (p.kind == 2) {
      const i32 gg = p.g_old;
      factor_apply(p, p.fac_prev - p.fac_val);
      fac[size_t(gg)] = p.fac_prev;
    }
  }

  inline void answer_state(i32 v, i64* out) {
    const size_t base = size_t(v) * size_t(w.q);
    for (i32 k = 0; k < w.q; ++k) {
      const i32 i = i32(qlb[base + size_t(k)]), j = i32(qhb[base + size_t(k)]);
      out[k] = gfen.prefix(j) - gfen.prefix(i);
    }
  }
};

}  // namespace bwh
