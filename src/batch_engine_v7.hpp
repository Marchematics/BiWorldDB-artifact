// batch_engine_v7.hpp — version-tree batch execution, v7: block-parallel chain execution.
//
// v7 keeps the v6 parallel setup (declared workload/query stream, offline plan, base scan,
// query-frame classes) and replaces the serial chain execute with a block-parallel algorithm:
//
//   1. the chain's states are partitioned into B blocks; a cheap sequential pre-pass walks the
//      declared edits once to record, at every block start, the factor vector and the sparse
//      per-group deviation slices for the groups that a factor edit in that block will touch;
//   2. every block then runs INDEPENDENTLY in parallel: starting from a zero local prefix it
//      replays its own edits (row deltas weighted by the globally-known factor state, factor
//      edits over the group's merged base+deviation support), accumulates a local Fenwick and
//      records (a) the block's total dense V-leaf delta and (b) the local prefix value at each
//      of its states' declared query endpoints;
//   3. a tiny serial pass over blocks accumulates the dense V start-leaf and materialises the
//      carry prefix array per block (O(B*C));
//   4. a parallel combine pass adds the carry prefix to the stored local endpoint prefixes;
//      the answer vector is exact (integer additions only).
//
// Only the O(B*C) carry pass and the O(V) pre-pass are serial; all edit/query work is block
// parallel.  Exactness vs v5/v6/rf/shipped is one SHA.  GPU/CUDA is not used.
//
// v6 = v5 executor with the setup phase parallelised with OpenMP (CPU, single socket):
//   * declared query-stream construction: per-state RNG draws and endpoint collection are
//     embarrassingly parallel (per-thread endpoint buffers, concatenated then deduped);
//   * offline plan: row-state resolution is parallelised over row-id groups (each row's edit
//     chain is independent), support marking uses per-thread class bitmaps merged by OR,
//     factor plans are resolved per group, support->class lists and the O(1) support index are
//     filled per group, and per-state query-frame classes are computed per state;
//   * base scan: parallel row scan with per-thread V-leaves (integer reduction) and atomic
//     per-group base-mass accumulation; per-group base-class lists are built afterwards.
// The execute phase is intrinsically sequential on a chain (each state's query depends on the
// prefix of all previous edits); it is left single-threaded and reported separately.  Integer
// additions make the parallel reductions exact, so the answer vector is bit-identical to v5.
// GPU/CUDA is not used.
//
// v5 = v4 blocked-snapshot prefix executor plus the measured pipeline optimisations:
//   * uninitialised POD plan storage (`new NodePlan[kStates]`, no 19 MB value-initialisation);
//   * support-loop software prefetch for the factor-edit base/deviation lists;
//   * (panel level) SHA-NI/OpenSSL extract path, applied identically to this engine and to
//     the aligned PGD-G comparison panel (sha_fast.hpp, BWH_SHA=sw reproduces the software
//     path for an apples-to-apples report of both).
// The arithmetic and exactness contract are unchanged from v4.
//
// v4 replaces the per-query Fenwick with a dense prefix array `pref` over the unified
// weighted-mass leaf V.  `pref` is rebuilt from V at snapshot points (every B states and at
// every factor edit); between snapshots, row edits only update their V point(s) and append
// (class, weight) delta events to a small pending list.  A query is
//     pref[c] + sum_{pending events with class < c} weight
// where full pending blocks are aggregated by a tiny Fenwick over 64-class blocks and only
// the events inside the endpoint's own block are scanned (a handful).  Factor edits update V
// at the group support and rebuild `pref` immediately (env BWH_V4_FACLAZY=1 defers that
// rebuild to the next query; both were measured).  Snapshot interval B is set by BWH_V4_B
// (default 32; sweep 16/32/64/128).  The prefix rebuild is vectorised with AVX2 when
// available.  Everything else (offline plan, support bitmap, unified V algebra, exactness)
// is inherited from v3.
//
// v3 changes over v2 (cross-version sharing / cheaper factor edits):
//   * chain-specialised offline plan: no preorder CSR, no tin/tout ancestor structure — the
//     edit sequence is the version order, so previous row/factor states are the last states
//     seen per row/group;
//   * merged per-group factor-edit support: the group's base-mass classes and deviation
//     classes are merged once at build time into a single (class, base amount, deviation
//     index) list, so a factor edit does ONE pass over the union (one read + one write per
//     class) instead of two passes with duplicate writes at shared classes;
//   * the Fenwick rebuild is still O(classes) per factor edit, but the support work drops to
//     the union size; the remaining cost model and its floor are documented in
//     results/batch_v3_20261031/NOTE.md.
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

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include <omp.h>
#include <memory>
#include "common.hpp"

namespace bwh {

// ------------------------------------------------------------------ radix sorts
// LSD radix sort of u64 keys with their u32 payloads (11-bit digits).
inline void v7_radix_sort_u64(std::vector<u64>& key, std::vector<u32>& val) {
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
inline void v7_radix_sort_u32(std::vector<u32>& key, std::vector<u32>& val) {
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
struct BatchWorkloadV7 {
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
  // v5 fused query stream: 2*u8 draw values per (state,level) instead of materialised
  // Query objects; qdraws.size() == 0 when the stream was materialised.
  std::vector<u8> qdraws;
  inline const Query& query(i32 v, i32 k) const {
    return per_state ? queries[size_t(v) * size_t(q) + size_t(k)] : queries[size_t(k)];
  }
};

// First q iterations of the shipped gen_nested (bit-identical for k < q; the shipped
// version runs kMaxQ iterations, but values for k < q depend only on the first q draws).
inline void v7_gen_nested_q(Rng& r, i64 tmin, i64 tmax, i32 q, Query* out) {
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
inline BatchWorkloadV7 make_workload_fast_v7(const Fixture& f, i32 topology, i32 seed,
                                        i32 factor_period, i32 q, bool per_state_ranges) {
  BatchWorkloadV7 w;
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
  const bool fuse_queries = [] {
    const char* e = std::getenv("BWH_V5_FUSE");
    return !(e && std::strcmp(e, "0") == 0);  // default on; BWH_V5_FUSE=0 materialises
  }();
  if (fuse_queries) {
    const size_t nst = per_state_ranges ? size_t(kStates - 1) : 1;
    w.qdraws.resize(nst * size_t(2 * q));
#pragma omp parallel for schedule(static)
    for (size_t s = 0; s < nst; ++s) {
      Rng qr(per_state_ranges
                 ? u64(seed) * 1000000007ULL + u64(topology) * 65537ULL +
                       u64(factor_period) * 131ULL + u64(s + 1) * 2654435761ULL + 17ULL
                 : u64(seed) * 1000000007ULL + u64(topology) * 65537ULL +
                       u64(factor_period) * 131ULL + 17ULL);
      u8* d = &w.qdraws[s * size_t(2 * q)];
      for (i32 k = 0; k < q; ++k) { d[2 * k] = u8(qr.bounded(16)); d[2 * k + 1] = u8(qr.bounded(16)); }
    }
  } else if (per_state_ranges) {
    w.queries.resize(size_t(kStates) * size_t(q));
    for (i32 v = 1; v < kStates; ++v) {
      Rng qr(u64(seed) * 1000000007ULL + u64(topology) * 65537ULL +
             u64(factor_period) * 131ULL + u64(v) * 2654435761ULL + 17ULL);
      v7_gen_nested_q(qr, tmin, tmax, q, &w.queries[size_t(v) * size_t(q)]);
    }
  } else {
    w.queries.resize(size_t(q));
    Rng qr(u64(seed) * 1000000007ULL + u64(topology) * 65537ULL +
           u64(factor_period) * 131ULL + 17ULL);
    v7_gen_nested_q(qr, tmin, tmax, q, w.queries.data());
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
  auto replay = [&](const u8* d, i64& lo, i64& hi, i32 k) {
    // identical arithmetic to v7_gen_nested_q, driven by the stored draw values
    (void)k;
  };
  {
    const int NT = omp_get_max_threads();
    std::vector<std::vector<u64>> tkeys(static_cast<size_t>(NT));
    if (!w.qdraws.empty()) {
      const bool ps = per_state_ranges;
      const size_t nst = ps ? size_t(kStates - 1) : 1;
#pragma omp parallel
      {
        std::vector<u64>& lk = tkeys[static_cast<size_t>(omp_get_thread_num())];
#pragma omp for schedule(static)
        for (long long si = 0; si < (long long)nst; ++si) {
          const size_t s = static_cast<size_t>(si);
          const u8* d = &w.qdraws[s * size_t(2 * q)];
          i64 lo = tmin, hi = tmax + 1;
          const i64 span0 = hi - lo;
          for (i32 k = 0; k < q; ++k) {
            i64 span = hi - lo;
            if (span <= 4096) { lo = tmin + span0 / 2; hi = lo + 2048; span = hi - lo; }
            lo += span * i64(3 + d[2 * k]) / 100;
            hi -= span * i64(3 + d[2 * k + 1]) / 100;
            if (hi - lo < 1024) hi = lo + 1024;
            lk.push_back(u64(lo - tmin));
            lk.push_back(u64(hi - tmin));
          }
        }
      }
    } else {
#pragma omp parallel
      {
        std::vector<u64>& lk = tkeys[static_cast<size_t>(omp_get_thread_num())];
#pragma omp for schedule(static)
        for (long long vi = 1; vi < (long long)kStates; ++vi) {
          const i32 v = static_cast<i32>(vi);
          for (i32 k = 0; k < q; ++k) {
            const Query& qq = w.query(v, k);
            lk.push_back(u64(qq.lo - tmin));
            lk.push_back(u64(qq.hi - tmin));
          }
        }
      }
    }
    size_t tot = 0;
    for (auto& v : tkeys) tot += v.size();
    keys.clear();
    keys.reserve(tot);
    for (auto& v : tkeys) keys.insert(keys.end(), v.begin(), v.end());
  }
  (void)replay;
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
    v7_radix_sort_u64(keys, dummy);
    w.endpoints.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i)
      if (i == 0 || keys[i] != keys[i - 1]) w.endpoints.push_back(i64(keys[i]) + tmin);
  }
  return w;
}

// ---------------------------------------------------------------- dense Fenwick
struct DenseFenwickV7 {
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
class BatchEngineV7 {
 public:
  const Fixture& f;
  const BatchWorkloadV7& w;
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
  // v4 blocked-snapshot prefix over V (replaces the per-query Fenwick)
  std::vector<i64> pref;         // pref[k] = sum_{class c < k} V[c]  (size ncls+1)
  i32 nblocks = 0;               // 64-class blocks
  std::vector<i64> pfen;         // Fenwick over pending event weights per block
  std::vector<std::vector<i32>> pblk;  // per-block pending event indices
  std::vector<i32> pcls;         // pending event class
  std::vector<i64> pw;           // pending event weight
  std::vector<i32> ptouched;     // blocks with pending events (for cheap clearing)
  i32 states_since = 0;          // states since last snapshot rebuild
  i32 rebuild_b = 64;            // snapshot interval B (best measured; env BWH_V4_B)
  bool pf_support = false;      // env BWH_V5_PREFETCH: prefetch the factor support lists
  bool fac_lazy = false;         // defer the factor-edit rebuild to the next query
  bool dirty = false;            // V changed by a factor edit, pref stale
  i64 t_rebuild_ns = 0;          // instrumentation

  // ---------------- v7 block-parallel chain execution ----------------
  bool v7_ok = false;
  i32 v7_blocks = 0;
  std::vector<i32> v7_v0, v7_v1;                 // block state ranges [v0, v1)
  std::vector<std::vector<i32>> v7_groups;       // factor-edited groups per block
  std::vector<std::vector<i64>> v7_fac;          // factor vector at block start
  std::vector<std::vector<i64>> v7_snap_off;     // per-group snapshot offset (i64 index)
  std::vector<i64> v7_snap_data;                 // sparse deviation snapshots
  std::vector<i64> v7_Vbase;                     // V leaf at the root
  std::vector<i64> v7_lvals;                     // local endpoint prefixes per state
  std::vector<i64> v7_dV;                        // dense V-leaf delta per block
  std::vector<i64> v7_carrypref;                 // dense carry prefix per block
  std::vector<i32> v7_bidx;                      // state -> block index
  std::vector<i32> ti_map;      // (group row, class) -> support index, or -1 (O(1) lookup)
  // v3: merged factor-edit support per touched group: union of base-mass classes and
  // deviation classes, one (class, base amount, deviation index) entry per class.
  std::vector<i32> row2g;       // active row -> group id
  std::vector<i32> ms_off;      // n_active+1 offsets
  std::vector<i64> ms_cls;      // class
  std::vector<i64> ms_base;     // base-mass amount at that class
  std::vector<i32> ms_dev;      // deviation support index, or -1

  // dynamic per-group deviations (sparse support)
  std::vector<std::vector<i64>> tcoord;  // per group sorted touched classes
  std::vector<std::vector<i64>> tdev;    // per group deviations at those classes

  std::vector<i64> fac, s0;
  std::vector<i64> qlb, qhb;  // [v*q + k]

  // offline plan
  struct NodePlan {  // POD: uninitialised allocation; every edited node sets its fields
    i32 g_old, c_old, ti_old;
    i32 g_new, c_new, ti_new;
    i64 amt_old, amt_new;
    i64 fac_val, fac_prev;
    u8 kind;  // 0 none, 1 row, 2 factor
    u8 ex_old, ex_new;
  };
  std::unique_ptr<NodePlan[]> plan_buf;  // kStates, uninitialised
  NodePlan* plan = nullptr;
  std::vector<i32> order;      // preorder of nodes
  std::vector<i32> parent;     // kStates

  explicit BatchEngineV7(const Fixture& fi, const BatchWorkloadV7& wo) : f(fi), w(wo) {}

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
    rebuild_b = 64;
    if (const char* be = std::getenv("BWH_V4_B")) { const int x = std::atoi(be); if (x > 0) rebuild_b = x; }
    fac_lazy = std::getenv("BWH_V4_FACLAZY") != nullptr;
    pf_support = std::getenv("BWH_V5_PREFETCH") != nullptr;
    pref.assign(size_t(ncls) + 1, 0);
    nblocks = (ncls + 63) / 64;
    pfen.assign(size_t(nblocks) + 1, 0);
    pblk.assign(static_cast<size_t>(nblocks), {});
    pcls.reserve(128); pw.reserve(128); ptouched.reserve(32);
    // ---- offline plan
    auto pt0 = std::chrono::steady_clock::now();
    auto b0 = std::chrono::steady_clock::now();
    if (!build_plan()) { if (std::getenv("BWH_BATCH_DEBUG")) std::fprintf(stderr, "batch: plan unsupported\n"); return false; }
    auto b1 = std::chrono::steady_clock::now();
    if (std::getenv("BWH_BATCH_DEBUG")) std::fprintf(stderr, "batch: plan ok order=%zu\n", order.size());
    (void)pt0;
    // ---- static base arrays: bucketed by class (sequential accumulation)
    if (!build_base()) { if (std::getenv("BWH_BATCH_DEBUG")) std::fprintf(stderr, "batch: base unsupported\n"); return false; }
    auto b2 = std::chrono::steady_clock::now();
    if (std::getenv("BWH_BATCH_DEBUG")) std::fprintf(stderr, "batch: base ok\n");
    build_merged_support();
    auto b3 = std::chrono::steady_clock::now();
    // ---- query frame bounds (fused draw stream, or materialised Query objects)
    qlb.assign(size_t(kStates) * size_t(w.q), 0);
    qhb.assign(size_t(kStates) * size_t(w.q), 0);
    if (!w.qdraws.empty()) {
      const bool ps = w.per_state;
#pragma omp parallel for schedule(static)
      for (long long vi = 1; vi < (long long)kStates; ++vi) {
        const i32 v = static_cast<i32>(vi);
        const size_t s = ps ? size_t(v - 1) : 0;
        const u8* d = &w.qdraws[s * size_t(2 * w.q)];
        i64 lo = w.t_min, hi = w.t_max + 1;
        const i64 span0 = hi - lo;
        for (i32 k = 0; k < w.q; ++k) {
          i64 span = hi - lo;
          if (span <= 4096) { lo = w.t_min + span0 / 2; hi = lo + 2048; span = hi - lo; }
          lo += span * i64(3 + d[2 * k]) / 100;
          hi -= span * i64(3 + d[2 * k + 1]) / 100;
          if (hi - lo < 1024) hi = lo + 1024;
          const size_t qidx = size_t(v) * size_t(w.q) + size_t(k);
          qlb[qidx] = class_of(lo);
          qhb[qidx] = class_of(hi);
        }
      }
    } else {
#pragma omp parallel for schedule(static)
      for (long long vi = 1; vi < (long long)kStates; ++vi) {
        const i32 v = static_cast<i32>(vi);
        for (i32 k = 0; k < w.q; ++k) {
          const Query& qq = w.query(v, k);
          qlb[size_t(v) * size_t(w.q) + size_t(k)] = class_of(qq.lo);
          qhb[size_t(v) * size_t(w.q) + size_t(k)] = class_of(qq.hi);
        }
      }
    }
    tdev.assign(static_cast<size_t>(G), {});
    for (i32 g = 0; g < G; ++g) tdev[size_t(g)].assign(tcoord[size_t(g)].size(), 0);
    supported = true;
    v7_ok = build_v7_blocks();
    return true;
  }

  // ---------------- v7: block-parallel chain execution ----------------
  bool build_v7_blocks() {
    if (w.topology != 0) return false;
    if (!corr_on) return false;  // needs the merged base/deviation support lists
    const i32 VS = kStates - 1;
    if (VS < 2) return false;
    i32 B = omp_get_max_threads();
    if (const char* e = std::getenv("BWH_V7_BLOCKS")) { const int x = std::atoi(e); if (x > 0) B = x; }
    if (B < 1) B = 1;
    if (B > VS) B = VS;
    v7_blocks = B;
    v7_v0.assign(static_cast<size_t>(B) + 1, 0);
    v7_v1.assign(static_cast<size_t>(B) + 1, 0);
    for (i32 b = 0; b <= B; ++b) {
      v7_v0[size_t(b)] = 1 + i32((i64(VS) * i64(b)) / i64(B));
      v7_v1[size_t(b)] = (b == B) ? kStates : 1 + i32((i64(VS) * i64(b + 1)) / i64(B));
    }
    v7_bidx.assign(static_cast<size_t>(kStates), 0);
    // gather factor-edited groups per block, mapping states to blocks via the boundaries
    v7_groups.assign(static_cast<size_t>(B), {});
    {
      i32 b = 0;
      for (i32 v = 1; v < kStates; ++v) {
        while (b + 1 < B && v >= v7_v1[size_t(b)]) ++b;
        v7_bidx[size_t(v)] = b;
        const NodePlan& p = plan[size_t(v)];
        if (p.kind == 2) v7_groups[size_t(b)].push_back(p.g_old);
      }
    }
    for (i32 b = 0; b < B; ++b) {
      auto& g = v7_groups[size_t(b)];
      std::sort(g.begin(), g.end());
      g.erase(std::unique(g.begin(), g.end()), g.end());
    }
    // snapshot offsets
    v7_snap_off.assign(static_cast<size_t>(B), {});
    v7_snap_data.clear();
    {
      size_t tot = 0;
      for (i32 b = 0; b < B; ++b)
        for (i32 g : v7_groups[size_t(b)]) tot += tcoord[size_t(g)].size();
      v7_snap_data.assign(tot, 0);
    }
    {
      size_t pos = 0;
      for (i32 b = 0; b < B; ++b) {
        auto& off = v7_snap_off[size_t(b)];
        off.resize(v7_groups[size_t(b)].size());
        for (size_t i = 0; i < v7_groups[size_t(b)].size(); ++i) {
          off[i] = i64(pos);
          pos += tcoord[size_t(v7_groups[size_t(b)][i])].size();
        }
      }
    }
    v7_fac.assign(static_cast<size_t>(B), std::vector<i64>(size_t(G), 0));
    auto pv0 = std::chrono::steady_clock::now();
    auto pv_now = []() { return std::chrono::steady_clock::now(); };
    auto pv_us = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
      return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    // ---- block-start factor vectors (serial, O(V + B*G))
    v7_fac.assign(static_cast<size_t>(B), std::vector<i64>(size_t(G), 0));
    {
      std::vector<i64> fcur = s0;
      i32 b = 0;
      for (i32 v = 1; v < kStates; ++v) {
        while (b < B && v >= v7_v1[size_t(b)]) ++b;
        if (b >= B) break;
        if (v == v7_v0[size_t(b)]) v7_fac[size_t(b)] = fcur;
        const NodePlan& p = plan[size_t(v)];
        if (p.kind == 2) fcur[size_t(p.g_old)] = p.fac_val;
      }
    }
    auto pv1 = std::chrono::steady_clock::now();
    // ---- deviation events grouped by row-edit group (serial gather, node order preserved)
    struct DevEv { i32 node; i32 ti; i64 delta; };
    std::vector<std::vector<DevEv>> ev(static_cast<size_t>(G));
    {
      size_t tot = 0;
      for (i32 v = 1; v < kStates; ++v) {
        const NodePlan& p = plan[size_t(v)];
        if (p.kind != 1) continue;
        if (p.ex_old && p.amt_old != 0) { ev[size_t(p.g_old)].push_back({v, p.ti_old, -p.amt_old}); ++tot; }
        if (p.ex_new && p.amt_new != 0) { ev[size_t(p.g_new)].push_back({v, p.ti_new, p.amt_new}); ++tot; }
      }
      (void)tot;
    }
    auto pv2 = std::chrono::steady_clock::now();
    // inverse map: for each group, the (block, snapshot offset) pairs where it is needed
    std::vector<std::vector<std::pair<i32, i64>>> gslots(static_cast<size_t>(G));
    for (i32 b = 0; b < B; ++b)
      for (size_t i = 0; i < v7_groups[size_t(b)].size(); ++i)
        gslots[size_t(v7_groups[size_t(b)][i])].push_back({b, v7_snap_off[size_t(b)][i]});
    auto pv3 = std::chrono::steady_clock::now();
    // ---- parallel pre-pass: per-group deviation timeline -> block-start snapshots
#pragma omp parallel for schedule(dynamic, 1)
    for (long long gi = 0; gi < (long long)G; ++gi) {
      const i32 g = static_cast<i32>(gi);
      const auto& my = ev[size_t(g)];
      if (my.empty()) continue;  // deviations stay zero for this group
      std::vector<i64> dv(tcoord[size_t(g)].size(), 0);
      size_t e = 0;
      for (const auto& slot : gslots[size_t(g)]) {
        const i32 b = slot.first;
        const i64 off = slot.second;
        const i32 v0 = v7_v0[size_t(b)];
        while (e < my.size() && my[e].node < v0) {
          dv[size_t(my[e].ti)] += my[e].delta;
          ++e;
        }
        i64* dst = &v7_snap_data[size_t(off)];
        for (size_t k = 0; k < dv.size(); ++k) dst[k] = dv[k];
      }
    }
    v7_Vbase = V;  // V is still the root weighted-base leaf at this point
    {
      auto pv4 = pv_now();
      if (std::getenv("BWH_V7PROF"))
        std::fprintf(stderr, "V7PREP fac_us=%lld ev_us=%lld slot_us=%lld snap_us=%lld total_us=%lld\n",
                     (long long)pv_us(pv0, pv1), (long long)pv_us(pv1, pv2),
                     (long long)pv_us(pv2, pv3), (long long)pv_us(pv3, pv4),
                     (long long)pv_us(pv0, pv4));
    }
    return true;
  }

  void run_blocks(std::vector<i64>& answers) {
    const i32 B = v7_blocks;
    const i32 q = w.q;
    const i32 VS = kStates - 1;
    answers.assign(size_t(VS) * size_t(q), 0);
    v7_lvals.assign(size_t(VS) * size_t(q) * 2, 0);
    v7_dV.assign(size_t(B) * size_t(Cpfx), 0);
    v7_carrypref.assign(size_t(B) * size_t(Cpfx), 0);
    const bool prof = std::getenv("BWH_V7PROF") != nullptr;
    auto now = []() { return std::chrono::steady_clock::now(); };
    auto el = [](std::chrono::steady_clock::time_point a) {
      return std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::steady_clock::now() - a)
          .count();
    };
    auto t0 = now();
    // ---- phase 1: independent block-local execution (parallel)
#pragma omp parallel
    {
      std::vector<i64> fen(static_cast<size_t>(Cpfx), 0);   // local Fenwick over classes
      std::vector<i64> lleaf(static_cast<size_t>(Cpfx), 0); // local dense V-leaf delta
      std::vector<i32> need(size_t(G), -1);
      std::vector<std::vector<i64>> devcopy;
      const i64* cpref = nullptr;
      (void)cpref;
#pragma omp for schedule(dynamic, 1)
      for (long long bi = 0; bi < (long long)B; ++bi) {
        const i32 b = static_cast<i32>(bi);
        const i32 v0 = v7_v0[size_t(b)], v1 = v7_v1[size_t(b)];
        if (v0 >= v1) continue;
        std::fill(fen.begin(), fen.end(), 0);
        std::fill(lleaf.begin(), lleaf.end(), 0);
        std::fill(need.begin(), need.end(), -1);
        std::vector<i64> locfac = v7_fac[size_t(b)];
        const auto& bg = v7_groups[size_t(b)];
        devcopy.assign(bg.size(), {});
        for (size_t i = 0; i < bg.size(); ++i) {
          const i32 g = bg[i];
          need[size_t(g)] = static_cast<i32>(i);
          const auto& tc = tcoord[size_t(g)];
          devcopy[i].resize(tc.size());
          const i64* src = &v7_snap_data[size_t(v7_snap_off[size_t(b)][i])];
          for (size_t k = 0; k < tc.size(); ++k) devcopy[i][k] = src[k];
        }
        auto fen_add = [&](i32 idx1, i64 v) {
          for (i32 i = idx1; i <= ncls; i += i & (-i)) fen[size_t(i)] += v;
        };
        auto fen_pref = [&](i32 x) {
          i64 r = 0;
          for (i32 i = x; i > 0; i -= i & (-i)) r += fen[size_t(i)];
          return r;
        };
        for (i32 v = v0; v < v1; ++v) {
          const NodePlan& p = plan[size_t(v)];
          if (p.kind == 1) {
            if (p.ex_old && p.amt_old != 0) {
              const i64 wv = locfac[size_t(p.g_old)] * p.amt_old;
              lleaf[size_t(p.c_old) + 1] -= wv;
              fen_add(p.c_old + 1, -wv);
              const i32 ni = need[size_t(p.g_old)];
              if (ni >= 0) devcopy[size_t(ni)][size_t(p.ti_old)] -= p.amt_old;
            }
            if (p.ex_new && p.amt_new != 0) {
              const i64 wv = locfac[size_t(p.g_new)] * p.amt_new;
              lleaf[size_t(p.c_new) + 1] += wv;
              fen_add(p.c_new + 1, wv);
              const i32 ni = need[size_t(p.g_new)];
              if (ni >= 0) devcopy[size_t(ni)][size_t(p.ti_new)] += p.amt_new;
            }
          } else if (p.kind == 2) {
            const i32 g = p.g_old;
            const i64 ds = p.fac_val - p.fac_prev;
            const i32 row = g2row[size_t(g)];
            const i32 ni = need[size_t(g)];
            if (row >= 0 && ds != 0) {
              const i32 kb = ms_off[size_t(row)], ke = ms_off[size_t(row) + 1];
              const i64* dv = (ni >= 0) ? devcopy[size_t(ni)].data() : nullptr;
              for (i32 k = kb; k < ke; ++k) {
                const i32 di = ms_dev[size_t(k)];
                const i64 dev = (di >= 0 && dv) ? dv[di] : 0;
                const i64 d = ds * (ms_base[size_t(k)] + dev);
                if (d != 0) {
                  const i32 c1 = static_cast<i32>(ms_cls[size_t(k)]) + 1;
                  lleaf[size_t(c1)] += d;
                  fen_add(c1, d);
                }
              }
            }
            locfac[size_t(g)] = p.fac_val;
          }
          const size_t lb = (size_t(v) - 1) * size_t(q);
          for (i32 kk = 0; kk < q; ++kk) {
            const i32 i = i32(qlb[size_t(v) * size_t(q) + size_t(kk)]);
            const i32 j = i32(qhb[size_t(v) * size_t(q) + size_t(kk)]);
            v7_lvals[(lb + size_t(kk)) * 2] = fen_pref(i);
            v7_lvals[(lb + size_t(kk)) * 2 + 1] = fen_pref(j);
          }
        }
        i64* dst = &v7_dV[size_t(b) * size_t(Cpfx)];
        for (i32 c = 0; c < Cpfx; ++c) dst[size_t(c)] = lleaf[size_t(c)];
      }
    }
    auto t1 = now();
    // ---- phase 2: serial block-carry pass (dense C arrays)
    {
      std::vector<i64> vstart = v7_Vbase;
      for (i32 b = 0; b < B; ++b) {
        i64* cp = &v7_carrypref[size_t(b) * size_t(Cpfx)];
        cp[0] = 0;
        for (i32 k = 1; k < Cpfx; ++k) cp[size_t(k)] = cp[size_t(k - 1)] + vstart[size_t(k)];
        const i64* dV = &v7_dV[size_t(b) * size_t(Cpfx)];
        for (i32 k = 1; k < Cpfx; ++k) vstart[size_t(k)] += dV[size_t(k)];
      }
    }
    auto t2 = now();
    // ---- phase 3: combine carry prefix + stored local endpoint prefixes (parallel)
#pragma omp parallel for schedule(static)
    for (long long vi = 1; vi < (long long)kStates; ++vi) {
      const i32 v = static_cast<i32>(vi);
      const i32 b = v7_bidx[size_t(v)];
      const i64* cp = &v7_carrypref[size_t(b) * size_t(Cpfx)];
      const size_t lb = (size_t(v) - 1) * size_t(q);
      for (i32 kk = 0; kk < q; ++kk) {
        const i32 i = i32(qlb[size_t(v) * size_t(q) + size_t(kk)]);
        const i32 j = i32(qhb[size_t(v) * size_t(q) + size_t(kk)]);
        answers[lb + size_t(kk)] = (cp[size_t(j)] - cp[size_t(i)]) +
                                   (v7_lvals[(lb + size_t(kk)) * 2 + 1] -
                                    v7_lvals[(lb + size_t(kk)) * 2]);
      }
    }
    if (prof) {
      auto t3 = now();
      std::fprintf(stderr, "V7PROF blocks=%d phase1_ms=%.1f carry_ms=%.1f combine_ms=%.1f\\n",
                   B, double(el(t0)) / 1e3, double(el(t1)) / 1e3, double(el(t2)) / 1e3);
      (void)t3;
    }
  }

  void run(std::vector<i64>& answers) {
    if (v7_ok) { run_blocks(answers); return; }
    answers.assign(size_t(kStates - 1) * size_t(w.q), 0);
    // Chain topology: the version tree is a path, so there is nothing to undo and the
    // preorder is the node order itself.
    if (w.topology == 0) {
      const i64* qlbp = qlb.data();
      const i64* qhbp = qhb.data();
      const bool prof = std::getenv("BWH_V4PROF") != nullptr;
      i64 t_row = 0, t_fac = 0, t_q = 0;
      const i64 t_rebuild0 = t_rebuild_ns;
      auto nw = []() { return std::chrono::steady_clock::now(); };
      auto el = [](std::chrono::steady_clock::time_point a) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - a).count();
      };
      for (i32 v = 1; v < kStates; ++v) {
        if (prof) {
          const u8 kind = plan[size_t(v)].kind;
          auto a = nw();
          apply_node(v);
          const i64 dt = el(a);
          if (kind == 2) t_fac += dt; else t_row += dt;
        } else {
          apply_node(v);
        }
        // snapshot policy: materialise a deferred factor rebuild, then the periodic rebuild
        if (dirty || ++states_since >= rebuild_b) rebuild_pref();
        const size_t base = size_t(v) * size_t(w.q);
        i64* out = answers.data() + size_t(v - 1) * size_t(w.q);
        if (prof) {
          auto a = nw();
          for (i32 k = 0; k < w.q; ++k) {
            const i32 i = i32(qlbp[base + size_t(k)]), j = i32(qhbp[base + size_t(k)]);
            out[k] = pref_query(j) - pref_query(i);
          }
          t_q += el(a);
          continue;
        }
        for (i32 k = 0; k < w.q; ++k) {
          const i32 i = i32(qlbp[base + size_t(k)]), j = i32(qhbp[base + size_t(k)]);
          out[k] = pref_query(j) - pref_query(i);
        }
      }
      if (prof)
        std::fprintf(stderr,
                     "V4PROF row_ms=%.1f fac_ms=%.1f query_ms=%.1f rebuild_ms=%.1f B=%d lazy=%d\n",
                     double(t_row) / 1e6, double(t_fac) / 1e6, double(t_q) / 1e6,
                     double(t_rebuild_ns - t_rebuild0) / 1e6, rebuild_b, int(fac_lazy));
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
        if (dirty || ++states_since >= rebuild_b) rebuild_pref();
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
    const bool prof = std::getenv("BWH_V4PROF") != nullptr;
    auto nw = []() { return std::chrono::steady_clock::now(); };
    auto el = [](std::chrono::steady_clock::time_point a) {
      return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - a).count();
    };
    auto p0 = nw();
    parent = w.parent;
    plan_buf.reset(new NodePlan[kStates]);
    plan = plan_buf.get();
    plan[0].kind = 0;  // root has no edit; no other field of the root is ever read
    auto p1 = nw();
    // preorder + tin/tout (chain: the node sequence is the preorder, no CSR/DFS needed)
    std::vector<i32> tin(static_cast<size_t>(kStates)), tout(static_cast<size_t>(kStates));
    order.clear();
    order.reserve(size_t(kStates));
    if (w.topology == 0) {
      order.resize(size_t(kStates));
#pragma omp parallel for schedule(static)
      for (long long vi = 0; vi < (long long)kStates; ++vi) {
        const i32 v = static_cast<i32>(vi);
        order[size_t(v)] = v; tin[size_t(v)] = v; tout[size_t(v)] = kStates;
      }
    } else {
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
    auto p2 = nw();
    // ---- row edits grouped by row id
    std::vector<u32> keys, vals;
    keys.reserve(size_t(kStates));
    for (i32 v = 1; v < kStates; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if (e.op != OP_FACTOR) { keys.push_back(u32(e.id)); vals.push_back(u32(v)); }
    }
    v7_radix_sort_u32(keys, vals);
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
    // group boundaries by row id (serial, ~280k steps)
    std::vector<size_t> gstart;
    gstart.reserve(1024);
    {
      size_t i = 0;
      while (i < keys.size()) {
        gstart.push_back(i);
        const u64 k = keys[i];
        ++i;
        while (i < keys.size() && keys[i] == k) ++i;
      }
      gstart.push_back(keys.size());
    }
    if (w.topology == 0) {
      // chain: each row's edit sequence is independent -> parallel over row groups
      const long long ng = (long long)gstart.size() - 1;
#pragma omp parallel
      {
        std::vector<Ent> stack;
        stack.reserve(16);
#pragma omp for schedule(dynamic, 64)
        for (long long gi = 0; gi < ng; ++gi) {
          const size_t i = gstart[size_t(gi)], j = gstart[size_t(gi) + 1];
          const i64 id = i64(keys[i]);
          u8 ex; i32 g, c; i64 amt;
          base_of(id, ex, g, c, amt);
          stack.clear();
          for (size_t a = i; a < j; ++a) {
            const i32 v = i32(vals[a]);
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
        }
      }
    } else {
      std::vector<Ent> stack;
      for (size_t gi = 0; gi + 1 < gstart.size(); ++gi) {
        const size_t i = gstart[gi], j = gstart[gi + 1];
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
      }
    }
    auto p3 = nw();
    // ---- per-group support from the row plan (v3: one class bitmap per group instead of
    // vector push + per-group sort; the bitmap is G*C/8 bytes and stays in L1/L2)
    tcoord.assign(static_cast<size_t>(G), {});
    {
      const i32 wpr = (ncls + 63) / 64;
      const int NT = omp_get_max_threads();
      std::vector<u64> tbm(size_t(NT) * size_t(G) * size_t(wpr), 0);
#pragma omp parallel
      {
        u64* my = &tbm[size_t(omp_get_thread_num()) * size_t(G) * size_t(wpr)];
#pragma omp for schedule(static)
        for (long long vi = 1; vi < (long long)kStates; ++vi) {
          const NodePlan& p = plan[size_t(vi)];
          if (p.kind != 1) continue;
          if (p.ex_old && p.amt_old != 0)
            my[size_t(p.g_old) * size_t(wpr) + size_t(p.c_old >> 6)] |= (1ULL << (p.c_old & 63));
          if (p.ex_new && p.amt_new != 0)
            my[size_t(p.g_new) * size_t(wpr) + size_t(p.c_new >> 6)] |= (1ULL << (p.c_new & 63));
        }
      }
      std::vector<u64> bm(size_t(G) * size_t(wpr), 0);
#pragma omp parallel for schedule(static)
      for (long long wi = 0; wi < (long long)(size_t(G) * size_t(wpr)); ++wi) {
        u64 x = 0;
        for (int t = 0; t < NT; ++t) x |= tbm[size_t(t) * size_t(G) * size_t(wpr) + size_t(wi)];
        bm[size_t(wi)] = x;
      }
#pragma omp parallel for schedule(dynamic, 4)
      for (long long gi = 0; gi < (long long)G; ++gi) {
        const i32 g = static_cast<i32>(gi);
        const u64* w = &bm[size_t(g) * size_t(wpr)];
        auto& v = tcoord[size_t(g)];
        for (i32 wi = 0; wi < wpr; ++wi) {
          u64 bits = w[wi];
          while (bits) {
            const i32 b = __builtin_ctzll(bits);
            v.push_back(i64(wi) * 64 + b);
            bits &= bits - 1;
          }
        }
      }
    }
    auto p4 = nw();
    // O(1) (group,class) -> support index map
    ti_map.assign(size_t(G) * size_t(ncls), -1);
#pragma omp parallel for schedule(dynamic, 4)
    for (long long gi = 0; gi < (long long)G; ++gi) {
      const i32 g = static_cast<i32>(gi);
      const auto& tc = tcoord[size_t(g)];
      if (tc.empty()) continue;
      i32* row = &ti_map[size_t(g) * size_t(ncls)];
      for (size_t k = 0; k < tc.size(); ++k) row[size_t(tc[k])] = i32(k);
    }
    {
      i32 ti_missing = 0;
#pragma omp parallel for schedule(static) reduction(| : ti_missing)
      for (long long vi = 1; vi < (long long)kStates; ++vi) {
        const i32 v = static_cast<i32>(vi);
        NodePlan& p = plan[size_t(v)];
        if (p.kind != 1) continue;
        if (p.ex_old && p.amt_old != 0) {
          p.ti_old = ti_map[size_t(p.g_old) * size_t(ncls) + size_t(p.c_old)];
          if (p.ti_old < 0) {
            if (std::getenv("BWH_BATCH_DEBUG")) std::fprintf(stderr, "batch: ti_old missing v=%d g=%d c=%d\n", v, p.g_old, p.c_old);
            ti_missing |= 1;
          }
        }
        if (p.ex_new && p.amt_new != 0) {
          p.ti_new = ti_map[size_t(p.g_new) * size_t(ncls) + size_t(p.c_new)];
          if (p.ti_new < 0) {
            if (std::getenv("BWH_BATCH_DEBUG")) std::fprintf(stderr, "batch: ti_new missing v=%d g=%d c=%d\n", v, p.g_new, p.c_new);
            ti_missing |= 1;
          }
        }
      }
      if (ti_missing) return false;
    }
    auto p5 = nw();
    // ---- factor plan: previous factor value per factor edit node
    {
      std::vector<std::vector<i32>> fg(static_cast<size_t>(G));
      for (i32 v = 1; v < kStates; ++v)
        if (w.edit[size_t(v)].op == OP_FACTOR) fg[size_t(w.edit[size_t(v)].g)].push_back(v);
      struct FEnt { i32 node; i64 val; };
#pragma omp parallel for schedule(dynamic, 4)
      for (long long gi = 0; gi < (long long)G; ++gi) {
        const i32 g = static_cast<i32>(gi);
        std::vector<FEnt> st2;
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
    if (prof) {
      auto p6 = nw();
      std::fprintf(stderr, "V4PLAN pre=%lld sort=%lld rowloop=%lld support=%lld ti=%lld fac=%lld\n",
                   (long long)el(p0), (long long)el(p1), (long long)el(p2), (long long)el(p3),
                   (long long)el(p4), (long long)el(p6));
    }
    return true;
  }

  bool build_base() {
    // Parallel row scan: per-thread initial-factor-weighted V leaves (exact integer
    // reduction) and atomic per-group base-mass accumulation (rows of a group are spread over
    // classes, so contention on a given cache line is low).  Per-group base-class lists are
    // then built per active row in parallel.
    const BaseRec* rows = f.rows.data();
    const int NT = omp_get_max_threads();
    if (corr_on) gbase.assign(size_t(n_active) * size_t(Cpfx), 0);
    std::vector<std::vector<i64>> tV(static_cast<size_t>(NT), std::vector<i64>(size_t(Cpfx), 0));
    {
#pragma omp parallel
      {
        i64* lv = tV[static_cast<size_t>(omp_get_thread_num())].data();
#pragma omp for schedule(static)
        for (long long ii = 0; ii < (long long)N; ++ii) {
          const BaseRec& br = rows[size_t(ii)];
          const i32 c = i32(class_of(br.t));
          lv[size_t(c) + 1] += br.amount * s0[size_t(br.g)];
          const i32 r = g2row[size_t(br.g)];
          if (r >= 0) {
            i64* dst = &gbase[size_t(r) * size_t(Cpfx) + size_t(c) + 1];
#pragma omp atomic
            *dst += br.amount;
          }
        }
      }
    }
    V.assign(size_t(Cpfx), 0);
#pragma omp parallel for schedule(static)
    for (long long ci = 0; ci < (long long)Cpfx; ++ci) {
      i64 x = 0;
      for (int t = 0; t < NT; ++t) x += tV[size_t(t)][size_t(ci)];
      V[size_t(ci)] = x;
    }
    // per-group base-class lists: count non-zero classes per active row, prefix, fill in parallel
    gb_off.assign(static_cast<size_t>(n_active) + 1, 0);
    std::vector<i32> cnt(static_cast<size_t>(n_active), 0);
    if (corr_on) {
#pragma omp parallel for schedule(static)
      for (long long ri = 0; ri < (long long)n_active; ++ri) {
        const i64* row = &gbase[size_t(ri) * size_t(Cpfx)];
        i32 k = 0;
        for (i32 c = 0; c < ncls; ++c) k += (row[size_t(c) + 1] != 0);
        cnt[size_t(ri)] = k;
      }
    }
    i32 tot = 0;
    for (i32 r = 0; r < n_active; ++r) { gb_off[size_t(r)] = tot; tot += cnt[size_t(r)]; }
    gb_off[size_t(n_active)] = tot;
    gb_cls.assign(static_cast<size_t>(tot), 0);
    gb_amt.assign(static_cast<size_t>(tot), 0);
    if (corr_on) {
#pragma omp parallel for schedule(static)
      for (long long ri = 0; ri < (long long)n_active; ++ri) {
        const i64* row = &gbase[size_t(ri) * size_t(Cpfx)];
        i32 pos = gb_off[size_t(ri)];
        for (i32 c = 0; c < ncls; ++c) {
          const i64 a = row[size_t(c) + 1];
          if (a != 0) { gb_cls[size_t(pos)] = c; gb_amt[size_t(pos)] = a; ++pos; }
        }
      }
    }
    rebuild_pref();
    clear_pending();
    return true;
  }

  // ---------------- v4 blocked-snapshot prefix and pending deltas ----------------
  static void prefix_scalar(const i64* x, i64* y, i32 n) {
    i64 c = 0;
    for (i32 i = 0; i < n; ++i) { c += x[i]; y[i] = c; }
  }
#if defined(__x86_64__) || defined(__i386__)
  __attribute__((target("avx2")))
  static void prefix_avx2(const i64* x, i64* y, i32 n) {
    i64 carry = 0;
    i32 i = 0;
    for (; i + 4 <= n; i += 4) {
      __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
      // within each 128-bit lane: [a, a+b] and [c, c+d]
      v = _mm256_add_epi64(v, _mm256_slli_si256(v, 8));
      // add the lower lane's total (a+b) to the upper two lanes
      __m256i up = _mm256_permute4x64_epi64(v, 0x55);
      up = _mm256_and_si256(up, _mm256_setr_epi64x(0, 0, -1, -1));
      v = _mm256_add_epi64(v, up);
      v = _mm256_add_epi64(v, _mm256_set1_epi64x(carry));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(y + i), v);
      carry = _mm256_extract_epi64(v, 3);
    }
    for (; i < n; ++i) { carry += x[i]; y[i] = carry; }
  }
#endif
  static bool have_avx2() {
#if defined(__x86_64__) || defined(__i386__)
    static const bool ok = __builtin_cpu_supports("avx2");
    return ok;
#else
    return false;
#endif
  }
  // rebuild pref from V and drop the pending delta list (snapshot point)
  void rebuild_pref() {
    auto t0 = std::chrono::steady_clock::now();
    pref[0] = 0;
    const i64* src = V.data() + 1;   // V[c+1] is the leaf of class c
    i64* dst = pref.data() + 1;
#if defined(__x86_64__) || defined(__i386__)
    if (have_avx2()) prefix_avx2(src, dst, ncls);
    else prefix_scalar(src, dst, ncls);
#else
    prefix_scalar(src, dst, ncls);
#endif
    clear_pending();
    states_since = 0;
    dirty = false;
    t_rebuild_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
  }
  void clear_pending() {
    for (i32 b : ptouched) pblk[size_t(b)].clear();
    ptouched.clear();
    pcls.clear();
    pw.clear();
    std::fill(pfen.begin(), pfen.end(), 0);
  }
  inline void pending_add(i32 cls, i64 w) {
    if (w == 0) return;
    const i32 b = cls >> 6;
    const i32 idx = i32(pcls.size());
    pcls.push_back(cls);
    pw.push_back(w);
    if (pblk[size_t(b)].empty()) ptouched.push_back(b);
    pblk[size_t(b)].push_back(idx);
    for (i32 i = b + 1; i <= nblocks; i += i & (-i)) pfen[size_t(i)] += w;
  }
  // exact prefix of the current weighted mass over classes < c
  inline i64 pref_query(i32 c) const {
    i64 r = pref[size_t(c)];
    const i32 b = c >> 6;
    if (b >= nblocks) {
      for (i32 i = nblocks; i > 0; i -= i & (-i)) r += pfen[size_t(i)];
      return r;
    }
    for (i32 i = b; i > 0; i -= i & (-i)) r += pfen[size_t(i)];
    const auto& lst = pblk[size_t(b)];
    for (i32 idx : lst)
      if (pcls[size_t(idx)] < c) r += pw[size_t(idx)];
    return r;
  }

  inline void row_apply(const NodePlan& p, i64 sgn) {
    // sgn = +1 applies the edit, -1 reverses it.  Row edits change only the deviation
    // part of V, so the base-mass leaves are untouched.
    if (sgn > 0) {
      if (p.ex_old && p.amt_old != 0) {
        const i64 wv = fac[size_t(p.g_old)] * p.amt_old;
        tdev[size_t(p.g_old)][size_t(p.ti_old)] -= p.amt_old;
        V[size_t(p.c_old) + 1] -= wv;
        pending_add(p.c_old, -wv);
      }
      if (p.ex_new && p.amt_new != 0) {
        const i64 wv = fac[size_t(p.g_new)] * p.amt_new;
        tdev[size_t(p.g_new)][size_t(p.ti_new)] += p.amt_new;
        V[size_t(p.c_new) + 1] += wv;
        pending_add(p.c_new, wv);
      }
    } else {
      if (p.ex_new && p.amt_new != 0) {
        const i64 wv = fac[size_t(p.g_new)] * p.amt_new;
        tdev[size_t(p.g_new)][size_t(p.ti_new)] -= p.amt_new;
        V[size_t(p.c_new) + 1] -= wv;
        pending_add(p.c_new, -wv);
      }
      if (p.ex_old && p.amt_old != 0) {
        const i64 wv = fac[size_t(p.g_old)] * p.amt_old;
        tdev[size_t(p.g_old)][size_t(p.ti_old)] += p.amt_old;
        V[size_t(p.c_old) + 1] += wv;
        pending_add(p.c_old, wv);
      }
    }
  }

  inline void factor_apply(const NodePlan& p, i64 ds) {
    if (ds == 0) return;
    const bool pf = pf_support;  // software prefetch of the support lists (env BWH_V5_PREFETCH)
    const i32 gg = p.g_old;  // factor nodes keep their group in the g_old slot
    const i32 r = g2row[size_t(gg)];
    bool touched_any = false;
    if (r >= 0) {
      const i32 b = gb_off[size_t(r)], e = gb_off[size_t(r) + 1];
      for (i32 k = b; k < e; ++k) {
        if (pf && k + 12 < e) {
          __builtin_prefetch(&gb_cls[size_t(k) + 12], 0, 1);
          __builtin_prefetch(&gb_amt[size_t(k) + 12], 0, 1);
          __builtin_prefetch(&V[size_t(gb_cls[size_t(k) + 6]) + 1], 1, 3);
        }
        const i64 a = gb_amt[size_t(k)];
        if (a == 0) continue;
        V[size_t(gb_cls[size_t(k)]) + 1] += ds * a;
        touched_any = true;
      }
      auto& tc = tcoord[size_t(gg)];
      auto& dv = tdev[size_t(gg)];
      for (size_t k = 0; k < tc.size(); ++k) {
        if (pf && k + 12 < tc.size()) {
          __builtin_prefetch(&tc[k + 12], 0, 1);
          __builtin_prefetch(&dv[k + 12], 0, 1);
          __builtin_prefetch(&V[size_t(tc[k + 6]) + 1], 1, 3);
        }
        const i64 d = dv[k];
        if (d == 0) continue;
        V[size_t(tc[k]) + 1] += ds * d;
        touched_any = true;
      }
    }
    if (touched_any) {
      if (fac_lazy) {
        dirty = true;
      } else {
        rebuild_pref();
      }
    }
  }

  // v3 offline cross-version sharing: merge each touched group's base-mass classes with its
  // deviation classes ONCE, so a factor edit walks the union with a single read+write per
  // class instead of two passes (base classes and deviation classes overlap heavily).
  void build_merged_support() {
    row2g.assign(static_cast<size_t>(n_active), -1);
    for (i32 g = 0; g < G; ++g) {
      const i32 r = g2row[size_t(g)];
      if (r >= 0) row2g[size_t(r)] = g;
    }
    ms_off.assign(static_cast<size_t>(n_active) + 1, 0);
    size_t total = 0;
    for (i32 r = 0; r < n_active; ++r) {
      total += size_t(gb_off[size_t(r) + 1] - gb_off[size_t(r)]) + tcoord[size_t(row2g[size_t(r)])].size();
    }
    ms_cls.clear(); ms_base.clear(); ms_dev.clear();
    ms_cls.reserve(total); ms_base.reserve(total); ms_dev.reserve(total);
    for (i32 r = 0; r < n_active; ++r) {
      const i32 g = row2g[size_t(r)];
      const auto& tc = tcoord[size_t(g)];
      const i32 b = gb_off[size_t(r)], e = gb_off[size_t(r) + 1];
      i32 bi = b, di = 0;
      while (bi < e || di < i32(tc.size())) {
        i64 cls;
        i64 base = 0;
        i32 dev = -1;
        if (di >= i32(tc.size()) || (bi < e && gb_cls[size_t(bi)] < tc[size_t(di)])) {
          cls = gb_cls[size_t(bi)];
          base = gb_amt[size_t(bi)];
          ++bi;
        } else if (bi >= e || tc[size_t(di)] < gb_cls[size_t(bi)]) {
          cls = tc[size_t(di)];
          dev = di;
          ++di;
        } else {  // both
          cls = gb_cls[size_t(bi)];
          base = gb_amt[size_t(bi)];
          ++bi;
          dev = di;
          ++di;
        }
        ms_cls.push_back(cls);
        ms_base.push_back(base);
        ms_dev.push_back(dev);
      }
      ms_off[size_t(r) + 1] = i32(ms_cls.size());
    }
    if (std::getenv("BWH_BATCH_DEBUG")) {
      size_t gb = 0, dev = 0;
      for (i32 r = 0; r < n_active; ++r) {
        gb += size_t(gb_off[size_t(r) + 1] - gb_off[size_t(r)]);
        dev += tcoord[size_t(row2g[size_t(r)])].size();
      }
      std::fprintf(stderr, "v3 support: gb=%zu dev=%zu merged=%zu\n", gb, dev, ms_cls.size());
    }
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
      out[k] = pref_query(j) - pref_query(i);
    }
  }
};

}  // namespace bwh
