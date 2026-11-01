// deri_index.hpp — Dynamic Endpoint Rank Index (DERI), opt-in backend for ad-hoc/mixed
// endpoint workloads.  New file (2026-11-01); the declared fast path (engine.hpp/panel.cpp)
// is untouched and keeps producing the same answers for declared workloads.
//
// Design
//   * endpoint domain = sorted distinct raw timestamps any row can occupy (base timestamps +
//     every declared insert/move destination).  Any query endpoint t is mapped to a rank with a
//     binary search over this domain (O(log C)); no 250 ms rebuild, no per-query resorting, and
//     the domain is extended only when the declared edit set introduces a new destination
//     (exact insertion, O(log C)).
//   * per-group structures: for every group a compressed rank list (base ranks of the group plus
//     the destination ranks of inserts/moves into it) and a Fenwick tree over that list holding
//     the group's *unweighted* current amounts.  Row edits update one rank (O(log n_g)).
//   * the factor vector is kept as a plain per-group array, so a factor edit is O(1): DERI never
//     eagerly materialises a factor change over the group's row list (that eager walk is what
//     makes the RF fallback build ~250 ms).
//   * a query answer at the current replay state is
//         sum_g fac[g] * (mass_g(rank(lo)) .. mass_g(rank(hi)))
//     with two binary searches + Fenwick prefixes per group: O(G log C), exact, for arbitrary
//     [lo,hi) endpoints (declared or ad-hoc).
//   * replay applies row replace/insert/delete/move and factor edits with the same absolute-edit
//     semantics as the declared workload, so the answers are bit-identical to the shipped fast
//     path on declared endpoints.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "common.hpp"

namespace bwh {

struct DeriIndex {
  const Fixture& f;
  i64 tmin = 0, tmax = 0;
  i32 G = 0;
  std::vector<i64> domain;                    // sorted unique raw timestamps
  std::vector<std::vector<i32>> gr;           // per group compressed ranks (i32 rank of domain)
  std::vector<std::vector<i64>> gsum;         // Fenwick over gr[g] (1-based)
  std::vector<std::vector<i64>> gbasepref;    // prefix of base amounts over gr[g] (unused after load)
  std::vector<i64> fac;                        // current factor per group
  std::vector<i64> cur_t, cur_amt;             // current row state
  std::vector<i32> cur_g, cur_rank;
  std::vector<u8> cur_ex;
  std::vector<i64> base_amt;                   // original amount (for replay deltas)
  i64 n = 0;

  explicit DeriIndex(const Fixture& fi) : f(fi) {}

  static void radix_sort_i64(std::vector<i64>& v, std::vector<i32>* idx = nullptr);

  // rank of timestamp t: number of distinct domain values < t
  inline i32 rank_of(i64 t) const {
    return i32(std::lower_bound(domain.begin(), domain.end(), t) - domain.begin());
  }

  void build(const std::vector<Edit>& edits);
  void reset();
  // replay one absolute edit (row replace/insert/delete/move or factor edit)
  void apply(const Edit& e);
  // range mass of group g at ranks [r0, r1)
  inline i64 group_range(i32 g, i32 r0, i32 r1) const {
    const auto& r = gr[size_t(g)];
    const i32 i0 = i32(std::lower_bound(r.begin(), r.end(), r0) - r.begin());
    const i32 i1 = i32(std::lower_bound(r.begin(), r.end(), r1) - r.begin());
    return fen_prefix(g, i1) - fen_prefix(g, i0);
  }
  inline i64 fen_prefix(i32 g, i32 count) const {
    i64 s = 0;
    const auto& t = gsum[size_t(g)];
    for (i32 i = count; i > 0; i -= i & (-i)) s += t[size_t(i)];
    return s;
  }
  inline void fen_add(i32 g, i32 pos1, i64 d) {   // pos1: 1-based compressed index
    auto& t = gsum[size_t(g)];
    const i32 m = i32(t.size()) - 1;
    for (i32 i = pos1; i <= m; i += i & (-i)) t[size_t(i)] += d;
  }
  inline i32 comp_index(i32 g, i32 rank) const {   // 1-based compressed index or 0
    const auto& r = gr[size_t(g)];
    return i32(std::lower_bound(r.begin(), r.end(), rank) - r.begin()) + 1;
  }
  // exact answer for an arbitrary half-open timestamp range at the current replay state
  inline i64 query(i64 lo, i64 hi) const {
    if (hi <= lo) return 0;
    const i32 r0 = rank_of(lo), r1 = rank_of(hi);
    i64 s = 0;
    for (i32 g = 0; g < G; ++g) {
      const i64 d = group_range(g, r0, r1);
      if (d) s += fac[size_t(g)] * d;
    }
    return s;
  }
};

void DeriIndex::radix_sort_i64(std::vector<i64>& v, std::vector<i32>* idx) {
  // 4-pass 16-bit LSD radix over the sign-flipped representation (order preserving)
  const size_t n = v.size();
  if (n < 2) return;
  std::vector<u64> a(n), b(n);
  std::vector<i32> ia, ib;
  const bool want_idx = idx != nullptr;
  for (size_t i = 0; i < n; ++i) a[i] = u64(v[i]) ^ (1ull << 63);
  if (want_idx) { ia.resize(n); for (size_t i = 0; i < n; ++i) ia[i] = i32(i); }
  const i32 passes = 4;
  for (i32 p = 0; p < passes; ++p) {
    const i32 shift = p * 16;
    size_t cnt[65536];
    std::memset(cnt, 0, sizeof(cnt));
    for (size_t i = 0; i < n; ++i) ++cnt[(a[i] >> shift) & 0xFFFF];
    size_t acc = 0;
    for (i32 k = 0; k < 65536; ++k) { const size_t c = cnt[k]; cnt[k] = acc; acc += c; }
    b.resize(n);
    if (want_idx) ib.resize(n);
    for (size_t i = 0; i < n; ++i) {
      const size_t d = cnt[(a[i] >> shift) & 0xFFFF]++;
      b[d] = a[i];
      if (want_idx) ib[d] = ia[i];
    }
    a.swap(b);
    if (want_idx) ia.swap(ib);
  }
  for (size_t i = 0; i < n; ++i) v[i] = i64(a[i] ^ (1ull << 63));
  if (want_idx) idx->swap(ia);
}

void DeriIndex::build(const std::vector<Edit>& edits) {
  n = f.h.n;
  G = f.h.groups;
  tmin = f.h.t_min;
  tmax = f.h.t_max;
  // ---- 1. base rows sorted by timestamp (LSD radix), ranks follow the domain order
  std::vector<i64> ts(static_cast<size_t>(n));
  std::vector<i32> id(static_cast<size_t>(n));
  for (i64 i = 0; i < n; ++i) { ts[size_t(i)] = f.rows[size_t(i)].t; id[size_t(i)] = i32(i); }
  radix_sort_i64(ts, &id);
  // ---- 2. endpoint domain = unique(base timestamps + declared insert/move destinations)
  std::vector<i64> dest;
  for (const Edit& e : edits)
    if (e.op == OP_INSERT || e.op == OP_MOVE) dest.push_back(e.t);
  radix_sort_i64(dest);
  dest.erase(std::unique(dest.begin(), dest.end()), dest.end());
  domain.clear();
  domain.reserve(ts.size() + dest.size());
  {
    size_t a = 0, b = 0;
    while (a < ts.size() || b < dest.size()) {
      i64 x;
      if (b >= dest.size() || (a < ts.size() && ts[a] <= dest[b])) x = ts[a++];
      else x = dest[b++];
      if (domain.empty() || domain.back() != x) domain.push_back(x);
    }
  }
  // ---- 3. base row ranks (two-pointer; both arrays sorted)
  std::vector<i32> brank(static_cast<size_t>(n));
  {
    size_t d = 0;
    for (i64 i = 0; i < n; ++i) {
      while (d < domain.size() && domain[d] < ts[size_t(i)]) ++d;
      brank[size_t(static_cast<size_t>(id[size_t(i)]))] = i32(d);
    }
  }
  // ---- 4. stable counting sort of the rank-ascending rows by group
  std::vector<i32> cstart(static_cast<size_t>(G) + 1, 0);
  for (i64 i = 0; i < n; ++i) ++cstart[size_t(f.rows[size_t(i)].g) + 1];
  for (i32 g = 0; g < G; ++g) cstart[size_t(g) + 1] += cstart[size_t(g)];
  std::vector<i32> gsorted(static_cast<size_t>(n));
  {
    std::vector<i32> cur(cstart.begin(), cstart.end());
    for (i64 i = 0; i < n; ++i) {
      const i32 r = id[size_t(i)];
      gsorted[size_t(cur[size_t(f.rows[size_t(r)].g)]++)] = r;
    }
  }
  // ---- 5. destination ranks per group (insert/move destinations into that group)
  std::vector<std::vector<i32>> gdest(static_cast<size_t>(G));
  for (const Edit& e : edits) {
    if ((e.op == OP_INSERT || e.op == OP_MOVE) && e.g >= 0) gdest[size_t(e.g)].push_back(rank_of(e.t));
  }
  // ---- 6. per-group merged compressed rank list + base leaf, then O(m) Fenwick build
  gr.resize(static_cast<size_t>(G));
  gsum.resize(static_cast<size_t>(G));
  cur_t.assign(static_cast<size_t>(n), 0);
  cur_amt.assign(static_cast<size_t>(n), 0);
  cur_g.assign(static_cast<size_t>(n), 0);
  cur_rank.assign(static_cast<size_t>(n), 0);
  cur_ex.assign(static_cast<size_t>(n), 0);
  base_amt.assign(static_cast<size_t>(n), 0);
  for (i32 g = 0; g < G; ++g) {
    auto& dv = gdest[size_t(g)];
    std::sort(dv.begin(), dv.end());
    dv.erase(std::unique(dv.begin(), dv.end()), dv.end());
    const i32 b0 = cstart[size_t(g)], b1 = cstart[size_t(g) + 1];
    std::vector<i32> merged;
    merged.reserve(size_t(b1 - b0) + dv.size());
    {
      i32 a = b0; size_t b = 0;
      while (a < b1 || b < dv.size()) {
        const i32 ra = (a < b1) ? brank[size_t(gsorted[size_t(a)])] : 0;
        const i32 x = (b >= dv.size() || (a < b1 && ra <= dv[b])) ? ra : dv[b];
        if (merged.empty() || merged.back() != x) merged.push_back(x);
        if (b >= dv.size() || (a < b1 && ra <= dv[b])) ++a; else ++b;
      }
    }
    auto& v = merged;
    gr[size_t(g)] = v;
    auto& fen = gsum[size_t(g)];
    fen.assign(v.size() + 1, 0);
    // fill leaves from the group's base rows (their ranks are the merged entries)
    i32 a = b0;
    for (size_t k = 0; k < v.size() && a < b1; ++k) {
      while (a < b1 && brank[size_t(gsorted[size_t(a)])] == v[k]) {
        const i32 r = gsorted[size_t(a)];
        fen[k + 1] += f.rows[size_t(r)].amount;
        cur_t[size_t(r)] = f.rows[size_t(r)].t;
        cur_amt[size_t(r)] = f.rows[size_t(r)].amount;
        cur_g[size_t(r)] = g;
        cur_rank[size_t(r)] = v[k];
        cur_ex[size_t(r)] = 1;
        base_amt[size_t(r)] = f.rows[size_t(r)].amount;
        ++a;
      }
    }
    // O(m) Fenwick build from the leaf array
    const i32 m = i32(v.size());
    for (i32 i = 1; i <= m; ++i) {
      const i32 j = i + (i & (-i));
      if (j <= m) fen[size_t(j)] += fen[size_t(i)];
    }
  }
  fac = f.base_factor;
  if (fac.size() != size_t(G)) fac.assign(static_cast<size_t>(G), 1000);
  // rows of groups without factor edits still need their base state in cur_* arrays
  for (i64 i = 0; i < n; ++i) {
    if (cur_ex[size_t(i)]) continue;
    const BaseRec& r = f.rows[size_t(i)];
    cur_t[size_t(i)] = r.t; cur_amt[size_t(i)] = r.amount; cur_g[size_t(i)] = r.g;
    cur_rank[size_t(i)] = brank[size_t(i)];
    cur_ex[size_t(i)] = 1;
    base_amt[size_t(i)] = r.amount;
  }
}

void DeriIndex::reset() {
  for (i32 g = 0; g < G; ++g) std::fill(gsum[size_t(g)].begin(), gsum[size_t(g)].end(), 0);
  fac = f.base_factor;
  if (fac.size() != size_t(G)) fac.assign(static_cast<size_t>(G), 1000);
  for (i64 i = 0; i < n; ++i) {
    const BaseRec& r = f.rows[size_t(i)];
    cur_t[size_t(i)] = r.t; cur_amt[size_t(i)] = r.amount; cur_g[size_t(i)] = r.g;
    cur_ex[size_t(i)] = 1;
    cur_rank[size_t(i)] = rank_of(r.t);
    fen_add(r.g, comp_index(r.g, cur_rank[size_t(i)]), r.amount);
  }
}

inline void deri_remove(DeriIndex& di, i64 id) {
  if (!di.cur_ex[size_t(id)] || di.cur_amt[size_t(id)] == 0) return;
  di.fen_add(di.cur_g[size_t(id)], di.comp_index(di.cur_g[size_t(id)], di.cur_rank[size_t(id)]),
             -di.cur_amt[size_t(id)]);
}

void DeriIndex::apply(const Edit& e) {
  switch (e.op) {
    case OP_REPLACE:
    case OP_DELETE:
    case OP_MOVE: {
      deri_remove(*this, e.id);
      if (e.op == OP_REPLACE) {
        cur_amt[size_t(e.id)] = e.val;
        cur_ex[size_t(e.id)] = 1;
      } else if (e.op == OP_DELETE) {
        cur_ex[size_t(e.id)] = 0;
        cur_amt[size_t(e.id)] = 0;
      } else {
        cur_t[size_t(e.id)] = e.t;
        cur_g[size_t(e.id)] = e.g;
        cur_amt[size_t(e.id)] = e.val;
        cur_rank[size_t(e.id)] = rank_of(e.t);
        cur_ex[size_t(e.id)] = 1;
      }
      if (e.op != OP_DELETE && e.val != 0) {
        fen_add(cur_g[size_t(e.id)], comp_index(cur_g[size_t(e.id)], cur_rank[size_t(e.id)]),
                cur_amt[size_t(e.id)]);
      }
      break;
    }
    case OP_INSERT: {
      const size_t id = size_t(e.id);
      if (id >= cur_t.size()) {
        cur_t.resize(id + 1); cur_amt.resize(id + 1); cur_g.resize(id + 1);
        cur_rank.resize(id + 1); cur_ex.resize(id + 1); base_amt.resize(id + 1, 0);
      }
      cur_t[id] = e.t; cur_g[id] = e.g; cur_amt[id] = e.val; cur_ex[id] = 1;
      cur_rank[id] = rank_of(e.t);
      fen_add(e.g, comp_index(e.g, cur_rank[id]), cur_amt[id]);
      break;
    }
    case OP_FACTOR:
      fac[size_t(e.g)] = e.val;
      break;
    default: break;
  }
}

}  // namespace bwh
