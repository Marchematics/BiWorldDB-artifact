// sparse_engine.hpp — counting-sort CSR sparse exact base representation for the
// BiWorldDB version-tree range-aggregate contract (2026-10-31, sparse-csr).
//
// New file: the shipped engine.hpp / panel.cpp / common.hpp / arms_strong.hpp are
// untouched.  Two arms live here:
//
//   SparseArm      the shipped hybrid decomposition
//                    Q_v(I) = base0(I) + DYN_v(I) + sum_g (s_v(g)-s0(g)) * B_g(I)
//                  with the immutable per-group base masses B_g stored as a
//                  counting-sort CSR of only the nonzero (endpoint class, base
//                  amount) supports, instead of the dense n_active x (C+1) prefix
//                  table.  B_g(i,j) is evaluated per changed-factor group as a
//                  lower_bound into that group's sorted class slice plus a
//                  per-group cumulative-amount prefix read.
//                  base0, gfen, tgf, the row-state updates and the factor handling
//                  are the shipped ones, verbatim; only the static representation
//                  changes, so every answer is bit-identical.
//
//   SparseBaseArm  BWH_SPARSE_BASE=0|1|auto selector.  It delegates to the real
//                  shipped HybridArm (dense base, = the shipped engine) when the
//                  dense representation is chosen, and to SparseArm when the CSR is
//                  chosen, so one binary/run can A/B the two representations with
//                  identical workload construction and identical dynamic code.
//
// Counting-sort build (no per-row pair temporary, so the construction peak stays
// O(N) bytes for the 32-bit class cache + row order and O(nnz) for the CSR, never
// O(N) x 16-byte pairs as in the earlier prototype):
//   pass 1  one sequential row scan: class_of() once per row (bucketed LUT when
//           enabled), cache the class, accumulate base0, set one bit in a
//           n_active x ceil(C/64) bitmap that counts the distinct classes per group.
//   prefix  popcount the bitmap to get the exact support count per group and the
//           CSR offsets; enumerate the set bits to fill each group's class slice
//           (already ascending, so no per-group sort is required).
//   pass 2  counting sort the row order by class; a single ascending-class sweep
//           aggregates the base amount of every (group, class) support into the CSR
//           and turns it into the per-group cumulative prefix.
//
// A per-group 64-class bucket LUT (BWH_SPARSE_QUERY=lut, default) shortens the
// query-time lower_bound: two bucket reads + a <=6-entry binary search instead of a
// full log2(support) bisection.  BWH_SPARSE_QUERY=bisect keeps the plain bisection.

#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.hpp"
#include "engine.hpp"

namespace bwh {

// ---------------------------------------------------------------------------
// Counting-sort CSR of the immutable per-active-group base supports.
//
// v2 query layer (rank LUT, no per-query lower_bound):
//   * bits[n_active * wpr]   one u64 class bitmap per group (C bits: 1 = class has a support)
//   * rlut[n_active * wpr+1] cumulative popcount at every 64-class word boundary
//   * pre[nnz]               cumulative support amount inside every group slice
//   rank_g(c) = rlut[r][c>>6] + popcount(bits[r][c>>6] & mask(c&63))
//   B_g(i,j)  = pre[rank(j)] - pre[rank(i)]        (two u64 loads + popcount + two i64 loads)
// so a correction term costs one bitmap word load, one rank-LUT read and one amount-prefix
// read per endpoint instead of a dependent bisection over the support array.
// ---------------------------------------------------------------------------
struct CsrBase {
  std::vector<u32> ptr;   // n_active+1 support offsets
  std::vector<u64> bits;  // n_active * wpr class bitmap (1 = support at that class)
  std::vector<u32> rlut;  // n_active * (wpr+1) cumulative support counts per 64-class word
  std::vector<i64> pre;   // nnz cumulative base amounts inside every group slice
  i32 wpr = 0;            // u64 words per group = ceil(C / 64)
  i64 nnz = 0;

  inline i32 rank(i32 r, i64 c) const {
    if (c < 0) return 0;
    const i64 b = c >> 6;
    if (b >= i64(wpr)) return i32(i64(ptr[size_t(r) + 1]) - i64(ptr[size_t(r)]));
    const u32 base = rlut[size_t(r) * size_t(wpr + 1) + size_t(b)];
    const i32 rem = i32(c) & 63;
    if (rem == 0) return i32(base);
    const u64 w = bits[size_t(r) * size_t(wpr) + size_t(b)];
    return i32(base) + __builtin_popcountll(w & ((u64(1) << rem) - 1));
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
           rlut.size() * sizeof(u32) + pre.size() * sizeof(i64);
  }
};

struct SparseStats {
  i32 mode = 0;   // 1 = CSR selected, 0 = dense (shipped HybridArm delegated)
  i32 n_active = 0;
  i64 c_domain = 0;
  i64 nnz = 0;
  i64 csr_bytes = 0;      // persistent CSR: ptr + bitmap + rank LUT + amount prefix
  i64 dense_bytes = 0;    // the dense per-group base prefix table it replaces
  i64 bitmap_bytes = 0;
  i64 lut_bytes = 0;
  i64 transient_bytes = 0;  // construction scratch (v2: none beyond the CSR itself)
  i64 est_nnz = 0;
  i64 build_us = 0;
};

// ---------------------------------------------------------------------------
// auto policy.  The CSR only wins where the class domain is large (a group's dense
// prefix row is long), the declared query count is in the shipped range (each query
// pays one bucket lower_bound per changed-factor group) and the relation is small
// enough that the two 32-bit construction scans stay a bounded transient.  The byte
// test uses the expected per-group class occupancy 1-exp(-rows_per_group / C) and
// requires a 2x byte win over the dense table before paying the lower_bound cost.
// ---------------------------------------------------------------------------
namespace sparse_policy {

inline int env_choice() {
  const char* e = std::getenv("BWH_SPARSE_BASE");
  if (!e || !*e) return -1;
  if (std::strcmp(e, "1") == 0) return 1;
  if (std::strcmp(e, "0") == 0) return 0;
  return -1;  // "auto" (and anything unrecognised)
}
inline const char* env_name() {
  const char* e = std::getenv("BWH_SPARSE_BASE");
  if (!e || !*e) return "auto";
  return e;
}

struct Inputs {
  i64 N = 0;
  i64 C = 0;
  i32 G = 1;
  i32 q = 0;
  i32 n_active = 0;
  bool dense_frame = false;
};

// v2 auto gate (rank-LUT CSR), calibrated on the measured crossover:
//   * large class count:          C >= 16,384
//   * dense table >= 24 MiB:      n_active * C * 8, i.e. the table has left a 32 MiB shared
//                                 LLC, so the dense scatter/prefix/query pay DRAM while the
//                                 bitmap + rank LUT + amount prefix stay cache-resident
//   * genuinely sparse supports:  expected per-group class occupancy
//                                 occ = 1 - exp(-(N/G)/C) <= 0.20.  This is the measured
//                                 discriminator: the dense prefix row is only expensive per
//                                 query when a group hits many classes; e.g. the real intel
//                                 panel at Q32 has occ ~ 0.26 (2.31M rows / 61 groups) and the
//                                 dense arm wins by 7-13%, while the 1M/10M/300k synthetic
//                                 relations and the taxi/retail Q32 panels have occ 0.04-0.15
//                                 and the CSR wins.  Rows-per-group is a property of the
//                                 declared relation, available before any scan.
//   * predicted CSR peak <= half the dense table: nnz*8 amount prefix + bitmap (C/8) +
//     rank LUT (C/16) + base0 (8*C).
// No N cap: the v2 construction scratch is O(C) plus two sequential fixture scans, so large
// relations are allowed; the occupancy test is what keeps the build/query balance right.
inline bool choose_sparse(const Inputs& x, i64* est_nnz_out = nullptr) {
  const i64 kMinClasses = 16384;
  const double kMinDenseBytes = 24.0 * 1024.0 * 1024.0;
  const double kMaxOccupancy = 0.20;
  if (est_nnz_out) *est_nnz_out = 0;
  if (x.dense_frame) return false;
  if (x.n_active <= 0) return false;
  if (x.C < kMinClasses) return false;
  const double dense_bytes = double(x.n_active) * double(x.C) * 8.0;
  if (dense_bytes < kMinDenseBytes) return false;
  const double rpg = double(x.N) / double(x.G > 0 ? x.G : 1);
  const double occ = 1.0 - std::exp(-rpg / double(x.C));
  const double nnz_est = double(x.n_active) * double(x.C) * occ;
  if (est_nnz_out) *est_nnz_out = i64(nnz_est);
  if (occ > kMaxOccupancy) return false;
  const double sparse_bytes = nnz_est * 8.0 + double(x.n_active) * double(x.C) * (3.0 / 16.0) +
                              8.0 * double(x.C);
  return sparse_bytes * 2.0 <= dense_bytes;
}

}  // namespace sparse_policy

// ---------------------------------------------------------------------------
// SparseArm: shipped hybrid algebra, CSR static base.
// ---------------------------------------------------------------------------
struct SparseArm : Arm {
  const Fixture& f;
  const Workload& w;
  i32 G = 0;
  i64 n_rows = 0;
  bool dense = false;  // coordinate frame: raw calendar (true) or projected classes
  i32 ncoord = 0;
  i64 E1 = 0;
  i32 n_active = 0;
  bool active_only = false;
  std::vector<i32> g2row;  // group -> CSR row, or -1 when never factor-touched
  std::vector<i64> s0, fac;
  std::vector<i64> base0;  // static base prefix weighted by the initial factor
  CsrBase csr;
  std::vector<i64> baseP;  // dense-frame fallback only: row-major n_active x C
  SparseStats st;
  // bucketed endpoint-class lookup (same accelerator as the shipped sparse frame)
  std::vector<i32> cl_lut;
  i64 lut_tmin = 0, lut_width = 1;
  i32 lut_buckets = 0;
  i32 lut_shift = 0;
  // row state: overlay of edited rows when BWH_OVERLAY is set (shipped behaviour)
  struct CurRec {
    i64 t;
    i64 amount;
    i32 g;
    u8 ex;
  };
  bool overlay = false;
  std::vector<CurRec> cur_ov;
  std::unordered_map<i64, i32> cur_idx;
  std::vector<i64> cur_amt, cur_t;
  std::vector<u8> cur_ex;
  std::vector<i32> cur_g;
  std::vector<std::vector<i64>> tcoord;  // per group sorted touched coordinates
  std::vector<std::vector<i64>> tdev;    // per group deviation values
  GroupFenwick tgf;
  std::vector<i64> gunion;
  FlatFenwick gfen;
  std::vector<i64> qlb, qhb;  // [state*qn + k] declared query frame bounds
  i32 qn = 0;
  std::vector<u32> ucnt;  // class -> union rank (sparse frame)

  SparseArm(const Fixture& fi, const Workload& wo) : f(fi), w(wo) {}
  const char* name() const override { return "sparse"; }

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
  inline i64 frame_of(i64 t) const { return dense ? (t - f.h.t_min) : class_of(t); }
  inline i32 find_coord(i32 g, i64 c) const {
    const auto& v = tcoord[size_t(g)];
    const auto it = std::lower_bound(v.begin(), v.end(), c);
    return (it == v.end() || *it != c) ? -1 : i32(it - v.begin());
  }
  inline i32 ucl(i64 c) const {
    if (!dense && c >= 0 && c < i64(ucnt.size())) return i32(ucnt[size_t(c)]);
    return i32(std::lower_bound(gunion.begin(), gunion.end(), c) - gunion.begin());
  }

  // -------- static base ----------------------------------------------------
  void build_dense_frame_base(i64 C) {
    baseP.assign(size_t(n_active) * size_t(C), 0);
    for (i64 i = 0; i < f.h.n; ++i) {
      const BaseRec& br = f.rows[size_t(i)];
      const i64 c = (br.t - f.h.t_min) + 1;
      base0[size_t(c)] += br.amount * s0[size_t(br.g)];
      const i32 r = g2row[size_t(br.g)];
      if (r >= 0) baseP[size_t(r) * size_t(C) + size_t(c)] += br.amount;
    }
    for (i32 r = 0; r < n_active; ++r) {
      i64* row = &baseP[size_t(r) * size_t(C)];
      for (i64 k = 1; k < C; ++k) row[k] += row[k - 1];
    }
  }

  // Counting-sort CSR build (v2).  Two sequential row scans, no per-row temporary:
  //   pass 1: class_of once per row, accumulate base0, set the per-group class bit;
  //   popcount the bitmap -> exact support counts / CSR offsets and the 64-class rank LUT
  //   (the support amount prefix is indexed by rank, so no sorted class array is stored);
  //   pass 2: class_of again, rank(r, class) -> amt[ptr[r] + rank] += amount.
  // Construction scratch is O(n_active * C / 8) bitmap + O(n_active * C / 16) LUT, i.e. the
  // persistent CSR itself; the old 8 bytes/row (class cache + row order) is gone.
  void build_csr_base(i64 C) {
    const i64 N = f.h.n;
    st.c_domain = C;
    st.dense_bytes = i64(n_active) * C * 8;
    const i32 wpr = i32((C + 63) >> 6);  // u64 words per group
    csr.wpr = wpr;
    csr.ptr.assign(size_t(n_active) + 1, 0);
    std::vector<u64> bits(n_active > 0 ? size_t(n_active) * size_t(wpr) : 0, 0);
    const i32* g2r = g2row.data();
    // ---- pass 1: base0 + per-group class bitmap (sequential fixture scan)
    for (i64 i = 0; i < N; ++i) {
      const BaseRec& br = f.rows[size_t(i)];
      const i64 c = class_of(br.t);
      base0[size_t(c) + 1] += br.amount * s0[size_t(br.g)];
      const i32 r = g2r[br.g];
      if (r >= 0) bits[size_t(r) * size_t(wpr) + size_t(c >> 6)] |= u64(1) << (c & 63);
    }
    for (i64 k = 1; k < C; ++k) base0[size_t(k)] += base0[size_t(k - 1)];
    if (n_active == 0) {
      st.transient_bytes = 0;
      return;
    }
    // ---- support offsets and the 64-class cumulative rank LUT
    csr.rlut.assign(size_t(n_active) * size_t(wpr + 1), 0);
    for (i32 r = 0; r < n_active; ++r) {
      const u64* w = &bits[size_t(r) * size_t(wpr)];
      u32* L = &csr.rlut[size_t(r) * size_t(wpr + 1)];
      u32 local = 0;  // LOCAL cumulative support count inside this group slice
      L[0] = 0;
      for (i32 b = 0; b < wpr; ++b) {
        local += u32(__builtin_popcountll(w[b]));
        L[size_t(b) + 1] = local;
      }
      csr.ptr[size_t(r) + 1] = csr.ptr[size_t(r)] + local;
    }
    csr.nnz = i64(csr.ptr[size_t(n_active)]);
    csr.bits = std::move(bits);
    std::vector<i64> amt(size_t(csr.nnz), 0);
    // ---- pass 2: rank-indexed support amount accumulation (sequential fixture scan)
    const u32* ptr = csr.ptr.data();
    const u64* cb = csr.bits.data();
    const u32* crl = csr.rlut.data();
    for (i64 i = 0; i < N; ++i) {
      const BaseRec& br = f.rows[size_t(i)];
      const i32 r = g2r[br.g];
      if (r < 0) continue;
      const i64 c = class_of(br.t);
      i32 b = i32(c >> 6);
      i32 off;
      if (b >= wpr) {
        off = i32(ptr[size_t(r) + 1] - ptr[size_t(r)]);
      } else {
        const u32 base = crl[size_t(r) * size_t(wpr + 1) + size_t(b)];
        const i32 rem = i32(c) & 63;
        off = i32(base) + (rem == 0
                               ? 0
                               : __builtin_popcountll(cb[size_t(r) * size_t(wpr) + size_t(b)] &
                                                      ((u64(1) << rem) - 1)));
      }
      const u32 slice = ptr[size_t(r) + 1] - ptr[size_t(r)];
      if (u32(off) >= slice) continue;  // cannot happen (every set bit has a support)
      amt[size_t(ptr[size_t(r)] + u32(off))] += br.amount;
    }
    // ---- per-group cumulative amount prefix
    csr.pre.assign(size_t(csr.nnz), 0);
    for (i32 r = 0; r < n_active; ++r) {
      i64 sum = 0;
      for (i64 k = i64(csr.ptr[size_t(r)]); k < i64(csr.ptr[size_t(r) + 1]); ++k) {
        sum += amt[size_t(k)];
        csr.pre[size_t(k)] = sum;
      }
    }
    st.nnz = csr.nnz;
    st.bitmap_bytes = i64(csr.bits.size()) * 8;
    st.lut_bytes = i64(csr.rlut.size()) * 4;
    st.csr_bytes = i64(csr.bytes());
    st.transient_bytes = 0;
  }

  void build() override {
    const auto pn = []() { return std::chrono::steady_clock::now(); };
    const auto pd = [](auto a, auto b) {
      return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    const auto t_start = pn();
    const bool prof = std::getenv("BWH_PROFILE") != nullptr;
    G = f.h.groups;
    n_rows = i64(f.h.n) + i64(w.n_inserted);
    s0 = f.base_factor;
    fac = f.base_factor;
    qn = w.q > 0 ? w.q : 1;
    E1 = i64(w.endpoints.size()) + 1;
    i64 U = 0;
    for (i32 v = 1; v < kStates; ++v) {
      const u8 op = w.edit[size_t(v)].op;
      if (op == OP_REPLACE || op == OP_INSERT || op == OP_DELETE || op == OP_MOVE) ++U;
    }
    const i64 span = f.h.t_max - f.h.t_min + 1;
    dense = Plan::predicate_dense(f.h.n, U, G, span);
    ncoord = dense ? i32(span) : i32(E1);
    const i64 C = i64(ncoord) + 1;
    // ---- bucketed endpoint-class lookup (shipped accelerator, env-gated)
    if (!dense && std::getenv("BWH_CLASS_LUT") != nullptr && w.endpoints.size() >= 1024) {
      i64 want = std::max<i64>(4096, std::min<i64>(i64(1) << 20, 4 * i64(w.endpoints.size()) + 1));
      lut_buckets = 1;
      while (i64(lut_buckets) < want) lut_buckets <<= 1;
      lut_tmin = f.h.t_min;
      const i64 sspan = f.h.t_max - f.h.t_min + 1;
      {
        const i64 target = i64(lut_buckets);
        i64 width = std::max<i64>(1, (sspan + target - 1) / target);
        lut_shift = 0;
        while ((i64(1) << lut_shift) < width) ++lut_shift;
        lut_width = i64(1) << lut_shift;
        lut_buckets = i32((sspan + lut_width - 1) / lut_width);
        if (lut_buckets < 1) lut_buckets = 1;
        if (lut_buckets > (1 << 20)) lut_buckets = 1 << 20;
      }
      cl_lut.assign(size_t(lut_buckets) + 1, 0);
      size_t ep = 0;
      for (i32 j = 0; j <= lut_buckets; ++j) {
        const i64 val = lut_tmin + i64(j) * lut_width;
        while (ep < w.endpoints.size() && w.endpoints[size_t(ep)] <= val) ++ep;
        cl_lut[size_t(j)] = i32(ep);
      }
    }
    // ---- immutable per-group prefix rows are only needed for factor-touched groups
    const char* gp_env = std::getenv("BWH_GROUP_PREFIX");
    active_only = gp_env && std::strcmp(gp_env, "active") == 0;
    {
      std::vector<u8> touched(size_t(G), 0);
      if (active_only) {
        for (i32 v = 1; v < kStates; ++v)
          if (w.edit[size_t(v)].op == OP_FACTOR) touched[size_t(w.edit[size_t(v)].g)] = 1;
      } else {
        for (i32 g = 0; g < G; ++g) touched[size_t(g)] = 1;
      }
      n_active = 0;
      g2row.assign(size_t(G), -1);
      for (i32 g = 0; g < G; ++g)
        if (touched[size_t(g)]) g2row[size_t(g)] = n_active++;
    }
    st.n_active = n_active;
    const auto t_alloc = pn();
    base0.assign(size_t(C), 0);
    if (dense) {
      build_dense_frame_base(C);
      st.dense_bytes = i64(n_active) * C * 8;
    } else {
      build_csr_base(C);
    }
    st.build_us = pd(t_start, pn());
    // ---- touched coordinates and the dynamic per-group / global Fenwicks
    tcoord.assign(size_t(G), {});
    for (i32 v = 1; v < kStates; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if (e.op == OP_INSERT || e.op == OP_MOVE) tcoord[size_t(e.g)].push_back(frame_of(e.t));
      if ((e.op == OP_REPLACE || e.op == OP_DELETE || e.op == OP_MOVE) && e.id < i64(f.h.n)) {
        const BaseRec& br = f.rows[size_t(e.id)];
        tcoord[size_t(br.g)].push_back(frame_of(br.t));
      }
    }
    std::vector<i32> sizes(size_t(G), 0);
    gunion.clear();
    for (i32 g = 0; g < G; ++g) {
      auto& v = tcoord[size_t(g)];
      std::sort(v.begin(), v.end());
      v.erase(std::unique(v.begin(), v.end()), v.end());
      sizes[size_t(g)] = i32(v.size());
      for (i64 c : v) gunion.push_back(c);
    }
    std::sort(gunion.begin(), gunion.end());
    gunion.erase(std::unique(gunion.begin(), gunion.end()), gunion.end());
    if (!dense) {
      ucnt.assign(size_t(C), 0);
      size_t p = 0;
      for (i64 k = 0; k < C; ++k) {
        while (p < gunion.size() && gunion[p] < k) ++p;
        ucnt[size_t(k)] = u32(p);
      }
    }
    tdev.assign(size_t(G), {});
    for (i32 g = 0; g < G; ++g) tdev[size_t(g)].assign(tcoord[size_t(g)].size(), 0);
    tgf.init(sizes);
    gfen.init(i32(gunion.size()));
    // ---- row state (overlay: only edited rows get a record)
    overlay = std::getenv("BWH_OVERLAY") != nullptr;
    if (!overlay) {
      cur_amt.assign(size_t(n_rows), 0);
      cur_t.assign(size_t(n_rows), 0);
      cur_ex.assign(size_t(n_rows), 0);
      cur_g.assign(size_t(n_rows), 0);
      for (i64 i = 0; i < f.h.n; ++i) {
        cur_amt[size_t(i)] = f.rows[size_t(i)].amount;
        cur_ex[size_t(i)] = 1;
        cur_t[size_t(i)] = f.rows[size_t(i)].t;
        cur_g[size_t(i)] = f.rows[size_t(i)].g;
      }
      for (i32 k = 0; k < w.n_inserted; ++k) {
        const i64 id = i64(f.h.n) + k;
        cur_amt[size_t(id)] = 0;
        cur_ex[size_t(id)] = 0;
        cur_t[size_t(id)] = w.ins_t[size_t(k)];
        cur_g[size_t(id)] = w.ins_g[size_t(k)];
      }
    }
    // ---- declared query frame bounds (shared scope repeats them)
    qlb.assign(size_t(kStates) * size_t(qn), 0);
    qhb.assign(size_t(kStates) * size_t(qn), 0);
    for (i32 v = 1; v < kStates; ++v) {
      for (i32 k = 0; k < qn; ++k) {
        const Query& qq = query_for(w, v, k);
        const size_t qidx = size_t(v) * size_t(qn) + size_t(k);
        qlb[qidx] = dense ? (qq.lo - f.h.t_min) : class_of(qq.lo);
        qhb[qidx] = dense ? (qq.hi - f.h.t_min) : class_of(qq.hi);
      }
    }
    if (prof) {
      std::fprintf(stderr,
                   "SPARSE2 us: total=%lld build_us=%lld | frame=%s G=%d E=%zu ncoord=%d "
                   "n_active=%d nnz=%lld csr_bytes=%lld dense_bytes=%lld transient=%lld q=%d\n",
                   (long long)pd(t_start, pn()), (long long)st.build_us,
                   dense ? "dense-calendar" : "projected", G, w.endpoints.size(), ncoord,
                   n_active, (long long)st.nnz, (long long)st.csr_bytes,
                   (long long)st.dense_bytes, (long long)st.transient_bytes, qn);
    }
  }

  // -------- dynamic updates (verbatim shipped algebra) ---------------------
  inline void dyn_add(i32 g, i64 amount, i64 coord, i64 sign) {
    if (amount == 0) return;
    const i32 ti = find_coord(g, coord);
    if (ti < 0) return;
    const i64 d = amount * sign;
    tdev[size_t(g)][size_t(ti)] += d;
    tgf.add(g, ti + 1, d);
    gfen.add(ucl(coord) + 1, fac[size_t(g)] * d);
  }
  inline void get_cur(i64 id, i64& t, i32& g, i64& amount, u8& ex) const {
    if (!overlay) {
      t = cur_t[size_t(id)];
      g = cur_g[size_t(id)];
      amount = cur_amt[size_t(id)];
      ex = cur_ex[size_t(id)];
      return;
    }
    auto it = cur_idx.find(id);
    if (it != cur_idx.end()) {
      const CurRec& r = cur_ov[size_t(it->second)];
      t = r.t;
      g = r.g;
      amount = r.amount;
      ex = r.ex;
      return;
    }
    const i64 n = i64(f.h.n);
    if (id < n) {
      const BaseRec& b = f.rows[size_t(id)];
      t = b.t;
      g = b.g;
      amount = b.amount;
      ex = 1;
    } else {
      const size_t k = size_t(id - n);
      t = w.ins_t[k];
      g = w.ins_g[k];
      amount = 0;
      ex = 0;
    }
  }
  inline void put_cur(i64 id, i64 t, i32 g, i64 amount, u8 ex) {
    if (!overlay) {
      cur_t[size_t(id)] = t;
      cur_g[size_t(id)] = g;
      cur_amt[size_t(id)] = amount;
      cur_ex[size_t(id)] = ex;
      return;
    }
    auto it = cur_idx.find(id);
    if (it != cur_idx.end()) {
      cur_ov[size_t(it->second)] = CurRec{t, amount, g, ex};
    } else {
      cur_idx.emplace(id, i32(cur_ov.size()));
      cur_ov.push_back(CurRec{t, amount, g, ex});
    }
  }
  inline void remove_cur(i64 id) {
    i64 t, a;
    i32 g;
    u8 e;
    get_cur(id, t, g, a, e);
    if (!e) return;
    dyn_add(g, a, frame_of(t), -1);
  }
  inline void add_cur(i64 id) {
    i64 t, a;
    i32 g;
    u8 e;
    get_cur(id, t, g, a, e);
    if (!e) return;
    dyn_add(g, a, frame_of(t), +1);
  }
  void set_row(i64 id, i64 t, i32 g, i64 amount, u8 ex) {
    remove_cur(id);
    put_cur(id, t, g, amount, ex);
    add_cur(id);
  }
  void set_factor(i32 g, i64 nf) {
    if (fac[size_t(g)] == nf) return;
    const i64 ds = nf - fac[size_t(g)];
    const auto& cv = tcoord[size_t(g)];
    const auto& dv = tdev[size_t(g)];
    for (size_t k = 0; k < cv.size(); ++k)
      if (dv[k] != 0) gfen.add(ucl(cv[k]) + 1, ds * dv[k]);
    fac[size_t(g)] = nf;
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
      i64 ct, ca;
      i32 cg;
      u8 ce;
      get_cur(e.id, ct, cg, ca, ce);
      s.t = ct;
      s.g = cg;
      s.amount = ca;
      s.ex = ce;
    }
    return s;
  }
  void apply(const Edit& e) override {
    switch (e.op) {
      case OP_REPLACE: {
        i64 ct, ca;
        i32 cg;
        u8 ce;
        get_cur(e.id, ct, cg, ca, ce);
        set_row(e.id, ct, cg, e.val, 1);
      } break;
      case OP_INSERT: set_row(e.id, e.t, e.g, e.val, 1); break;
      case OP_DELETE: {
        i64 ct, ca;
        i32 cg;
        u8 ce;
        get_cur(e.id, ct, cg, ca, ce);
        set_row(e.id, ct, cg, ca, 0);
      } break;
      case OP_MOVE: set_row(e.id, e.t, e.g, e.val, 1); break;
      case OP_FACTOR: set_factor(e.g, e.val); break;
    }
  }
  void undo(const Edit& e, const SavedRow& s) override {
    (void)e;
    if (s.kind == 1) set_factor(s.g, s.amount);
    else set_row(s.id, s.t, s.g, s.amount, s.ex);
  }

  i64 query(i32 state, i32 k) const override {
    const size_t qidx = size_t(state) * size_t(qn) + size_t(k);
    const i64 i = qlb[qidx], j = qhb[qidx];
    i64 ans = base0[size_t(j)] - base0[size_t(i)];
    ans += gfen.prefix(ucl(j)) - gfen.prefix(ucl(i));
    if (dense) {
      for (i32 g = 0; g < G; ++g) {
        const i64 d = fac[size_t(g)] - s0[size_t(g)];
        if (d == 0) continue;
        const i32 r = g2row[size_t(g)];
        if (r < 0) continue;
        const i64* row = baseP.data() + size_t(r) * (size_t(ncoord) + 1);
        ans += d * (row[j] - row[i]);
      }
    } else {
      for (i32 g = 0; g < G; ++g) {
        const i64 d = fac[size_t(g)] - s0[size_t(g)];
        if (d == 0) continue;
        const i32 r = g2row[size_t(g)];
        if (r < 0) continue;
        ans += d * csr.range(r, i, j);
      }
    }
    return ans;
  }
  void query_batch(i32 state, i64* out, i32 nq) const override {
    const size_t base = size_t(state) * size_t(qn);
    for (i32 k = 0; k < nq; ++k) {
      const i64 i = qlb[base + size_t(k)], j = qhb[base + size_t(k)];
      out[k] = base0[size_t(j)] - base0[size_t(i)] +
               gfen.prefix(ucl(j)) - gfen.prefix(ucl(i));
    }
    if (dense) {
      for (i32 g = 0; g < G; ++g) {
        const i64 d = fac[size_t(g)] - s0[size_t(g)];
        if (d == 0) continue;
        const i32 r = g2row[size_t(g)];
        if (r < 0) continue;
        const i64* row = baseP.data() + size_t(r) * (size_t(ncoord) + 1);
        for (i32 k = 0; k < nq; ++k) {
          const i64 i = qlb[base + size_t(k)], j = qhb[base + size_t(k)];
          out[k] += d * (row[j] - row[i]);
        }
      }
    } else {
      // rank-LUT correction: per changed group and query, 2 rank LUT+bitmap reads and
      // 2 amount-prefix reads; no dependent lower_bound
      for (i32 g = 0; g < G; ++g) {
        const i64 d = fac[size_t(g)] - s0[size_t(g)];
        if (d == 0) continue;
        const i32 r = g2row[size_t(g)];
        if (r < 0) continue;
        for (i32 k = 0; k < nq; ++k) {
          const i64 i = qlb[base + size_t(k)], j = qhb[base + size_t(k)];
          out[k] += d * csr.range(r, i, j);
        }
      }
    }
  }
};

// ---------------------------------------------------------------------------
// BWH_SPARSE_BASE=0|1|auto selector.  Dense mode is the real shipped HybridArm,
// so a "dense" run is byte-for-byte the shipped engine; sparse mode is SparseArm.
// ---------------------------------------------------------------------------
struct SparseBaseArm : Arm {
  const Fixture& f;
  const Workload& w;
  Arm* active_arm = nullptr;
  i32 mode = 0;
  int env_choice = -1;
  SparseStats st;
  SparseBaseArm(const Fixture& fi, const Workload& wo) : f(fi), w(wo) {}
  ~SparseBaseArm() override { delete active_arm; }
  const char* name() const override { return mode ? "sparse" : "hybrid"; }
  i32 base_mode() const { return mode; }
  void build() override {
    const i32 G = f.h.groups;
    i64 U = 0;
    for (i32 v = 1; v < kStates; ++v) {
      const u8 op = w.edit[size_t(v)].op;
      if (op == OP_REPLACE || op == OP_INSERT || op == OP_DELETE || op == OP_MOVE) ++U;
    }
    std::vector<u8> touched(size_t(G), 0);
    for (i32 v = 1; v < kStates; ++v)
      if (w.edit[size_t(v)].op == OP_FACTOR) touched[size_t(w.edit[size_t(v)].g)] = 1;
    i32 n_active = 0;
    for (i32 g = 0; g < G; ++g) n_active += touched[size_t(g)] ? 1 : 0;
    const i64 span = f.h.t_max - f.h.t_min + 1;
    const bool dense_frame = Plan::predicate_dense(f.h.n, U, G, span);
    const i64 C = dense_frame ? span : i64(w.endpoints.size()) + 2;  // = ncoord + 1
    sparse_policy::Inputs in;
    in.N = f.h.n;
    in.C = C;
    in.G = G;
    in.q = w.q;
    in.n_active = n_active;
    in.dense_frame = dense_frame;
    env_choice = sparse_policy::env_choice();
    if (env_choice == 1) {
      mode = 1;
    } else if (env_choice == 0) {
      mode = 0;
    } else {
      i64 est = 0;
      mode = sparse_policy::choose_sparse(in, &est) ? 1 : 0;
      st.est_nnz = est;
    }
    st.mode = mode;
    st.n_active = n_active;
    st.c_domain = C;
    if (mode) {
      SparseArm* a = new SparseArm(f, w);
      active_arm = a;
      a->build();
      st = a->st;
      st.mode = 1;
    } else {
      active_arm = new HybridArm(f, w);
      active_arm->build();
      st.dense_bytes = i64(n_active) * C * 8;
    }
  }
  SavedRow snapshot(const Edit& e) const override { return active_arm->snapshot(e); }
  void apply(const Edit& e) override { active_arm->apply(e); }
  void undo(const Edit& e, const SavedRow& s) override { active_arm->undo(e, s); }
  i64 query(i32 state, i32 k) const override { return active_arm->query(state, k); }
  void query_batch(i32 state, i64* out, i32 nq) const override {
    active_arm->query_batch(state, out, nq);
  }
};

inline Arm* make_sparse_arm(const std::string& name, const Fixture& f, const Workload& w) {
  if (name == "sparse") return new SparseBaseArm(f, w);
  if (name == "hybrid") return new HybridArm(f, w);  // pristine shipped dense baseline
  if (name == "full_raw") return new FullStateArm(f, w, 0, "full_raw");
  if (name == "full_projected") return new FullStateArm(f, w, 1, "full_projected");
  return nullptr;
}

}  // namespace bwh
