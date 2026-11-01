// engine.hpp — authored reimplementation of the BiWorldDB hybrid-density design and two
// genuine full-state controls for the 2026-10-25 formal matched panel.
//
// Arms (all exact over the declared contract):
//   hybrid          dense-calendar direct index or endpoint-projected stable base,
//                   selected by the frozen density predicate G*(hi-lo+1) <= 2(N+U),
//                   plus sparse per-row/per-group dynamic corrections.
//   full_raw        genuine current-mass per-group Fenwick over raw coordinates
//                   (base rows + declared future-write destinations); every group is
//                   queried for every range.
//   full_projected  same, but coordinates projected through the declared endpoint
//                   classes with future-write support (zero support retained).
//
// Hybrid decomposition (exact; handles row moves across time and group):
//   Q_v(I) = BASE0(I) + DYN_v(I) + sum_{g in C_v} (s_v(g)-s0(g)) * (B_g(I)+Delta_g(I))
// BASE0 is initial-factor-weighted immutable base mass, DYNs_v = sum_g s_v(g) Delta_g,
// C_v = groups whose current factor differs from the initial factor, B_g immutable base
// mass, Delta_g current mass deviation from immutable base mass.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "common.hpp"

namespace bwh {

// ---------------------------------------------------------------- grouped Fenwick
struct GroupFenwick {
  std::vector<i64> bit;
  std::vector<i64> offs;
  std::vector<i32> sizes;
  void init(const std::vector<i32>& sz) {
    sizes = sz;
    offs.assign(sz.size() + 1, 0);
    i64 total = 0;
    for (size_t g = 0; g < sz.size(); ++g) { offs[g] = total; total += i64(sz[g]) + 1; }
    offs[sz.size()] = total;
    bit.assign(size_t(total), 0);
  }
  inline i64* leaves(i32 g) { return bit.data() + offs[size_t(g)]; }
  inline void add(i32 g, i32 i1, i64 v) {
    i64* b = bit.data() + offs[size_t(g)];
    const i32 n = sizes[size_t(g)];
    for (i32 i = i1; i <= n; i += i & (-i)) b[i] += v;
  }
  inline i64 prefix(i32 g, i32 i1) const {
    const i64* b = bit.data() + offs[size_t(g)];
    i64 s = 0;
    for (i32 i = i1; i > 0; i -= i & (-i)) s += b[i];
    return s;
  }
  void build_from_leaves() {
    for (size_t g = 0; g < sizes.size(); ++g) {
      i64* b = bit.data() + offs[g];
      const i32 n = sizes[g];
      for (i32 i = 1; i <= n; ++i) {
        const i32 p = i + (i & (-i));
        if (p <= n) b[p] += b[i];
      }
    }
  }
};

// ---------------------------------------------------------------- flat Fenwick
struct FlatFenwick {
  std::vector<i64> tree;
  i32 n = 0;
  void init(i32 nn) { n = nn; tree.assign(size_t(nn) + 1, 0); }
  inline void add(i32 i1, i64 v) { for (i32 i = i1; i <= n; i += i & (-i)) tree[size_t(i)] += v; }
  inline i64 prefix(i32 i1) const {
    i64 s = 0;
    for (i32 i = i1; i > 0; i -= i & (-i)) s += tree[size_t(i)];
    return s;
  }
  void build_from_leaves() {
    for (i32 i = 1; i <= n; ++i) {
      const i32 p = i + (i & (-i));
      if (p <= n) tree[size_t(p)] += tree[size_t(i)];
    }
  }
};

// ---------------------------------------------------------------- radix sort
struct RadixSorter {
  std::vector<u64> k0, k1;
  std::vector<u32> v0, v1;
  std::vector<u32> cnt;
  void sort(std::vector<u64>& key, std::vector<u32>& val) {
    const size_t n = key.size();
    if (n < 2) return;
    k0 = key;
    v0 = val;
    k1.resize(n);
    v1.resize(n);
    u64 all = 0;
    for (u64 k : k0) all |= k;
    int bits = 0;
    while (all) { ++bits; all >>= 1; }
    int passes = (bits + 7) / 8;
    if (passes < 1) passes = 1;
    cnt.assign(256, 0);
    std::vector<u64>* sk = &k0;
    std::vector<u32>* sv = &v0;
    std::vector<u64>* dk = &k1;
    std::vector<u32>* dv = &v1;
    for (int p = 0; p < passes; ++p) {
      const int shift = p * 8;
      std::fill(cnt.begin(), cnt.end(), 0u);
      for (size_t i = 0; i < n; ++i) ++cnt[size_t(((*sk)[i] >> shift) & 255)];
      u32 sum = 0;
      for (int b = 0; b < 256; ++b) { const u32 c = cnt[size_t(b)]; cnt[size_t(b)] = sum; sum += c; }
      for (size_t i = 0; i < n; ++i) {
        const u32 b = u32(((*sk)[i] >> shift) & 255);
        const u32 pos = cnt[b]++;
        (*dk)[pos] = (*sk)[i];
        (*dv)[pos] = (*sv)[i];
      }
      std::swap(sk, dk);
      std::swap(sv, dv);
    }
    key.assign(sk->begin(), sk->end());
    val.assign(sv->begin(), sv->end());
  }
};

// ---------------------------------------------------------------- shared plan helpers
// Coordinate conventions:
//  * dense frame: coordinate = timestamp - t_min in [0, H-1]; H = t_max-t_min+1.
//  * sparse frame: coordinate = class(t) = upper_bound(E,t) in [0, |E|].
// For a declared query [lo,hi), frame bounds are [ql,qh):
//  * dense : ql = lo-t_min, qh = hi-t_min
//  * sparse: ql = class(lo), qh = class(hi)   (classes ql..qh-1)
// Static/dynamic prefix arrays use the convention A[k] = sum over coordinates < k.
struct Plan {
  static bool predicate_dense(i64 N, i64 U, i32 G, i64 span) {
    const __int128 S = __int128(N) + __int128(U);
    return __int128(G) * __int128(span) <= 2 * S;
  }
};

struct SavedRow {
  i32 kind = 0;       // 0 row edit, 1 factor edit
  i64 id = 0;
  i64 t = 0;
  i32 g = 0;
  i64 amount = 0;
  u8 ex = 0;
};
static_assert(sizeof(SavedRow) == 48, "saved size");

struct Arm {
  virtual ~Arm() = default;
  virtual const char* name() const = 0;
  virtual void build() = 0;
  virtual SavedRow snapshot(const Edit& e) const = 0;
  virtual void apply(const Edit& e) = 0;
  virtual void undo(const Edit& e, const SavedRow& s) = 0;
  virtual i64 query(i32 state, i32 k) const = 0;
  virtual void query_batch(i32 state, i64* out, i32 nq) const {
    for (i32 k = 0; k < nq; ++k) out[k] = query(state, k);
  }
};

// ============================================================ full-state control
struct FullStateArm : Arm {
  const Fixture& f;
  const Workload& w;
  const int frame;  // 0 raw, 1 projected classes
  const char* nm;
  i32 G = 0;
  i64 n_rows = 0;
  i64 E1 = 0;  // class count = |E|+1 when frame==1
  std::vector<std::vector<i64>> coords;
  GroupFenwick gf;
  std::vector<i32> dmap;  // frame==1: G*E1 -> coord index (0-based) or -1
  std::vector<i64> cur_amt, cur_t;
  std::vector<u8> cur_ex;
  std::vector<i32> cur_g;
  std::vector<i64> fac;

  FullStateArm(const Fixture& fi, const Workload& wo, int fr, const char* name)
      : f(fi), w(wo), frame(fr), nm(name) {}
  const char* name() const override { return nm; }

  inline i64 class_of(i64 t) const {
    return i64(std::upper_bound(w.endpoints.begin(), w.endpoints.end(), t) - w.endpoints.begin());
  }
  inline i64 frame_of(i64 t) const { return frame == 0 ? t : class_of(t); }
  inline i32 coord_index(i32 g, i64 t) const {
    if (frame == 0) {
      const auto& cv = coords[size_t(g)];
      return i32(std::lower_bound(cv.begin(), cv.end(), t) - cv.begin()) + 1;
    }
    const i64 c = class_of(t);
    return dmap[size_t(g) * size_t(E1) + size_t(c)] + 1;
  }

  void build() override {
    G = f.h.groups;
    n_rows = i64(f.h.n) + i64(w.n_inserted);
    E1 = i64(w.endpoints.size()) + 1;
    coords.assign(size_t(G), {});
    std::vector<std::vector<i64>> extra(static_cast<size_t>(G));
    for (i32 v = 1; v < kStates; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if (e.op == OP_INSERT || e.op == OP_MOVE) extra[size_t(e.g)].push_back(frame_of(e.t));
      if ((e.op == OP_REPLACE || e.op == OP_DELETE || e.op == OP_MOVE) && e.id < i64(f.h.n)) {
        const BaseRec& br = f.rows[size_t(e.id)];
        extra[size_t(br.g)].push_back(frame_of(br.t));
      }
    }
    if (frame == 0) {
      // radix sort base rows by time, then stable counting sort by group
      std::vector<u64> keys(size_t(f.h.n));
      std::vector<u32> vals(size_t(f.h.n));
      for (i64 i = 0; i < f.h.n; ++i) {
        keys[size_t(i)] = u64(f.rows[size_t(i)].t - f.h.t_min);
        vals[size_t(i)] = u32(i);
      }
      RadixSorter rs;
      rs.sort(keys, vals);
      std::vector<i32> bounds(size_t(G) + 1, 0);
      for (i64 i = 0; i < f.h.n; ++i) ++bounds[size_t(f.rows[size_t(vals[size_t(i)])].g) + 1];
      for (i32 g = 0; g < G; ++g) bounds[size_t(g) + 1] += bounds[size_t(g)];
      std::vector<u32> bygroup(size_t(f.h.n));
      std::vector<i32> pos = bounds;
      for (i64 i = 0; i < f.h.n; ++i) {
        const u32 id = vals[size_t(i)];
        bygroup[size_t(pos[size_t(f.rows[size_t(id)].g)]++)] = id;
      }
      for (i32 g = 0; g < G; ++g) {
        auto& cv = coords[size_t(g)];
        for (i32 i = bounds[size_t(g)]; i < bounds[size_t(g) + 1]; ++i) {
          const i64 t = f.rows[size_t(bygroup[size_t(i)])].t;
          if (cv.empty() || cv.back() != t) cv.push_back(t);
        }
        for (i64 x : extra[size_t(g)]) cv.push_back(x);
        std::sort(cv.begin(), cv.end());
        cv.erase(std::unique(cv.begin(), cv.end()), cv.end());
      }
      std::vector<i32> sizes(static_cast<size_t>(G));
      for (i32 g = 0; g < G; ++g) sizes[size_t(g)] = i32(coords[size_t(g)].size());
      gf.init(sizes);
      for (i64 i = 0; i < f.h.n; ++i) {
        const BaseRec& br = f.rows[size_t(i)];
        gf.leaves(br.g)[coord_index(br.g, br.t)] += br.amount;
      }
    } else {
      std::vector<std::vector<i64>> cls(static_cast<size_t>(G));
      for (i64 i = 0; i < f.h.n; ++i)
        cls[size_t(f.rows[size_t(i)].g)].push_back(class_of(f.rows[size_t(i)].t));
      for (i32 g = 0; g < G; ++g) {
        auto& cv = coords[size_t(g)];
        cv = cls[size_t(g)];
        for (i64 x : extra[size_t(g)]) cv.push_back(x);
        std::sort(cv.begin(), cv.end());
        cv.erase(std::unique(cv.begin(), cv.end()), cv.end());
      }
      std::vector<i32> sizes(static_cast<size_t>(G));
      for (i32 g = 0; g < G; ++g) sizes[size_t(g)] = i32(coords[size_t(g)].size());
      gf.init(sizes);
      dmap.assign(size_t(G * E1), -1);
      for (i32 g = 0; g < G; ++g)
        for (size_t k = 0; k < coords[size_t(g)].size(); ++k)
          dmap[size_t(g) * size_t(E1) + size_t(coords[size_t(g)][k])] = i32(k);
      for (i64 i = 0; i < f.h.n; ++i) {
        const BaseRec& br = f.rows[size_t(i)];
        gf.leaves(br.g)[coord_index(br.g, br.t)] += br.amount;
      }
    }
    gf.build_from_leaves();
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
    fac = f.base_factor;
  }

  void set_row(i64 id, i64 t, i32 g, i64 amount, u8 ex) {
    if (cur_ex[size_t(id)]) {
      const i64 a = cur_amt[size_t(id)];
      if (a != 0) gf.add(cur_g[size_t(id)], coord_index(cur_g[size_t(id)], cur_t[size_t(id)]), -a);
    }
    cur_t[size_t(id)] = t;
    cur_g[size_t(id)] = g;
    cur_amt[size_t(id)] = amount;
    cur_ex[size_t(id)] = ex;
    if (ex && amount != 0) gf.add(g, coord_index(g, t), amount);
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
  void apply(const Edit& e) override {
    switch (e.op) {
      case OP_REPLACE: set_row(e.id, cur_t[size_t(e.id)], cur_g[size_t(e.id)], e.val, 1); break;
      case OP_INSERT: set_row(e.id, e.t, e.g, e.val, 1); break;
      case OP_DELETE: set_row(e.id, cur_t[size_t(e.id)], cur_g[size_t(e.id)], cur_amt[size_t(e.id)], 0); break;
      case OP_MOVE: set_row(e.id, e.t, e.g, e.val, 1); break;
      case OP_FACTOR: fac[size_t(e.g)] = e.val; break;
    }
  }
  void undo(const Edit& e, const SavedRow& s) override {
    (void)e;
    if (s.kind == 1) fac[size_t(s.g)] = s.amount;
    else set_row(s.id, s.t, s.g, s.amount, s.ex);
  }

  i64 query(i32 state, i32 k) const override {
    const Query& qq = query_for(w, state, k);
    const i64 ql = frame == 0 ? qq.lo : class_of(qq.lo);
    const i64 qh = frame == 0 ? qq.hi : class_of(qq.hi);
    i64 ans = 0;
    for (i32 g = 0; g < G; ++g) {
      const i64 v = fac[size_t(g)];
      if (v == 0) continue;
      const auto& cv = coords[size_t(g)];
      const i32 lo = i32(std::lower_bound(cv.begin(), cv.end(), ql) - cv.begin());
      const i32 hi = i32(std::lower_bound(cv.begin(), cv.end(), qh) - cv.begin());
      if (hi == lo) continue;
      ans += v * (gf.prefix(g, hi) - gf.prefix(g, lo));
    }
    return ans;
  }
};

// ============================================================ hybrid
struct HybridArm : Arm {
  const Fixture& f;
  const Workload& w;
  i32 G = 0;
  i64 n_rows = 0;
  bool dense = false;
  i32 ncoord = 0;    // number of coordinate values in the frame
  i64 E1 = 0;
  // static base
  std::vector<i64> baseP;  // n_active*(ncoord+1), prefix over coordinates.
  // optional eager correction-prefix path: when the class domain and factor-edit count are small,
  // maintain a per-class correction term and its prefix at factor-update time (O(C) per factor
  // edit), replacing the per-query active-group correction loop with two array reads.
  bool corr_pref_on = false;
  bool derive_base0 = false;  // build base0 from an all-group base prefix (one random write/row)
  std::vector<i64> corr_pref;  // prefix of the factor correction, size C+1
  std::vector<i64> basePA;     // row-major copy of the per-group base prefixes (corr update path)
                           // n_active rows are allocated only for groups that any declared
                           // factor edit can change (BWH_GROUP_PREFIX=active); all groups
                           // otherwise. Queries only ever read changed-factor groups.
  std::vector<i64> base0;  // (ncoord+1), initial-factor weighted prefix
  std::vector<i32> g2row;  // group -> row in baseP, or -1 when not allocated
  i32 n_active = 0;
  bool active_only = false;
  // transposed prefix rows: baseP is stored class-major ([class][active group]) so the
  // per-query correction loop reads a contiguous row of group values for each query endpoint
  // instead of one cache line per changed group
  bool tp_base = false;
  // lazy factor-value aggregation (BWH_AGG2): per-group compact (class, amount) slices plus one
  // raw/prefix array per distinct current factor value. A factor edit moves one group's slice
  // between two value arrays and refreshes those prefix arrays in O(C); a query needs one
  // prefix lookup per distinct value instead of a loop over changed groups.
  bool agg2 = false;
  std::vector<u32> a2_cls;
  std::vector<i64> a2_amt;
  std::vector<i32> a2_off, a2_val;
  std::vector<i64> a2_vval;
  std::vector<std::vector<i64>> a2_raw, a2_pref;
  i32 a2_ensure(i64 v) {
    for (size_t i = 0; i < a2_vval.size(); ++i)
      if (a2_vval[i] == v) return i32(i);
    a2_vval.push_back(v);
    a2_raw.emplace_back(size_t(ncoord) + 2, 0);
    a2_pref.emplace_back(size_t(ncoord) + 2, 0);
    return i32(a2_vval.size() - 1);
  }
  void a2_reprefix(i32 idx) {
    auto& raw = a2_raw[size_t(idx)];
    auto& pref = a2_pref[size_t(idx)];
    pref[0] = 0;
    for (size_t k = 1; k < raw.size(); ++k) pref[k] = pref[k - 1] + raw[k - 1];
  }
  std::vector<i64> fac, s0;
  // dynamic: base rows are read from the mapped fixture; only rows that an edit touches get
  // an overlay record, so construction no longer copies the whole relation
  struct CurRec { i64 t; i64 amount; i32 g; u8 ex; };
  bool overlay = false;
  std::vector<CurRec> cur_ov;
  std::unordered_map<i64, i32> cur_idx;
  std::vector<i64> cur_amt, cur_t;
  std::vector<u8> cur_ex;
  std::vector<i32> cur_g;
  std::vector<std::vector<i64>> tcoord;  // per group sorted touched coords
  std::vector<std::vector<i64>> tdev;    // per group deviation values
  GroupFenwick tgf;                      // per-group Fenwick over touched coords
  std::vector<i64> gunion;
  FlatFenwick gfen;  // global DYNs_v index over union touched coords
  std::vector<i64> qlb, qhb;  // [state*kMaxQ + k] declared query frame bounds
  i32 qn = 0;
  // bucketed class lookup over the declared endpoint domain: class_of(t) becomes a bucket
  // index plus a short local scan instead of a full binary search per row
  std::vector<i32> cl_lut;
  i64 lut_tmin = 0, lut_width = 1;
  i32 lut_buckets = 0;
  i32 lut_shift = 0;  // power-of-two bucket width: class_of uses a shift instead of division

  HybridArm(const Fixture& fi, const Workload& wo) : f(fi), w(wo) {}
  const char* name() const override { return "hybrid"; }

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
  inline i32 union_cnt_le(i64 c) const {  // count of union coords < c
    return i32(std::lower_bound(gunion.begin(), gunion.end(), c) - gunion.begin());
  }
  // precomputed class-index -> union-rank map for the sparse branch (removes two bisections
  // per query); the dense branch keeps the bisection
  std::vector<u32> ucnt;
  inline i32 ucl(i64 c) const {
    if (!dense && c >= 0 && c < i64(ucnt.size())) return i32(ucnt[size_t(c)]);
    return union_cnt_le(c);
  }

  void build() override {
    const bool prof2 = std::getenv("BWH_PROFILE") != nullptr;
    auto pn = []() { return std::chrono::steady_clock::now(); };
    auto pd = [](auto a, auto b) {
      return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    auto t_start = pn();
    G = f.h.groups;
    n_rows = i64(f.h.n) + i64(w.n_inserted);
    s0 = f.base_factor;
    fac = f.base_factor;
    qn = w.q;
    E1 = i64(w.endpoints.size()) + 1;
    i64 U = 0;
    for (i32 v = 1; v < kStates; ++v) {
      const u8 op = w.edit[size_t(v)].op;
      if (op == OP_REPLACE || op == OP_INSERT || op == OP_DELETE || op == OP_MOVE) ++U;
    }
    const i64 span = f.h.t_max - f.h.t_min + 1;
    dense = Plan::predicate_dense(f.h.n, U, G, span);
    ncoord = dense ? i32(span) : i32(E1);
    const i64 C = ncoord + 1;
    // ---- bucketed class lookup (only for the sparse coordinate domain)
    if (!dense && std::getenv("BWH_CLASS_LUT") != nullptr &&
        w.endpoints.size() >= 1024) {
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
    // ---- which groups need an immutable per-group prefix row?
    // Only groups that a declared factor edit can change are ever read by the
    // correction term; allocating rows for the others is pure waste on bases whose
    // group count far exceeds the number of factor-touched groups.
    const char* gp_env = std::getenv("BWH_GROUP_PREFIX");
    active_only = gp_env && std::strcmp(gp_env, "active") == 0;
    {
      const char* b0_env = std::getenv("BWH_BASE0_MODE");
      if (b0_env && std::strcmp(b0_env, "derive") == 0 && !dense &&
          i64(G) * C <= 4000000LL) {
        derive_base0 = true;
        active_only = false;  // allocate one prefix row per group; base0 is derived after the scan
      }
    }
    g2row.assign(size_t(G), -1);
    {
      std::vector<u8> touched(size_t(G), 0);
      if (active_only) {
        for (i32 v = 1; v < kStates; ++v)
          if (w.edit[size_t(v)].op == OP_FACTOR) touched[size_t(w.edit[size_t(v)].g)] = 1;
      } else {
        for (i32 g = 0; g < G; ++g) touched[size_t(g)] = 1;
      }
      n_active = 0;
      for (i32 g = 0; g < G; ++g)
        if (touched[size_t(g)]) g2row[size_t(g)] = n_active++;
    }
    // ---- decide the base representation up front: the aggregated path does not build or
    // touch the dense (n_active x C) table at all
    bool want_agg2 = false;
    if (!dense) {
      const char* a2e0 = std::getenv("BWH_AGG2");
      if (a2e0) {
        want_agg2 = true;
        if (a2e0[0] == 'a') {
          i64 nf0 = 0;
          for (i32 v = 1; v < kStates; ++v)
            if (w.edit[size_t(v)].op == OP_FACTOR) ++nf0;
          // the aggregated path trades a bounded per-factor-edit move for O(1) queries. The
          // paired roster A/B showed it must not be selected when the moved slice covers most
          // of the relation: Intel Q32 factor-heavy (61 groups, 38k rows per slice, 512 factor
          // edits) was 2x slower, while the same regime on the large taxi base (260 groups,
          // 11k rows per slice) is +37%. Require Q>=32 and either sparse factor edits or a
          // large active-group count.
          want_agg2 = (w.q >= 32) && ((nf0 * 16 < i64(kStates)) || (n_active >= 128));
        }
      }
    }
    // ---- optional correction-prefix path (small class domain and few factor edits)
    {
      const char* cpe = std::getenv("BWH_CORR_PREFIX");
      if (cpe && (!dense || cpe[0] == '1') && !want_agg2) {
        i64 nf0 = 0;
        for (i32 v = 1; v < kStates; ++v)
          if (w.edit[size_t(v)].op == OP_FACTOR) ++nf0;
        const bool force = cpe[0] == '1';
        const bool aut = std::strcmp(cpe, "auto") == 0;
        const i64 active = std::min<i64>(G, std::max<i64>(nf0, 1));
        const i64 query_work = i64(kStates) * i64(std::max<i32>(w.q, 1)) * active;
        // enable only when the O(C)-per-factor-edit maintenance is small relative to the
        // per-query active-group loop that it removes
        if (force || (aut && kStates >= 8192 && nf0 > 0 && nf0 * C <= 60000000LL && nf0 * C <= 2 * query_work))
          corr_pref_on = true;
      }
    }
    if (derive_base0 && want_agg2) derive_base0 = false;
    // ---- static base: bucket then prefix (A[k] = sum over coords < k)
    auto t_alloc = pn();
    if (!want_agg2) {
      baseP.assign(size_t(n_active) * size_t(C), 0);
      base0.assign(size_t(C), 0);
      if (corr_pref_on) corr_pref.assign(size_t(C) + 1, 0);
    }
    {
      const char* tpv = std::getenv("BWH_TRANSPOSE_BASEP");
      if (tpv && !dense) {
        if (tpv[0] == '1') {
          tp_base = true;
        } else if (std::strcmp(tpv, "auto") == 0) {
          // automatic rule: use the class-major layout when factor edits are frequent (at
          // least one per sixteen states; many groups differ from their initial factor and the
          // correction loop dominates), or when the endpoint class domain is large enough that a
          // row-major prefix row (C * 8 bytes) no longer streams well (measured on the 100k/300k
          // version runs: class-major is 4%-17% faster at C >= 2^18 with no regression at the
          // small-panel class counts)
          i64 n_factor = 0;
          for (i32 v = 1; v < kStates; ++v)
            if (w.edit[size_t(v)].op == OP_FACTOR) ++n_factor;
          tp_base = (n_factor * 16 >= i64(kStates)) || (C >= 262144);
        }
      }
    }
    if (derive_base0) tp_base = true;
    if (dense) {
      for (i64 i = 0; i < f.h.n; ++i) {
        const BaseRec& br = f.rows[size_t(i)];
        const i64 c = (br.t - f.h.t_min) + 1;
        base0[size_t(c)] += br.amount * s0[size_t(br.g)];
        const i32 r = g2row[size_t(br.g)];
        if (r >= 0) baseP[size_t(r) * size_t(C) + size_t(c)] += br.amount;
      }
    } else {
      agg2 = want_agg2;
      if (agg2) {
        std::vector<u8> touched(size_t(G), 0);
        for (i32 v = 1; v < kStates; ++v)
          if (w.edit[size_t(v)].op == OP_FACTOR) touched[size_t(w.edit[size_t(v)].g)] = 1;
        a2_val.assign(size_t(G), -1);
        for (i32 g = 0; g < G; ++g) a2_val[size_t(g)] = a2_ensure(s0[size_t(g)]);
        std::vector<u32> cnt(size_t(G), 0);
        for (i64 i = 0; i < f.h.n; ++i) {
          const BaseRec& br = f.rows[size_t(i)];
          if (touched[size_t(br.g)]) ++cnt[size_t(br.g)];
        }
        a2_off.assign(size_t(G) + 1, 0);
        for (i32 g = 0; g < G; ++g) a2_off[size_t(g) + 1] = a2_off[size_t(g)] + i32(cnt[size_t(g)]);
        a2_cls.assign(static_cast<size_t>(a2_off[size_t(G)]), 0);
        a2_amt.assign(static_cast<size_t>(a2_off[size_t(G)]), 0);
        std::vector<u32> pos(size_t(G), 0);
        for (i32 g = 0; g < G; ++g) pos[size_t(g)] = u32(a2_off[size_t(g)]);
        for (i64 i = 0; i < f.h.n; ++i) {
          const BaseRec& br = f.rows[size_t(i)];
          const u32 c = u32(class_of(br.t));
          const i32 g = br.g;
          a2_raw[size_t(a2_val[size_t(g)])][size_t(c)] += br.amount;
          if (touched[size_t(g)]) {
            const u32 k = pos[size_t(g)]++;
            a2_cls[size_t(k)] = c;
            a2_amt[size_t(k)] = br.amount;
          }
        }
        for (size_t vi = 0; vi < a2_vval.size(); ++vi) a2_reprefix(i32(vi));
      } else {
        if (derive_base0) {
          for (i64 i = 0; i < f.h.n; ++i) {
            const BaseRec& br = f.rows[size_t(i)];
            const i64 c = class_of(br.t) + 1;
            baseP[size_t(c) * size_t(n_active) + size_t(br.g)] += br.amount;
          }
        } else {
          for (i64 i = 0; i < f.h.n; ++i) {
            const BaseRec& br = f.rows[size_t(i)];
            const i64 c = class_of(br.t) + 1;
            base0[size_t(c)] += br.amount * s0[size_t(br.g)];
            const i32 r = g2row[size_t(br.g)];
            if (r >= 0) {
              if (tp_base) baseP[size_t(c) * size_t(n_active) + size_t(r)] += br.amount;
              else baseP[size_t(r) * size_t(C) + size_t(c)] += br.amount;
            }
          }
        }
      }
    }
    if (prof2) {
      auto c0 = pn();
      volatile i64 acc = 0;
      for (i64 i = 0; i < f.h.n; ++i) acc += class_of(f.rows[size_t(i)].t);
      auto c1 = pn();
      std::fprintf(stderr, "PROFILE2 classonly us=%lld acc=%lld\n",
                   (long long)pd(c0, c1), (long long)acc);
    }
    auto t_scatter = pn();
    if (!want_agg2) {
      if (derive_base0) {
        base0.assign(size_t(C) + 1, 0);
        for (i64 c = 1; c < C; ++c) {
          const i64* row = &baseP[size_t(c) * size_t(n_active)];
          i64 v = 0;
          for (i32 g = 0; g < G; ++g) v += row[size_t(g)] * s0[size_t(g)];
          base0[size_t(c)] = v;
        }
      }
      if (tp_base) {
        for (i64 k = 1; k < C; ++k) {
          i64* row = &baseP[size_t(k) * size_t(n_active)];
          const i64* prev = &baseP[size_t(k - 1) * size_t(n_active)];
          for (i32 r = 0; r < n_active; ++r) row[size_t(r)] += prev[size_t(r)];
        }
      } else {
        for (i32 r = 0; r < n_active; ++r) {
          i64* row = &baseP[size_t(r) * size_t(C)];
          for (i64 k = 1; k < C; ++k) row[k] += row[k - 1];
        }
      }
      for (i64 k = 1; k < C; ++k) base0[size_t(k)] += base0[size_t(k - 1)];
    }
    auto t_prefix = pn();
    if (corr_pref_on && tp_base) {
      // class-major baseP needs a row-major copy for a contiguous correction-prefix update
      basePA.assign(size_t(n_active) * size_t(C), 0);
      for (i32 r = 0; r < n_active; ++r) {
        i64* dst = &basePA[size_t(r) * size_t(C)];
        for (i32 k = 0; k < i32(C); ++k) dst[size_t(k)] = baseP[size_t(k) * size_t(n_active) + size_t(r)];
      }
    }
    // ---- touched coordinates
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
    auto t_idx = pn();
    // ---- row state: with BWH_OVERLAY=1 the mapped base rows are read through get_cur and only
    // edited rows receive an overlay record; otherwise the whole relation is copied as before
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
    // ---- declared query frame bounds for every state (shared scope repeats them)
    // only the declared q frames are ever queried; allocating the full kMaxQ stride wasted
    // V*(kMaxQ-q)*2 i32 entries and, at large V, millions of unnecessary class lookups
    qlb.assign(size_t(kStates) * size_t(qn > 0 ? qn : 1), 0);
    qhb.assign(size_t(kStates) * size_t(qn > 0 ? qn : 1), 0);
    for (i32 v = 1; v < kStates; ++v) {
      for (i32 k = 0; k < qn; ++k) {
        const Query& qq = query_for(w, v, k);
        const size_t qidx = size_t(v) * size_t(qn) + size_t(k);
        qlb[qidx] = dense ? (qq.lo - f.h.t_min) : class_of(qq.lo);
        qhb[qidx] = dense ? (qq.hi - f.h.t_min) : class_of(qq.hi);
      }
    }
    if (prof2) {
      auto t_end = pn();
      std::fprintf(stderr,
                   "PROFILE2 us: alloc=%lld scatter=%lld prefix=%lld idx=%lld rows_q=%lld total=%lld | G=%d E=%zu C=%lld n_active=%d\n",
                   (long long)pd(t_start, t_alloc), (long long)pd(t_alloc, t_scatter),
                   (long long)pd(t_scatter, t_prefix), (long long)pd(t_prefix, t_idx),
                   (long long)pd(t_idx, t_end), (long long)pd(t_start, t_end),
                   G, w.endpoints.size(), (long long)C, n_active);
    }
  }

  // Add a signed current-mass contribution to the dynamic structures.
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
      t = cur_t[size_t(id)]; g = cur_g[size_t(id)]; amount = cur_amt[size_t(id)];
      ex = cur_ex[size_t(id)];
      return;
    }
    auto it = cur_idx.find(id);
    if (it != cur_idx.end()) {
      const CurRec& r = cur_ov[size_t(it->second)];
      t = r.t; g = r.g; amount = r.amount; ex = r.ex;
      return;
    }
    const i64 n = i64(f.h.n);
    if (id < n) {
      const BaseRec& b = f.rows[size_t(id)];
      t = b.t; g = b.g; amount = b.amount; ex = 1;
    } else {
      const size_t k = size_t(id - n);
      t = w.ins_t[k]; g = w.ins_g[k]; amount = 0; ex = 0;
    }
  }
  inline void put_cur(i64 id, i64 t, i32 g, i64 amount, u8 ex) {
    if (!overlay) {
      cur_t[size_t(id)] = t; cur_g[size_t(id)] = g; cur_amt[size_t(id)] = amount;
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
    i64 t, a; i32 g; u8 e;
    get_cur(id, t, g, a, e);
    if (!e) return;
    dyn_add(g, a, frame_of(t), -1);
  }
  inline void add_cur(i64 id) {
    i64 t, a; i32 g; u8 e;
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
    if (agg2) {
      const i32 oldi = a2_val[size_t(g)];
      const i32 newi = a2_ensure(nf);
      if (oldi != newi) {
        auto& ro = a2_raw[size_t(oldi)];
        auto& rn = a2_raw[size_t(newi)];
        const u32 b = u32(a2_off[size_t(g)]), e = u32(a2_off[size_t(g) + 1]);
        for (u32 k = b; k < e; ++k) {
          const u32 c = a2_cls[size_t(k)];
          const i64 a = a2_amt[size_t(k)];
          ro[size_t(c)] -= a;
          rn[size_t(c)] += a;
        }
        a2_reprefix(oldi);
        a2_reprefix(newi);
        a2_val[size_t(g)] = newi;
      }
    }
    const auto& cv = tcoord[size_t(g)];
    const auto& dv = tdev[size_t(g)];
    for (size_t k = 0; k < cv.size(); ++k)
      if (dv[k] != 0) gfen.add(ucl(cv[k]) + 1, ds * dv[k]);
    if (corr_pref_on) {
      const i32 r = g2row[size_t(g)];
      if (r >= 0) {
        // baseP[k] is the base mass of groups g at classes < k, so one pass over the
        // per-group prefix row updates the whole correction prefix for this factor edit.
        const i32 ncl = ncoord;
        const i64* row = tp_base ? &basePA[size_t(r) * size_t(ncoord + 1)]
                                 : &baseP[size_t(r) * size_t(ncoord + 1)];
        for (i32 k = 1; k <= ncl; ++k) corr_pref[size_t(k)] += ds * row[size_t(k)];
      }
    }
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
      i64 ct, ca; i32 cg; u8 ce;
      get_cur(e.id, ct, cg, ca, ce);
      s.t = ct; s.g = cg; s.amount = ca; s.ex = ce;
    }
    return s;
  }
  void apply(const Edit& e) override {
    switch (e.op) {
      case OP_REPLACE: { i64 ct, ca; i32 cg; u8 ce; get_cur(e.id, ct, cg, ca, ce);
                         set_row(e.id, ct, cg, e.val, 1); } break;
      case OP_INSERT: set_row(e.id, e.t, e.g, e.val, 1); break;
      case OP_DELETE: { i64 ct, ca; i32 cg; u8 ce; get_cur(e.id, ct, cg, ca, ce);
                        set_row(e.id, ct, cg, ca, 0); } break;
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
    if (agg2) {
      i64 a = gfen.prefix(ucl(j)) - gfen.prefix(ucl(i));
      for (size_t vi = 0; vi < a2_vval.size(); ++vi) {
        const auto& pref = a2_pref[vi];
        a += a2_vval[vi] * (pref[size_t(j)] - pref[size_t(i)]);
      }
      return a;
    }
    i64 ans = base0[size_t(j)] - base0[size_t(i)];
    ans += gfen.prefix(ucl(j)) - gfen.prefix(ucl(i));
    if (corr_pref_on) return ans + corr_pref[size_t(j)] - corr_pref[size_t(i)];
    for (i32 g = 0; g < G; ++g) {
      const i64 d = fac[size_t(g)] - s0[size_t(g)];
      if (d == 0) continue;
      const i32 r = g2row[size_t(g)];
      if (r < 0) continue;  // cannot happen: only factor-touched groups are changed
      if (tp_base) {
        ans += d * (baseP[size_t(j) * size_t(n_active) + size_t(r)] -
                    baseP[size_t(i) * size_t(n_active) + size_t(r)]);
      } else {
        const i64* row = &baseP[size_t(r) * size_t(ncoord + 1)];
        ans += d * (row[j] - row[i]);
      }
    }
    return ans;
  }
  void query_batch(i32 state, i64* out, i32 nq) const override {
    if (agg2 || corr_pref_on) {
      for (i32 k = 0; k < nq; ++k) out[k] = query(state, k);
      return;
    }
    const size_t base = size_t(state) * size_t(qn);
    for (i32 k = 0; k < nq; ++k) {
      const i32 i = i32(qlb[base + size_t(k)]), j = i32(qhb[base + size_t(k)]);
      out[k] = base0[size_t(j)] - base0[size_t(i)] +
               gfen.prefix(ucl(j)) - gfen.prefix(ucl(i));
    }
    for (i32 g = 0; g < G; ++g) {
      const i64 d = fac[size_t(g)] - s0[size_t(g)];
      if (d == 0) continue;
      const i32 r = g2row[size_t(g)];
      if (r < 0) continue;
      if (tp_base) {
        for (i32 k = 0; k < nq; ++k) {
          const i32 i = i32(qlb[base + size_t(k)]), j = i32(qhb[base + size_t(k)]);
          out[k] += d * (baseP[size_t(j) * size_t(n_active) + size_t(r)] -
                         baseP[size_t(i) * size_t(n_active) + size_t(r)]);
        }
      } else {
        const i64* row = &baseP[size_t(r) * size_t(ncoord + 1)];
        for (i32 k = 0; k < nq; ++k) {
          const i32 i = i32(qlb[base + size_t(k)]), j = i32(qhb[base + size_t(k)]);
          out[k] += d * (row[j] - row[i]);
        }
      }
    }
  }
};

inline Arm* make_arm(const std::string& name, const Fixture& f, const Workload& w) {
  if (name == "hybrid") return new HybridArm(f, w);
  if (name == "full_raw") return new FullStateArm(f, w, 0, "full_raw");
  if (name == "full_projected") return new FullStateArm(f, w, 1, "full_projected");
  return nullptr;
}

}  // namespace bwh
