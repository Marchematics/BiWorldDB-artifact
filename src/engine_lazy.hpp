// engine_lazy.hpp — Factor-Lazy Correction variant of hybrid/engine.hpp.
//
// This file is a COPY of the shipped engine (hybrid/engine.hpp, which is not edited):
// the arm set, the `Arm` interface, the coordinate conventions and the eager execution
// path are preserved bit-for-bit in behaviour so the same panel driver can select it.
//
// What is new here is the factor-correction strategy, selected at process start by
//   BWH_FACTOR_MODE = eager | lazy | auto      (default eager; hybrid is also accepted)
// plus explicit arm aliases in make_arm() for A/B measurement:
//   lazyhybrid / lazyhybrid_auto / lazyhybrid_lazy / lazyhybrid_hybrid / lazyhybrid_eager
//
// The shipped decomposition (exact) is
//   Q_v(I) = BASE0(I) + sum_g fac_v(g)*Delta_g(I)
//          = BASE0(I) + gfen(I) + sum_{g: fac_v(g)!=s0(g)} (fac_v(g)-s0(g)) * B_g(I)
// with B_g the immutable base mass and Delta_g the current mass deviation of group g.
// The shipped eager path keeps gfen = sum_g fac_v(g)*Delta_g exact incrementally, so a
// group-factor update costs O(nonzero deviation entries of that group) and is paid again
// on every DFS undo.
//
// Lazy Factor Correction keeps gfen0 = sum_g s0(g)*Delta_g instead (updated only by row
// edits) and moves the factor-weighting difference into the query:
//   Q_v(I) = BASE0(I) + gfen0(I)
//          + sum_{g active} (fac_v(g)-s0(g)) * (B_g(I) + Delta_g(I))
// where Delta_g(I) is read from the existing per-group deviation Fenwick tgf. Therefore
//   set_factor: O(1)   (record the scalar; previous value comes from the undo snapshot)
//   undo      : O(1)
//   query     : O(log M) + |active| * (2 bisections + 2 per-group Fenwick prefixes)
// Exactness is unchanged: row edits still go through the same deviation maps/Fenwicks,
// factors are absolute, and undo restores the scalar through the same snapshot interface.
//
// AUTO/HYBRID use an additive per-group cost model over the declared workload (see
// LazyHybridArm::plan_factor_mode): heavy groups (large eager update work relative to
// their lazy query work) stay eager in a second weighted-difference index hfen, light
// groups are evaluated lazily, and auto picks eager / lazy / hybrid by predicted cost.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
struct LazyHybridArm : Arm {
  const Fixture& f;
  const Workload& w;
  i32 G = 0;
  i64 n_rows = 0;
  bool dense = false;
  i32 ncoord = 0;    // number of coordinate values in the frame
  i64 E1 = 0;
  // static base
  std::vector<i64> baseP;  // n_active*(ncoord+1), prefix over coordinates.
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
  // ---------------------------------------------------------------- factor mode
  enum FMode { FM_EAGER = 0, FM_LAZY = 1, FM_HYBRID = 2, FM_AUTO = 3 };
  FMode fmode = FM_EAGER;   // resolved mode (auto resolves at the end of build())
  FMode freq = FM_EAGER;    // requested mode
  bool lazy_path = false;   // fmode != FM_EAGER
  std::vector<u8> hflag;    // FM_HYBRID: 1 = group's weighted-difference index is eager
  FlatFenwick hfen;         // FM_HYBRID: sum_{g:hflag} (fac_g - s0_g) * Delta_g
  std::vector<i32> act;     // groups with fac != s0, unordered
  std::vector<i32> act_pos; // position in act, or -1
  struct PlanTrace {
    bool ran = false;
    const char* requested = "eager";
    const char* chosen = "eager";
    double eager_units = 0, lazy_units = 0, hybrid_units = 0, chosen_units = 0;
    i64 eager_dv_iters = 0, lazy_group_queries = 0;
    i64 n_factor_edits = 0, n_row_edits = 0;
    // raw workload features used to fit/verify the calibrated cost model
    double log_m = 0.0;
    i64 escan_units = 0, edv_units = 0, rc_units = 0, lazy_raw = 0;
    double queries = 0.0;
    // per-group raw features, 5 i64 per group: A, Escan, Edv, Rc, |tcoord|
    std::vector<i64> gfeat;
    i32 groups = 0, factor_groups = 0, heavy_groups = 0, active_groups = 0;
  } ptrace;
  std::vector<std::vector<i64>> tcoord;  // per group sorted touched coords
  std::vector<std::vector<i64>> tdev;    // per group deviation values
  GroupFenwick tgf;                      // per-group Fenwick over touched coords
  std::vector<i64> gunion;
  FlatFenwick gfen;  // global DYNs_v index over union touched coords
  std::vector<i64> qlb, qhb;  // [state*kMaxQ + k] declared query frame bounds
  i32 qn = 0;
  bool pre_eager = false;  // auto resolved eagerly before the build (shallow version trees)
  // bucketed class lookup over the declared endpoint domain: class_of(t) becomes a bucket
  // index plus a short local scan instead of a full binary search per row
  std::vector<i32> cl_lut;
  i64 lut_tmin = 0, lut_width = 1;
  i32 lut_buckets = 0;

  LazyHybridArm(const Fixture& fi, const Workload& wo, const char* mode_override = nullptr)
      : f(fi), w(wo) {
    const char* m = mode_override ? mode_override : std::getenv("BWH_FACTOR_MODE");
    if (!m || !*m) m = "eager";
    if (!std::strcmp(m, "lazy")) freq = FM_LAZY;
    else if (!std::strcmp(m, "hybrid")) freq = FM_HYBRID;
    else if (!std::strcmp(m, "auto")) freq = FM_AUTO;
    else freq = FM_EAGER;
    // Shallow version trees never amortize the lazy build: measured out of sample, eager is the
    // best candidate on every 2,049-state panel, and the lazy-compatible build alone cost up to
    // 46% of the eager build. Auto therefore resolves eagerly *before* building for shallow
    // trees; the cost-model planner is kept for deep trees where deferred corrections win.
    if (freq == FM_AUTO && kStates < 8192) {
      fmode = FM_EAGER;
      pre_eager = true;
    } else {
      // deeper trees: run the lazy-compatible build and resolve from plan_factor_mode()
      fmode = (freq == FM_AUTO) ? FM_LAZY : freq;
    }
    lazy_path = (fmode != FM_EAGER);
    ptrace.requested = mode_name(freq);
    if (pre_eager) {
      ptrace.chosen = "eager";
      ptrace.chosen_units = 0;
    }
  }
  const char* name() const override { return "lazyhybrid"; }
  static const char* mode_name(FMode m) {
    switch (m) {
      case FM_EAGER: return "eager";
      case FM_LAZY: return "lazy";
      case FM_HYBRID: return "hybrid";
      default: return "auto";
    }
  }
  const char* resolved_mode() const { return mode_name(fmode); }

  inline i64 class_of(i64 t) const {
    if (cl_lut.empty())
      return i64(std::upper_bound(w.endpoints.begin(), w.endpoints.end(), t) - w.endpoints.begin());
    i64 j = (t - lut_tmin) / lut_width;
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
      lut_width = std::max<i64>(1, (sspan + i64(lut_buckets) - 1) / i64(lut_buckets));
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
    if (!lazy_path) {
      active_only = gp_env && std::strcmp(gp_env, "active") == 0;
    } else {
      // lazy/hybrid/auto read baseP only for groups a declared factor edit can activate;
      // allocate immutable rows for those groups only (BWH_GROUP_PREFIX=all restores the
      // shipped dense allocation). Exactness is unchanged: d==0 groups are never read.
      active_only = !(gp_env && std::strcmp(gp_env, "all") == 0);
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
    if (!dense && !lazy_path) {
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
    // ---- static base: bucket then prefix (A[k] = sum over coords < k)
    auto t_alloc = pn();
    if (!want_agg2) {
      baseP.assign(size_t(n_active) * size_t(C), 0);
      base0.assign(size_t(C), 0);
    }
    {
      const char* tpv = std::getenv("BWH_TRANSPOSE_BASEP");
      if (tpv && !dense) {
        if (tpv[0] == '1') {
          tp_base = true;
        } else if (std::strcmp(tpv, "auto") == 0) {
          // automatic rule: use the class-major layout only when factor edits are frequent
          // (at least one per sixteen states), where many groups differ from their initial
          // factor and the correction loop dominates the query phase
          i64 n_factor = 0;
          for (i32 v = 1; v < kStates; ++v)
            if (w.edit[size_t(v)].op == OP_FACTOR) ++n_factor;
          tp_base = (n_factor * 16 >= i64(kStates));
        }
      }
    }
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
    hfen.init(i32(gunion.size()));
    hflag.assign(size_t(G), 0);
    act.clear();
    act_pos.assign(size_t(G), -1);
    auto t_idx = pn();
    if (freq == FM_AUTO || freq == FM_HYBRID || std::getenv("BWH_PLANNER_TRACE") != nullptr)
      plan_factor_mode();
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
    qlb.assign(size_t(kStates) * kMaxQ, 0);
    qhb.assign(size_t(kStates) * kMaxQ, 0);
    for (i32 v = 1; v < kStates; ++v) {
      for (i32 k = 0; k < kMaxQ; ++k) {
        const Query& qq = query_for(w, v, k);
        qlb[size_t(v) * kMaxQ + size_t(k)] = dense ? (qq.lo - f.h.t_min) : class_of(qq.lo);
        qhb[size_t(v) * kMaxQ + size_t(k)] = dense ? (qq.hi - f.h.t_min) : class_of(qq.hi);
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
    if (!lazy_path) {
      gfen.add(ucl(coord) + 1, fac[size_t(g)] * d);
    } else {
      // initial-factor weighting: row edits must not read the mutable factor
      gfen.add(ucl(coord) + 1, s0[size_t(g)] * d);
      if (fmode == FM_HYBRID && hflag[size_t(g)]) {
        const i64 w = fac[size_t(g)] - s0[size_t(g)];
        if (w != 0) hfen.add(ucl(coord) + 1, w * d);
      }
    }
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
  inline void set_active(i32 g, bool on) {
    if (on) {
      if (act_pos[size_t(g)] < 0) {
        act_pos[size_t(g)] = i32(act.size());
        act.push_back(g);
      }
    } else {
      const i32 p = act_pos[size_t(g)];
      if (p >= 0) {
        const i32 last = act.back();
        act[size_t(p)] = last;
        act_pos[size_t(last)] = p;
        act.pop_back();
        act_pos[size_t(g)] = -1;
      }
    }
  }

  void set_factor(i32 g, i64 nf) {
    if (fac[size_t(g)] == nf) return;
    const i64 ds = nf - fac[size_t(g)];
    if (!lazy_path) {
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
    } else if (fmode == FM_HYBRID && hflag[size_t(g)]) {
      // heavy group: keep the (fac-s0)-weighted index exact by paying the update now
      const auto& cv = tcoord[size_t(g)];
      const auto& dv = tdev[size_t(g)];
      for (size_t k = 0; k < cv.size(); ++k)
        if (dv[k] != 0) hfen.add(ucl(cv[k]) + 1, ds * dv[k]);
    }
    // lazy: O(1) — only the scalar factor and the active-group membership change
    fac[size_t(g)] = nf;
    if (lazy_path) set_active(g, nf != s0[size_t(g)]);
  }


  // ---------------------------------------------------------------- factor planner
  // Additive per-group cost model over the declared workload; it never executes a query.
  // The DFS below tracks, for every group g on the current root-to-node path,
  //   nz[g]      : number of coordinates with a nonzero current deviation
  // and records, over all declared edits,
  //   Escan[g]   : |tcoord[g]| entries scanned by the shipped eager set_factor loop
  //   Edv[g]     : nonzero deviation entries updated by the eager set_factor loop
  //   Rc[g]      : dynamic row-edit index updates touching g (extra hfen add when eager)
  //   A[g]       : (state, query) pairs on which g is in the active set (q*active states)
  // The two candidate per-group costs in index-operation units are
  //   eager(g) = Escan + Edv*(1+log2 M) + Rc*log2 M                     [M = |union coords|]
  //   lazy(g)  = A*(4*log2 d_g + 1)                                     [d_g = |tcoord[g]|]
  // (lazy: two bisections + two per-group Fenwick prefixes per active group per query;
  // the log2 factors approximate the Fenwick/bisection depths). Auto compares the predicted
  // all-eager, all-lazy and optimal-partition totals; explicit hybrid uses the partition.
  void plan_factor_mode() {
    if (freq == FM_EAGER || pre_eager) return;
    ptrace.ran = true;
    ptrace.groups = G;
    const i32 n = kStates;
    std::vector<i64> Edv(size_t(G), 0), Escan(size_t(G), 0), A(size_t(G), 0), Rc(size_t(G), 0);
    std::vector<i64> nz(size_t(G), 0);
    std::vector<i64> simf = s0;             // simulated current factor per group
    std::vector<u8> actv(size_t(G), 0);     // simulated active set
    std::vector<i64> astart(size_t(G), 0);  // state-entry counter of the open active run
    std::vector<i64> oldfac(size_t(n), 0);   // factor value before the node edit
    std::vector<std::vector<u64>> newkeys;
    std::vector<std::vector<i32>> kids;
    newkeys.resize(size_t(n));
    kids.resize(size_t(n));
    for (i32 v = 1; v < n; ++v) kids[size_t(w.parent[size_t(v)])].push_back(v);
    std::unordered_set<u64> live;
    live.reserve(size_t(n) * 2 + 16);
    i64 cnt = 0, dv_iters = 0, fac_edits = 0, row_edits = 0;
    // Coordinates fit in 32 bits for every declared workload (dense span <= 2(N+U), sparse
    // class count <= 2*kMaxQ*kStates); the key packs group and coordinate.
    auto touch = [&](i32 v, i32 g, i64 c) {
      const u64 key = (u64(u32(g)) << 32) | u64(u32(c));
      ++Rc[size_t(g)];
      if (live.insert(key).second) {
        ++nz[size_t(g)];
        newkeys[size_t(v)].push_back(key);
      }
    };
    auto frame = [&](i64 t) { return frame_of(t); };
    struct Ev { i32 v; u8 exit; };
    std::vector<Ev> st;
    st.reserve(size_t(n) + 8);
    st.push_back({0, 0});
    while (!st.empty()) {
      const Ev ev = st.back();
      st.pop_back();
      const i32 v = ev.v;
      if (ev.exit) {
        if (v > 0) {
          const Edit& e = w.edit[size_t(v)];
          if (e.op == OP_FACTOR) {
            const i32 g = e.g;
            simf[size_t(g)] = oldfac[size_t(v)];
            const bool was = actv[size_t(g)] != 0;
            const bool old_active = simf[size_t(g)] != s0[size_t(g)];
            if (old_active && !was) {
              actv[size_t(g)] = 1;
              astart[size_t(g)] = cnt + 1;
            } else if (!old_active && was) {
              A[size_t(g)] += i64(w.q) * (cnt - astart[size_t(g)] + 1);
              actv[size_t(g)] = 0;
            }
          } else {
            for (u64 key : newkeys[size_t(v)]) {
              live.erase(key);
              const i32 g = i32(key >> 32);
              --nz[size_t(g)];
            }
          }
        }
        continue;
      }
      if (v > 0) {
        ++cnt;
        const Edit& e = w.edit[size_t(v)];
        if (e.op == OP_FACTOR) {
          const i32 g = e.g;
          const i64 curf = simf[size_t(g)];
          oldfac[size_t(v)] = curf;
          const i64 nf = e.val;
          if (nf != curf) {
            ++fac_edits;
            Edv[size_t(g)] += nz[size_t(g)];
            dv_iters += nz[size_t(g)];
            Escan[size_t(g)] += i64(tcoord[size_t(g)].size());
            const bool was = actv[size_t(g)] != 0;
            const bool now = nf != s0[size_t(g)];
            if (now && !was) {
              actv[size_t(g)] = 1;
              astart[size_t(g)] = cnt;
            } else if (!now && was) {
              A[size_t(g)] += i64(w.q) * (cnt - astart[size_t(g)]);
              actv[size_t(g)] = 0;
            }
            simf[size_t(g)] = nf;
          }
        } else {
          ++row_edits;
          if (e.op == OP_INSERT) {
            touch(v, e.g, frame(e.t));
          } else if (e.op == OP_MOVE) {
            if (e.id < i64(f.h.n)) {
              const BaseRec& br = f.rows[size_t(e.id)];
              touch(v, br.g, frame(br.t));
            }
            touch(v, e.g, frame(e.t));
          } else {  // OP_REPLACE / OP_DELETE
            if (e.id < i64(f.h.n)) {
              const BaseRec& br = f.rows[size_t(e.id)];
              touch(v, br.g, frame(br.t));
            } else {
              const size_t k = size_t(e.id - i64(f.h.n));
              if (k < w.ins_t.size()) touch(v, w.ins_g[k], frame(w.ins_t[k]));
            }
          }
        }
      }
      st.push_back({v, 1});
      for (auto it = kids[size_t(v)].rbegin(); it != kids[size_t(v)].rend(); ++it)
        st.push_back({*it, 0});
    }
    // ---------------------------------------------------------------- calibrated cost model
    // All totals below are in estimated execute-phase cost units (~nanoseconds), fitted by
    // least squares against the measured execute phase of the remote benchmark panel
    // (33 cases: q in {2,8,32,128}, 3 bases, factor-sparse/factor-heavy, tree depths
    // 2,049/16,385/65,537).  The fit reproduces the measured eager/lazy/hybrid ranking with
    // median regret 1.00x and max regret 1.15x on that panel (see
    // results/lazy_20261029/planner_fit.json).  Operation classes:
    //   W_ADD  global gfen/hfen Fenwick step over |union| coordinates (largest working set,
    //          ~3.8 ns/step measured);
    //   W_SCAN one tcoord slot scanned by the shipped eager set_factor loop (cache-resident);
    //   W_GRP  one group slot of the eager query loop (d==0 test + active base-prefix rows);
    //   W_BIS/W_FIX  lazy per-active-group query work: two bisections + two per-group Fenwick
    //          prefixes (log2 d_g scaled by W_BIS) plus the fixed pair of base-prefix loads;
    //   W_HEAVY/W_LIST  hybrid heavy group (base-prefix loads only, delta already in hfen) and
    //          the act-list slot itself;  W_HFEN the extra global hfen prefix pair per query;
    //   UPD    every declared factor edit is applied on enter and undone on exit (x2).
    const double logM = std::log2(double(gunion.size()) + 2.0);
    const double W_SCAN = 1.0, W_ADD = 3.8, W_GRP = 2.3, W_BIS = 2.4, W_FIX = 1.1,
                 W_HEAVY = 1.7, W_LIST = 0.7, W_HFEN = 2.4, UPD = 2.0;
    const double queries = double(i64(w.q) * i64(n - 1));
    std::vector<double> cEupd(size_t(G), 0), cL(size_t(G), 0);
    double eager_units = 0, lazy_units = 0, hybrid_units = 0;
    i64 lazy_iters = 0;
    i32 active_groups = 0, heavy = 0;
    bool any_heavy = false;
    for (i32 g = 0; g < G; ++g) {
      const double dg = std::log2(double(std::max<i64>(2, i64(tcoord[size_t(g)].size()))));
      // eager: factor-update work paid per declared edit (scans + global Fenwick adds) and the
      // per-query correction-scan slot; lazy: deferred per-active-group query work
      const double upd = UPD * (W_SCAN * double(Escan[size_t(g)]) +
                                W_ADD * logM * (double(Edv[size_t(g)]) + double(Rc[size_t(g)])));
      cEupd[size_t(g)] = upd;
      cL[size_t(g)] = double(A[size_t(g)]) * (W_BIS * dg + W_FIX);
      eager_units += upd + W_GRP * queries;
      lazy_units += cL[size_t(g)];
      // hybrid: a heavy group pays its hard update now and only the cheap base-prefix delta at
      // query time; a light group is fully deferred to the query
      const double hheavy = upd + W_HEAVY * double(A[size_t(g)]);
      if (hheavy <= cL[size_t(g)]) { hflag[size_t(g)] = 1; any_heavy = true; ++heavy; }
      else hflag[size_t(g)] = 0;
      hybrid_units += hflag[size_t(g)] ? hheavy : cL[size_t(g)];
      if (A[size_t(g)] > 0) ++active_groups;
      lazy_iters += A[size_t(g)];
    }
    // the lazy/hybrid query walks the active list; hybrid additionally pays the global hfen
    // prefix pair on every query when at least one group is heavy
    const double loop_lazy = W_LIST * double(lazy_iters);
    lazy_units += loop_lazy;
    hybrid_units += loop_lazy + (any_heavy ? W_HFEN * queries : 0.0);
    ptrace.eager_units = eager_units;
    ptrace.lazy_units = lazy_units;
    ptrace.hybrid_units = hybrid_units;
    ptrace.eager_dv_iters = dv_iters;
    ptrace.lazy_group_queries = lazy_iters;
    ptrace.log_m = logM;
    ptrace.queries = queries;
    for (i32 g = 0; g < G; ++g) {
      ptrace.escan_units += Escan[size_t(g)];
      ptrace.edv_units += Edv[size_t(g)];
      ptrace.rc_units += Rc[size_t(g)];
      const double dg = std::log2(double(std::max<i64>(2, i64(tcoord[size_t(g)].size()))));
      ptrace.lazy_raw += i64(double(A[size_t(g)]) * (4.0 * dg + 1.0));
      ptrace.gfeat.push_back(A[size_t(g)]);
      ptrace.gfeat.push_back(Escan[size_t(g)]);
      ptrace.gfeat.push_back(Edv[size_t(g)]);
      ptrace.gfeat.push_back(Rc[size_t(g)]);
      ptrace.gfeat.push_back(i64(tcoord[size_t(g)].size()));
    }
    ptrace.n_factor_edits = fac_edits;
    ptrace.n_row_edits = row_edits;
    ptrace.active_groups = active_groups;
    {
      std::vector<u8> touched(size_t(G), 0);
      i32 fg = 0;
      for (i32 v = 1; v < n; ++v)
        if (w.edit[size_t(v)].op == OP_FACTOR) touched[size_t(w.edit[size_t(v)].g)] = 1;
      for (i32 g = 0; g < G; ++g) fg += touched[size_t(g)];
      ptrace.factor_groups = fg;
    }
    FMode chosen;
    if (freq == FM_HYBRID) {
      chosen = FM_HYBRID;
    } else {
      const double best_pure = std::min(eager_units, lazy_units);
      // 2% model-risk margin: keep a pure mode unless the partition wins by more than that
      if (hybrid_units < 0.98 * best_pure) chosen = FM_HYBRID;
      else chosen = (eager_units <= lazy_units) ? FM_EAGER : FM_LAZY;
    }
    bool any_h = false, any_l = false;
    for (i32 g = 0; g < G; ++g) {
      if (hflag[size_t(g)]) any_h = true; else any_l = true;
    }
    if (chosen == FM_HYBRID && !any_l) chosen = FM_EAGER;
    else if (chosen == FM_HYBRID && !any_h) chosen = FM_LAZY;
    fmode = chosen;
    lazy_path = (fmode != FM_EAGER);
    if (fmode != FM_HYBRID) std::fill(hflag.begin(), hflag.end(), 0);
    ptrace.heavy_groups = (fmode == FM_HYBRID) ? heavy : 0;
    ptrace.chosen = mode_name(fmode);
    ptrace.chosen_units = (fmode == FM_EAGER) ? eager_units
                        : (fmode == FM_LAZY ? lazy_units : hybrid_units);
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
    const size_t qidx = size_t(state) * kMaxQ + size_t(k);
    const i64 i = qlb[qidx], j = qhb[qidx];
    if (!lazy_path) {
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
    // ---- lazy / hybrid: factor weighting is applied at query time
    i64 ans = base0[size_t(j)] - base0[size_t(i)];
    ans += gfen.prefix(ucl(j)) - gfen.prefix(ucl(i));
    if (fmode == FM_HYBRID) ans += hfen.prefix(ucl(j)) - hfen.prefix(ucl(i));
    for (size_t ai = 0; ai < act.size(); ++ai) {
      const i32 g = act[ai];
      const i64 d = fac[size_t(g)] - s0[size_t(g)];
      const i32 r = g2row[size_t(g)];
      if (r >= 0) {
        if (tp_base) {
          ans += d * (baseP[size_t(j) * size_t(n_active) + size_t(r)] -
                      baseP[size_t(i) * size_t(n_active) + size_t(r)]);
        } else {
          const i64* row = &baseP[size_t(r) * size_t(ncoord + 1)];
          ans += d * (row[j] - row[i]);
        }
      }
      if (fmode == FM_HYBRID && hflag[size_t(g)]) continue;  // delta already in hfen
      const auto& cv = tcoord[size_t(g)];
      const i32 lo = i32(std::lower_bound(cv.begin(), cv.end(), i) - cv.begin());
      const i32 hi = i32(std::lower_bound(cv.begin(), cv.end(), j) - cv.begin());
      if (hi != lo) ans += d * (tgf.prefix(g, hi) - tgf.prefix(g, lo));
    }
    return ans;
  }
};

inline Arm* make_arm(const std::string& name, const Fixture& f, const Workload& w) {
  // "lazyhybrid" honours BWH_FACTOR_MODE; the explicit aliases force one mode so an A/B
  // driver can interleave candidates inside one window without touching the environment.
  if (name == "lazyhybrid" || name == "lazyhybrid_auto")
    return new LazyHybridArm(f, w, name == "lazyhybrid_auto" ? "auto" : nullptr);
  if (name == "lazyhybrid_eager") return new LazyHybridArm(f, w, "eager");
  if (name == "lazyhybrid_lazy") return new LazyHybridArm(f, w, "lazy");
  if (name == "lazyhybrid_hybrid") return new LazyHybridArm(f, w, "hybrid");
  if (name == "hybrid") return new LazyHybridArm(f, w, "eager");  // shipped alias
  if (name == "full_raw") return new FullStateArm(f, w, 0, "full_raw");
  if (name == "full_projected") return new FullStateArm(f, w, 1, "full_projected");
  return nullptr;
}

}  // namespace bwh
