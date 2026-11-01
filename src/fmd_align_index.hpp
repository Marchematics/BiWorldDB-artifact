// fmd_align_index.hpp — dynamic dense 2D baseline for the FMDI alignment experiment (2026-11-01).
//
// New file only (results/fmdi_align_20261101/); the shipped engine, panel and paper are untouched.
//
// Baseline semantics: exactly the FMDI contract (see fmd_index.hpp) — the same 2D workload,
// versions, reversible row/factor edits and rectangle queries as `FmdArm`; answer = sum over
// current rows in [lo0,hi0)x[lo1,hi1) of amount * fac(group).
//
// Representation: one dense 2D Fenwick per group over the C0 x C1 class grid
// (`G * (C0+1) * (C1+1)` int64 cells; G groups, no product C0*C1 across groups) with:
//   * point row update: O(log C0 * log C1) (replace / insert / delete / move),
//   * factor edit: O(1) (a scalar, applied at query time),
//   * rectangle query: for each group, 4 2D-Fenwick prefix sums -> O(G * log C0 * log C1),
//   * reversible versions: snapshot/apply/undo of a SavedRow re-applies the inverse point
//     updates, so bushy/star traversals and undo are exact.
//
// This is the "dynamic 2D index" baseline the FMDI comparison was missing: unlike `NaiveArm`
// (per-group dense 2D prefix rebuilt per state) it never rebuilds a prefix grid per version;
// unlike FMDI it pays the full dense G*C0*C1 memory and every query scans all G groups.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

#include "fmd_index.hpp"

namespace fmd {

struct DynArm : Arm {
  const Workload& w;
  Config cfg;
  RowState rs;
  i32 C0 = 0, C1 = 0, G = 0;
  i64 stride1 = 1;                     // C1 + 1
  std::vector<i64> ft;                 // G * (C0+1) * (C1+1) 2D Fenwick cells
  explicit DynArm(const Workload& wo) : w(wo), cfg(wo.cfg) {}
  const char* name() const override { return "dyn2d"; }

  inline i64* gbase(i32 g) { return ft.data() + size_t(g) * size_t(C0 + 1) * size_t(stride1); }
  inline const i64* gbase(i32 g) const { return ft.data() + size_t(g) * size_t(C0 + 1) * size_t(stride1); }

  inline void point_add(i32 g, i32 c0, i32 c1, i64 v) {
    if (v == 0) return;
    i64* t = gbase(g);
    for (i32 i = c0 + 1; i <= C0; i += i & -i) {
      i64* row = t + size_t(i) * size_t(stride1);
      for (i32 j = c1 + 1; j <= C1; j += j & -j) row[size_t(j)] += v;
    }
  }
  inline i64 prefix(i32 g, i32 x, i32 y) const {   // sum over c0 < x, c1 < y
    if (x <= 0 || y <= 0) return 0;
    const i64* t = gbase(g);
    i64 s = 0;
    for (i32 i = x; i > 0; i -= i & -i) {
      const i64* row = t + size_t(i) * size_t(stride1);
      for (i32 j = y; j > 0; j -= j & -j) s += row[size_t(j)];
    }
    return s;
  }
  void build() override {
    rs.init(w);
    C0 = cfg.c0; C1 = cfg.c1; G = cfg.groups; stride1 = i64(C1) + 1;
    ft.assign(size_t(G) * size_t(C0 + 1) * size_t(stride1), 0);
    for (i64 id = 0; id < rs.n_tot; ++id) {
      if (!rs.c_ex[size_t(id)]) continue;
      point_add(rs.c_g[size_t(id)], rs.c_c0[size_t(id)], rs.c_c1[size_t(id)], rs.c_amt[size_t(id)]);
    }
  }
  void set_row(i64 id, i32 c0, i32 c1, i32 g, i64 amount, u8 ex) {
    const size_t i = size_t(id);
    if (rs.c_ex[i]) point_add(rs.c_g[i], rs.c_c0[i], rs.c_c1[i], -rs.c_amt[i]);
    rs.c_c0[i] = c0; rs.c_c1[i] = c1; rs.c_g[i] = g; rs.c_amt[i] = amount; rs.c_ex[i] = ex;
    if (ex) point_add(g, c0, c1, amount);
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
    if (s.kind == 1) { rs.fac[size_t(s.g)] = s.fac; return; }
    if (rs.c_ex[size_t(s.id)]) { /* nothing: set_row handles the old state below */ }
    // restore the exact previous row state (set_row first removes the current state)
    const size_t i = size_t(s.id);
    // current row must be removed before restoring; set_row does remove-then-add
    set_row(s.id, s.c0, s.c1, s.g, s.amount, s.ex);
    (void)i;
  }
  i64 query(i32 state, i32 k) override {
    const Query2& qq = w.query[size_t(state) * size_t(cfg.q) + size_t(k)];
    i64 ans = 0;
    for (i32 g = 0; g < G; ++g) {
      const i64 v = prefix(g, qq.hi0, qq.hi1) - prefix(g, qq.lo0, qq.hi1) -
                    prefix(g, qq.hi0, qq.lo1) + prefix(g, qq.lo0, qq.lo1);
      if (v) ans += rs.fac[size_t(g)] * v;
    }
    return ans;
  }
  void stats(i64& sb, i64& pb, i64& wb) const override {
    const i64 cells = i64(G) * i64(C0 + 1) * i64(C1 + 1);
    sb = cells * 8;
    pb = i64(cfg.c0) * i64(cfg.c1) * 8;
    wb = cells * 8;
  }
};

inline Arm* make_arm_align(const std::string& name, const Workload& w) {
  if (name == "dyn2d") return new DynArm(w);
  return nullptr;
}

}  // namespace fmd
