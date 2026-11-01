// sparse2d_arm.hpp — sparse dynamic 2D index baseline for the FMDI comparison (2026-11-02).
//
// New file under results/fmdi_sparse_20261101/.  FMDI itself is the existing fmd_index.hpp
// (unchanged); this file adds the fair "sparse dynamic 2D" competitor with exactly the same
// 2D version-tree semantics:
//
//   * per factor group, the set of (dim0 class, dim1 class) supports that any row of the group
//     can occupy over the whole declared session is precomputed (base row positions + every
//     insert/move destination); the state is a per-group sparse 2D Fenwick-of-vectors over
//     those supports (outer Fenwick over dim0, each node a sorted dim1 vector + inner i64
//     Fenwick) holding the current raw amounts.
//   * point update (row replace/insert/delete/move) = remove/add one point, O(log C0 log s);
//     rectangle query per group = two outer prefixes with inner dim1 prefixes, O(log^2);
//     factor edits are a scalar at query time:  answer = sum_g fac_g * Range_g(rect).
//   * memory grows with the occupied supports (sum over groups of supports*log C0), not with
//     the dense product G*C0*C1 -- this is the support-sparse 2D baseline par excellence.
//
// It has no FMDI-style hierarchical shared base over dim0 and no separate deviation layer: it
// stores the full current relation sparsely per group, so it is the natural "sparse 2D index"
// competitor for FMDI and for the dynamic dense 2D Fenwick.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

#include "fmd_index.hpp"

namespace fmd {

struct Sparse2DArm : Arm {
  const Workload& w;
  Config cfg;
  RowState rs;
  std::vector<Bit2DGroup> gb;  // one sparse 2D Fenwick per group
  i64 static_bytes = 0, work_bytes = 0, supports_total = 0;
  i64 updates = 0, queries = 0;

  explicit Sparse2DArm(const Workload& wo) : w(wo), cfg(wo.cfg) {}
  const char* name() const override { return "sparse2d"; }

  void build() override {
    rs.init(w);
    gb.resize(size_t(cfg.groups));
    // supports each group can ever hold: base row positions of its base rows + every
    // insert/move destination written with that group (replace/delete keep the position).
    std::vector<std::vector<std::pair<i32, i32>>> sup(size_t(cfg.groups));
    for (i64 i = 0; i < cfg.rows; ++i)
      sup[size_t(w.b_g[size_t(i)])].emplace_back(w.b_c0[size_t(i)], w.b_c1[size_t(i)]);
    for (i64 v = 1; v < cfg.states; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if (e.op == OP_INSERT || e.op == OP_MOVE)
        sup[size_t(e.g)].emplace_back(e.c0, e.c1);
    }
    for (i32 g = 0; g < cfg.groups; ++g) {
      auto& s = sup[size_t(g)];
      std::sort(s.begin(), s.end());
      s.erase(std::unique(s.begin(), s.end()), s.end());
      supports_total += i64(s.size());
      if (!s.empty()) {
        gb[size_t(g)].build(cfg.c0, s);
        static_bytes += i64(gb[size_t(g)].bytes());
      }
    }
    for (i64 i = 0; i < cfg.rows; ++i) {
      const i32 g = w.b_g[size_t(i)];
      gb[size_t(g)].add(w.b_c0[size_t(i)], w.b_c1[size_t(i)], w.b_amt[size_t(i)]);
    }
    work_bytes = static_bytes;
  }

  inline void point(i32 g, i32 c0, i32 c1, i64 v) {
    if (v == 0) return;
    gb[size_t(g)].add(c0, c1, v);
  }
  void set_row(i64 id, i32 c0, i32 c1, i32 g, i64 amount, u8 ex) {
    if (rs.c_ex[size_t(id)])
      point(rs.c_g[size_t(id)], rs.c_c0[size_t(id)], rs.c_c1[size_t(id)], -rs.c_amt[size_t(id)]);
    rs.c_c0[size_t(id)] = c0;
    rs.c_c1[size_t(id)] = c1;
    rs.c_g[size_t(id)] = g;
    rs.c_amt[size_t(id)] = amount;
    rs.c_ex[size_t(id)] = ex;
    if (ex) point(g, c0, c1, amount);
    ++updates;
  }
  SavedRow snapshot(const Edit& e) const override {
    SavedRow s{};
    if (e.op == OP_FACTOR) {
      s.kind = 1;
      s.g = e.g;
      s.fac = rs.fac[size_t(e.g)];
    } else {
      s.kind = 0;
      s.id = e.id;
      s.prev_edited = 0;
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
      case OP_REPLACE:
        set_row(e.id, rs.c_c0[size_t(e.id)], rs.c_c1[size_t(e.id)], rs.c_g[size_t(e.id)], e.val, 1);
        break;
      case OP_INSERT: set_row(e.id, e.c0, e.c1, e.g, e.val, 1); break;
      case OP_DELETE:
        set_row(e.id, rs.c_c0[size_t(e.id)], rs.c_c1[size_t(e.id)], rs.c_g[size_t(e.id)],
                rs.c_amt[size_t(e.id)], 0);
        break;
      case OP_MOVE: set_row(e.id, e.c0, e.c1, e.g, e.val, 1); break;
      case OP_FACTOR: rs.fac[size_t(e.g)] = e.val; break;
    }
  }
  void undo(const Edit& e, const SavedRow& s) override {
    (void)e;
    if (s.kind == 1) rs.fac[size_t(s.g)] = s.fac;
    else set_row(s.id, s.c0, s.c1, s.g, s.amount, s.ex);
  }
  i64 query(i32 state, i32 k) override {
    const Query2& q = w.query[size_t(state) * size_t(cfg.q) + size_t(k)];
    i64 ans = 0;
    for (i32 g = 0; g < cfg.groups; ++g) {
      const i64 f = rs.fac[size_t(g)];
      if (f == 0) continue;
      ans += f * gb[size_t(g)].query(q.lo0, q.hi0, q.lo1, q.hi1);
    }
    ++queries;
    return ans;
  }
  void stats(i64& sb, i64& pb, i64& wb) const override {
    sb = static_bytes;
    pb = i64(cfg.groups) * i64(cfg.c0) * i64(cfg.c1) * 8;  // the dense product it avoids
    wb = supports_total;
  }
};

}  // namespace fmd
