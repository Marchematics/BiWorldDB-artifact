// dense_fenwick_arm.hpp — dynamic dense 2D Fenwick baseline for the large-domain FMDI
// comparison (2026-11-01, FMDI large-domain study).
//
// New file under results/fmdi_large_20261101/.  The FMDI index itself is the existing
// fmd_index.hpp (unchanged); this file only adds the fair "dynamic dense 2D Fenwick"
// competitor for the same 2D version-tree contract:
//
//   * one dense 2D Fenwick tree per factor group: G arrays of (C0+1)*(C1+1) i64 counters,
//     indexed over the dim0/dim1 integer class domains; point update and rectangle query are
//     O(log C0 log C1); factor edits are a scalar at query time (factor * group amount).
//   * memory = G*(C0+1)*(C1+1)*8 bytes -- exactly the product-class memory the FMDI avoids.
//   * the worker guards the allocation against the host's currently available memory and
//     exits with code 3 and a machine-readable INFEASIBLE_MEMORY line instead of risking an
//     OOM kill on a shared host; the benchmark records that as a dense-infeasible cell.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "fmd_index.hpp"

namespace fmd {

struct DenseFenwickArm : Arm {
  const Workload& w;
  Config cfg;
  RowState rs;
  std::vector<i64> bit;  // G * (C0+1) * (C1+1) i64 counters
  i64 static_bytes = 0, product_bytes = 0, need_bytes = 0;
  i64 updates = 0, queries = 0;

  explicit DenseFenwickArm(const Workload& wo) : w(wo), cfg(wo.cfg) {}
  const char* name() const override { return "densefen"; }

  static i64 mem_available_bytes() {
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    i64 kb = -1;
    while (std::fgets(line, sizeof(line), f)) {
      long long v = 0;
      if (std::sscanf(line, "MemAvailable: %lld kB", &v) == 1) { kb = v; break; }
    }
    std::fclose(f);
    return kb >= 0 ? kb * 1024 : -1;
  }

  inline i64 plane(i32 g) const { return i64(g) * i64(cfg.c0 + 1) * i64(cfg.c1 + 1); }
  inline void bit_add(i32 g, i32 x, i32 y, i64 v) {
    if (v == 0) return;
    const i64 base = plane(g);
    const i64 W = cfg.c1 + 1;
    for (i32 i = x + 1; i <= cfg.c0; i += i & (-i)) {
      i64* row = bit.data() + base + i64(i) * W;
      for (i32 j = y + 1; j <= cfg.c1; j += j & (-j)) row[j] += v;
    }
  }
  inline i64 bit_pref(i32 g, i32 x, i32 y) const {  // sum over c0 < x, c1 < y
    i64 s = 0;
    const i64 base = plane(g);
    const i64 W = cfg.c1 + 1;
    for (i32 i = x; i > 0; i -= i & (-i)) {
      const i64* row = bit.data() + base + i64(i) * W;
      for (i32 j = y; j > 0; j -= j & (-j)) s += row[j];
    }
    return s;
  }
  inline i64 bit_rect(i32 g, i32 lo0, i32 hi0, i32 lo1, i32 hi1) const {
    return bit_pref(g, hi0, hi1) - bit_pref(g, lo0, hi1) - bit_pref(g, hi0, lo1) +
           bit_pref(g, lo0, lo1);
  }
  void build() override {
    need_bytes = i64(cfg.groups) * i64(cfg.c0 + 1) * i64(cfg.c1 + 1) * 8;
    product_bytes = i64(cfg.c0) * i64(cfg.c1) * 8;
    const i64 avail = mem_available_bytes();
    if (avail > 0 && need_bytes > avail - (i64(4) << 30)) {
      std::fprintf(stderr,
                   "INFEASIBLE_MEMORY need_bytes=%lld available_bytes=%lld "
                   "groups=%d c0=%d c1=%d\n",
                   (long long)need_bytes, (long long)avail, cfg.groups, cfg.c0, cfg.c1);
      std::exit(3);
    }
    bit.assign(size_t(need_bytes / 8), 0);
    rs.init(w);
    for (i64 id = 0; id < rs.n_tot; ++id) {
      if (!rs.c_ex[size_t(id)]) continue;
      bit_add(rs.c_g[size_t(id)], rs.c_c0[size_t(id)], rs.c_c1[size_t(id)], rs.c_amt[size_t(id)]);
    }
    static_bytes = need_bytes;
  }
  void set_row(i64 id, i32 c0, i32 c1, i32 g, i64 amount, u8 ex) {
    if (rs.c_ex[size_t(id)])
      bit_add(rs.c_g[size_t(id)], rs.c_c0[size_t(id)], rs.c_c1[size_t(id)], -rs.c_amt[size_t(id)]);
    rs.c_c0[size_t(id)] = c0;
    rs.c_c1[size_t(id)] = c1;
    rs.c_g[size_t(id)] = g;
    rs.c_amt[size_t(id)] = amount;
    rs.c_ex[size_t(id)] = ex;
    if (ex) bit_add(g, c0, c1, amount);
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
    if (s.kind == 1) rs.fac[size_t(s.g)] = s.fac;
    else set_row(s.id, s.c0, s.c1, s.g, s.amount, s.ex);
  }
  i64 query(i32 state, i32 k) override {
    const Query2& q = w.query[size_t(state) * size_t(cfg.q) + size_t(k)];
    i64 ans = 0;
    for (i32 g = 0; g < cfg.groups; ++g) {
      const i64 f = rs.fac[size_t(g)];
      if (f == 0) continue;
      ans += f * bit_rect(g, q.lo0, q.hi0, q.lo1, q.hi1);
    }
    ++queries;
    return ans;
  }
  void stats(i64& sb, i64& pb, i64& wb) const override {
    sb = static_bytes;
    pb = product_bytes;
    wb = need_bytes;
  }
};

}  // namespace fmd
