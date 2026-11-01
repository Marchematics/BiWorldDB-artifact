// hros_factor_index.hpp — factor-weighted holistic statistics, structural HROS (2026-11-01).
//
// New file only (results/hros_factor_20261101/); the shipped engine/panel/paper are untouched.
//
// Why this operator: a plain order-statistic Fenwick over materialised row values is the optimal
// dynamic 1D order-statistic index *when values are static*, but a FACTOR edit re-scales every row
// of one group (weighted value = amount * fac[group]), which forces the materialising index to
// delete/insert all rows of that group.  HROS-F keeps the per-group amount distributions and
// applies the factor as a scalar at query time: a factor edit is O(1).
//
// Contract (this round):
//   rows carry a raw integer amount and a group; weighted value w = amount * fac[group];
//   per version the query is GLOBAL (no 1D range) and returns, over the current relation:
//       count     number of active rows
//       sum       sum of weighted values (i64)
//       median    lower median of the weighted values (0 when empty)
//       distinct  number of distinct (group, amount) pairs (cross-group weighted-value collisions
//                 are NOT merged; this is the distinct notion both HROS-F and the baseline can
//                 maintain in O(1); documented limitation)
//   version edits: replace / insert / delete / move (group or amount changes) / factor update.
//   A future round can compose the global weighted selection with the 1D-range structure from
//   results/hros_20261101 (the range restriction is orthogonal to the factor handling measured here).
//
// HROS-F structure:
//   * global amount universe (base amounts + every edit/insert amount), coordinate-compressed;
//   * per group: a reversible Fenwick over amount ranks (active row counts) + group sum +
//     per-group presence array for pair-distinct bookkeeping;
//   * factor edit = scalar update O(1) (total sum adjusted by (fac_new-fac_old)*group_sum);
//   * row edit = O(log U) point updates;
//   * median = binary search over the weighted-value integer range with
//       count_w(x) = sum_g Fenwick_g.prefix(rank(floor(x / fac[g]))),
//     then the minimal x with count_w(x) >= k is an existing weighted value, so median = x;
//   * count/sum/distinct are O(1) (maintained counters).
//
// Baseline (FenwickMat, for the comparison):
//   one global Fenwick over the universe of all possible weighted values (all compressed amounts x
//   the finite set of factor values in the workload); factor edit must walk every active row of the
//   group and move its weighted value (O(n_g log U_w)); query median is a Fenwick k-th O(log U_w);
//   count/sum/distinct maintained.  This is the "materialising Fenwick-only" baseline.
#pragma once
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace hf {

using i32 = int32_t;
using i64 = int64_t;
using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;

struct Rng {
  u64 s;
  explicit Rng(u64 seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
  inline u64 next() {
    u64 x = s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    s = x;
    return x;
  }
  inline i64 range(i64 lo, i64 hi) { return lo + i64(next() % u64(hi - lo + 1)); }
  inline i32 bounded(i32 n) { return i32(next() % u64(n)); }
};

enum Op : u8 { OP_REPLACE = 0, OP_INSERT = 1, OP_DELETE = 2, OP_MOVE = 3, OP_FACTOR = 4 };

struct Edit {
  u8 op;
  u8 pad[3];
  i32 g;
  i64 id;
  i64 val;  // absolute amount (replace/insert/move) or factor value (factor)
};

struct Config {
  i64 rows = 100000;
  i32 groups = 4;
  i64 states = 2049;
  i32 q = 2;
  i32 factor_period = 4;  // every fp-th state is a factor edit
  i32 topology = 0;       // 0 chain, 1 bushy heap
  i64 seed = 7;
};

struct Workload {
  Config cfg;
  std::vector<i32> parent;
  std::vector<Edit> edit;      // edit[0] unused
  std::vector<i32> b_g;        // base row groups
  std::vector<i64> b_amt;      // base amounts
  i32 n_ins = 0;
  std::vector<i32> ins_g;      // per inserted row: group
  std::vector<i64> ins_amt;    // per inserted row: amount
  std::vector<i64> fac_val;    // distinct factor values in the workload (incl. initial 1000)
  std::vector<i32> fac_init;   // initial per-group factor index into fac_val
};

inline Workload make_workload(const Config& cfg) {
  Workload w;
  w.cfg = cfg;
  const i64 N = cfg.rows, V = cfg.states;
  Rng rng(u64(cfg.seed) * 1000003ULL + u64(cfg.groups) * 7919ULL + 20261101ULL);
  w.fac_val = {1000, 950, 1050};
  w.fac_init.assign(size_t(cfg.groups), 0);
  w.b_g.resize(size_t(N));
  w.b_amt.resize(size_t(N));
  for (i64 i = 0; i < N; ++i) {
    w.b_g[size_t(i)] = rng.bounded(cfg.groups);
    // positive skewed amounts
    double u = double(rng.next() % 10000) / 10000.0;
    w.b_amt[size_t(i)] = i64(1000.0 * (1.0 + 3.0 * u * u)) + 1;
  }
  w.parent.resize(size_t(V));
  w.edit.resize(size_t(V));
  for (i64 v = 0; v < V; ++v) {
    const i32 topo = cfg.topology;
    w.parent[size_t(v)] = (v <= 0) ? -1 : (topo == 0 ? i32(v - 1) : i32((v - 1) / 2));
  }
  i32 next_ins = 0;
  for (i64 v = 1; v < V; ++v) {
    Edit e{};
    Rng er(u64(cfg.seed) * 1000000007ULL + u64(v) * 2654435761ULL + 12345ULL);
    const bool factor_edit = (cfg.factor_period > 0) && (v % cfg.factor_period == 0);
    if (factor_edit) {
      e.op = OP_FACTOR;
      e.g = er.bounded(cfg.groups);
      const i32 fi = 1 + er.bounded(2);  // 950 or 1050, never the current one for a visible change
      e.val = w.fac_val[size_t(fi)];
    } else {
      const i32 roll = er.bounded(100);
      if (roll < 40) {  // replace amount on a base row
        e.op = OP_REPLACE;
        e.id = er.range(0, N - 1);
        e.val = w.b_amt[size_t(e.id)] + er.range(-200, 200);
      } else if (roll < 70) {  // insert a fresh row (id N+k)
        e.op = OP_INSERT;
        e.id = N + next_ins++;
        e.g = er.bounded(cfg.groups);
        e.val = w.b_amt[size_t(er.range(0, N - 1))] + er.range(-200, 200);
        w.ins_g.push_back(e.g);
        w.ins_amt.push_back(e.val);
      } else if (roll < 85) {  // delete a base row
        e.op = OP_DELETE;
        e.id = er.range(0, N - 1);
      } else {  // move a base row to another group (and a new amount)
        e.op = OP_MOVE;
        e.id = er.range(0, N - 1);
        e.g = er.bounded(cfg.groups);
        e.val = w.b_amt[size_t(e.id)] + er.range(-200, 200);
      }
    }
    w.edit[size_t(v)] = e;
  }
  w.n_ins = next_ins;
  return w;
}

// ---------------------------------------------------------------- helper: amount universe
struct AmountUniverse {
  std::vector<i64> vals;
  void build(const Workload& w) {
    vals.reserve(size_t(w.cfg.rows) + size_t(w.n_ins) + size_t(w.cfg.states));
    for (i64 a : w.b_amt) vals.push_back(a);
    for (i64 a : w.ins_amt) vals.push_back(a);
    for (i64 v = 1; v < w.cfg.states; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if (e.op == OP_REPLACE || e.op == OP_INSERT || e.op == OP_MOVE) vals.push_back(e.val);
    }
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
  }
  inline i32 rank(i64 a) const {
    return i32(std::lower_bound(vals.begin(), vals.end(), a) - vals.begin());
  }
};

// ---------------------------------------------------------------- HROS-F (proposed)
struct HrosFactor {
  const Workload& w;
  i32 G = 0, U = 0;
  AmountUniverse au;
  std::vector<i64> fac;          // current group factors (actual values, not indices)
  std::vector<i32> fac_cur_idx;  // current index into w.fac_val (for snapshot)
  // per-group Fenwick over amount ranks
  std::vector<std::vector<i32>> fen;   // fen[g] size U+1
  // per-group amount multiplicity map: exact distinct bookkeeping (pair-distinct)
  std::vector<std::unordered_map<i64, i32>> amount_count;
  std::vector<i64> gsum;               // per-group sum of raw amounts
  std::vector<i32> gdistinct;          // per-group distinct amounts
  i64 total_sum = 0;                   // sum of weighted values
  i64 n_active = 0;
  i64 total_distinct = 0;              // sum of per-group distinct (pair-distinct)
  // rows
  i64 n_rows = 0;
  std::vector<i32> row_g, row_rank;
  std::vector<i64> row_amt;
  std::vector<u8> row_active;
  // snapshot stacks (per version)
  struct Saved { u8 kind; u8 was_active; i32 g, rank; i64 amt; i64 fac; };
  std::vector<Saved> saved;

  explicit HrosFactor(const Workload& wo) : w(wo) {}
  void build() {
    G = w.cfg.groups; au.build(w); U = i32(au.vals.size());
    fac.assign(size_t(G), 1000); fac_cur_idx.assign(size_t(G), 0);
    fen.assign(size_t(G), {});
    amount_count.assign(size_t(G), {});
    for (i32 g = 0; g < G; ++g) fen[size_t(g)].assign(size_t(U) + 1, 0);
    gsum.assign(size_t(G), 0); gdistinct.assign(size_t(G), 0);
    n_rows = w.cfg.rows + w.n_ins;
    row_g.assign(size_t(n_rows), 0); row_rank.assign(size_t(n_rows), 0);
    row_amt.assign(size_t(n_rows), 0); row_active.assign(size_t(n_rows), 0);
    saved.assign(size_t(w.cfg.states), Saved{});
    for (i64 i = 0; i < w.cfg.rows; ++i) {
      row_g[size_t(i)] = w.b_g[size_t(i)];
      row_amt[size_t(i)] = w.b_amt[size_t(i)];
      row_rank[size_t(i)] = au.rank(row_amt[size_t(i)]);
      add_row(i32(i), 1);
    }
  }
  inline void fen_add(i32 g, i32 r, i32 d) {
    auto& f = fen[size_t(g)];
    for (i32 i = r + 1; i <= U; i += i & -i) f[size_t(i)] += d;
  }
  inline i32 fen_pref(i32 g, i32 r_excl) const {
    i32 s = 0;
    const auto& f = fen[size_t(g)];
    for (i32 i = r_excl; i > 0; i -= i & -i) s += f[size_t(i)];
    return s;
  }
  inline void activate(i32 g, i32 r, i64 amt) {
    fen_add(g, r, +1);
    gsum[size_t(g)] += amt;
    ++n_active;
    auto& m = amount_count[size_t(g)];
    if (m[amt]++ == 0) { ++gdistinct[size_t(g)]; ++total_distinct; }
  }
  inline void deactivate(i32 g, i32 r, i64 amt) {
    fen_add(g, r, -1);
    gsum[size_t(g)] -= amt;
    --n_active;
    auto& m = amount_count[size_t(g)];
    auto it = m.find(amt);
    if (--it->second == 0) { m.erase(it); --gdistinct[size_t(g)]; --total_distinct; }
  }
  inline void add_row(i32 id, i32 on) {
    const size_t i = size_t(id);
    if (!on) { if (row_active[i]) { deactivate(row_g[i], row_rank[i], row_amt[i]); row_active[i] = 0; } return; }
    if (!row_active[i]) { activate(row_g[i], row_rank[i], row_amt[i]); row_active[i] = 1; }
  }
  // set the row to (g, amt) and to active/inactive (REPLACE/MOVE restore existence, as in the
  // shipped edit semantics; DELETE drops it; INSERT inserts a row that starts inactive and is
  // then activated)
  inline void set_row(i32 id, i32 g, i64 amt, i32 on) {
    const size_t i = size_t(id);
    if (row_active[i]) deactivate(row_g[i], row_rank[i], row_amt[i]);
    row_g[i] = g; row_amt[i] = amt; row_rank[i] = au.rank(amt);
    if (on) { activate(row_g[i], row_rank[i], row_amt[i]); row_active[i] = 1; }
    else row_active[i] = 0;
  }
  inline i64 weighted_sum() const {
    i64 s = 0;
    for (i32 g = 0; g < G; ++g) s += fac[size_t(g)] * gsum[size_t(g)];
    return s;
  }
  // count of active rows whose weighted value <= x
  inline i64 count_weighted_le(i64 x) const {
    i64 s = 0;
    for (i32 g = 0; g < G; ++g) {
      const i64 f = fac[size_t(g)];
      // largest amount a with a*f <= x  <=>  a <= floor(x/f)
      i64 q = (x >= 0) ? (x / f) : -(((-x) + f - 1) / f);
      const i32 r = i32(std::upper_bound(au.vals.begin(), au.vals.end(), q) - au.vals.begin());
      s += fen_pref(g, r);
    }
    return s;
  }
  i64 median_weighted() const {
    if (n_active == 0) return 0;
    const i64 k = (n_active - 1) / 2 + 1;  // lower median, 1-based
    // search bounds: all weighted values lie in [min_amt*min_f, max_amt*max_f]
    i64 lo = au.vals.front() * fac[size_t(0)], hi = au.vals.back() * fac[size_t(0)];
    for (i32 g = 1; g < G; ++g) {
      lo = std::min(lo, au.vals.front() * fac[size_t(g)]);
      hi = std::max(hi, au.vals.back() * fac[size_t(g)]);
    }
    if (lo >= hi) return lo;
    // invariant: count(<=lo) < k <= count(<=hi); binary search the minimal x with count>=k
    i64 a = lo, b = hi;
    while (a < b) {
      const i64 mid = a + (b - a) / 2;
      if (count_weighted_le(mid) >= k) b = mid; else a = mid + 1;
    }
    return a;
  }
  // ---- reversible edit application
  void snapshot(i32 v) {
    const Edit& e = w.edit[size_t(v)];
    Saved s{};
    if (e.op == OP_FACTOR) { s.kind = 1; s.g = e.g; s.fac = fac[size_t(e.g)]; }
    else {
      s.kind = 0;
      s.was_active = row_active[size_t(e.id)];
      s.g = row_g[size_t(e.id)]; s.rank = row_rank[size_t(e.id)]; s.amt = row_amt[size_t(e.id)];
    }
    saved[size_t(v)] = s;
  }
  void apply(i32 v) {
    const Edit& e = w.edit[size_t(v)];
    switch (e.op) {
      case OP_FACTOR: fac[size_t(e.g)] = e.val; break;
      case OP_REPLACE: set_row(i32(e.id), row_g[size_t(e.id)], e.val, 1); break;
      case OP_INSERT: set_row(i32(e.id), e.g, e.val, 1); break;
      case OP_DELETE: set_row(i32(e.id), row_g[size_t(e.id)], row_amt[size_t(e.id)], 0); break;
      case OP_MOVE: set_row(i32(e.id), e.g, e.val, 1); break;
    }
  }
  void undo(i32 v) {
    const Edit& e = w.edit[size_t(v)];
    const Saved& s = saved[size_t(v)];
    if (e.op == OP_FACTOR) { fac[size_t(e.g)] = s.fac; return; }
    if (e.op == OP_INSERT) { add_row(i32(e.id), 0); return; }
    const size_t i = size_t(e.id);
    if (row_active[i]) deactivate(row_g[i], row_rank[i], row_amt[i]);
    row_g[i] = s.g; row_amt[i] = s.amt; row_rank[i] = s.rank;
    if (s.was_active) { activate(row_g[i], row_rank[i], row_amt[i]); row_active[i] = 1; }
    else row_active[i] = 0;
  }
  // ---- query: (count, weighted sum, weighted lower median, pair-distinct)
  void query(i64* out) const {
    out[0] = n_active;
    out[1] = weighted_sum();
    out[2] = median_weighted();
    out[3] = total_distinct;
  }
  i64 static_bytes() const {
    return i64(G) * (i64(U) + 1) * i64(sizeof(i32)) +
           i64(G) * i64(U) * i64(8) +   // amount multiplicity maps (upper bound of the std bucket array)
           i64(n_rows) * i64(sizeof(i32) * 2 + sizeof(i64) * 2);
  }
};

}  // namespace hf
