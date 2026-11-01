// panel_hros_factor.cpp — worker for the factor-weighted holistic-statistics comparison (2026-11-01).
//
// New file only (results/hros_factor_20261101/); shipped engine/panel/paper untouched.
// Modes:
//   hros     HROS-F v2: per-group amount Fenwicks, O(1)-or-deferred factor edits, weighted median
//            by binary search (hros_factor_index_v2.hpp); --variant eager|skip|lazy selects the
//            factor-assignment path, --burst K inserts K-1 extra consecutive assignments before
//            each declared factor edit to exercise coalescing.
//   fenwick  Fenwick-only baseline that MATERIALISES weighted values in one global order-statistic
//            Fenwick; a factor edit walks every active row of the group and moves its weighted value
//   oracle   brute-force reference: per query, collect the current weighted values and sort
// All three answer the same tuple (count, weighted sum, weighted lower median, pair-distinct) over
// the GLOBAL current relation (no 1D range in this round; ranges are orthogonal and compose with
// results/hros_20261101). Answers are written as little-endian int64 tuples in (v,k) order and
// compared by sha256.
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "hros_factor_index_v2.hpp"

namespace bwh { using u8 = uint8_t; using u32 = uint32_t; using u64 = uint64_t; }
#include "sha256.hpp"

using namespace hf;

static i64 ns_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

// ---------------------------------------------------------------- generic mode interface
struct Mode {
  const Workload& w;
  explicit Mode(const Workload& wo) : w(wo) {}
  virtual ~Mode() = default;
  virtual const char* name() const = 0;
  virtual void build() = 0;
  virtual void snapshot(i32 v) = 0;
  virtual void apply(i32 v) = 0;
  virtual void undo(i32 v) = 0;
  virtual void query(i64* out) = 0;
  virtual void set_variant(const std::string& v) { variant = v; }
  virtual void factor_extra(i32, i64) = 0;
  virtual void counters(i64* out) const { out[0]=out[1]=out[2]=out[3]=0; }
  virtual i64 static_bytes() const { return 0; }
  std::string variant = "eager";
};

struct SavedRow {
  u8 kind;  // 0 row, 1 factor
  u8 was_active;
  i32 g, rank;
  i64 amt, fac;
  i64 pend = 0;
  u8 pend_dirty = 0;
};

// ---------------------------------------------------------------- HROS-F
struct HrosMode : Mode {
  HrosFactor h;
  std::vector<SavedRow> saved;
  explicit HrosMode(const Workload& wo) : Mode(wo), h(wo) {}
  const char* name() const override { return "hros"; }
  void build() override {
    h.skip_noop = (variant != "eager");
    h.lazy = (variant == "lazy");
    h.build(); saved.assign(size_t(w.cfg.states), SavedRow{});
  }
  void factor_extra(i32 g, i64 val) override { h.assign_factor(g, val); }
  void counters(i64* out) const override { out[0]=h.skipped_same; out[1]=h.skipped_empty; out[2]=h.deferred_assign; out[3]=h.lazy_flushes; }
  void snapshot(i32 v) override { h.snapshot(v); }
  void apply(i32 v) override { h.apply(v); }
  void undo(i32 v) override { h.undo(v); }
  void query(i64* out) override { h.query(out); }
  i64 static_bytes() const override { return h.static_bytes(); }
};

// ---------------------------------------------------------------- materialising Fenwick
struct FenwickMode : Mode {
  const Workload& w;
  Config cfg;
  AmountUniverse au;
  std::vector<i64> wvals;                 // sorted distinct weighted values
  std::vector<i32> bit;                  // Fenwick counts over wvals
  std::vector<i64> fac;                  // current factors (actual values)
  std::vector<i64> gsum;                 // per-group raw amount sum
  std::vector<i32> gdistinct;            // per-group distinct amounts
  std::vector<std::unordered_map<i64, i32>> amount_count;  // per-group amount multiplicity
  std::vector<u8> active;
  std::vector<i32> rg;
  std::vector<i64> ramt;
  i64 n_active = 0, total_sum = 0, total_distinct = 0;
  std::vector<i32> next_id, prev_id, ghead;  // per-group active row list
  std::vector<SavedRow> saved;
  i64 row_edits = 0, factor_row_updates = 0;
  // v2 skip / deferral
  bool skip_noop = true, lazy = false;
  std::vector<i64> pend_fac; std::vector<u8> pend_dirty;
  i64 skipped_same = 0, skipped_empty = 0, deferred_assign = 0, lazy_flushes = 0;
  i64 gactive_cnt = 0; std::vector<i32> gactive;

  explicit FenwickMode(const Workload& wo) : Mode(wo), w(wo), cfg(wo.cfg) {}
  const char* name() const override { return "fenwick"; }
  i64 wrank(i64 val) const { return std::lower_bound(wvals.begin(), wvals.end(), val) - wvals.begin(); }
  void bit_add(i32 r, i32 d) {
    for (i32 i = r + 1; i <= i32(wvals.size()); i += i & -i) bit[size_t(i)] += d;
  }
  void build() override {
    au.build(w);
    std::vector<i64> wv;
    wv.reserve(au.vals.size() * w.fac_val.size());
    for (i64 a : au.vals) for (i64 f : w.fac_val) wv.push_back(a * f);
    std::sort(wv.begin(), wv.end());
    wv.erase(std::unique(wv.begin(), wv.end()), wv.end());
    wvals = std::move(wv);
    bit.assign(wvals.size() + 1, 0);
    fac.assign(size_t(cfg.groups), 1000);
    gsum.assign(size_t(cfg.groups), 0);
    gdistinct.assign(size_t(cfg.groups), 0);
    amount_count.assign(size_t(cfg.groups), {});
    const i64 n_rows = cfg.rows + w.n_ins;
    active.assign(size_t(n_rows), 0); rg.assign(size_t(n_rows), 0); ramt.assign(size_t(n_rows), 0);
    next_id.assign(size_t(n_rows), -1); prev_id.assign(size_t(n_rows), -1);
    ghead.assign(size_t(cfg.groups), -1);
    saved.assign(size_t(cfg.states), SavedRow{});
    pend_fac.assign(size_t(cfg.groups), 0); pend_dirty.assign(size_t(cfg.groups), 0);
    gactive.assign(size_t(cfg.groups), 0);
    for (i64 i = 0; i < cfg.rows; ++i) { rg[size_t(i)] = w.b_g[size_t(i)]; ramt[size_t(i)] = w.b_amt[size_t(i)]; activate(i32(i)); }
  }
  void list_remove(i32 id) {
    const i32 p = prev_id[size_t(id)], nx = next_id[size_t(id)];
    if (p >= 0) next_id[size_t(p)] = nx; else ghead[size_t(rg[size_t(id)])] = nx;
    if (nx >= 0) prev_id[size_t(nx)] = p;
    prev_id[size_t(id)] = next_id[size_t(id)] = -1;
  }
  void list_insert_front(i32 id) {
    const i32 h = ghead[size_t(rg[size_t(id)])];
    prev_id[size_t(id)] = -1; next_id[size_t(id)] = h;
    if (h >= 0) prev_id[size_t(h)] = id;
    ghead[size_t(rg[size_t(id)])] = id;
  }
  void activate(i32 id) {
    active[size_t(id)] = 1;
    bit_add(i32(wrank(ramt[size_t(id)] * fac[size_t(rg[size_t(id)])])), +1);
    ++n_active;
    total_sum += ramt[size_t(id)] * fac[size_t(rg[size_t(id)])];
    gsum[size_t(rg[size_t(id)])] += ramt[size_t(id)];
    ++gactive[size_t(rg[size_t(id)])];
    auto& m = amount_count[size_t(rg[size_t(id)])];
    if (m[ramt[size_t(id)]]++ == 0) { ++gdistinct[size_t(rg[size_t(id)])]; ++total_distinct; }
    list_insert_front(id);
  }
  void deactivate(i32 id) {
    active[size_t(id)] = 0;
    bit_add(i32(wrank(ramt[size_t(id)] * fac[size_t(rg[size_t(id)])])), -1);
    --n_active;
    total_sum -= ramt[size_t(id)] * fac[size_t(rg[size_t(id)])];
    gsum[size_t(rg[size_t(id)])] -= ramt[size_t(id)];
    --gactive[size_t(rg[size_t(id)])];
    auto& m = amount_count[size_t(rg[size_t(id)])];
    auto it = m.find(ramt[size_t(id)]);
    if (--it->second == 0) { m.erase(it); --gdistinct[size_t(rg[size_t(id)])]; --total_distinct; }
    list_remove(id);
  }
  void set_row(i32 id, i32 g, i64 amt, i32 on) {
    if (active[size_t(id)]) deactivate(id);
    rg[size_t(id)] = g; ramt[size_t(id)] = amt;
    if (on) activate(id);
  }
  inline void materialise_factor(i32 g, i64 new_f) {
    const i64 old_f = fac[size_t(g)];
    for (i32 id = ghead[size_t(g)]; id >= 0; id = next_id[size_t(id)]) {
      const i64 a = ramt[size_t(id)];
      bit_add(i32(wrank(a * old_f)), -1);
      bit_add(i32(wrank(a * new_f)), +1);
      total_sum += a * (new_f - old_f);
      ++factor_row_updates;
    }
    fac[size_t(g)] = new_f;
  }
  inline void factor_assign(i32 g, i64 val) {
    if (skip_noop) {
      const i64 cur = pend_dirty[size_t(g)] ? pend_fac[size_t(g)] : fac[size_t(g)];
      if (val == cur) { ++skipped_same; return; }
      if (gactive[size_t(g)] == 0) {
        ++skipped_empty;
        if (lazy) { pend_fac[size_t(g)] = val; pend_dirty[size_t(g)] = 1; ++deferred_assign; }
        else fac[size_t(g)] = val;
        return;
      }
    } else if (gactive[size_t(g)] == 0) ++skipped_empty;
    if (lazy) { pend_fac[size_t(g)] = val; pend_dirty[size_t(g)] = 1; ++deferred_assign; return; }
    materialise_factor(g, val);
  }
  inline void flush_pending() {
    if (!lazy) return;
    for (i32 g = 0; g < cfg.groups; ++g)
      if (pend_dirty[size_t(g)]) { materialise_factor(g, pend_fac[size_t(g)]); pend_dirty[size_t(g)] = 0; }
    ++lazy_flushes;
  }
  void set_variant(const std::string& v) override {
    variant = v; skip_noop = (v != "eager"); lazy = (v == "lazy");
  }
  void factor_extra(i32 g, i64 val) override { factor_assign(g, val); }
  void counters(i64* out) const override { out[0]=skipped_same; out[1]=skipped_empty; out[2]=deferred_assign; out[3]=lazy_flushes; }
  void snapshot(i32 v) override {
    const Edit& e = w.edit[size_t(v)];
    SavedRow s{};
    if (e.op == OP_FACTOR) { s.kind = 1; s.g = e.g; s.fac = fac[size_t(e.g)]; s.pend = pend_fac[size_t(e.g)]; s.pend_dirty = pend_dirty[size_t(e.g)]; }
    else {
      s.kind = 0; s.was_active = active[size_t(e.id)];
      s.g = rg[size_t(e.id)]; s.amt = ramt[size_t(e.id)];
    }
    saved[size_t(v)] = s;
  }
  void apply(i32 v) override {
    const Edit& e = w.edit[size_t(v)];
    switch (e.op) {
      case OP_FACTOR: factor_assign(e.g, e.val); break;
      case OP_REPLACE: set_row(i32(e.id), rg[size_t(e.id)], e.val, 1); ++row_edits; break;
      case OP_INSERT: set_row(i32(e.id), e.g, e.val, 1); ++row_edits; break;
      case OP_DELETE: set_row(i32(e.id), rg[size_t(e.id)], ramt[size_t(e.id)], 0); ++row_edits; break;
      case OP_MOVE: set_row(i32(e.id), e.g, e.val, 1); ++row_edits; break;
    }
  }
  void undo(i32 v) override {
    const Edit& e = w.edit[size_t(v)];
    const SavedRow& s = saved[size_t(v)];
    if (e.op == OP_FACTOR) {
      const i32 g = e.g;
      const i64 new_f = fac[size_t(g)], old_f = s.fac;
      for (i32 id = ghead[size_t(g)]; id >= 0; id = next_id[size_t(id)]) {
        const i64 a = ramt[size_t(id)];
        bit_add(i32(wrank(a * new_f)), -1);
        bit_add(i32(wrank(a * old_f)), +1);
        total_sum += a * (old_f - new_f);
      }
      fac[size_t(g)] = old_f;
      pend_fac[size_t(g)] = s.pend; pend_dirty[size_t(g)] = s.pend_dirty;
      return;
    }
    if (e.op == OP_INSERT) { set_row(i32(e.id), rg[size_t(e.id)], ramt[size_t(e.id)], 0); return; }
    if (active[size_t(e.id)]) deactivate(i32(e.id));
    rg[size_t(e.id)] = s.g; ramt[size_t(e.id)] = s.amt;
    if (s.was_active) activate(i32(e.id));
  }
  i64 kth_weighted(i64 k) const {  // 1-based k-th smallest weighted value, Fenwick binary lifting
    if (n_active == 0) return 0;
    i32 pos = 0; i32 rem = i32(k);
    const i32 N = i32(wvals.size());
    for (i32 pw = 1; pw <= N; pw <<= 1) {}
    for (i32 pw = (1 << 30); pw > 0; pw >>= 1) {
      if (pos + pw <= N && bit[size_t(pos + pw)] < rem) { pos += pw; rem -= bit[size_t(pos)]; }
    }
    return wvals[size_t(pos)];  // pos == number of values with count fully below; next value is pos
  }
  void query(i64* out) override {
    flush_pending();
    out[0] = n_active;
    out[1] = total_sum;
    out[2] = (n_active == 0) ? 0 : kth_weighted((n_active - 1) / 2 + 1);
    out[3] = total_distinct;
  }
  i64 static_bytes() const override {
    return i64(wvals.size() + 1) * i64(sizeof(i32)) + i64(cfg.rows + w.n_ins) * 24 +
           i64(cfg.groups) * i64(au.vals.size()) * i64(sizeof(i64)) +
           i64(cfg.groups) * i64(au.vals.size()) * i64(sizeof(i32) * 2);
  }
};

// ---------------------------------------------------------------- brute-force oracle
struct OracleMode : Mode {
  const Workload& w;
  Config cfg;
  std::vector<i64> fac;
  std::vector<u8> active;
  std::vector<i32> rg;
  std::vector<i64> ramt;
  std::vector<SavedRow> saved;
  i64 extra_assigns = 0;
  explicit OracleMode(const Workload& wo) : Mode(wo), w(wo), cfg(wo.cfg) {}
  const char* name() const override { return "oracle"; }
  void build() override {
    fac.assign(size_t(cfg.groups), 1000);
    const i64 n = cfg.rows + w.n_ins;
    active.assign(size_t(n), 0); rg.assign(size_t(n), 0); ramt.assign(size_t(n), 0);
    saved.assign(size_t(cfg.states), SavedRow{});
    for (i64 i = 0; i < cfg.rows; ++i) { active[size_t(i)] = 1; rg[size_t(i)] = w.b_g[size_t(i)]; ramt[size_t(i)] = w.b_amt[size_t(i)]; }
  }
  void set_row(i32 id, i32 g, i64 amt, i32 on) { rg[size_t(id)] = g; ramt[size_t(id)] = amt; active[size_t(id)] = (u8)on; }
  void factor_extra(i32 g, i64 val) override { fac[size_t(g)] = val; ++extra_assigns; }
  void counters(i64* out) const override { out[0]=out[1]=out[2]=out[3]=0; out[3]=extra_assigns; }
  void snapshot(i32 v) override {
    const Edit& e = w.edit[size_t(v)];
    SavedRow s{};
    if (e.op == OP_FACTOR) { s.kind = 1; s.g = e.g; s.fac = fac[size_t(e.g)]; }
    else { s.kind = 0; s.was_active = active[size_t(e.id)]; s.g = rg[size_t(e.id)]; s.amt = ramt[size_t(e.id)]; }
    saved[size_t(v)] = s;
  }
  void apply(i32 v) override {
    const Edit& e = w.edit[size_t(v)];
    switch (e.op) {
      case OP_FACTOR: fac[size_t(e.g)] = e.val; break;
      case OP_REPLACE: set_row(i32(e.id), rg[size_t(e.id)], e.val, 1); break;
      case OP_INSERT: set_row(i32(e.id), e.g, e.val, 1); break;
      case OP_DELETE: set_row(i32(e.id), rg[size_t(e.id)], ramt[size_t(e.id)], 0); break;
      case OP_MOVE: set_row(i32(e.id), e.g, e.val, 1); break;
    }
  }
  void undo(i32 v) override {
    const Edit& e = w.edit[size_t(v)];
    const SavedRow& s = saved[size_t(v)];
    if (e.op == OP_FACTOR) { fac[size_t(e.g)] = s.fac; return; }
    set_row(i32(e.id), s.g, s.amt, s.was_active);
  }
  void query(i64* out) override {
    std::vector<i64> wv;
    std::vector<std::pair<i32, i64>> pairs;
    wv.reserve(size_t(active.size()));
    for (size_t i = 0; i < active.size(); ++i) {
      if (!active[i]) continue;
      wv.push_back(ramt[i] * fac[size_t(rg[i])]);
      pairs.emplace_back(rg[i], ramt[i]);
    }
    i64 sum = 0;
    for (i64 x : wv) sum += x;
    std::sort(wv.begin(), wv.end());
    const i64 med = wv.empty() ? 0 : wv[size_t((i64(wv.size()) - 1) / 2)];
    std::sort(pairs.begin(), pairs.end());
    i64 dis = 0;
    for (size_t i = 0; i < pairs.size(); ++i) if (i == 0 || pairs[i] != pairs[i - 1]) ++dis;
    out[0] = i64(wv.size()); out[1] = sum; out[2] = med; out[3] = dis;
  }
};

int main(int argc, char** argv) {
  std::string mode = "hros", answers_path, json_path, variant = "eager";
  Config cfg;
  i32 repeat = 0, max_v = -1, burst = 1;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](const char* k) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k); std::exit(2); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--mode")) mode = next(argv[i]);
    else if (!std::strcmp(argv[i], "--rows")) cfg.rows = std::atoll(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--groups")) cfg.groups = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--states")) cfg.states = std::atoll(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--q")) cfg.q = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--factor-period")) cfg.factor_period = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--topology")) cfg.topology = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--seed")) cfg.seed = std::atoll(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--repeat")) repeat = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--max-v")) max_v = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--variant")) variant = next(argv[i]);
    else if (!std::strcmp(argv[i], "--burst")) burst = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--answers")) answers_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--json")) json_path = next(argv[i]);
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  if (json_path.empty()) { std::fprintf(stderr, "usage: panel_hros_factor --mode hros|fenwick|oracle ... --json J\n"); return 2; }
  const i64 V = cfg.states;
  const i32 last_v = (max_v < 0 || max_v >= V - 1) ? i32(V - 1) : max_v;
  const auto t0 = std::chrono::steady_clock::now();
  Workload w = make_workload(cfg);
  const auto t1 = std::chrono::steady_clock::now();
  Mode* m = nullptr;
  if (mode == "hros") m = new HrosMode(w);
  else if (mode == "fenwick") m = new FenwickMode(w);
  else if (mode == "oracle") m = new OracleMode(w);
  else { std::fprintf(stderr, "unknown mode\n"); return 3; }
  m->set_variant(variant);
  m->build();
  const auto t2 = std::chrono::steady_clock::now();
  std::vector<i64> answers(size_t(last_v) * size_t(cfg.q) * 4, 0);
  i64 update_ns = 0, query_ns = 0, n_factor_edits = 0, n_extra_assigns = 0;
  auto apply_one = [&](i32 v) {
    const Edit& e = w.edit[size_t(v)];
    const auto a0 = std::chrono::steady_clock::now();
    if (e.op == OP_FACTOR) {
      ++n_factor_edits;
      for (i32 j = 1; j < burst; ++j) {
        const i32 idx = 1 + ((j + i32(e.g)) % 2);
        m->factor_extra(e.g, w.fac_val[size_t(idx)]);
        ++n_extra_assigns;
      }
    }
    m->apply(v);
    update_ns += ns_between(a0, std::chrono::steady_clock::now());
  };
  auto query_one = [&](i32 v) {
    const auto q0 = std::chrono::steady_clock::now();
    for (i32 k = 0; k < cfg.q; ++k) m->query(&answers[(size_t(v - 1) * size_t(cfg.q) + size_t(k)) * 4]);
    query_ns += ns_between(q0, std::chrono::steady_clock::now());
  };
  std::vector<i32> kids;
  if (cfg.topology != 0) {
    kids.assign(size_t(V + 1), -1);
    std::vector<i32> first(size_t(V), -1), nxt(size_t(V), -1);
    for (i64 v = 1; v < V; ++v) {
      const i32 p = w.parent[size_t(v)];
      nxt[size_t(v)] = first[size_t(p)];
      first[size_t(p)] = i32(v);
    }
    // reuse first/nxt arrays for DFS via explicit stack (below)
    std::vector<i32> stack;
    stack.push_back(0);
    std::vector<u8> exit_flag(size_t(V), 0);
    // simple iterative preorder with (v, exit) encoded in two stacks
    std::vector<std::pair<i32, u8>> st;
    st.emplace_back(0, 0);
    while (!st.empty()) {
      auto [v, ex] = st.back(); st.pop_back();
      if (ex) { if (v > 0) m->undo(v); continue; }
      if (v > 0 && v <= last_v) {
        m->snapshot(v);
        apply_one(v);
        query_one(v);
      } else if (v > 0 && v > last_v) continue;
      st.emplace_back(v, 1);
      for (i32 c = first[size_t(v)]; c >= 0; c = nxt[size_t(c)]) st.emplace_back(c, 0);
    }
    (void)kids; (void)exit_flag; (void)stack;
  } else {
    for (i32 v = 1; v <= last_v; ++v) {
      apply_one(v);
      query_one(v);
    }
  }
  const auto t3 = std::chrono::steady_clock::now();
  const u64 cks = 1469598103934665603ULL;
  u64 fnv = cks;
  {
    const u8* p = reinterpret_cast<const u8*>(answers.data());
    for (size_t i = 0; i < answers.size() * sizeof(i64); ++i) fnv = (fnv ^ p[i]) * 1099511628211ULL;
  }
  const std::string sha = bwh::sha::of(answers.data(), answers.size() * sizeof(i64));
  if (!answers_path.empty()) {
    FILE* f = std::fopen(answers_path.c_str(), "wb");
    if (f) { std::fwrite(answers.data(), sizeof(i64), answers.size(), f); std::fclose(f); }
  }
  const auto t4 = std::chrono::steady_clock::now();
  const i64 static_bytes = m->static_bytes();
  i64 cnt[4] = {0,0,0,0};
  m->counters(cnt);
  delete m;
  const auto t5 = std::chrono::steady_clock::now();
  i64 peak_kb = 0;
  {
    FILE* s = std::fopen("/proc/self/status", "r");
    if (s) { char line[256]; while (std::fgets(line, sizeof(line), s)) { long v = 0; if (std::sscanf(line, "VmHWM: %ld kB", &v) == 1) { peak_kb = v; break; } } std::fclose(s); }
  }
  const i64 gen_ns = ns_between(t0, t1), build_ns = ns_between(t1, t2);
  const i64 exec_ns = ns_between(t2, t3), extract_ns = ns_between(t3, t4), cleanup_ns = ns_between(t4, t5);
  const i64 full_ns = gen_ns + build_ns + exec_ns + extract_ns + cleanup_ns;
  FILE* jf = std::fopen((json_path + ".tmp").c_str(), "w");
  if (!jf) return 4;
  std::fprintf(jf, "{\n  \"mode\": \"%s\",\n  \"rows\": %" PRId64 ", \"groups\": %d, \"states\": %" PRId64
                   ", \"q\": %d, \"factor_period\": %d, \"topology\": %d, \"seed\": %" PRId64 ",\n",
               mode.c_str(), cfg.rows, cfg.groups, cfg.states, cfg.q, cfg.factor_period, cfg.topology, cfg.seed);
  std::fprintf(jf, "  \"repeat\": %d, \"max_v\": %d,\n  \"variant\": \"%s\", \"burst\": %d,\n  \"n_answers\": %zu,\n",
               repeat, last_v, variant.c_str(), burst, answers.size());
  std::fprintf(jf, "  \"counters\": {\"skipped_same\": %" PRId64 ", \"skipped_empty\": %" PRId64
                   ", \"deferred_assign\": %" PRId64 ", \"lazy_flushes\": %" PRId64
                   ", \"factor_edits\": %" PRId64 ", \"extra_assigns\": %" PRId64 "},\n",
               cnt[0], cnt[1], cnt[2], cnt[3], n_factor_edits, n_extra_assigns);
  std::fprintf(jf, "  \"split_ns\": {\"update\": %" PRId64 ", \"query\": %" PRId64 "},\n", update_ns, query_ns);
  std::fprintf(jf, "  \"answers_fn\": \"%016" PRIx64 "\",\n  \"answers_sha256\": \"%s\",\n", fnv, sha.c_str());
  std::fprintf(jf, "  \"phases_ns\": {\"gen\": %" PRId64 ", \"build\": %" PRId64 ", \"execute\": %" PRId64
                   ", \"extract\": %" PRId64 ", \"cleanup\": %" PRId64 "},\n", gen_ns, build_ns, exec_ns, extract_ns, cleanup_ns);
  std::fprintf(jf, "  \"full_ns\": %" PRId64 ",\n  \"peak_rss_kb\": %" PRId64 ",\n  \"static_bytes\": %" PRId64 "\n}\n",
               full_ns, peak_kb, static_bytes);
  std::fclose(jf);
  std::rename((json_path + ".tmp").c_str(), json_path.c_str());
  std::printf("mode=%s rows=%" PRId64 " G=%d V=%" PRId64 " q=%d fp=%d topo=%d full_ns=%" PRId64 " rss_kb=%" PRId64 "\n",
              mode.c_str(), cfg.rows, cfg.groups, cfg.states, cfg.q, cfg.factor_period, cfg.topology, full_ns, peak_kb);
  return 0;
}
