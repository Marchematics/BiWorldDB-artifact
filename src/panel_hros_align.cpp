// panel_hros_align.cpp — HROS alignment worker: HROS vs a Fenwick-only backend vs an
// augmented-AVL-tree backend, on the identical holistic order-statistics contract.
// (2026-11-01, P0 HROS alignment; new file under results/hros_align_20261101/.)
//
// Same CLI / phase accounting / answer-serialization contract as panel_hros.cpp:
//   --mode hros     the HROS prototype (reversible value-rank treap + presence Fenwick)
//   --mode fenwick  Fenwick-only: value-rank cnt/sum Fenwicks + presence Fenwick for distinct;
//                   lower median by Fenwick binary lifting (no balanced tree)
//   --mode avl      augmented AVL tree (independent implementation, own rotations) with
//                   subtree count/sum/distinct-node metadata (no presence Fenwick)
//   --mode rebuild  independent brute-force reference (scan + sort per query; correctness)
// Query tuple: (count, sum, lower_median, exact_distinct) over a 1D measure range [lo,hi).
// All backends get the same deterministic workload, edits, undo path and query ranges.
//
// Phases: load, setup (index build), execute (DFS apply/undo + queries; update_ns/query_ns
// sub-counters), extract (answer vector + fnv1a + sha256), cleanup.  peak RSS = VmHWM.
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <sys/resource.h>

#include "common.hpp"
#include "hros_index.hpp"
#include "rank_backends.hpp"
#include "sha256.hpp"

using namespace bwh;
using namespace bwh::hros;

static i64 ns_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

// ---------------------------------------------------------------- uniform backend interface
struct Backend {
  virtual ~Backend() = default;
  virtual void build() = 0;
  virtual void apply(i32 v) = 0;
  virtual void undo(i32 v) = 0;
  virtual void query(i64 lo, i64 hi, i64* out) = 0;
  virtual i32 sigma() const = 0;
  virtual i32 n_rows() const = 0;
  virtual i64 updates() const = 0;
};

struct HrosAdapter : Backend {
  HrosIndex h;
  HrosAdapter(const Fixture& f, const Workload& w) : h(f, w) {}
  void build() override { h.build(); }
  void apply(i32 v) override { h.apply(v); }
  void undo(i32 v) override { h.undo(v); }
  void query(i64 lo, i64 hi, i64* out) override { h.query(lo, hi, out); }
  i32 sigma() const override { return h.sigma; }
  i32 n_rows() const override { return h.n_rows; }
  i64 updates() const override { return h.updates; }
};

struct FenAdapter : Backend {
  FenwickBackend b;
  FenAdapter(const Fixture& f, const Workload& w) : b(f, w) {}
  void build() override { b.build(); }
  void apply(i32 v) override { b.apply(v); }
  void undo(i32 v) override { b.undo(v); }
  void query(i64 lo, i64 hi, i64* out) override { b.query(lo, hi, out); }
  i32 sigma() const override { return b.sigma; }
  i32 n_rows() const override { return b.n_rows; }
  i64 updates() const override { return b.updates; }
};

struct AvlAdapter : Backend {
  AvlBackend b;
  AvlAdapter(const Fixture& f, const Workload& w) : b(f, w) {}
  void build() override { b.build(); }
  void apply(i32 v) override { b.apply(v); }
  void undo(i32 v) override { b.undo(v); }
  void query(i64 lo, i64 hi, i64* out) override { b.query(lo, hi, out); }
  i32 sigma() const override { return b.sigma; }
  i32 n_rows() const override { return b.n_rows; }
  i64 updates() const override { return b.updates; }
};

// brute-force reference: scan the current relation per query and sort the matching values
struct RebuildAdapter : Backend {
  const Fixture& f;
  const Workload& w;
  i32 n_rows_ = 0;
  std::vector<u8> row_active;
  std::vector<i64> row_value;
  std::vector<u8> saved_active;
  std::vector<i64> saved_value;
  std::vector<i64> scratch;
  i64 updates_ = 0;
  RebuildAdapter(const Fixture& fi, const Workload& wo) : f(fi), w(wo) {}
  void build() override {
    i32 max_id = i32(f.h.n);
    for (i32 v = 1; v < kStates; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if (e.op == OP_REPLACE || e.op == OP_DELETE || e.op == OP_MOVE || e.op == OP_INSERT) {
        if (e.id + 1 > i64(max_id)) max_id = i32(e.id) + 1;
      }
    }
    n_rows_ = max_id + 1;
    row_active.assign(size_t(n_rows_), 0);
    row_value.assign(size_t(n_rows_), 0);
    saved_active.assign(size_t(kStates), 0);
    saved_value.assign(size_t(kStates), 0);
    for (i64 i = 0; i < f.h.n; ++i) { row_active[size_t(i)] = 1; row_value[size_t(i)] = f.rows[size_t(i)].amount; }
  }
  void apply(i32 v) override {
    const Edit& e = w.edit[size_t(v)];
    if (e.op == OP_FACTOR) { ++updates_; return; }
    const size_t id = size_t(e.id);
    saved_active[size_t(v)] = row_active[id];
    saved_value[size_t(v)] = row_value[id];
    if (e.op == OP_DELETE) row_active[id] = 0;
    else { row_active[id] = 1; row_value[id] = e.val; }
    ++updates_;
  }
  void undo(i32 v) override {
    const Edit& e = w.edit[size_t(v)];
    if (e.op == OP_FACTOR) { ++updates_; return; }
    const size_t id = size_t(e.id);
    row_active[id] = saved_active[size_t(v)];
    row_value[id] = saved_value[size_t(v)];
    ++updates_;
  }
  void query(i64 lo, i64 hi, i64* out) override {
    scratch.clear();
    for (size_t i = 0; i < size_t(n_rows_); ++i) {
      if (!row_active[i]) continue;
      const i64 val = row_value[i];
      if (val >= lo && val < hi) scratch.push_back(val);
    }
    std::sort(scratch.begin(), scratch.end());
    i64 sum = 0, dis = 0;
    for (size_t i = 0; i < scratch.size(); ++i) {
      sum += scratch[i];
      if (i == 0 || scratch[i] != scratch[i - 1]) ++dis;
    }
    out[0] = i64(scratch.size());
    out[1] = sum;
    out[2] = scratch.empty() ? 0 : scratch[size_t((scratch.size() - 1) / 2)];
    out[3] = dis;
  }
  i32 sigma() const override { return 0; }
  i32 n_rows() const override { return n_rows_; }
  i64 updates() const override { return updates_; }
};

int main(int argc, char** argv) {
  std::string mode = "hros", fixture_path, base_id, answers_path, json_path;
  i32 topology = 0, seed = 7, factor_period = 4, q = 2, repeat = 0, max_v = -1;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](const char* k) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k); std::exit(2); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--mode")) mode = next(argv[i]);
    else if (!std::strcmp(argv[i], "--fixture")) fixture_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--base-id")) base_id = next(argv[i]);
    else if (!std::strcmp(argv[i], "--topology")) topology = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--seed")) seed = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--factor-period")) factor_period = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--q")) q = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--repeat")) repeat = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--max-v")) max_v = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--answers")) answers_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--json")) json_path = next(argv[i]);
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  if (fixture_path.empty() || answers_path.empty() || json_path.empty()) {
    std::fprintf(stderr, "usage: panel_hros_align --mode hros|fenwick|avl|rebuild --fixture PATH "
                         "--base-id ID --topology T --seed S --factor-period P --q Q --repeat R "
                         "--max-v N --answers A --json J\n");
    return 2;
  }
  if (q < 1 || q > kMaxQ) { std::fprintf(stderr, "unsupported q\n"); return 2; }
  const i32 last_v = (max_v < 0 || max_v >= kStates - 1) ? kStates - 1 : max_v;

  const auto t0 = std::chrono::steady_clock::now();
  Fixture f;
  if (!load_fixture(fixture_path, f)) { std::fprintf(stderr, "cannot load fixture\n"); return 3; }
  const auto t1 = std::chrono::steady_clock::now();

  Workload w = make_workload(f, topology, seed, factor_period, q, /*per_state=*/true);
  i64 vmin = 0, vmax = 0;
  {
    bool first = true;
    for (i64 i = 0; i < f.h.n; ++i) {
      const i64 a = f.rows[size_t(i)].amount;
      if (first) { vmin = vmax = a; first = false; } else { vmin = std::min(vmin, a); vmax = std::max(vmax, a); }
    }
    for (i32 v = 1; v < kStates; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if (e.op == OP_REPLACE || e.op == OP_INSERT || e.op == OP_MOVE) {
        if (first) { vmin = vmax = e.val; first = false; } else { vmin = std::min(vmin, e.val); vmax = std::max(vmax, e.val); }
      }
    }
    if (vmin > vmax) { vmin = 0; vmax = 1; }
  }
  std::vector<i64> answers(size_t(last_v) * size_t(q) * 4, 0);
  i64 update_ns = 0, query_ns = 0;
  i64 n_queries = 0;

  std::unique_ptr<Backend> be;
  if (mode == "hros") be.reset(new HrosAdapter(f, w));
  else if (mode == "fenwick") be.reset(new FenAdapter(f, w));
  else if (mode == "avl") be.reset(new AvlAdapter(f, w));
  else if (mode == "rebuild") be.reset(new RebuildAdapter(f, w));
  else { std::fprintf(stderr, "unknown mode %s\n", mode.c_str()); return 3; }
  be->build();
  const auto t2 = std::chrono::steady_clock::now();

  std::vector<std::vector<i32>> kids(static_cast<size_t>(kStates));
  for (i32 v = 1; v < kStates; ++v) kids[size_t(w.parent[size_t(v)])].push_back(v);
  struct Ev { i32 v; u8 exit; };
  std::vector<Ev> st;
  st.reserve(size_t(kStates) + 8);
  st.push_back({0, 0});
  while (!st.empty()) {
    const Ev ev = st.back();
    st.pop_back();
    const i32 v = ev.v;
    if (ev.exit) {
      if (v > 0 && v <= last_v) {
        const auto u0 = std::chrono::steady_clock::now();
        be->undo(v);
        update_ns += ns_between(u0, std::chrono::steady_clock::now());
      }
      continue;
    }
    if (v > 0) {
      if (v > last_v) continue;
      const auto u0 = std::chrono::steady_clock::now();
      be->apply(v);
      update_ns += ns_between(u0, std::chrono::steady_clock::now());
      // this state's declared measure ranges
      const auto q0 = std::chrono::steady_clock::now();
      std::vector<i64> rl(size_t(q), 0), rh(size_t(q), 0);
      Rng qr(u64(seed) * 1000000007ULL + u64(topology) * 65537ULL + u64(factor_period) * 131ULL +
             u64(v) * 2654435761ULL + 17ULL);
      gen_value_ranges(qr, vmin, vmax, q, rl.data(), rh.data());
      for (i32 k = 0; k < q; ++k) {
        be->query(rl[size_t(k)], rh[size_t(k)],
                  &answers[(size_t(v - 1) * size_t(q) + size_t(k)) * 4]);
      }
      query_ns += ns_between(q0, std::chrono::steady_clock::now());
      n_queries += q;
    }
    st.push_back({v, 1});
    const auto& ch = kids[size_t(v)];
    for (auto it = ch.rbegin(); it != ch.rend(); ++it) st.push_back({*it, 0});
  }
  const auto t3 = std::chrono::steady_clock::now();

  FILE* out = std::fopen(answers_path.c_str(), "wb");
  if (!out) { std::fprintf(stderr, "cannot write answers\n"); return 4; }
  const bool wok = std::fwrite(answers.data(), sizeof(i64), answers.size(), out) == answers.size();
  std::fclose(out);
  if (!wok) { std::fprintf(stderr, "short answer write\n"); return 4; }
  const u64 cks = fnv1a(answers.data(), answers.size() * sizeof(i64));
  const std::string sha = sha::of(answers.data(), answers.size() * sizeof(i64));
  const auto t4 = std::chrono::steady_clock::now();
  const i64 reported_updates = be->updates();
  const i32 sigma = be->sigma();
  const i32 n_rows = be->n_rows();
  be.reset();
  f.release_rows();
  const auto t5 = std::chrono::steady_clock::now();

  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
  i64 peak_kb = 0;
  {
    FILE* s = std::fopen("/proc/self/status", "r");
    if (s) {
      char line[256];
      while (std::fgets(line, sizeof(line), s)) {
        long vv = 0;
        if (std::sscanf(line, "VmHWM: %ld kB", &vv) == 1) { peak_kb = vv; break; }
      }
      std::fclose(s);
    }
  }
  const i64 load_ns = ns_between(t0, t1);
  const i64 setup_ns = ns_between(t1, t2);
  const i64 exec_ns = ns_between(t2, t3);
  const i64 extract_ns = ns_between(t3, t4);
  const i64 cleanup_ns = ns_between(t4, t5);
  const i64 full_ns = load_ns + setup_ns + exec_ns + extract_ns + cleanup_ns;
  FILE* jf = std::fopen((json_path + ".tmp").c_str(), "w");
  if (!jf) { std::fprintf(stderr, "cannot write json\n"); return 4; }
  std::fprintf(jf, "{\n  \"mode\": \"%s\",\n  \"base_id\": \"%s\",\n", mode.c_str(), base_id.c_str());
  std::fprintf(jf, "  \"fixture\": \"%s\",\n  \"topology\": %d,\n  \"seed\": %d,\n  \"factor_period\": %d,\n",
               fixture_path.c_str(), topology, seed, factor_period);
  std::fprintf(jf, "  \"q\": %d,\n  \"repeat\": %d,\n  \"n_states\": %d,\n  \"max_v\": %d,\n",
               q, repeat, kStates, last_v);
  std::fprintf(jf, "  \"n_rows\": %d,\n  \"sigma\": %d,\n", n_rows, sigma);
  std::fprintf(jf, "  \"n_answers\": %zu,\n  \"n_queries\": %" PRId64 ", \"n_updates\": %" PRId64 ",\n",
               answers.size(), n_queries, reported_updates);
  std::fprintf(jf, "  \"range_scope\": \"measure_1d\",\n");
  std::fprintf(jf, "  \"answers_fn\": \"%016" PRIx64 "\",\n", cks);
  std::fprintf(jf, "  \"answers_sha256\": \"%s\",\n", sha.c_str());
  std::fprintf(jf, "  \"phases_ns\": {\"load\": %" PRId64 ", \"setup\": %" PRId64
                   ", \"execute\": %" PRId64 ", \"extract\": %" PRId64 ", \"cleanup\": %" PRId64 "},\n",
               load_ns, setup_ns, exec_ns, extract_ns, cleanup_ns);
  std::fprintf(jf, "  \"update_ns\": %" PRId64 ",\n  \"query_ns\": %" PRId64 ",\n", update_ns, query_ns);
  std::fprintf(jf, "  \"full_ns\": %" PRId64 ",\n  \"peak_rss_kb\": %" PRId64 "\n}\n", full_ns, peak_kb);
  std::fclose(jf);
  std::rename((json_path + ".tmp").c_str(), json_path.c_str());
  std::printf("mode=%s base=%s t%d s%d f%d q%d r%d mv=%d full_ns=%" PRId64 " peak_kb=%" PRId64
              " sigma=%d rows=%d updates=%" PRId64 "\n",
              mode.c_str(), base_id.c_str(), topology, seed, factor_period, q, repeat, last_v,
              full_ns, peak_kb, sigma, n_rows, reported_updates);
  return 0;
}
