// panel_hros.cpp — one timed worker for the HROS (exact holistic order statistics) prototype.
//
// New file (2026-11-01).  Same CLI / phase accounting / answer-serialization contract as the
// shipped panel, with --mode selecting the backend under test:
//   hros     HROS index (reversible treap over value ranks + presence Fenwick)  [the proposal]
//   rebuild  full-state rebuild reference: per query, scan the current relation from scratch,
//            filter to [lo,hi), sort, take the lower median and exact distinct (this is the
//            independent brute-force reference for correctness)
//   sortall  per-version sort reference: rebuild + sort the whole active relation once per
//            version, then answer every query on the sorted array
//
// Query semantics (identical in every mode): for state v = 1..kStates-1, for k = 0..q-1, over
// the 1D MEASURE range [lo,hi) generated deterministically from (seed, topology, factor_period,
// q, v, k) over the fixture's possible-amount domain, answer the 4-tuple
//   (count, sum, lower_median, exact_distinct_count)
// written as little-endian int64s in (v,k) order: 4 values per query.  The shipped prefix
// backend answers only additive sums over TIME ranges and cannot answer these holistic
// statistics at all (reported separately as capability context).
//
// Phases (all charged): load, setup (index build / reference preparation), execute
// (version traversal, edit apply/undo, queries; update_ns and query_ns are sub-counters),
// extract (complete answer vector + fnv1a + sha256), cleanup.
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/resource.h>

#include "common.hpp"
#include "hros_index.hpp"
#include "sha256.hpp"

using namespace bwh;
using namespace bwh::hros;

static i64 ns_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

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
    std::fprintf(stderr, "usage: panel_hros --mode hros|rebuild|sortall --fixture PATH --base-id ID "
                         "--topology T --seed S --factor-period P --q Q --repeat R "
                         "--max-v N --answers A --json J\n");
    return 2;
  }
  if (q < 1 || q > kMaxQ) { std::fprintf(stderr, "unsupported q\n"); return 2; }
  const i32 last_v = (max_v < 0 || max_v >= kStates - 1) ? kStates - 1 : max_v;

  const auto t0 = std::chrono::steady_clock::now();
  Fixture f;
  if (!load_fixture(fixture_path, f)) { std::fprintf(stderr, "cannot load fixture\n"); return 3; }
  const auto t1 = std::chrono::steady_clock::now();

  // declared workload: edits + parents (queries are regenerated on the measure axis below)
  Workload w = make_workload(f, topology, seed, factor_period, q, /*per_state=*/true);

  // possible-amount domain (base amounts + every absolute row-edit amount)
  i64 vmin = 0, vmax = 0;
  {
    bool first = true;
    for (i64 i = 0; i < f.h.n; ++i) {
      i64 a = f.rows[size_t(i)].amount;
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
  i64 update_ns = 0, query_ns = 0, n_updates = 0, n_queries = 0;
  i64 n_rows = 0;
  i32 sigma = 0;
  HrosIndex* hros = nullptr;
  // rebuild/sortall row state
  std::vector<u8> row_active;
  std::vector<i64> row_value;
  std::vector<u8> saved_active;
  std::vector<i64> saved_value;
  std::vector<i64> scratch, sorted_all;
  std::vector<std::vector<i32>> kids;

  // ---- setup: index build / reference state
  if (mode == "hros") {
    hros = new HrosIndex(f, w);
    hros->build();
    sigma = hros->sigma;
    n_rows = hros->n_rows;
  } else {
    i32 max_id = i32(f.h.n);
    for (i32 v = 1; v < kStates; ++v) {
      const Edit& e = w.edit[size_t(v)];
      if (e.op == OP_REPLACE || e.op == OP_DELETE || e.op == OP_MOVE || e.op == OP_INSERT) {
        if (e.id + 1 > i64(max_id)) max_id = i32(e.id) + 1;
      }
    }
    n_rows = max_id + 1;
    row_active.assign(size_t(n_rows), 0);
    row_value.assign(size_t(n_rows), 0);
    saved_active.assign(size_t(kStates), 0);
    saved_value.assign(size_t(kStates), 0);
    for (i64 i = 0; i < f.h.n; ++i) { row_active[size_t(i)] = 1; row_value[size_t(i)] = f.rows[size_t(i)].amount; }
    sorted_all.reserve(size_t(n_rows));
  }
  kids.assign(size_t(kStates), {});
  for (i32 v = 1; v < kStates; ++v) kids[size_t(w.parent[size_t(v)])].push_back(v);
  const auto t2 = std::chrono::steady_clock::now();

  // ---- execute: iterative preorder traversal with apply/undo
  struct Ev { i32 v; u8 exit; };
  std::vector<Ev> st;
  st.reserve(size_t(kStates) + 8);
  st.push_back({0, 0});
  while (!st.empty()) {
    const Ev ev = st.back(); st.pop_back();
    const i32 v = ev.v;
    if (ev.exit) {
      if (v > 0 && mode != "hros" && w.edit[size_t(v)].op != OP_FACTOR) {
        // restore previous row state
        const size_t id = size_t(w.edit[size_t(v)].id);
        row_active[id] = saved_active[size_t(v)];
        row_value[id] = saved_value[size_t(v)];
        ++update_ns;
      }
      if (v > 0 && mode == "hros") {
        const auto u0 = std::chrono::steady_clock::now();
        hros->undo(v);
        update_ns += ns_between(u0, std::chrono::steady_clock::now());
      }
      continue;
    }
    if (v > 0) {
      const Edit& e = w.edit[size_t(v)];
      if (v > last_v) continue;
      if (mode == "hros") {
        const auto u0 = std::chrono::steady_clock::now();
        hros->apply(v);
        update_ns += ns_between(u0, std::chrono::steady_clock::now());
      } else {
        // snapshot + apply edit to the mutable row state
        if (e.op != OP_FACTOR) {
          const size_t id = size_t(e.id);
          saved_active[size_t(v)] = row_active[id];
          saved_value[size_t(v)] = row_value[id];
          if (e.op == OP_DELETE) row_active[id] = 0;
          else { row_active[id] = 1; row_value[id] = e.val; }
        }
        ++update_ns;
      }
      // per-version reference sort (sortall mode)
      if (mode == "sortall") {
        sorted_all.clear();
        for (size_t i = 0; i < size_t(n_rows); ++i)
          if (row_active[i]) sorted_all.push_back(row_value[i]);
        std::sort(sorted_all.begin(), sorted_all.end());
      }
      // answer this state's q queries (measure ranges are generated once per state)
      const auto q0 = std::chrono::steady_clock::now();
      std::vector<i64> rl((size_t(q))), rh((size_t(q)));
      Rng qr(u64(seed) * 1000000007ULL + u64(topology) * 65537ULL + u64(factor_period) * 131ULL +
             u64(v) * 2654435761ULL + 17ULL);
      gen_value_ranges(qr, vmin, vmax, q, rl.data(), rh.data());
      if (mode == "hros") {
        for (i32 k = 0; k < q; ++k) {
          hros->query(rl[size_t(k)], rh[size_t(k)],
                      &answers[(size_t(v - 1) * size_t(q) + size_t(k)) * 4]);
        }
      } else {
        for (i32 k = 0; k < q; ++k) {
          const i64 lo = rl[size_t(k)], hi = rh[size_t(k)];
          i64 cnt = 0, sum = 0, median = 0, dis = 0;
          auto put = [&](i64 c, i64 s, i64 m, i64 d) {
            size_t base = (size_t(v - 1) * size_t(q) + size_t(k)) * 4;
            answers[base + 0] = c; answers[base + 1] = s; answers[base + 2] = m; answers[base + 3] = d;
          };
          if (mode == "sortall") {
            auto it0 = std::lower_bound(sorted_all.begin(), sorted_all.end(), lo);
            auto it1 = std::lower_bound(sorted_all.begin(), sorted_all.end(), hi);
            cnt = i64(it1 - it0);
            for (auto it = it0; it != it1; ++it) sum += *it;
            if (cnt > 0) median = *(it0 + (cnt - 1) / 2);
            for (auto it = it0; it != it1; ++it) if (it == it0 || *it != *(it - 1)) ++dis;
          } else {  // rebuild: full-state scan + sort per query (independent brute force)
            scratch.clear();
            for (size_t i = 0; i < size_t(n_rows); ++i) {
              if (!row_active[i]) continue;
              i64 val = row_value[i];
              if (val >= lo && val < hi) scratch.push_back(val);
            }
            cnt = i64(scratch.size());
            std::sort(scratch.begin(), scratch.end());
            for (i64 x : scratch) sum += x;
            if (cnt > 0) median = scratch[size_t((cnt - 1) / 2)];
            dis = 0;
            for (size_t i = 0; i < scratch.size(); ++i) if (i == 0 || scratch[i] != scratch[i - 1]) ++dis;
          }
          put(cnt, sum, median, dis);
        }
      }
      query_ns += ns_between(q0, std::chrono::steady_clock::now());
      n_queries += q;
      n_updates += 2;  // apply + undo counted per edited version
    }
    st.push_back({v, 1});
    {
      const auto& ch = kids[size_t(v)];
      for (auto it = ch.rbegin(); it != ch.rend(); ++it) st.push_back({*it, 0});
    }
  }
  const auto t3 = std::chrono::steady_clock::now();

  // ---- extract
  FILE* out = std::fopen(answers_path.c_str(), "wb");
  if (!out) { std::fprintf(stderr, "cannot write answers\n"); return 4; }
  const bool wok = std::fwrite(answers.data(), sizeof(i64), answers.size(), out) == answers.size();
  std::fclose(out);
  if (!wok) { std::fprintf(stderr, "short answer write\n"); return 4; }
  const u64 cks = fnv1a(answers.data(), answers.size() * sizeof(i64));
  const std::string sha = sha::of(answers.data(), answers.size() * sizeof(i64));
  const auto t4 = std::chrono::steady_clock::now();
  const i64 hros_updates_saved = hros ? hros->updates : 0;
  delete hros;
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
        long v = 0;
        if (std::sscanf(line, "VmHWM: %ld kB", &v) == 1) { peak_kb = v; break; }
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
  std::fprintf(jf, "{\n");
  std::fprintf(jf, "  \"mode\": \"%s\",\n  \"base_id\": \"%s\",\n", mode.c_str(), base_id.c_str());
  std::fprintf(jf, "  \"fixture\": \"%s\",\n  \"topology\": %d,\n  \"seed\": %d,\n  \"factor_period\": %d,\n",
               fixture_path.c_str(), topology, seed, factor_period);
  std::fprintf(jf, "  \"q\": %d,\n  \"repeat\": %d,\n  \"n_states\": %d,\n  \"max_v\": %d,\n",
               q, repeat, kStates, last_v);
  std::fprintf(jf, "  \"n_rows\": %" PRId64 ",\n  \"sigma\": %d,\n", n_rows, sigma);
  const i64 reported_updates = hros ? hros_updates_saved : n_updates;
  std::fprintf(jf, "  \"n_answers\": %zu,\n  \"n_queries\": %" PRId64 ",\n  \"n_updates\": %" PRId64 ",\n",
               answers.size(), n_queries, reported_updates);
  std::fprintf(jf, "  \"range_scope\": \"measure_1d\",\n");
  std::fprintf(jf, "  \"answers_fn\": \"%016" PRIx64 "\",\n", cks);
  std::fprintf(jf, "  \"answers_sha256\": \"%s\",\n", sha.c_str());
  std::fprintf(jf, "  \"phases_ns\": {\"load\": %" PRId64 ", \"setup\": %" PRId64 ", \"execute\": %" PRId64
                   ", \"extract\": %" PRId64 ", \"cleanup\": %" PRId64 "},\n",
               load_ns, setup_ns, exec_ns, extract_ns, cleanup_ns);
  std::fprintf(jf, "  \"update_ns\": %" PRId64 ",\n  \"query_ns\": %" PRId64 ",\n", update_ns, query_ns);
  std::fprintf(jf, "  \"full_ns\": %" PRId64 ",\n  \"peak_rss_kb\": %" PRId64 "\n}\n", full_ns, peak_kb);
  std::fclose(jf);
  std::rename((json_path + ".tmp").c_str(), json_path.c_str());
  std::printf("mode=%s base=%s t%d s%d f%d q%d r%d mv=%d full_ns=%" PRId64 " peak_kb=%" PRId64 "\n",
              mode.c_str(), base_id.c_str(), topology, seed, factor_period, q, repeat, last_v, full_ns, peak_kb);
  return 0;
}
