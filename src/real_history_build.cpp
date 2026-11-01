// real_history_build.cpp — build the real version-modification history from a committed real
// base fixture (2026-10-31, real-history P1).  New file; shipped files untouched.
//
// Input: a committed BWBASE02 fixture built from a real public dataset
//   * retail2009.bin  UCI Online Retail II (DOI 10.24432/C5CG6D), 1,067,371 real invoice
//     lines, 25 real months (2009-12 .. 2011-12), 43 country groups, t = minutes since
//     2009-12-01 00:00, amount = cents of Quantity*UnitPrice (negatives are real
//     cancellations/credits);
//   * taxi2024.bin    NYC TLC yellow_tripdata_2024-01, t = microseconds since 2024-01-01,
//     amount = fare cents, 260 groups (PULocationID).
//
// Output:
//   <base>.bin    the version-0 snapshot: every real row with t < period_lo, in original
//                 order (verbatim records, real amounts/timestamps/groups); header keeps the
//                 source support bounds so declared ranges may exceed the snapshot's t range.
//   <spec>.bin    the declared real history: chain parent list, one edit per version, the
//                 per-state declared ranges, inserted-row declarations and the endpoint domain.
//
// History construction (real data only):
//   * versions 1..V-1 follow the real row stream with period_lo <= t < period_hi in strict
//     (t, source row id) order;
//   * positive-amount line  -> real append: OP_INSERT of that real (t, g, amount) row;
//   * negative-amount line  -> real cancellation/credit: find a present positive line of the
//     same group with the same magnitude and reverse it (OP_DELETE), else append the real
//     negative row (credit append);
//   * every factor_period-th version is a declared group-factor update (the shipped contract
//     declares these as workload parameters: 950/1000/1050 thousandths, groups round-robin).
//     These slots do not consume real rows.
//   * per-state declared ranges (q of them) are the month-to-date window quantised to the next
//     real time bucket and the full real month: [period_lo, next_bucket(cursor)) and
//     [period_lo, period_hi), where cursor is the timestamp of the latest real event applied.
//
// CLI: real_history_build --source S --period-lo L --period-hi H --bucket B --factor-period F
//          --versions V --out-base B --out-spec J [--probe]
#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.hpp"

using namespace bwh;

namespace {

struct RowRef {
  i64 t;
  i64 a;
  i32 g;
  i32 src;
};

struct Stat {
  i64 n_inserts = 0, n_deletes = 0, n_credits = 0, n_factor = 0, n_row_events = 0;
  i64 first_t = 0, last_t = 0;
};

bool write_all(FILE* fp, const void* p, size_t n) { return std::fwrite(p, 1, n, fp) == n; }

}  // namespace

int main(int argc, char** argv) {
  std::string source, out_base, out_spec;
  i64 period_lo = 0, period_hi = 0, bucket = 60;
  i32 fp = 37, versions = 0;
  bool probe = false;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](const char* k) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k); std::exit(2); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--source")) source = next(argv[i]);
    else if (!std::strcmp(argv[i], "--out-base")) out_base = next(argv[i]);
    else if (!std::strcmp(argv[i], "--out-spec")) out_spec = next(argv[i]);
    else if (!std::strcmp(argv[i], "--period-lo")) period_lo = std::atoll(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--period-hi")) period_hi = std::atoll(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--bucket")) bucket = std::atoll(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--factor-period")) fp = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--versions")) versions = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--probe")) probe = true;
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  if (source.empty() || bucket <= 0 || period_hi <= period_lo) {
    std::fprintf(stderr, "usage: real_history_build --source S --period-lo L --period-hi H "
                         "--bucket B --factor-period F --versions V --out-base B --out-spec J "
                         "[--probe]\n");
    return 2;
  }
  Fixture f;
  if (!load_fixture(source, f)) { std::fprintf(stderr, "cannot load source fixture\n"); return 3; }
  const i64 N = f.h.n;
  const i32 G = f.h.groups;

  // ---- real rows in strict (t, source id) order; base = rows before the window
  std::vector<i32> order(static_cast<size_t>(N));
  for (i64 i = 0; i < N; ++i) order[size_t(i)] = i32(i);
  std::sort(order.begin(), order.end(), [&](i32 x, i32 y) {
    const i64 tx = f.rows[size_t(x)].t, ty = f.rows[size_t(y)].t;
    return tx != ty ? tx < ty : x < y;
  });
  std::vector<i32> base_src;  // source ids of the snapshot, original order
  i64 n_stream = 0;
  for (i64 i = 0; i < N; ++i) {
    const BaseRec& r = f.rows[size_t(i)];
    if (r.t < period_lo) base_src.push_back(i32(i));
    else if (r.t < period_hi) ++n_stream;
  }
  const i64 n_base = i64(base_src.size());

  // ---- number of row-event versions for the requested total version count
  auto row_events_for = [&](i32 V, i64* row_events, i64* n_factor) {
    i64 re = 0, nf = 0;
    for (i32 v = 1; v < V; ++v) {
      if (fp > 0 && v % fp == 0) ++nf;
      else ++re;
    }
    *row_events = re;
    *n_factor = nf;
  };
  i64 need_rows = 0, n_factor = 0;
  if (versions > 0) {
    row_events_for(versions, &need_rows, &n_factor);
    if (need_rows > n_stream) {
      std::fprintf(stderr, "window stream too short: need %" PRId64 " rows, have %" PRId64 "\n",
                   need_rows, n_stream);
      return 4;
    }
  }
  // probe mode: report enough real structure for the NOTE (and to pick the version count)
  {
    i64 pos = 0, neg = 0, first_neg = -1;
    for (i64 i = 0; i < N; ++i) {
      const BaseRec& r = f.rows[size_t(i)];
      if (r.t >= period_lo && r.t < period_hi) {
        if (r.amount >= 0) ++pos; else { ++neg; if (first_neg < 0) first_neg = r.t; }
      }
    }
    i32 best_V = 0;
    {
      i64 re = 0, nf = 0;
      for (i32 v = 1; re < n_stream; ++v) {
        if (fp > 0 && v % fp == 0) ++nf; else ++re;
        best_V = v + 1;
      }
    }
    std::printf("{\"source\":\"%s\",\"n\":%" PRId64 ",\"groups\":%d,\"n_base\":%" PRId64
                ",\"n_stream\":%" PRId64 ",\"stream_pos\":%" PRId64 ",\"stream_neg\":%" PRId64
                ",\"period_lo\":%" PRId64 ",\"period_hi\":%" PRId64 ",\"bucket\":%" PRId64
                ",\"full_month_versions\":%d",
                source.c_str(), N, G, n_base, n_stream, pos, neg, period_lo, period_hi, bucket,
                best_V);
    if (versions > 0)
      std::printf(",\"requested_versions\":%d,\"requested_row_events\":%" PRId64
                  ",\"requested_factor_edits\":%" PRId64,
                  versions, need_rows, n_factor);
    std::printf("}\n");
    if (probe) return 0;
  }
  if (versions <= 0) { std::fprintf(stderr, "--versions required unless --probe\n"); return 2; }
  const i32 V = versions;

  // ---- build the declared history
  std::vector<i32> parent(static_cast<size_t>(V), -1);
  std::vector<Edit> edit(static_cast<size_t>(V));
  std::vector<Query> queries(static_cast<size_t>(V) * 2, Query{0, 0});
  std::vector<i64> ins_t;
  std::vector<i32> ins_g;
  ins_t.reserve(size_t(need_rows));
  ins_g.reserve(size_t(need_rows));
  std::vector<i64> endpoints;
  endpoints.reserve(size_t(need_rows) + 8);
  std::vector<i64> gfactor(static_cast<size_t>(G), 1000);
  Stat st;
  // present-positive-line index for real cancellations: (group, magnitude) -> live row ids
  std::unordered_map<u64, std::vector<i32>> present;
  present.reserve(size_t(n_base));
  // base rows get new ids 0..n_base-1; inserted rows get n_base + k
  for (i64 i = 0; i < n_base; ++i) {
    const BaseRec& r = f.rows[size_t(base_src[size_t(i)])];
    if (r.amount > 0) {
      const u64 key = (u64(u32(r.g)) << 40) | u64(r.amount);
      present[key].push_back(i32(i));
    }
    endpoints.push_back(r.t);  // placeholder; replaced below by declared endpoints
  }
  endpoints.clear();
  i64 cursor = period_lo;
  i32 next_group = 0;
  i64 stream_idx = 0;  // index into the ordered stream rows
  i64 n_ins = 0;
  // ordered stream ids (only rows in window), built once
  std::vector<i32> stream_src;
  stream_src.reserve(size_t(n_stream));
  for (i32 id : order) {
    const BaseRec& r = f.rows[size_t(id)];
    if (r.t >= period_lo && r.t < period_hi) stream_src.push_back(id);
  }
  for (i32 v = 1; v < V; ++v) {
    parent[size_t(v)] = v - 1;
    Edit e{};
    if (fp > 0 && v % fp == 0) {
      // declared group-factor update (contract parameter)
      const i32 g = next_group++ % G;
      const i64 cur = gfactor[size_t(g)];
      const i64 nf = (cur == 950) ? 1000 : (cur == 1000 ? 1050 : 950);
      gfactor[size_t(g)] = nf;
      e.op = OP_FACTOR;
      e.g = g;
      e.val = nf;
      ++st.n_factor;
    } else {
      if (stream_idx >= i64(stream_src.size())) { std::fprintf(stderr, "stream exhausted\n"); return 5; }
      const i32 sid = stream_src[size_t(stream_idx++)];
      const BaseRec& r = f.rows[size_t(sid)];
      ++st.n_row_events;
      cursor = r.t;
      if (r.amount >= 0) {
        e.op = OP_INSERT;
        e.id = n_base + n_ins;
        e.t = r.t;
        e.g = r.g;
        e.val = r.amount;
        ins_t.push_back(r.t);
        ins_g.push_back(r.g);
        const u64 key = (u64(u32(r.g)) << 40) | u64(r.amount);
        present[key].push_back(i32(e.id));
        ++n_ins;
        ++st.n_inserts;
      } else {
        const u64 key = (u64(u32(r.g)) << 40) | u64(-r.amount);
        auto it = present.find(key);
        i32 hit = -1;
        if (it != present.end()) {
          auto& vec = it->second;
          while (!vec.empty() && vec.back() < 0) vec.pop_back();
          if (!vec.empty()) { hit = vec.back(); vec.pop_back(); }
        }
        if (hit >= 0) {
          e.op = OP_DELETE;
          e.id = hit;
          e.val = 0;
          ++st.n_deletes;
        } else {
          e.op = OP_INSERT;
          e.id = n_base + n_ins;
          e.t = r.t;
          e.g = r.g;
          e.val = r.amount;  // real credit line (negative amount)
          ins_t.push_back(r.t);
          ins_g.push_back(r.g);
          ++n_ins;
          ++st.n_credits;
        }
      }
    }
    edit[size_t(v)] = e;
    if (st.first_t == 0 && st.n_row_events > 0) st.first_t = cursor;
    st.last_t = cursor;
    const i64 hi0 = std::min(period_hi, ((cursor + bucket - 1) / bucket) * bucket);
    queries[size_t(v) * 2 + 0] = Query{period_lo, std::max(period_lo, hi0)};
    queries[size_t(v) * 2 + 1] = Query{period_lo, period_hi};
    endpoints.push_back(queries[size_t(v) * 2 + 0].lo);
    endpoints.push_back(queries[size_t(v) * 2 + 0].hi);
    endpoints.push_back(queries[size_t(v) * 2 + 1].lo);
    endpoints.push_back(queries[size_t(v) * 2 + 1].hi);
  }
  edit[0].op = OP_REPLACE;  // unused slot: make it a benign value
  edit[0].id = 0;
  edit[0].val = 0;
  std::sort(endpoints.begin(), endpoints.end());
  endpoints.erase(std::unique(endpoints.begin(), endpoints.end()), endpoints.end());
  if (i64(n_ins) != i64(ins_t.size())) { std::fprintf(stderr, "inserted-row declaration mismatch\n"); return 5; }

  if (probe) {
    std::printf("{\"versions\":%d,\"row_events\":%" PRId64 ",\"inserts\":%" PRId64
                ",\"deletes\":%" PRId64 ",\"credits\":%" PRId64 ",\"factor_edits\":%" PRId64
                ",\"n_ins\":%" PRId64 ",\"endpoints\":%zu,\"first_t\":%" PRId64
                ",\"last_t\":%" PRId64 "}\n",
                V, st.n_row_events, st.n_inserts, st.n_deletes, st.n_credits, st.n_factor, n_ins,
                endpoints.size(), st.first_t, st.last_t);
    return 0;
  }
  if (out_base.empty() || out_spec.empty()) { std::fprintf(stderr, "missing output paths\n"); return 2; }

  // ---- write the version-0 snapshot fixture (real records verbatim, original order)
  {
    FILE* fp_out = std::fopen(out_base.c_str(), "wb");
    if (!fp_out) { std::fprintf(stderr, "cannot write base fixture\n"); return 6; }
    FixtureHeader h = f.h;
    h.n = n_base;
    h.original_lines = n_base;
    bool ok = write_all(fp_out, &h, sizeof(h));
    for (i64 i = 0; i < n_base && ok; ++i) {
      const BaseRec& r = f.rows[size_t(base_src[size_t(i)])];
      ok = write_all(fp_out, &r, sizeof(r));
    }
    ok = ok && write_all(fp_out, f.base_factor.data(), size_t(G) * sizeof(i64));
    ok = ok && write_all(fp_out, f.group_labels.data(), size_t(G) * sizeof(i32));
    std::fclose(fp_out);
    if (!ok) { std::fprintf(stderr, "short base fixture write\n"); return 6; }
  }
  // ---- write the declared history spec
  {
    FILE* fp_out = std::fopen(out_spec.c_str(), "wb");
    if (!fp_out) { std::fprintf(stderr, "cannot write spec\n"); return 6; }
    bool ok = write_all(fp_out, "BWHHIST1", 8);
    const i32 kst = V, q = 2, fpp = fp, gg = G;  // kstates must equal the requested chain length
    ok = ok && write_all(fp_out, &kst, 4) && write_all(fp_out, &q, 4) &&
         write_all(fp_out, &fpp, 4) && write_all(fp_out, &gg, 4);
    const i64 nb = n_base, ni = n_ins, tmn = f.h.t_min, tmx = f.h.t_max,
              nep = i64(endpoints.size()), plo = period_lo, phi = period_hi, bk = bucket,
              nins = st.n_inserts, ndel = st.n_deletes, ncred = st.n_credits, nfac = st.n_factor;
    ok = ok && write_all(fp_out, &nb, 8) && write_all(fp_out, &ni, 8) &&
         write_all(fp_out, &tmn, 8) && write_all(fp_out, &tmx, 8) &&
         write_all(fp_out, &nep, 8) && write_all(fp_out, &plo, 8) &&
         write_all(fp_out, &phi, 8) && write_all(fp_out, &bk, 8) &&
         write_all(fp_out, &nins, 8) && write_all(fp_out, &ndel, 8) &&
         write_all(fp_out, &ncred, 8) && write_all(fp_out, &nfac, 8);
    ok = ok && write_all(fp_out, parent.data(), parent.size() * 4) &&
         write_all(fp_out, edit.data(), edit.size() * sizeof(Edit)) &&
         write_all(fp_out, queries.data(), queries.size() * sizeof(Query)) &&
         write_all(fp_out, ins_t.data(), ins_t.size() * 8) &&
         write_all(fp_out, ins_g.data(), ins_g.size() * 4) &&
         write_all(fp_out, endpoints.data(), endpoints.size() * 8);
    std::fclose(fp_out);
    if (!ok) { std::fprintf(stderr, "short spec write\n"); return 6; }
  }
  // ---- machine-readable construction receipt
  std::printf("{\"versions\":%d,\"base_rows\":%" PRId64 ",\"row_events\":%" PRId64
              ",\"inserts\":%" PRId64 ",\"deletes\":%" PRId64 ",\"credits\":%" PRId64
              ",\"factor_edits\":%" PRId64 ",\"n_ins\":%" PRId64 ",\"endpoints\":%zu"
              ",\"period_lo\":%" PRId64 ",\"period_hi\":%" PRId64 ",\"bucket\":%" PRId64
              ",\"first_t\":%" PRId64 ",\"last_t\":%" PRId64 "}\n",
              V, n_base, st.n_row_events, st.n_inserts, st.n_deletes, st.n_credits, st.n_factor,
              n_ins, endpoints.size(), period_lo, period_hi, bucket, st.first_t, st.last_t);
  return 0;
}
