// real_branch_spec.hpp — declared workload spec for the real branching version-history benchmark
// (2026-10-31, real-history P1).  New file; no shipped file is touched.
//
// The committed real base fixtures (retail2009.bin from UCI Online Retail II, taxi2024.bin
// from NYC TLC January 2024) are static snapshots.  A real temporal modification history is
// reconstructed from them as follows (see real_history_build.cpp):
//   * the snapshot is the real relation restricted to rows before a real instant T0 (the
//     version-0 relation);
//   * versions 1..V-1 follow the real row stream after T0 in strict real timestamp order:
//     a positive-amount line is a real append (OP_INSERT), a negative-amount line (a real
//     cancellation/credit) is matched against a present line of the same group and magnitude
//     and becomes a real reversal (OP_DELETE) when found, else a real credit append;
//   * every factor_period-th version is a declared group-factor update (the shipped contract
//     schedules these as declared workload parameters, not data edits).
// All row timestamps/amounts/groups, their order and the append/reversal split are the real
// data; only the factor schedule and the declared range predicates are declared parameters.
// The file stores that history so every engine under test reads the identical declaration.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common.hpp"

namespace bwh {

struct RealHistorySpec {
  i32 kstates = 0, q = 0, fp = 0, G = 0, topology = 0;
  i64 n_base = 0, n_ins = 0, t_min = 0, t_max = 0, n_ep = 0;
  i64 period_lo = 0, period_hi = 0, bucket = 0;  // real window and query time bucket
  i64 n_inserts = 0, n_deletes = 0, n_credits = 0, n_factor_edits = 0;
  std::vector<i32> parent;    // kstates
  std::vector<Edit> edit;     // kstates (edit[0] unused)
  std::vector<Query> queries; // kstates * q (per-state declared ranges)
  std::vector<i64> ins_t;     // n_ins inserted-row timestamps, id = n_base + k
  std::vector<i32> ins_g;     // n_ins inserted-row groups
  std::vector<i64> endpoints; // sorted unique declared endpoints
  inline const Query& query(i32 v, i32 k) const {
    return queries[size_t(v) * size_t(q) + size_t(k)];
  }
};

inline bool load_real_history_spec(const std::string& path, RealHistorySpec& s) {
  FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) return false;
  auto rd = [&](void* p, size_t n) { return std::fread(p, 1, n, fp) == n; };
  char magic[8];
  if (!rd(magic, 8) || std::memcmp(magic, "BWHHIST1", 8) != 0) { std::fclose(fp); return false; }
  if (!rd(&s.kstates, 4) || !rd(&s.q, 4) || !rd(&s.fp, 4) || !rd(&s.G, 4) ||
      !rd(&s.topology, 4)) { std::fclose(fp); return false; }
  if (!rd(&s.n_base, 8) || !rd(&s.n_ins, 8) || !rd(&s.t_min, 8) || !rd(&s.t_max, 8) ||
      !rd(&s.n_ep, 8) || !rd(&s.period_lo, 8) || !rd(&s.period_hi, 8) || !rd(&s.bucket, 8) ||
      !rd(&s.n_inserts, 8) || !rd(&s.n_deletes, 8) || !rd(&s.n_credits, 8) ||
      !rd(&s.n_factor_edits, 8)) { std::fclose(fp); return false; }
  if (s.kstates != kStates) { std::fclose(fp); return false; }
  s.parent.resize(size_t(s.kstates));
  s.edit.resize(size_t(s.kstates));
  s.queries.resize(size_t(s.kstates) * size_t(s.q));
  s.ins_t.resize(size_t(s.n_ins));
  s.ins_g.resize(size_t(s.n_ins));
  s.endpoints.resize(size_t(s.n_ep));
  const bool ok = rd(s.parent.data(), s.parent.size() * 4) &&
                  rd(s.edit.data(), s.edit.size() * sizeof(Edit)) &&
                  rd(s.queries.data(), s.queries.size() * sizeof(Query)) &&
                  rd(s.ins_t.data(), s.ins_t.size() * 8) && rd(s.ins_g.data(), s.ins_g.size() * 4) &&
                  rd(s.endpoints.data(), s.endpoints.size() * 8);
  std::fclose(fp);
  return ok;
}

// Shipped-Arm workload: per-state ranges are stored with the shipped kMaxQ stride so that
// query_for()/HybridArm read the identical declared ranges.
inline void fill_workload(const RealHistorySpec& s, Workload& w) {
  w.topology = s.topology;
  w.seed = 7;
  w.factor_period = s.fp;
  w.q = s.q;
  w.per_state = true;
  w.n_base = i32(s.n_base);
  w.n_inserted = i32(s.n_ins);
  w.groups = s.G;
  w.t_min = s.t_min;
  w.t_max = s.t_max;
  w.parent = s.parent;
  w.edit = s.edit;
  w.ins_t = s.ins_t;
  w.ins_g = s.ins_g;
  w.endpoints = s.endpoints;
  w.queries.assign(size_t(kStates) * size_t(kMaxQ), Query{0, 0});
  for (i32 v = 1; v < kStates; ++v)
    for (i32 k = 0; k < s.q; ++k)
      w.queries[size_t(v) * size_t(kMaxQ) + size_t(k)] = s.query(v, k);
}

// batch_engine.hpp / batch_engine_v2.hpp BatchWorkload: per-state ranges with stride q.
template <class BW>
inline void fill_batch_workload(const RealHistorySpec& s, BW& w) {
  w.topology = s.topology;
  w.seed = 7;
  w.factor_period = s.fp;
  w.q = s.q;
  w.per_state = true;
  w.n_base = i32(s.n_base);
  w.n_inserted = i32(s.n_ins);
  w.groups = s.G;
  w.t_min = s.t_min;
  w.t_max = s.t_max;
  w.parent = s.parent;
  w.edit = s.edit;
  w.queries = s.queries;
  w.ins_t = s.ins_t;
  w.ins_g = s.ins_g;
  w.endpoints = s.endpoints;
}

}  // namespace bwh
