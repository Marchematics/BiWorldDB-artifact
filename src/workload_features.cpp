// workload_features.cpp — extract declared-workload features for the cross-family planner.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "common.hpp"

using namespace bwh;

int main(int argc, char** argv) {
  std::string fixture_path, base_id;
  i32 topology = 0, seed = 7, factor_period = 4, q = 2;
  bool per_state = false;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](const char* k) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing %s\n", k); std::exit(2); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--fixture")) fixture_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--base-id")) base_id = next(argv[i]);
    else if (!std::strcmp(argv[i], "--topology")) topology = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--seed")) seed = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--factor-period")) factor_period = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--q")) q = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--per-state")) per_state = true;
    else { std::fprintf(stderr, "unknown %s\n", argv[i]); return 2; }
  }
  Fixture f;
  if (fixture_path.empty() || !load_fixture(fixture_path, f)) {
    std::fprintf(stderr, "cannot load fixture\n");
    return 3;
  }
  Workload w = make_workload(f, topology, seed, factor_period, q, per_state);
  i64 factor_edits = 0, row_edits = 0, inserts = 0, moves = 0, deletes = 0, replaces = 0;
  for (i32 v = 1; v < kStates; ++v) {
    const u8 op = w.edit[size_t(v)].op;
    if (op == OP_FACTOR) ++factor_edits;
    else ++row_edits;
    if (op == OP_INSERT) ++inserts;
    if (op == OP_MOVE) ++moves;
    if (op == OP_DELETE) ++deletes;
    if (op == OP_REPLACE) ++replaces;
  }
  const i64 span = f.h.t_max - f.h.t_min + 1;
  const i64 N = f.h.n;
  const double ng = double(N) / double(f.h.groups);
  std::printf(
      "{\"base\":\"%s\",\"rows\":%lld,\"groups\":%d,\"states\":%d,\"q\":%d,"
      "\"topology\":%d,\"seed\":%d,\"factor_period\":%d,"
      "\"edits\":%lld,\"factor_edits\":%lld,\"row_edits\":%lld,"
      "\"inserts\":%lld,\"moves\":%lld,\"deletes\":%lld,\"replaces\":%lld,"
      "\"endpoints\":%zu,\"span\":%lld,\"rows_per_group\":%.3f,"
      "\"dense_predicate\":%d,\"per_state\":%d}\n",
      base_id.empty() ? fixture_path.c_str() : base_id.c_str(), (long long)N, f.h.groups,
      kStates, w.q, topology, seed, factor_period, (long long)(kStates - 1),
      (long long)factor_edits, (long long)row_edits, (long long)inserts, (long long)moves,
      (long long)deletes, (long long)replaces, w.endpoints.size(), (long long)span, ng,
      (f.h.groups * span <= 2 * (N + (kStates - 1))) ? 1 : 0, per_state ? 1 : 0);
  return 0;
}
