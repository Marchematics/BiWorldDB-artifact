// real_branch_build.cpp — build a real timestamp-ordered *branching* version history from a
// committed real base fixture (2026-11-01, real-branch-history P1).  New file; shipped files
// untouched.
//
// Real data (committed fixtures, real provenance in their .meta.json):
//   * retail2009.bin  UCI Online Retail II, 1,067,371 real invoice lines, 25 real months
//     (2009-12..2011-12), 43 country groups, t = minutes since 2009-12-01, amount = cents of
//     Quantity*UnitPrice (negative amounts are real cancellations/credits);
//   * taxi2024.bin    NYC TLC yellow_tripdata_2024-01, t = microseconds since 2024-01-01.
//
// History shape (one edit per state, the shipped tree contract):
//
//   v0 ── trunk (K real events, real time order) ── P  (fork parent state)
//                                                    ├── FA (declared factor) ── A: NA real events ── A_tip
//                                                    └── FB (declared factor) ── B: NB real events ── B_tip
//   A_tip ── FM (declared factor, merges B's factor change) ── replay of B's NB real events ── merged tip
//
//   * trunk, branch A and branch B consume consecutive real events of the dataset after T0, in
//     strict real timestamp order, so branch A and branch B own *disjoint real event intervals*
//     (A's last real timestamp precedes B's first); the merged descendant replays branch B's real
//     events on top of A_tip, which is the tree-contract materialisation of the merge.
//   * branch row sets are disjoint by construction (row-id parity: branch A edits base rows with
//     even new ids / its own inserted block, branch B the odd base rows / its own inserted block),
//     so branch operations commute and the merged tip state is exactly P + A + B; the oracle
//     binary verifies this union against an independent reference merge.
//   * FA/FB/FM are **declared factor-contract assignments** (one group each, 950/1000/1050-style
//     values), not data events.  Everything else is real event data: real (timestamp, group,
//     amount) of real invoice lines / trips, real chronological order, real cancellation
//     reversal magnitudes.
//
// CLI: real_branch_build --source S --period-lo T0 --epoch E --unit U --trunk-events K
//          --branch-a-events NA --branch-b-events NB --q Q --seed 7 --out-base B --out-spec J
//          [--probe]
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
struct RowRef { i64 t; i64 a; i32 g; i32 src; };

bool write_all(FILE* fp, const void* p, size_t n) { return std::fwrite(p, 1, n, fp) == n; }
}  // namespace

int main(int argc, char** argv) {
  std::string source, out_base, out_spec, epoch = "?", unit = "?";
  i64 period_lo = 0;
  i32 K = 512, NA = 4096, NB = 4096, q = 2, seed = 7;
  bool probe = false;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](const char* k) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k); std::exit(2); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--source")) source = next(argv[i]);
    else if (!std::strcmp(argv[i], "--out-base")) out_base = next(argv[i]);
    else if (!std::strcmp(argv[i], "--out-spec")) out_spec = next(argv[i]);
    else if (!std::strcmp(argv[i], "--epoch")) epoch = next(argv[i]);
    else if (!std::strcmp(argv[i], "--unit")) unit = next(argv[i]);
    else if (!std::strcmp(argv[i], "--period-lo")) period_lo = std::atoll(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--trunk-events")) K = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--branch-a-events")) NA = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--branch-b-events")) NB = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--q")) q = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--seed")) seed = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--probe")) probe = true;
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  if (source.empty() || q < 1 || q > kMaxQ || K < 0 || NA < 1 || NB < 1) {
    std::fprintf(stderr, "usage: real_branch_build --source S --period-lo T0 --trunk-events K "
                         "--branch-a-events NA --branch-b-events NB --q Q --out-base B "
                         "--out-spec J [--probe]\n");
    return 2;
  }
  Fixture f;
  if (!load_fixture(source, f)) { std::fprintf(stderr, "cannot load source fixture\n"); return 3; }
  const i64 N = f.h.n;
  const i32 G = f.h.groups;
  // version layout
  const i32 V = K + NA + 2 * NB + 4;      // kStates
  const i32 vP = K;                       // fork parent state
  const i32 vFA = K + 1;
  const i32 vA0 = vFA + 1, vAtip = vFA + NA;
  const i32 vFB = vAtip + 1;
  const i32 vB0 = vFB + 1, vBtip = vFB + NB;
  const i32 vFM = vBtip + 1;
  const i32 vM0 = vFM + 1, vMtip = vFM + NB;
  if (vMtip != V - 1) { std::fprintf(stderr, "version layout bug\n"); return 4; }

  // ---- chronological real row stream after T0, and the version-0 snapshot
  std::vector<i32> order(static_cast<size_t>(N));
  for (i64 i = 0; i < N; ++i) order[size_t(i)] = i32(i);
  std::sort(order.begin(), order.end(), [&](i32 x, i32 y) {
    const i64 tx = f.rows[size_t(x)].t, ty = f.rows[size_t(y)].t;
    return tx != ty ? tx < ty : x < y;
  });
  std::vector<i32> base_src;
  std::vector<i32> stream;
  for (i32 id : order) {
    const BaseRec& r = f.rows[size_t(id)];
    if (r.t < period_lo) base_src.push_back(id);
    else stream.push_back(id);
  }
  const i64 n_base = i64(base_src.size());
  const size_t need = size_t(K) + size_t(NA) + size_t(NB);
  if (stream.size() < need) { std::fprintf(stderr, "real stream too short\n"); return 4; }

  // ---- per-branch real edit construction.
  // Branch row partition: branch A owns base new-ids with (id % 2 == 0), branch B the odd ids;
  // inserted rows use disjoint id blocks (A first, B after A) so the two branches never touch
  // the same row and their operations commute (the merge is exact).
  struct BranchEdit {
    std::vector<Edit> ev;   // real events, in real time order
    i64 appends = 0, reversals = 0, credits = 0;
    i64 first_t = 0, last_t = 0;
  };
  auto make_present = [&](int parity, std::unordered_map<u64, std::vector<i32>>& present) {
    for (i64 i = 0; i < n_base; ++i) {
      if (int(i & 1) != parity) continue;
      const BaseRec& r = f.rows[size_t(base_src[size_t(i)])];
      if (r.amount > 0) present[(u64(u32(r.g)) << 40) | u64(r.amount)].push_back(i32(i));
    }
  };
  auto build_branch = [&](const i32* ids, i32 n, int parity, i64 ins_base,
                          BranchEdit& be) {
    std::unordered_map<u64, std::vector<i32>> present;
    present.reserve(size_t(n_base / 2 + 1));
    make_present(parity, present);
    i64 n_ins = 0;
    for (i32 j = 0; j < n; ++j) {
      const BaseRec& r = f.rows[size_t(ids[j])];
      Edit e{};
      if (j == 0) be.first_t = r.t;
      be.last_t = r.t;
      if (r.amount >= 0) {
        e.op = OP_INSERT;
        e.id = ins_base + n_ins++;
        e.t = r.t; e.g = r.g; e.val = r.amount;
        present[(u64(u32(r.g)) << 40) | u64(r.amount)].push_back(i32(e.id));
        ++be.appends;
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
          ++be.reversals;
        } else {
          e.op = OP_INSERT;
          e.id = ins_base + n_ins++;
          e.t = r.t; e.g = r.g; e.val = r.amount;  // real credit line (negative)
          ++be.credits;
        }
      }
      be.ev.push_back(e);
    }
    return n_ins;
  };
  BranchEdit trunk, brA, brB;
  {
    std::vector<i32> tids(stream.begin(), stream.begin() + K);
    std::vector<i32> aids(stream.begin() + K, stream.begin() + K + NA);
    // branch B ids are only assigned after A's inserted block size is known; fill B later
    // (two passes: A first to get its inserted count, then B with its block).
    i64 dummy = 0;
    // trunk uses its own id block (after base) - trunk is on the shared path before the fork.
    {
      std::unordered_map<u64, std::vector<i32>> present;
      present.reserve(size_t(n_base + 1));
      for (i64 i = 0; i < n_base; ++i) {
        const BaseRec& r = f.rows[size_t(base_src[size_t(i)])];
        if (r.amount > 0) present[(u64(u32(r.g)) << 40) | u64(r.amount)].push_back(i32(i));
      }
      i64 n_ins = 0;
      for (i32 id : tids) {
        const BaseRec& r = f.rows[size_t(id)];
        Edit e{};
        if (trunk.ev.empty()) trunk.first_t = r.t;
        trunk.last_t = r.t;
        if (r.amount >= 0) {
          e.op = OP_INSERT; e.id = n_base + n_ins++; e.t = r.t; e.g = r.g; e.val = r.amount;
          present[(u64(u32(r.g)) << 40) | u64(r.amount)].push_back(i32(e.id));
          ++trunk.appends;
        } else {
          const u64 key = (u64(u32(r.g)) << 40) | u64(-r.amount);
          auto it = present.find(key);
          i32 hit = -1;
          if (it != present.end()) {
            auto& vec = it->second;
            while (!vec.empty() && vec.back() < 0) vec.pop_back();
            if (!vec.empty()) { hit = vec.back(); vec.pop_back(); }
          }
          if (hit >= 0) { e.op = OP_DELETE; e.id = hit; e.val = 0; ++trunk.reversals; }
          else {
            e.op = OP_INSERT; e.id = n_base + n_ins++; e.t = r.t; e.g = r.g; e.val = r.amount;
            ++trunk.credits;
          }
        }
        trunk.ev.push_back(e);
      }
      dummy = n_ins;
    }
    const i64 trunk_ins = dummy;
    i64 a_ins = build_branch(aids.data(), NA, 0, n_base + trunk_ins, brA);
    (void)a_ins;
    std::vector<i32> bids(stream.begin() + K + NA, stream.begin() + K + NA + NB);
    build_branch(bids.data(), NB, 1, n_base + trunk_ins + a_ins, brB);
  }
  // global inserted-row declarations, ordered by new id (trunk block, A block, B block)
  std::vector<i64> ins_t;
  std::vector<i32> ins_g;
  auto declare_block = [&](const BranchEdit& be) {
    for (const Edit& e : be.ev) {
      if (e.op == OP_INSERT) { ins_t.push_back(e.t); ins_g.push_back(e.g); }
    }
  };
  declare_block(trunk);
  declare_block(brA);
  declare_block(brB);
  const i64 n_ins = i64(ins_t.size());

  // ---- assemble versions
  std::vector<i32> parent(static_cast<size_t>(V), -1);
  std::vector<Edit> edit(static_cast<size_t>(V));
  const i32 gA = G - 1, gB = G - 2;  // declared factor-contract groups (real group ids)
  auto set_chain = [&](std::vector<Edit>& dst, i32 start, i32 par, const std::vector<Edit>& ev) {
    for (size_t j = 0; j < ev.size(); ++j) {
      const i32 v = start + i32(j);
      dst[size_t(v)] = ev[j];
      parent[size_t(v)] = (j == 0) ? par : v - 1;
    }
  };
  set_chain(edit, 1, 0, trunk.ev);                 // v1..vK
  { Edit e{}; e.op = OP_FACTOR; e.g = gA; e.val = 1050; edit[size_t(vFA)] = e; parent[size_t(vFA)] = vP; }
  set_chain(edit, vA0, vFA, brA.ev);
  { Edit e{}; e.op = OP_FACTOR; e.g = gB; e.val = 950; edit[size_t(vFB)] = e; parent[size_t(vFB)] = vP; }
  set_chain(edit, vB0, vFB, brB.ev);
  // merge factor node: re-applies branch B's declared factor on top of A_tip (A never changes gB)
  { Edit e{}; e.op = OP_FACTOR; e.g = gB; e.val = 950; edit[size_t(vFM)] = e; parent[size_t(vFM)] = vAtip; }
  set_chain(edit, vM0, vFM, brB.ev);               // replay B's real events -> merged tip
  edit[0].op = OP_REPLACE; edit[0].id = 0; edit[0].val = 0;  // unused root slot

  // ---- declared per-state nested query ranges (contract workload, deterministic)
  std::vector<Query> queries(static_cast<size_t>(V) * size_t(q), Query{0, 0});
  std::vector<Query> tmp(static_cast<size_t>(kMaxQ));
  std::vector<i64> endpoints;
  endpoints.reserve(size_t(V) * size_t(q) * 2);
  for (i32 v = 1; v < V; ++v) {
    Rng qr(u64(seed) * 1000000007ULL + u64(v) * 2654435761ULL + 17ULL);
    gen_nested(qr, f.h.t_min, f.h.t_max, kMaxQ, tmp.data());
    for (i32 k = 0; k < q; ++k) {
      queries[size_t(v) * size_t(q) + size_t(k)] = tmp[size_t(k)];
      endpoints.push_back(tmp[size_t(k)].lo);
      endpoints.push_back(tmp[size_t(k)].hi);
    }
  }
  std::sort(endpoints.begin(), endpoints.end());
  endpoints.erase(std::unique(endpoints.begin(), endpoints.end()), endpoints.end());
  if (endpoints.empty()) endpoints.push_back(f.h.t_min);

  const i64 real_events = i64(K) + i64(NA) + i64(NB) + i64(NB);  // merged replay repeats B
  const i64 real_reversals = trunk.reversals + brA.reversals + brB.reversals + brB.reversals;
  if (probe) {
    std::printf("{\"versions\":%d,\"base_rows\":%" PRId64 ",\"trunk_events\":%d,"
                "\"branch_a_events\":%d,\"branch_b_events\":%d,\"merge_replay_events\":%d,"
                "\"declared_factor_nodes\":3,\"real_events\":%" PRId64 ",\"real_reversals\":%" PRId64
                ",\"n_ins\":%" PRId64 ",\"endpoints\":%zu,\"q\":%d,\"version_layout\":{"
                "\"fork_parent\":%d,\"branch_a_factor\":%d,\"branch_a_start\":%d,"
                "\"branch_a_tip\":%d,\"branch_b_factor\":%d,\"branch_b_start\":%d,"
                "\"branch_b_tip\":%d,\"merge_factor\":%d,\"merged_tip\":%d},"
                "\"branch_intervals\":{\"trunk\":[%" PRId64 ",%" PRId64 "],"
                "\"a\":[%" PRId64 ",%" PRId64 "],\"b\":[%" PRId64 ",%" PRId64 "]},"
                "\"branch_mix\":{\"trunk\":{\"append\":%" PRId64 ",\"reversal\":%" PRId64
                ",\"credit\":%" PRId64 "},\"a\":{\"append\":%" PRId64 ",\"reversal\":%" PRId64
                ",\"credit\":%" PRId64 "},\"b\":{\"append\":%" PRId64 ",\"reversal\":%" PRId64
                ",\"credit\":%" PRId64 "}},\"epoch\":\"%s\",\"unit\":\"%s\"}\n",
                V, n_base, K, NA, NB, NB, real_events, real_reversals, n_ins, endpoints.size(), q,
                vP, vFA, vA0, vAtip, vFB, vB0, vBtip, vFM, vMtip,
                trunk.first_t, trunk.last_t, brA.first_t, brA.last_t, brB.first_t, brB.last_t,
                trunk.appends, trunk.reversals, trunk.credits, brA.appends, brA.reversals,
                brA.credits, brB.appends, brB.reversals, brB.credits, epoch.c_str(), unit.c_str());
    return 0;
  }
  if (out_base.empty() || out_spec.empty()) { std::fprintf(stderr, "missing output paths\n"); return 2; }
  // ---- version-0 snapshot fixture
  {
    FILE* fp = std::fopen(out_base.c_str(), "wb");
    if (!fp) { std::fprintf(stderr, "cannot write base fixture\n"); return 6; }
    FixtureHeader h = f.h;
    h.n = n_base;
    h.original_lines = n_base;
    bool ok = write_all(fp, &h, sizeof(h));
    for (i64 i = 0; i < n_base && ok; ++i) {
      const BaseRec& r = f.rows[size_t(base_src[size_t(i)])];
      ok = write_all(fp, &r, sizeof(r));
    }
    ok = ok && write_all(fp, f.base_factor.data(), size_t(G) * sizeof(i64));
    ok = ok && write_all(fp, f.group_labels.data(), size_t(G) * sizeof(i32));
    std::fclose(fp);
    if (!ok) { std::fprintf(stderr, "short base write\n"); return 6; }
  }
  // ---- declared history spec (BWHHIST1, same format as the chain real-history benchmark)
  {
    FILE* fp = std::fopen(out_spec.c_str(), "wb");
    if (!fp) { std::fprintf(stderr, "cannot write spec\n"); return 6; }
    bool ok = write_all(fp, "BWHHIST1", 8);
    const i32 kst = V, qq = q, fpp = 0, gg = G, topo = 1;  // 1 = branched/multi-child tree
    ok = ok && write_all(fp, &kst, 4) && write_all(fp, &qq, 4) && write_all(fp, &fpp, 4) &&
         write_all(fp, &gg, 4) && write_all(fp, &topo, 4);
    const i64 nb = n_base, ni = n_ins, tmn = f.h.t_min, tmx = f.h.t_max,
              nep = i64(endpoints.size()), plo = period_lo, phi = 0, bk = 0,
              nins = i64(real_events - real_reversals), ndel = real_reversals,
              ncred = trunk.credits + brA.credits + brB.credits + brB.credits, nfac = 3;
    ok = ok && write_all(fp, &nb, 8) && write_all(fp, &ni, 8) &&
         write_all(fp, &tmn, 8) && write_all(fp, &tmx, 8) && write_all(fp, &nep, 8) &&
         write_all(fp, &plo, 8) && write_all(fp, &phi, 8) && write_all(fp, &bk, 8) &&
         write_all(fp, &nins, 8) && write_all(fp, &ndel, 8) && write_all(fp, &ncred, 8) &&
         write_all(fp, &nfac, 8);
    ok = ok && write_all(fp, parent.data(), parent.size() * 4) &&
         write_all(fp, edit.data(), edit.size() * sizeof(Edit)) &&
         write_all(fp, queries.data(), queries.size() * sizeof(Query)) &&
         write_all(fp, ins_t.data(), ins_t.size() * 8) &&
         write_all(fp, ins_g.data(), ins_g.size() * 4) &&
         write_all(fp, endpoints.data(), endpoints.size() * 8);
    std::fclose(fp);
    if (!ok) { std::fprintf(stderr, "short spec write\n"); return 6; }
  }
  std::printf("{\"versions\":%d,\"base_rows\":%" PRId64 ",\"real_events\":%" PRId64
              ",\"real_reversals\":%" PRId64 ",\"n_ins\":%" PRId64 ",\"endpoints\":%zu"
              ",\"fork_parent\":%d,\"branch_a_tip\":%d,\"branch_b_tip\":%d,\"merged_tip\":%d}\n",
              V, n_base, real_events, real_reversals, n_ins, endpoints.size(), vP, vAtip, vBtip,
              vMtip);
  return 0;
}
