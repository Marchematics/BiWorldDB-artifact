// panel_real_branch_oracle.cpp — independent oracle for the real branching version history
// (2026-11-01, real-branch-history P1).  New file; shipped files untouched.
//
// Shares no index with the engines: it replays the declared version tree in plain row arrays
// (enter/exit DFS with explicit previous-value records) and recomputes every sampled state's
// answers by a from-scratch scan  Q(I) = sum_{present rows with lo<=t<hi} fac[g]*amount.
//
// Merge semantics are validated independently as well: the fork's two branch states are
// recomputed from the base snapshot by a plain sequential replay of each branch's real event
// interval + its declared factor node, and the merged descendant is recomputed as
// base + branch A events + branch B events + both declared factor assignments.  The oracle
// compares (a) the tree replay's branch-tip and merged-tip answers against those independent
// reference states and (b) the merged-tip row state against the reference union state,
// so the "merge/rejoin" is verified to be a genuine union of the two branches.
//
// CLI: panel_real_branch_oracle --fixture B --spec J --stride S
//        --fork F --fa FA --a-start AS --a-tip AT --fb FB --b-start BS --b-tip BT
//        --fm FM --merge M --answers A --json J
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.hpp"
#include "real_branch_spec.hpp"
#include "sha256.hpp"

using namespace bwh;

static bool write_all(FILE* fp, const void* p, size_t n) { return std::fwrite(p, 1, n, fp) == n; }

struct RowState {
  std::vector<i64> t, amt;
  std::vector<i32> g;
  std::vector<u8> ex;
  std::vector<i64> fac;
  void init(const Fixture& f, i64 n_total) {
    t.assign(size_t(n_total), 0);
    amt.assign(size_t(n_total), 0);
    g.assign(size_t(n_total), 0);
    ex.assign(size_t(n_total), 0);
    fac = f.base_factor;
    for (i64 i = 0; i < f.h.n; ++i) {
      const BaseRec& r = f.rows[size_t(i)];
      t[size_t(i)] = r.t; amt[size_t(i)] = r.amount; g[size_t(i)] = r.g; ex[size_t(i)] = 1;
    }
  }
  inline void apply(const Edit& e, i64 n_total) {
    switch (e.op) {
      case OP_REPLACE:
        if (e.id >= 0 && e.id < n_total) { amt[size_t(e.id)] = e.val; ex[size_t(e.id)] = 1; }
        break;
      case OP_INSERT:
      case OP_MOVE:
        if (e.id >= 0 && e.id < n_total) {
          t[size_t(e.id)] = e.t; g[size_t(e.id)] = e.g; amt[size_t(e.id)] = e.val;
          ex[size_t(e.id)] = 1;
        }
        break;
      case OP_DELETE:
        if (e.id >= 0 && e.id < n_total) ex[size_t(e.id)] = 0;
        break;
      case OP_FACTOR:
        if (e.g >= 0 && e.g < i32(fac.size())) fac[size_t(e.g)] = e.val;
        break;
      default: break;
    }
  }
  inline i64 answer(i64 lo, i64 hi, i64 n_total) const {
    i64 sum = 0;
    for (i64 id = 0; id < n_total; ++id) {
      if (!ex[size_t(id)]) continue;
      const i64 tt = t[size_t(id)];
      if (tt < lo || tt >= hi) continue;
      sum += fac[size_t(g[size_t(id)])] * amt[size_t(id)];
    }
    return sum;
  }
};

int main(int argc, char** argv) {
  std::string fixture_path, spec_path, answers_path, json_path;
  i32 stride = 1, repeat = 0;
  i32 vFork = -1, vFA = -1, vA0 = -1, vAtip = -1, vFB = -1, vB0 = -1, vBtip = -1, vFM = -1,
      vMerge = -1;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](const char* k) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k); std::exit(2); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--fixture")) fixture_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--spec")) spec_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--stride")) stride = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--repeat")) repeat = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--fork")) vFork = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--fa")) vFA = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--a-start")) vA0 = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--a-tip")) vAtip = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--fb")) vFB = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--b-start")) vB0 = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--b-tip")) vBtip = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--fm")) vFM = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--merge")) vMerge = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--answers")) answers_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--json")) json_path = next(argv[i]);
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  if (fixture_path.empty() || spec_path.empty() || answers_path.empty() || json_path.empty()) {
    std::fprintf(stderr, "usage: panel_real_branch_oracle --fixture B --spec J --stride S ...\n");
    return 2;
  }
  if (stride < 1) stride = 1;
  const auto t0 = std::chrono::steady_clock::now();
  Fixture f;
  if (!load_fixture(fixture_path, f)) { std::fprintf(stderr, "cannot load fixture\n"); return 3; }
  RealHistorySpec spec;
  if (!load_real_history_spec(spec_path, spec)) { std::fprintf(stderr, "cannot load spec\n"); return 3; }
  if (spec.n_base != f.h.n) { std::fprintf(stderr, "spec/fixture base mismatch\n"); return 3; }
  const i32 q = spec.q;
  const i64 n_total = spec.n_base + spec.n_ins;
  std::vector<i64> answers(size_t(kStates - 1) * size_t(q), 0);
  std::vector<u8> sampled(size_t(kStates - 1), 0);
  i64 n_sampled = 0;

  // ---- children lists for the enter/exit replay
  std::vector<i32> cnt(static_cast<size_t>(kStates) + 1, 0);
  for (i32 v = 1; v < kStates; ++v) ++cnt[size_t(spec.parent[size_t(v)]) + 1];
  for (i32 v = 0; v < kStates; ++v) cnt[size_t(v) + 1] += cnt[size_t(v)];
  std::vector<i32> kids(static_cast<size_t>(kStates - 1));
  {
    std::vector<i32> off(cnt.begin(), cnt.begin() + kStates);
    for (i32 v = 1; v < kStates; ++v) kids[size_t(off[size_t(spec.parent[size_t(v)])]++)] = v;
  }
  struct Undo {
    u8 kind = 0;  // 0 row, 1 factor
    i64 id = 0, t = 0, amt = 0;
    i32 g = 0;
    u8 ex = 0;
    i64 fval = 0;
  };
  std::vector<Undo> undo(static_cast<size_t>(kStates));
  RowState cur;
  cur.init(f, n_total);
  RowState merge_state;  // captured at the merged descendant
  merge_state.init(f, n_total);
  auto is_special = [&](i32 v) {
    return v == vFork || v == vAtip || v == vBtip || v == vMerge;
  };
  auto answer_state = [&](i32 v) {
    for (i32 k = 0; k < q; ++k) {
      const Query& qq = spec.query(v, k);
      answers[size_t(v - 1) * size_t(q) + size_t(k)] = cur.answer(qq.lo, qq.hi, n_total);
    }
    sampled[size_t(v - 1)] = 1;
    ++n_sampled;
  };
  auto enter = [&](i32 v) {
    const Edit& e = spec.edit[size_t(v)];
    Undo& u = undo[size_t(v)];
    if (e.op == OP_FACTOR) {
      u.kind = 1;
      u.g = e.g;
      u.fval = cur.fac[size_t(e.g)];
    } else {
      u.kind = 0;
      u.id = e.id;
      u.t = cur.t[size_t(e.id)];
      u.amt = cur.amt[size_t(e.id)];
      u.g = cur.g[size_t(e.id)];
      u.ex = cur.ex[size_t(e.id)];
    }
    cur.apply(e, n_total);
  };
  auto leave = [&](i32 v) {
    const Undo& u = undo[size_t(v)];
    if (u.kind == 1) {
      cur.fac[size_t(u.g)] = u.fval;
    } else {
      const i64 id = u.id;
      cur.t[size_t(id)] = u.t;
      cur.amt[size_t(id)] = u.amt;
      cur.g[size_t(id)] = u.g;
      cur.ex[size_t(id)] = u.ex;
    }
  };
  std::vector<std::pair<i32, u8>> st;
  st.reserve(static_cast<size_t>(kStates) + 8);
  st.push_back({0, 0});
  while (!st.empty()) {
    const auto ev = st.back();
    st.pop_back();
    const i32 v = ev.first;
    if (v == 0) {
      if (ev.second) continue;  // root exit: nothing to restore
      st.push_back({0, 1});
      for (i32 j = cnt[size_t(0) + 1] - 1; j >= cnt[size_t(0)]; --j) st.push_back({kids[size_t(j)], 0});
      continue;
    }
    if (ev.second) { leave(v); continue; }
    enter(v);
    if ((v % stride == 0) || is_special(v)) {
      answer_state(v);
      if (v == vMerge) merge_state = cur;  // copy of the replay state at the merge
    }
    st.push_back({v, 1});
    for (i32 j = cnt[size_t(v) + 1] - 1; j >= cnt[size_t(v)]; --j) st.push_back({kids[size_t(j)], 0});
  }
  const auto t1 = std::chrono::steady_clock::now();

  // ---- independent reference states: branch A, branch B, and the merge union.
  // They start from the fork parent state P = base snapshot + the real trunk path (versions
  // 1..fork), replayed by a plain sequential apply loop (not the DFS).
  auto init_P = [&](RowState& s) {
    s.init(f, n_total);
    for (i32 v = 1; v <= vFork; ++v) s.apply(spec.edit[size_t(v)], n_total);
  };
  auto ref_answers = [&](const std::vector<const Edit*>& evs, i32 factor_node,
                         const std::vector<Query>& qs, std::vector<i64>& out) {
    RowState s;
    init_P(s);
    if (factor_node >= 0) s.apply(spec.edit[size_t(factor_node)], n_total);
    for (const Edit* e : evs) s.apply(*e, n_total);
    out.resize(qs.size());
    for (size_t k = 0; k < qs.size(); ++k) out[k] = s.answer(qs[k].lo, qs[k].hi, n_total);
  };
  std::vector<const Edit*> aEvents, bEvents;
  for (i32 v = vA0; v <= vAtip; ++v) aEvents.push_back(&spec.edit[size_t(v)]);
  for (i32 v = vB0; v <= vBtip; ++v) bEvents.push_back(&spec.edit[size_t(v)]);
  std::vector<Query> qsAtip, qsBtip, qsMerge;
  for (i32 k = 0; k < q; ++k) {
    qsAtip.push_back(spec.query(vAtip, k));
    qsBtip.push_back(spec.query(vBtip, k));
    qsMerge.push_back(spec.query(vMerge, k));
  }
  std::vector<i64> refA, refB, refM;
  ref_answers(aEvents, vFA, qsAtip, refA);
  ref_answers(bEvents, vFB, qsBtip, refB);
  RowState refMerge;
  init_P(refMerge);
  refMerge.apply(spec.edit[size_t(vFA)], n_total);
  for (const Edit* e : aEvents) refMerge.apply(*e, n_total);
  refMerge.apply(spec.edit[size_t(vFM)], n_total);
  for (const Edit* e : bEvents) refMerge.apply(*e, n_total);
  refM.resize(qsMerge.size());
  for (size_t k = 0; k < qsMerge.size(); ++k)
    refM[k] = refMerge.answer(qsMerge[k].lo, qsMerge[k].hi, n_total);
  auto cmp = [&](i32 v, const std::vector<i64>& ref) {
    i64 bad = 0;
    for (i32 k = 0; k < q; ++k)
      if (answers[size_t(v - 1) * size_t(q) + size_t(k)] != ref[size_t(k)]) ++bad;
    return bad;
  };
  i64 merge_row_diffs = 0;
  for (i64 id = 0; id < n_total; ++id) {
    if (merge_state.ex[size_t(id)] != refMerge.ex[size_t(id)] ||
        merge_state.t[size_t(id)] != refMerge.t[size_t(id)] ||
        merge_state.g[size_t(id)] != refMerge.g[size_t(id)] ||
        merge_state.amt[size_t(id)] != refMerge.amt[size_t(id)])
      ++merge_row_diffs;
  }
  i64 factor_diffs = 0;
  for (size_t g = 0; g < merge_state.fac.size(); ++g)
    if (merge_state.fac[g] != refMerge.fac[g]) ++factor_diffs;

  std::vector<i64> picked;
  for (i64 s = 0; s < i64(kStates - 1); ++s)
    if (sampled[size_t(s)])
      for (i32 k = 0; k < q; ++k) picked.push_back(answers[size_t(s) * size_t(q) + size_t(k)]);
  const std::string sha = sha::of(picked.data(), picked.size() * sizeof(i64));
  const i64 ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  FILE* out = std::fopen(answers_path.c_str(), "wb");
  if (!out) { std::fprintf(stderr, "cannot write answers\n"); return 4; }
  const bool ok = write_all(out, answers.data(), answers.size() * sizeof(i64));
  std::fclose(out);
  if (!ok) { std::fprintf(stderr, "short answer write\n"); return 4; }
  FILE* jf = std::fopen((json_path + ".tmp").c_str(), "w");
  if (!jf) { std::fprintf(stderr, "cannot write json\n"); return 4; }
  std::fprintf(jf, "{\n  \"arm\": \"oracle_bruteforce\",\n  \"family\": \"real_branch\",\n");
  std::fprintf(jf, "  \"fixture\": \"%s\",\n  \"spec\": \"%s\",\n", fixture_path.c_str(), spec_path.c_str());
  std::fprintf(jf, "  \"repeat\": %d,\n  \"n_states\": %d,\n  \"q\": %d,\n  \"stride\": %d,\n",
               repeat, kStates, q, stride);
  std::fprintf(jf, "  \"sampled_states\": %" PRId64 ",\n  \"sampled_answers\": %zu,\n"
                   "  \"sampled_sha256\": \"%s\",\n  \"oracle_ns\": %" PRId64 ",\n",
               n_sampled, picked.size(), sha.c_str(), ns);
  std::fprintf(jf, "  \"merge_check\": {\n");
  std::fprintf(jf, "    \"fork\": %d, \"branch_a_tip\": %d, \"branch_b_tip\": %d, \"merged_tip\": %d,\n",
               vFork, vAtip, vBtip, vMerge);
  std::fprintf(jf, "    \"branch_a_tip_answer_mismatches\": %" PRId64 ",\n", cmp(vAtip, refA));
  std::fprintf(jf, "    \"branch_b_tip_answer_mismatches\": %" PRId64 ",\n", cmp(vBtip, refB));
  std::fprintf(jf, "    \"merged_tip_answer_mismatches\": %" PRId64 ",\n", cmp(vMerge, refM));
  std::fprintf(jf, "    \"merged_tip_row_state_diffs\": %" PRId64 ",\n", merge_row_diffs);
  std::fprintf(jf, "    \"merged_tip_factor_diffs\": %" PRId64 "\n  },\n", factor_diffs);
  std::fprintf(jf, "  \"branch_a_tip_answers\": [");
  for (i32 k = 0; k < q; ++k)
    std::fprintf(jf, "%s%" PRId64, k ? "," : "",
                 answers[size_t(vAtip - 1) * size_t(q) + size_t(k)]);
  std::fprintf(jf, "],\n  \"branch_b_tip_answers\": [");
  for (i32 k = 0; k < q; ++k)
    std::fprintf(jf, "%s%" PRId64, k ? "," : "",
                 answers[size_t(vBtip - 1) * size_t(q) + size_t(k)]);
  std::fprintf(jf, "],\n  \"merged_tip_answers\": [");
  for (i32 k = 0; k < q; ++k)
    std::fprintf(jf, "%s%" PRId64, k ? "," : "",
                 answers[size_t(vMerge - 1) * size_t(q) + size_t(k)]);
  std::fprintf(jf, "]\n}\n");
  std::fclose(jf);
  if (std::rename((json_path + ".tmp").c_str(), json_path.c_str()) != 0) {
    std::fprintf(stderr, "cannot rename json\n");
    return 4;
  }
  std::printf("arm=oracle_bruteforce stride=%d sampled=%" PRId64 " merge_rows_diffs=%" PRId64
              " merge_ans_mismatch=%" PRId64 " a/b_ans_mismatch=%" PRId64 "/%" PRId64 "\n",
              stride, n_sampled, merge_row_diffs, cmp(vMerge, refM), cmp(vAtip, refA),
              cmp(vBtip, refB));
  return 0;
}
