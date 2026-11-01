// active_density.cpp — report the true active-modification density of a declared workload.
//
// For the declared version tree this walks the same reversible DFS as the panel and reports,
// per state, the number of rows whose current (existence, timestamp, group, amount) differs from
// the base fixture, the number of nonzero per-group deviation coordinates (the D parameter of
// the complexity model), and the cumulative edit and factor-edit counts. This replaces the
// cumulative-edits/N proxy with the quantity the reviewer asked for:
//   rho_active = |{rows currently different from base}| / N.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "common.hpp"

using namespace bwh;

int main(int argc, char** argv) {
  std::string fixture_path, out_path;
  i32 topology = 0, seed = 7, factor_period = 4, q = 2;
  bool per_state = false;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](const char* k) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing %s\n", k); std::exit(2); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--fixture")) fixture_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--topology")) topology = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--seed")) seed = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--factor-period")) factor_period = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--q")) q = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--per-state")) per_state = true;
    else if (!std::strcmp(argv[i], "--out")) out_path = next(argv[i]);
    else { std::fprintf(stderr, "unknown %s\n", argv[i]); return 2; }
  }
  Fixture f;
  if (fixture_path.empty() || !load_fixture(fixture_path, f)) {
    std::fprintf(stderr, "cannot load fixture\n");
    return 3;
  }
  Workload w = make_workload(f, topology, seed, factor_period, q, per_state);
  const i64 N = f.h.n;
  const i64 n_rows = N + i64(w.n_inserted);
  const i32 G = f.h.groups;
  std::vector<i64> cur_t(static_cast<size_t>(n_rows), 0), cur_amt(static_cast<size_t>(n_rows), 0);
  std::vector<i32> cur_g(static_cast<size_t>(n_rows), 0);
  std::vector<u8> cur_ex(static_cast<size_t>(n_rows), 0);
  for (i64 i = 0; i < N; ++i) {
    cur_t[size_t(i)] = f.rows[size_t(i)].t;
    cur_amt[size_t(i)] = f.rows[size_t(i)].amount;
    cur_g[size_t(i)] = f.rows[size_t(i)].g;
    cur_ex[size_t(i)] = 1;
  }
  auto cls = [&](i64 t) {
    return i32(std::upper_bound(w.endpoints.begin(), w.endpoints.end(), t) - w.endpoints.begin());
  };
  std::vector<std::map<i32, i64>> dmap(static_cast<size_t>(G));
  std::vector<i64> base_t(static_cast<size_t>(n_rows), 0), base_amt(static_cast<size_t>(n_rows), 0);
  std::vector<i32> base_g(static_cast<size_t>(n_rows), 0);
  std::vector<u8> base_ex(static_cast<size_t>(n_rows), 0);
  for (i64 i = 0; i < N; ++i) {
    base_t[size_t(i)] = f.rows[size_t(i)].t;
    base_amt[size_t(i)] = f.rows[size_t(i)].amount;
    base_g[size_t(i)] = f.rows[size_t(i)].g;
    base_ex[size_t(i)] = 1;
  }
  auto is_dirty = [&](i64 id) {
    return cur_ex[size_t(id)] != base_ex[size_t(id)] ||
           (cur_ex[size_t(id)] && (cur_t[size_t(id)] != base_t[size_t(id)] ||
                                   cur_g[size_t(id)] != base_g[size_t(id)] ||
                                   cur_amt[size_t(id)] != base_amt[size_t(id)]));
  };
  auto delta_add = [&](i32 g, i32 c, i64 v) {
    if (v == 0) return;
    auto& m = dmap[size_t(g)];
    auto it = m.find(c);
    if (it == m.end()) m.emplace(c, v);
    else { it->second += v; if (it->second == 0) m.erase(it); }
  };
  auto set_row = [&](i64 id, i64 t, i32 g, i64 amt, u8 ex) {
    if (cur_ex[size_t(id)] && cur_amt[size_t(id)] != 0)
      delta_add(cur_g[size_t(id)], cls(cur_t[size_t(id)]), -cur_amt[size_t(id)]);
    cur_t[size_t(id)] = t; cur_g[size_t(id)] = g; cur_amt[size_t(id)] = amt; cur_ex[size_t(id)] = ex;
    if (ex && amt != 0) delta_add(g, cls(t), amt);
  };
  std::vector<i32> parent(static_cast<size_t>(kStates));
  std::vector<std::vector<i32>> kids(static_cast<size_t>(kStates));
  for (i32 v = 1; v < kStates; ++v) { parent[size_t(v)] = w.parent[size_t(v)]; kids[size_t(parent[size_t(v)])].push_back(v); }
  struct Ev { i32 v; i32 id; i32 op; i64 t; i32 g; i64 amt; u8 ex; u8 exit; };
  std::vector<Ev> st; st.reserve(size_t(kStates) * 2);
  st.push_back({0, 0, 0, 0, 0, 0, 0, 0});
  i64 active = 0;
  std::vector<i64> act; act.reserve(size_t(kStates));
  std::vector<i64> dsum; dsum.reserve(size_t(kStates));
  while (!st.empty()) {
    Ev ev = st.back(); st.pop_back();
    const i32 v = ev.v;
    if (ev.exit) {
      if (ev.op == OP_FACTOR) continue;
      const bool before = is_dirty(ev.id);
      set_row(ev.id, ev.t, ev.g, ev.amt, ev.ex);
      const bool after = is_dirty(ev.id);
      if (before && !after) --active;
      else if (!before && after) ++active;
      continue;
    }
    if (v > 0) {
      const Edit& e = w.edit[size_t(v)];
      Ev saved{}; saved.v = v; saved.exit = 1; saved.op = e.op;
      saved.id = e.id;
      saved.t = cur_t[size_t(e.id)];
      saved.g = cur_g[size_t(e.id)];
      saved.amt = cur_amt[size_t(e.id)];
      saved.ex = cur_ex[size_t(e.id)];
      if (e.op != OP_FACTOR) {
        const bool was = is_dirty(e.id);
        i64 t = cur_t[size_t(e.id)], amt = cur_amt[size_t(e.id)];
        i32 g = cur_g[size_t(e.id)];
        u8 ex = cur_ex[size_t(e.id)];
        if (e.op == OP_REPLACE) { amt = e.val; ex = 1; }
        else if (e.op == OP_DELETE) ex = 0;
        else if (e.op == OP_INSERT) { t = e.t; g = e.g; amt = e.val; ex = 1; }
        else if (e.op == OP_MOVE) { t = e.t; g = e.g; amt = e.val; ex = 1; }
        set_row(e.id, t, g, amt, ex);
        const bool now = is_dirty(e.id);
        if (!was && now) ++active;
        else if (was && !now) --active;
      }
      act.push_back(active);
      i64 d = 0; for (auto& m : dmap) d += i64(m.size());
      dsum.push_back(d);
      st.push_back(saved);
    }
    for (auto it = kids[size_t(v)].rbegin(); it != kids[size_t(v)].rend(); ++it) {
      Ev child{}; child.v = *it; child.exit = 0;
      // child edits are applied when the child event is processed
      st.push_back(child);
    }
  }
  std::sort(act.begin(), act.end());
  std::sort(dsum.begin(), dsum.end());
  auto med = [](const std::vector<i64>& a) { return a.empty() ? i64(0) : a[a.size() / 2]; };
  i64 edits = kStates - 1, fedits = 0;
  for (i32 v = 1; v < kStates; ++v) if (w.edit[size_t(v)].op == OP_FACTOR) ++fedits;
  std::string out = out_path.empty() ? std::string("-") : out_path;
  FILE* fp = out == "-" ? stdout : std::fopen(out.c_str(), "w");
  std::fprintf(fp,
    "{\"fixture\":\"%s\",\"states\":%d,\"rows\":%lld,\"groups\":%d,"
    "\"q\":%d,\"factor_period\":%d,\"edits\":%lld,\"factor_edits\":%lld,"
    "\"active_rows_max\":%lld,\"rho_active_max\":%.5f,"
    "\"active_rows_median\":%lld,\"rho_active_median\":%.5f,"
    "\"cumulative_edits_over_rows\":%.5f,"
    "\"nonzero_dev_coords_max\":%lld,\"nonzero_dev_coords_median\":%lld}\n",
    fixture_path.c_str(), kStates, (long long)N, G, q, factor_period, (long long)edits,
    (long long)fedits, (long long)act.back(), double(act.back()) / double(N),
    (long long)med(act), double(med(act)) / double(N), double(edits) / double(N),
    (long long)dsum.back(), (long long)med(dsum));
  if (fp != stdout) std::fclose(fp);
  return 0;
}
