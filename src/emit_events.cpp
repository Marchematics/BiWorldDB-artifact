// Emit a sequential external-engine event stream for the declared BiWorldDB workload.
// Used by the Rust Differential Dataflow probe: the C++ workload generator stays the
// single source of truth for edits, queries and DFS ordering.
//
// Event record (36 bytes): kind u32, state u32, qidx u32, g i32, w i32, a i64, b i64
//   kind 0 row delta : a=t, b=signed amount, g=group, w=+1 insert / -1 retract
//   kind 1 factor set: g=group, a=new multiplier (thousandths)
//   kind 2 query     : state=v, qidx=k, a=lo, b=hi
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common.hpp"

using namespace bwh;

struct SavedState { i64 t; i32 g; i64 amount; u8 ex; i64 factor; };

static void put_event(FILE* fp, u32 kind, u32 state, u32 qidx, i32 g, i32 w, i64 a, i64 b) {
  u32 k = kind, s = state, q = qidx;
  i32 gg = g, ww = w;
  std::fwrite(&k, 4, 1, fp); std::fwrite(&s, 4, 1, fp); std::fwrite(&q, 4, 1, fp);
  std::fwrite(&gg, 4, 1, fp); std::fwrite(&ww, 4, 1, fp);
  std::fwrite(&a, 8, 1, fp); std::fwrite(&b, 8, 1, fp);
}

int main(int argc, char** argv) {
  std::string fixture, out;
  i32 topology = 0, seed = 7, fp_period = 37, q = 2;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    auto nxt = [&]() -> const char* { return argv[++i]; };
    if (!std::strcmp(a, "--fixture")) fixture = nxt();
    else if (!std::strcmp(a, "--out")) out = nxt();
    else if (!std::strcmp(a, "--topology")) topology = std::atoi(nxt());
    else if (!std::strcmp(a, "--seed")) seed = std::atoi(nxt());
    else if (!std::strcmp(a, "--factor-period")) fp_period = std::atoi(nxt());
    else if (!std::strcmp(a, "--q")) q = std::atoi(nxt());
  }
  if (fixture.empty() || out.empty()) { std::fprintf(stderr, "usage\n"); return 2; }
  Fixture f;
  if (!load_fixture(fixture, f)) { std::fprintf(stderr, "bad fixture\n"); return 3; }
  Workload w = make_workload(f, topology, seed, fp_period, q, /*per_state_ranges=*/true);
  const i64 N = f.h.n;
  const i64 n_rows = N + w.n_inserted;
  std::vector<i64> t(size_t(n_rows), 0), amt(size_t(n_rows), 0);
  std::vector<i32> g(size_t(n_rows), 0);
  std::vector<u8> ex(size_t(n_rows), 0);
  std::vector<i64> mul = f.base_factor;
  for (i64 i = 0; i < N; ++i) {
    t[size_t(i)] = f.rows[size_t(i)].t; amt[size_t(i)] = f.rows[size_t(i)].amount;
    g[size_t(i)] = f.rows[size_t(i)].g; ex[size_t(i)] = 1;
  }
  for (i32 k = 0; k < w.n_inserted; ++k) {
    t[size_t(N + k)] = w.ins_t[size_t(k)];
    g[size_t(N + k)] = w.ins_g[size_t(k)];
  }
  FILE* out_fp = std::fopen((out + ".tmp").c_str(), "wb");
  if (!out_fp) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return 4; }
  const char magic[8] = {'B','W','L','E','V','0','0','1'};
  std::fwrite(magic, 8, 1, out_fp);
  u64 n_events_placeholder = 0;
  std::fwrite(&n_events_placeholder, 8, 1, out_fp);
  u64 n_events = 0;

  auto emit_row = [&](i64 id, i32 weight) {
    if (!ex[size_t(id)]) return;
    put_event(out_fp, 0, 0, 0, g[size_t(id)], weight, t[size_t(id)], amt[size_t(id)]);
    ++n_events;
  };
  auto emit_factor = [&](i32 grp, i64 value) {
    put_event(out_fp, 1, 0, 0, grp, 0, value, 0);
    ++n_events;
  };

  std::vector<SavedState> saved(static_cast<size_t>(kStates));
  std::vector<std::vector<i32>> kids(static_cast<size_t>(kStates));
  for (i32 v = 1; v < kStates; ++v) kids[size_t(w.parent[size_t(v)])].push_back(v);
  std::vector<std::pair<i32, u8>> stack;  // node, exit flag
  stack.push_back({0, 0});
  while (!stack.empty()) {
    auto [v, exiting] = stack.back();
    stack.pop_back();
    if (exiting) {
      if (v > 0) {
        const SavedState& s = saved[size_t(v)];
        const Edit& e = w.edit[size_t(v)];
        i64 id = e.id;
        if (e.op == OP_FACTOR) {
          mul[size_t(e.g)] = s.factor;
          emit_factor(e.g, s.factor);
        } else {
          emit_row(id, -1);  // retract current record
          t[size_t(id)] = s.t; g[size_t(id)] = s.g; amt[size_t(id)] = s.amount;
          ex[size_t(id)] = s.ex;
          emit_row(id, +1);  // insert restored record
        }
      }
      continue;
    }
    if (v > 0) {
      const Edit& e = w.edit[size_t(v)];
      if (e.op == OP_FACTOR) {
        SavedState& s = saved[size_t(v)];
        s.factor = mul[size_t(e.g)];
        mul[size_t(e.g)] = e.val;
        emit_factor(e.g, e.val);
      } else {
        i64 id = e.id;
        SavedState& s = saved[size_t(v)];
        s.t = t[size_t(id)]; s.g = g[size_t(id)]; s.amount = amt[size_t(id)]; s.ex = ex[size_t(id)];
        emit_row(id, -1);  // retract old record
        switch (e.op) {
          case OP_REPLACE: amt[size_t(id)] = e.val; ex[size_t(id)] = 1; break;
          case OP_INSERT: case OP_MOVE:
            t[size_t(id)] = e.t; g[size_t(id)] = e.g; amt[size_t(id)] = e.val; ex[size_t(id)] = 1; break;
          case OP_DELETE: ex[size_t(id)] = 0; break;
        }
        emit_row(id, +1);  // insert new record
      }
      for (i32 k = 0; k < q; ++k) {
        const Query& qq = query_for(w, v, k);
        put_event(out_fp, 2, u32(v), u32(k), 0, 0, qq.lo, qq.hi);
        ++n_events;
      }
    }
    stack.push_back({v, 1});
    for (auto it = kids[size_t(v)].rbegin(); it != kids[size_t(v)].rend(); ++it) stack.push_back({*it, 0});
  }
  std::fseek(out_fp, 8, SEEK_SET);
  std::fwrite(&n_events, 8, 1, out_fp);
  std::fclose(out_fp);
  if (std::rename((out + ".tmp").c_str(), out.c_str()) != 0) return 5;
  std::printf("events=%" PRIu64 "\n", n_events);
  return 0;
}
