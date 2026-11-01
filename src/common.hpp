// BiWorldDB hybrid-density continuation panel, 2026-10-25.
// Shared base fixture format + deterministic declared workload for the formal
// 1,296-worker timing panel (72 fixtures x 6 fresh-process repeats x 3 arms).
//
// Workload shape follows the recovered v7 manuscript (external/v7_20261024):
//   * 2,049 states (root + 2,048 single-edit version nodes);
//   * topologies chain / bushy / star over the 2,048 edited states;
//   * edits: row replacement, insertion, deletion, move, group-factor update;
//   * factors 950/1000/1050 thousandths, initial 1000;
//   * factor-update period 4 or 37 (one factor update every P-th edit);
//   * Q in {2,8,32} nested half-open ranges per state; the declared endpoint
//     domain E is the distinct set of endpoints over all states' first Q ranges.
// All mutations are absolute (set current value), so one declared edit list is
// valid on every root-to-node path. This file is authored; it is not the cloud
// implementation.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace bwh {

using i8 = int8_t;
using u8 = uint8_t;
using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;

// ---------------------------------------------------------------- RNG
struct Rng {
  u64 s;
  explicit Rng(u64 seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
  u64 next() {
    u64 z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  u32 bounded(u32 n) { return n ? u32(next() % n) : 0; }
  i64 range(i64 lo, i64 hi) {  // half-open
    if (hi <= lo) return lo;
    return lo + i64(next() % u64(hi - lo));
  }
};

// ---------------------------------------------------------------- fixture
struct FixtureHeader {
  char magic[8];       // "BWBASE02"
  i64 n;               // admitted numeric base rows
  i32 groups;          // dense group count
  i32 unit;            // 0 = microsecond timestamps, 1 = civil-hour labels
  i64 original_lines;  // source records read
  i64 missing_values;  // excluded NULL numeric values
  i64 t_min, t_max;    // raw support bounds (inclusive)
  i64 amount_scale;    // numeric amount = source value * 10^amount_scale
};
static_assert(sizeof(FixtureHeader) == 64, "header size");

struct BaseRec {
  i64 t;
  i64 amount;
  i32 g;
  i32 pad;
};
static_assert(sizeof(BaseRec) == 24, "record size");

// Zero-copy view of the mapped fixture rows. The file is mapped read-only once; the
// record bytes are used in place instead of being copied into a vector, so the load
// phase becomes a map + header parse and the page faults are charged to the phases
// that actually touch the rows.
struct RowView {
  const BaseRec* p = nullptr;
  size_t n = 0;
  inline const BaseRec& operator[](size_t i) const { return p[i]; }
  inline size_t size() const { return n; }
  inline const BaseRec* data() const { return p; }
  inline const BaseRec* begin() const { return p; }
  inline const BaseRec* end() const { return p + n; }
};

struct Fixture {
  FixtureHeader h{};
  RowView rows;
  std::vector<i64> base_factor;
  std::vector<i32> group_labels;
  void* map_base = nullptr;
  size_t map_len = 0;
  Fixture() = default;
  Fixture(const Fixture&) = delete;
  Fixture& operator=(const Fixture&) = delete;
  void release_rows() {
    if (map_base) {
      ::munmap(map_base, map_len);
      map_base = nullptr;
      map_len = 0;
      rows = RowView{};
    }
  }
  ~Fixture() { release_rows(); }
};

inline bool load_fixture(const std::string& path, Fixture& f) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return false;
  struct stat st {};
  if (::fstat(fd, &st) != 0 || st.st_size < i64(sizeof(FixtureHeader))) { ::close(fd); return false; }
  const size_t len = size_t(st.st_size);
  void* base = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (base == MAP_FAILED) return false;
  f.release_rows();
  f.map_base = base;
  f.map_len = len;
  std::memcpy(&f.h, base, sizeof(FixtureHeader));
  if (std::memcmp(f.h.magic, "BWBASE02", 8) != 0) { f.release_rows(); return false; }
  const i64 n = f.h.n;
  const i32 G = f.h.groups;
  if (n <= 0 || G <= 0 || n > (i64(1) << 40) || G > (1 << 20)) { f.release_rows(); return false; }
  const size_t off_rows = sizeof(FixtureHeader);
  const size_t off_fac = off_rows + size_t(n) * sizeof(BaseRec);
  const size_t off_lab = off_fac + size_t(G) * sizeof(i64);
  if (off_lab + size_t(G) * sizeof(i32) > len) { f.release_rows(); return false; }
  f.rows.p = reinterpret_cast<const BaseRec*>(static_cast<const char*>(base) + off_rows);
  f.rows.n = size_t(n);
  f.base_factor.resize(size_t(G), 1000);
  f.group_labels.resize(size_t(G), 0);
  std::memcpy(f.base_factor.data(), static_cast<const char*>(base) + off_fac, size_t(G) * sizeof(i64));
  std::memcpy(f.group_labels.data(), static_cast<const char*>(base) + off_lab, size_t(G) * sizeof(i32));
  // Optional validation of the mapped rows (BWH_VALIDATE_FIXTURE=1); disabled by default
  // because the rows are trusted artifacts and the scan is charged to the load phase.
  const char* val = std::getenv("BWH_VALIDATE_FIXTURE");
  if (val && val[0] == '1') {
    i64 tmin = f.rows[0].t, tmax = f.rows[0].t;
    for (i64 i = 0; i < n; ++i) {
      const BaseRec& r = f.rows[size_t(i)];
      if (r.g < 0 || r.g >= G) { f.release_rows(); return false; }
      if (r.t < tmin) tmin = r.t;
      if (r.t > tmax) tmax = r.t;
    }
    if (tmin != f.h.t_min || tmax != f.h.t_max) { f.release_rows(); return false; }
  }
  for (i32 g = 0; g < G; ++g)
    if (f.base_factor[size_t(g)] <= 0) { f.release_rows(); return false; }
  return true;
}

// ---------------------------------------------------------------- workload
enum : u8 {
  OP_REPLACE = 0,  // set amount of existing row, restore existence
  OP_INSERT = 1,   // create row at (t,g) with amount
  OP_DELETE = 2,   // set existence 0
  OP_MOVE = 3,     // set (t,g,amount), restore existence
  OP_FACTOR = 4,   // set group multiplier (thousandths)
};

struct Edit {
  u8 op;
  u8 pad[3];
  i32 g;    // group: OP_INSERT/OP_MOVE/OP_FACTOR
  i32 pad2;
  i64 id;   // row id (base 0..n-1; inserted n..)
  i64 val;  // amount or factor
  i64 t;    // timestamp: OP_INSERT/OP_MOVE
};
static_assert(sizeof(Edit) == 40, "edit size");

struct Query {
  i64 lo, hi;  // half open [lo, hi)
};

// Compile-time workload-size knobs (defaults reproduce the shipped panel).
// Build variants with -DBWH_STATES=... and/or -DBWH_MAXQ=... to run stress
// dimensions (version count, edits-per-state density via more states, Q up to 128).
#ifndef BWH_STATES
#define BWH_STATES 2049
#endif
#ifndef BWH_MAXQ
#define BWH_MAXQ 32
#endif
constexpr i32 kStates = BWH_STATES;
constexpr i32 kEdited = kStates - 1;
constexpr i32 kMaxQ = BWH_MAXQ;

struct Workload {
  i32 topology = 0;      // 0 chain, 1 bushy, 2 star
  i32 seed = 0;
  i32 factor_period = 0;
  i32 q = 0;             // queries used per state (2, 8 or 32)
  bool per_state = false;  // false: one nested set shared by all states;
                           // true: a fresh nested set declared for every state
  i32 n_base = 0;
  i32 n_inserted = 0;
  i32 groups = 0;
  i64 t_min = 0, t_max = 0;
  std::vector<i32> parent;      // kStates
  std::vector<Edit> edit;       // kStates (edit[0] unused)
  std::vector<Query> queries;   // shared scope: kMaxQ nested ranges at index 0..kMaxQ-1
                                // per-state scope: kStates*kMaxQ, state v at v*kMaxQ+k
  std::vector<i64> ins_t;       // inserted row timestamps, id = n_base + k
  std::vector<i32> ins_g;
  std::vector<i64> endpoints;   // distinct declared endpoints for this (case,q)
};

inline const char* topology_name(i32 t) {
  static const char* names[3] = {"chain", "bushy", "star"};
  return (t >= 0 && t < 3) ? names[t] : "?";
}

inline i32 version_parent(i32 topology, i32 v) {
  if (v <= 0) return -1;
  switch (topology) {
    case 0: return v - 1;        // chain: depth 2,048
    case 1: return (v - 1) / 2;  // bushy: binary heap, depth 11
    default: return 0;           // star: root with 2,048 children
  }
}

// Deterministic nested range generator for one state (kMaxQ ranges).
inline void gen_nested(Rng& r, i64 tmin, i64 tmax, i32 nq, Query* out) {
  i64 lo = tmin, hi = tmax + 1;
  const i64 span0 = hi - lo;
  for (i32 k = 0; k < nq; ++k) {
    i64 span = hi - lo;
    if (span <= 4096) {
      lo = tmin + span0 / 2;
      hi = lo + 2048;
      span = hi - lo;
    }
    i64 a = span * i64(3 + r.bounded(16)) / 100;
    i64 b = span * i64(3 + r.bounded(16)) / 100;
    lo += a;
    hi -= b;
    if (hi - lo < 1024) hi = lo + 1024;
    out[k] = {lo, hi};
  }
}

// Declared query k for state v under whichever range scope the workload was built with.
inline const Query& query_for(const Workload& w, i32 v, i32 k) {
  return w.per_state ? w.queries[size_t(v) * kMaxQ + size_t(k)] : w.queries[size_t(k)];
}

// Build the workload for (fixture, topology, seed, factor_period, q).
// Radix sort/unique for the declared endpoint vector (same values as std::sort+unique;
// avoids a comparison sort on the Q128-sized endpoint sets).
inline void radix_sort_unique_endpoints(std::vector<i64>& v) {
  if (v.size() < 256) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return;
  }
  std::vector<i64> tmp(v.size());
  std::vector<u32> cnt(65536);
  std::vector<i64>* src = &v;
  std::vector<i64>* dst = &tmp;
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 16 * pass;
    std::fill(cnt.begin(), cnt.end(), 0);
    for (i64 x : *src) ++cnt[u32(((u64(x) ^ (u64(1) << 63)) >> shift) & 0xFFFF)];
    u32 sum = 0;
    for (u32& c : cnt) { const u32 t = c; c = sum; sum += t; }
    for (i64 x : *src) (*dst)[cnt[u32(((u64(x) ^ (u64(1) << 63)) >> shift) & 0xFFFF)]++] = x;
    std::swap(src, dst);
  }
  if (src != &v) v = *src;
  v.erase(std::unique(v.begin(), v.end()), v.end());
}

inline Workload make_workload(const Fixture& f, i32 topology, i32 seed, i32 factor_period, i32 q,
                              bool per_state_ranges = false) {
  Workload w;
  w.per_state = per_state_ranges;
  w.topology = topology;
  w.seed = seed;
  w.factor_period = factor_period;
  w.q = q;
  w.n_base = i32(f.h.n);
  w.groups = f.h.groups;
  w.t_min = f.h.t_min;
  w.t_max = f.h.t_max;
  w.parent.resize(kStates);
  w.edit.resize(kStates);
  const i64 N = f.h.n;
  const i64 tmin = f.h.t_min, tmax = f.h.t_max;
  if (per_state_ranges) {
    w.queries.resize(size_t(kStates) * kMaxQ);
    for (i32 v = 1; v < kStates; ++v) {
      Rng qr(u64(seed) * 1000000007ULL + u64(topology) * 65537ULL +
             u64(factor_period) * 131ULL + u64(v) * 2654435761ULL + 17ULL);
      gen_nested(qr, tmin, tmax, q, &w.queries[size_t(v) * kMaxQ]);
    }
  } else {
    w.queries.resize(kMaxQ);
    Rng qr(u64(seed) * 1000000007ULL + u64(topology) * 65537ULL + u64(factor_period) * 131ULL + 17ULL);
    gen_nested(qr, tmin, tmax, q, w.queries.data());
  }
  const i32 G = f.h.groups;
  Rng rng(u64(seed) * 1000003ULL + u64(topology) * 7919ULL + u64(factor_period) * 104729ULL + 20261025ULL);

  for (i32 v = 0; v < kStates; ++v) w.parent[size_t(v)] = version_parent(topology, v);
  // Edits, one per edited state. Factor updates occupy every factor_period-th edit.
  std::vector<i64> gfactor(size_t(G), 1000);
  i32 next_group = 0;
  i32 next_ins = 0;
  for (i32 v = 1; v < kStates; ++v) {
    Edit e{};
    // Optional env override for the fraction of edits that are factor updates
    // (BWH_FACTOR_PCT in [0,100]); unset keeps the shipped factor-period pattern.
    static const i32 factor_pct = [] {
      const char* e = std::getenv("BWH_FACTOR_PCT");
      return e ? std::atoi(e) : -1;
    }();
    bool factor_step;
    if (factor_pct >= 0) factor_step = (rng.bounded(100) < u32(factor_pct));
    else factor_step = (factor_period > 0) && (v % factor_period == 0);
    if (factor_step) {
      i32 g = next_group++ % G;
      i64 cur = gfactor[size_t(g)];
      i64 nf = (cur == 950) ? 1000 : (cur == 1000 ? 1050 : 950);
      gfactor[size_t(g)] = nf;
      e.op = OP_FACTOR;
      e.g = g;
      e.val = nf;
    } else {
      u32 roll = rng.bounded(100);
      if (roll < 55) {
        e.op = OP_REPLACE;
        e.id = rng.range(0, N);
        e.val = f.rows[size_t(e.id)].amount + rng.range(-100000, 100000);
      } else if (roll < 75) {
        e.op = OP_INSERT;
        e.id = N + next_ins++;
        e.t = rng.range(tmin, tmax + 1);
        e.g = i32(rng.bounded(u32(G)));
        i64 base_amt = f.rows[size_t(rng.bounded(u32(N)))].amount;
        e.val = base_amt + rng.range(-100000, 100000);
        w.ins_t.push_back(e.t);
        w.ins_g.push_back(e.g);
      } else if (roll < 90) {
        e.op = OP_DELETE;
        e.id = rng.range(0, N);
      } else {
        e.op = OP_MOVE;
        e.id = rng.range(0, N);
        e.t = rng.range(tmin, tmax + 1);
        e.g = i32(rng.bounded(u32(G)));
        e.val = f.rows[size_t(e.id)].amount + rng.range(-100000, 100000);
      }
    }
    w.edit[size_t(v)] = e;
  }
  w.n_inserted = next_ins;
  // Declared endpoint domain: distinct endpoints over all states' first q ranges.
  w.endpoints.reserve(size_t(q) * 2 * (per_state_ranges ? size_t(kStates - 1) : 1));
  for (i32 v = 1; v < kStates; ++v) {
    for (i32 k = 0; k < q; ++k) {
      const Query& qq = query_for(w, v, k);
      w.endpoints.push_back(qq.lo);
      w.endpoints.push_back(qq.hi);
    }
  }
  radix_sort_unique_endpoints(w.endpoints);
  return w;
}

inline i64 total_answers(const Workload& w) { return i64(kStates - 1) * i64(w.q); }


// Preorder state indices (children ascending).
inline void preorder(const Workload& w, std::vector<i32>& order) {
  order.clear();
  order.reserve(kStates);
  std::vector<std::vector<i32>> kids(static_cast<size_t>(kStates));
  for (i32 v = 1; v < kStates; ++v) kids[size_t(w.parent[size_t(v)])].push_back(v);
  std::vector<i32> st;
  st.push_back(0);
  while (!st.empty()) {
    i32 v = st.back();
    st.pop_back();
    order.push_back(v);
    for (auto it = kids[size_t(v)].rbegin(); it != kids[size_t(v)].rend(); ++it) st.push_back(*it);
  }
}

// ---------------------------------------------------------------- utils
inline u64 fnv1a(const void* data, size_t n, u64 h = 1469598103934665603ULL) {
  const u8* p = static_cast<const u8*>(data);
  for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ULL;
  return h;
}

}  // namespace bwh
