// oracle.cpp — independent exact-answer oracle for the BiWorldDB hybrid panel.
//
// This program deliberately uses NO indexed / prefix / aggregate structure from
// any measurement arm.  It keeps a fully mutable row state (timestamp, amount,
// group, existence) plus the mutable per-group factor, walks the version tree
// depth-first (apply the node's edit on entry, restore on exit), and for every
// edited state answers that state's first q declared queries by one direct scan
// over all base + inserted rows.
//
// Version semantics (as shared by every arm):
//   state(v) = base state with edit[w] applied for every w on the root-to-v
//   path, w = 1..v along that path only (ancestors, never "all ids <= v").
//   A row contributes amount * factor[group] iff its existence flag is 1 and
//   lo <= t < hi.  Edits are absolute, so the same declared edit is valid on
//   every root-to-node path.
//
// Output: int64 answers, native little-endian, in order
//   for v = 1..kStates-1, for k = 0..q-1: answer[v][k].
// One JSON line is printed to stdout with the fixture path, case parameters,
// answer count, SHA-256 of the output bytes, and process wall time.
//
// Build: clang++ -std=c++17 -O2 -I. oracle.cpp -o oracle
//
// Standalone: C++17 standard library only (plus the shared header, which only
// provides the fixture format and the deterministic workload definition).

#include <cinttypes>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

#include "common.hpp"

using namespace bwh;

// ------------------------------------------------------------------ SHA-256
// Small self-contained FIPS 180-4 implementation (no external tools).
namespace sha {

struct Ctx {
  u32 h[8];
  u64 total;
  u8 buf[64];
  size_t fill;
};

static const u32 K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

static inline u32 rotr(u32 x, u32 n) { return (x >> n) | (x << (32 - n)); }

static void transform(Ctx& c, const u8* p) {
  u32 w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (u32(p[4 * i]) << 24) | (u32(p[4 * i + 1]) << 16) |
           (u32(p[4 * i + 2]) << 8) | u32(p[4 * i + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    u32 s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    u32 s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  u32 a = c.h[0], b = c.h[1], cc = c.h[2], d = c.h[3];
  u32 e = c.h[4], f = c.h[5], g = c.h[6], hh = c.h[7];
  for (int i = 0; i < 64; ++i) {
    u32 S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    u32 ch = (e & f) ^ (~e & g);
    u32 t1 = hh + S1 + ch + K[i] + w[i];
    u32 S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    u32 maj = (a & b) ^ (a & cc) ^ (b & cc);
    u32 t2 = S0 + maj;
    hh = g; g = f; f = e; e = d + t1;
    d = cc; cc = b; b = a; a = t1 + t2;
  }
  c.h[0] += a; c.h[1] += b; c.h[2] += cc; c.h[3] += d;
  c.h[4] += e; c.h[5] += f; c.h[6] += g; c.h[7] += hh;
}

static void init(Ctx& c) {
  c.h[0] = 0x6a09e667u; c.h[1] = 0xbb67ae85u; c.h[2] = 0x3c6ef372u;
  c.h[3] = 0xa54ff53au; c.h[4] = 0x510e527fu; c.h[5] = 0x9b05688cu;
  c.h[6] = 0x1f83d9abu; c.h[7] = 0x5be0cd19u;
  c.total = 0;
  c.fill = 0;
}

static void update(Ctx& c, const void* data, size_t n) {
  const u8* p = static_cast<const u8*>(data);
  c.total += u64(n);
  if (c.fill) {
    size_t take = 64 - c.fill;
    if (take > n) take = n;
    std::memcpy(c.buf + c.fill, p, take);
    c.fill += take;
    p += take;
    n -= take;
    if (c.fill == 64) {
      transform(c, c.buf);
      c.fill = 0;
    }
  }
  while (n >= 64) {
    transform(c, p);
    p += 64;
    n -= 64;
  }
  if (n) {
    std::memcpy(c.buf, p, n);
    c.fill = n;
  }
}

static void finish(Ctx& c, u8 out[32]) {
  const u64 bits = c.total * 8;
  u8 pad[72];
  size_t padlen = (c.fill < 56) ? (56 - c.fill) : (120 - c.fill);
  pad[0] = 0x80;
  std::memset(pad + 1, 0, padlen - 1);
  for (int i = 0; i < 8; ++i) pad[padlen + i] = u8(bits >> (56 - 8 * i));
  update(c, pad, padlen + 8);
  for (int i = 0; i < 8; ++i) {
    out[4 * i] = u8(c.h[i] >> 24);
    out[4 * i + 1] = u8(c.h[i] >> 16);
    out[4 * i + 2] = u8(c.h[i] >> 8);
    out[4 * i + 3] = u8(c.h[i]);
  }
}

static std::string hex(const u8 d[32]) {
  static const char* hx = "0123456789abcdef";
  std::string s(64, '0');
  for (int i = 0; i < 32; ++i) {
    s[size_t(2 * i)] = hx[d[i] >> 4];
    s[size_t(2 * i + 1)] = hx[d[i] & 15];
  }
  return s;
}

static std::string of(const void* data, size_t n) {
  Ctx c;
  init(c);
  update(c, data, n);
  u8 d[32];
  finish(c, d);
  return hex(d);
}

}  // namespace sha

// ------------------------------------------------------------------ helpers
static std::string json_escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (unsigned char ch : s) {
    switch (ch) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\b': o += "\\b"; break;
      case '\f': o += "\\f"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (ch < 0x20) {
          char b[8];
          std::snprintf(b, sizeof(b), "\\u%04x", unsigned(ch));
          o += b;
        } else {
          o += char(ch);
        }
    }
  }
  return o;
}

static bool parse_i64(const char* s, i64& out) {
  if (!s || !*s) return false;
  char* end = nullptr;
  errno = 0;
  long long v = std::strtoll(s, &end, 10);
  if (errno != 0 || !end || *end != '\0') return false;
  out = i64(v);
  return true;
}

static i64 checked_i64(__int128 x, const char* what) {
  if (x < __int128(INT64_MIN) || x > __int128(INT64_MAX)) {
    std::fprintf(stderr, "oracle: %s overflowed int64 (__int128=%lld%019lld)\n",
                 what, (long long)(x / 1000000000000000000LL),
                 (long long)(x % 1000000000000000000LL));
    std::exit(6);
  }
  return i64(x);
}

// ------------------------------------------------------------------ solver
struct Undo {
  u8 op;
  i32 g;
  i64 id;
  i64 old_amt;
  i64 old_t;
  i64 old_factor;
  i32 old_g;
  u8 old_ex;
};

struct Solver {
  const Fixture& f;
  const Workload& w;
  i32 N = 0;          // base rows
  i32 M = 0;          // base + inserted rows
  i32 G = 0;          // groups
  std::vector<i64> t, amt;
  std::vector<i32> g;
  std::vector<u8> ex;
  std::vector<i64> factor;
  std::vector<Undo> undo;
  std::vector<std::vector<i32>> kids;
  std::vector<i64> ans;  // (kStates-1) * q
  i32 q = 0;

  Solver(const Fixture& fixture, const Workload& workload)
      : f(fixture), w(workload) {
    N = w.n_base;
    M = w.n_base + w.n_inserted;
    G = w.groups;
    q = w.q;
    t.resize(size_t(M));
    amt.resize(size_t(M));
    g.resize(size_t(M));
    ex.resize(size_t(M));
    factor.resize(size_t(G));
    undo.resize(size_t(kStates));
    for (i32 i = 0; i < N; ++i) {
      const BaseRec& r = f.rows[size_t(i)];
      t[size_t(i)] = r.t;
      amt[size_t(i)] = r.amount;
      g[size_t(i)] = r.g;
      ex[size_t(i)] = 1;
    }
    for (i32 k = 0; k < w.n_inserted; ++k) {
      const i32 i = N + k;
      t[size_t(i)] = w.ins_t[size_t(k)];
      g[size_t(i)] = w.ins_g[size_t(k)];
      amt[size_t(i)] = 0;
      ex[size_t(i)] = 0;  // created only when its OP_INSERT is applied
    }
    for (i32 gg = 0; gg < G; ++gg) factor[size_t(gg)] = f.base_factor[size_t(gg)];

    kids.assign(size_t(kStates), {});
    for (i32 v = 1; v < kStates; ++v) {
      i32 p = w.parent[size_t(v)];
      if (p < 0 || p >= v) {
        std::fprintf(stderr, "oracle: bad parent %d for state %d\n", p, v);
        std::exit(7);
      }
      kids[size_t(p)].push_back(v);
    }
    ans.assign(size_t(kStates - 1) * size_t(q), 0);
  }

  inline void apply(i32 v) {
    const Edit& e = w.edit[size_t(v)];
    Undo& u = undo[size_t(v)];
    u.op = e.op;
    u.g = e.g;
    u.id = e.id;
    switch (e.op) {
      case OP_REPLACE: {
        u.old_amt = amt[size_t(e.id)];
        u.old_ex = ex[size_t(e.id)];
        amt[size_t(e.id)] = e.val;
        ex[size_t(e.id)] = 1;
        break;
      }
      case OP_INSERT: {
        u.old_amt = amt[size_t(e.id)];
        u.old_t = t[size_t(e.id)];
        u.old_g = g[size_t(e.id)];
        u.old_ex = ex[size_t(e.id)];
        t[size_t(e.id)] = e.t;
        g[size_t(e.id)] = e.g;
        amt[size_t(e.id)] = e.val;
        ex[size_t(e.id)] = 1;
        break;
      }
      case OP_DELETE: {
        u.old_ex = ex[size_t(e.id)];
        ex[size_t(e.id)] = 0;
        break;
      }
      case OP_MOVE: {
        u.old_amt = amt[size_t(e.id)];
        u.old_t = t[size_t(e.id)];
        u.old_g = g[size_t(e.id)];
        u.old_ex = ex[size_t(e.id)];
        t[size_t(e.id)] = e.t;
        g[size_t(e.id)] = e.g;
        amt[size_t(e.id)] = e.val;
        ex[size_t(e.id)] = 1;
        break;
      }
      case OP_FACTOR: {
        u.old_factor = factor[size_t(e.g)];
        factor[size_t(e.g)] = e.val;
        break;
      }
      default:
        std::fprintf(stderr, "oracle: unknown op %u at state %d\n", unsigned(e.op), v);
        std::exit(7);
    }
  }

  inline void revert(i32 v) {
    const Undo& u = undo[size_t(v)];
    switch (u.op) {
      case OP_REPLACE:
        amt[size_t(u.id)] = u.old_amt;
        ex[size_t(u.id)] = u.old_ex;
        break;
      case OP_INSERT:
      case OP_MOVE:
        t[size_t(u.id)] = u.old_t;
        g[size_t(u.id)] = u.old_g;
        amt[size_t(u.id)] = u.old_amt;
        ex[size_t(u.id)] = u.old_ex;
        break;
      case OP_DELETE:
        ex[size_t(u.id)] = u.old_ex;
        break;
      case OP_FACTOR:
        factor[size_t(u.g)] = u.old_factor;
        break;
      default:
        break;
    }
  }

  // One direct scan over every base + inserted row, answering this state's
  // first q ranges.  No precomputation, no index, no reordering.
  void answer_state(i32 v) {
    // The shared header has appeared in two layouts during development:
    //   (a) per-state ranges: queries.size() == kStates * kMaxQ
    //   (b) one 32-range set shared by every state: queries.size() == kMaxQ
    // Accept either; the answers always describe exactly the workload object
    // make_workload() returned for this compile.
    const Query* qs;
    if (w.queries.size() == size_t(kStates) * size_t(kMaxQ)) {
      qs = &w.queries[size_t(v) * size_t(kMaxQ)];
    } else if (w.queries.size() == size_t(kMaxQ)) {
      qs = w.queries.data();
    } else {
      std::fprintf(stderr,
                   "oracle: unexpected w.queries.size()=%zu (want %d or %zu)\n",
                   w.queries.size(), int(kMaxQ),
                   size_t(kStates) * size_t(kMaxQ));
      std::exit(7);
    }
    i64 lo[kMaxQ], hi[kMaxQ];
    bool nested = true;
    for (i32 k = 0; k < q; ++k) {
      lo[k] = qs[k].lo;
      hi[k] = qs[k].hi;
      if (k > 0 && (lo[k] < lo[k - 1] || hi[k] > hi[k - 1])) nested = false;
    }
    i64* out = &ans[size_t(v - 1) * size_t(q)];

    const i64* tp = t.data();
    const i64* ap = amt.data();
    const i32* gp = g.data();
    const u8* ep = ex.data();
    const i64* fp = factor.data();

    if (nested) {
      // Nested ranges: a row lies in a prefix [0..k0] of the ranges.  Bucket
      // each row's contribution by its deepest containing range, then suffix
      // sum.  This is still one full independent scan per state.
      __int128 bucket[kMaxQ];
      for (i32 k = 0; k < q; ++k) bucket[k] = 0;
      for (i32 i = 0; i < M; ++i) {
        if (!ep[i]) continue;
        const i64 ti = tp[i];
        i32 k = 0;
        while (k < q && ti >= lo[k] && ti < hi[k]) ++k;
        if (k) bucket[k - 1] += (__int128)ap[i] * (__int128)fp[gp[i]];
      }
      __int128 run = 0;
      for (i32 k = q - 1; k >= 0; --k) {
        run += bucket[k];
        out[k] = checked_i64(run, "query answer");
      }
    } else {
      __int128 acc[kMaxQ];
      for (i32 k = 0; k < q; ++k) acc[k] = 0;
      for (i32 i = 0; i < M; ++i) {
        if (!ep[i]) continue;
        const i64 ti = tp[i];
        const __int128 c = (__int128)ap[i] * (__int128)fp[gp[i]];
        for (i32 k = 0; k < q; ++k)
          if (ti >= lo[k] && ti < hi[k]) acc[k] += c;
      }
      for (i32 k = 0; k < q; ++k) out[k] = checked_i64(acc[k], "query answer");
    }
  }

  // Iterative depth-first traversal: enter a node (apply its edit, answer its
  // queries), descend through its children one at a time, then restore the
  // edit on the way out.  ITERATIVE on purpose: chain topology is 2,049 deep
  // and the per-state scan frame is too large for 2,049 native recursion
  // frames on an 8 MiB main-thread stack.
  void run() {
    struct Frame {
      i32 v;
      i32 next_child;
    };
    std::vector<Frame> stack;
    stack.reserve(size_t(kStates));
    stack.push_back({0, 0});  // root: base state, edit[0] is unused
    while (!stack.empty()) {
      Frame& fr = stack.back();
      const std::vector<i32>& ch = kids[size_t(fr.v)];
      if (fr.next_child < i32(ch.size())) {
        const i32 c = ch[size_t(fr.next_child++)];
        apply(c);
        answer_state(c);
        stack.push_back({c, 0});
      } else {
        if (fr.v > 0) revert(fr.v);
        stack.pop_back();
      }
    }
  }
};

// ------------------------------------------------------------------ main
int main(int argc, char** argv) {
  const auto t_start = std::chrono::steady_clock::now();
  std::string fixture_path, out_path;
  i64 topology = -1, seed = 0, factor_period = -1, q = -1;

  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "oracle: missing value for %s\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (!std::strcmp(a, "--fixture")) fixture_path = need("--fixture");
    else if (!std::strcmp(a, "--out")) out_path = need("--out");
    else if (!std::strcmp(a, "--topology")) { if (!parse_i64(need("--topology"), topology)) return 2; }
    else if (!std::strcmp(a, "--seed")) { if (!parse_i64(need("--seed"), seed)) return 2; }
    else if (!std::strcmp(a, "--factor-period")) { if (!parse_i64(need("--factor-period"), factor_period)) return 2; }
    else if (!std::strcmp(a, "--q")) { if (!parse_i64(need("--q"), q)) return 2; }
    else {
      std::fprintf(stderr, "oracle: unknown argument %s\n", a);
      return 2;
    }
  }
  if (fixture_path.empty() || out_path.empty() || topology < 0 || topology > 2 ||
      factor_period < 0 || q < 1 || q > kMaxQ) {
    std::fprintf(stderr,
                 "usage: oracle --fixture FIXTURE.bin --topology 0|1|2 --seed S "
                 "--factor-period P --q Q --out ANSWERS.bin\n");
    return 2;
  }

  Fixture f;
  if (!load_fixture(fixture_path, f)) {
    std::fprintf(stderr, "oracle: cannot load fixture %s\n", fixture_path.c_str());
    return 1;
  }
  const char* scope_env = std::getenv("BWH_RANGE_SCOPE");
  const bool per_state = scope_env && std::strcmp(scope_env, "per_state") == 0;
  Workload w = make_workload(f, i32(topology), i32(seed), i32(factor_period), i32(q), per_state);

  Solver s(f, w);
  s.run();

  const size_t n_answers = s.ans.size();
  FILE* fp = std::fopen(out_path.c_str(), "wb");
  if (!fp) {
    std::fprintf(stderr, "oracle: cannot write %s\n", out_path.c_str());
    return 4;
  }
  const bool wok =
      std::fwrite(s.ans.data(), sizeof(i64), n_answers, fp) == n_answers;
  if (std::fclose(fp) != 0 || !wok) {
    std::fprintf(stderr, "oracle: short write to %s\n", out_path.c_str());
    return 5;
  }

  const std::string digest = sha::of(s.ans.data(), n_answers * sizeof(i64));
  const auto t_end = std::chrono::steady_clock::now();
  const double wall =
      std::chrono::duration<double>(t_end - t_start).count();

  std::printf(
      "{\"fixture\":\"%s\",\"topology\":%" PRId64 ",\"seed\":%" PRId64
      ",\"factor_period\":%" PRId64 ",\"q\":%" PRId64
      ",\"n_answers\":%zu,\"answers_sha256\":\"%s\",\"wall_s\":%.6f}\n",
      json_escape(fixture_path).c_str(), topology, seed, factor_period, q,
      n_answers, digest.c_str(), wall);
  return 0;
}
