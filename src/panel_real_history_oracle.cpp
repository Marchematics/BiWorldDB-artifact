// panel_real_history_oracle.cpp — independent brute-force oracle for the real version-history
// benchmark (2026-10-31, real-history P1).  New file; shipped files untouched.
//
// The oracle does not share any index with the engines: it replays the real edit chain in
// memory (plain row arrays), and for sampled versions recomputes every declared answer by a
// direct scan of all current rows,  Q(I) = sum_{rows present, lo<=t<hi} fac[g]*amount.
// Full-vector exactness comes from the cross-arm SHA equality; this binary anchors sampled
// versions to a from-scratch evaluation.
//
// CLI: panel_real_history_oracle --fixture B --spec J --stride S --answers A --json J
//   stride S: evaluate versions whose index is a multiple of S (plus the last version).
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.hpp"
#include "real_history_spec.hpp"
#include "sha256.hpp"

using namespace bwh;

static bool write_all(FILE* fp, const void* p, size_t n) { return std::fwrite(p, 1, n, fp) == n; }

int main(int argc, char** argv) {
  std::string fixture_path, spec_path, answers_path, json_path;
  i32 stride = 1, repeat = 0;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](const char* k) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k); std::exit(2); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--fixture")) fixture_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--spec")) spec_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--stride")) stride = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--repeat")) repeat = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--answers")) answers_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--json")) json_path = next(argv[i]);
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  if (fixture_path.empty() || spec_path.empty() || answers_path.empty() || json_path.empty()) {
    std::fprintf(stderr, "usage: panel_real_history_oracle --fixture B --spec J --stride S "
                         "--answers A --json J\n");
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
  std::vector<i64> cur_t(size_t(n_total), 0), cur_amt(size_t(n_total), 0);
  std::vector<i32> cur_g(size_t(n_total), 0);
  std::vector<u8> cur_ex(size_t(n_total), 0);
  std::vector<i64> fac(f.base_factor);
  for (i64 i = 0; i < spec.n_base; ++i) {
    const BaseRec& r = f.rows[size_t(i)];
    cur_t[size_t(i)] = r.t;
    cur_amt[size_t(i)] = r.amount;
    cur_g[size_t(i)] = r.g;
    cur_ex[size_t(i)] = 1;
  }
  std::vector<i64> answers(size_t(kStates - 1) * size_t(q), 0);
  std::vector<u8> sampled(size_t(kStates - 1), 0);
  i64 n_sampled = 0;
  auto scan_answer = [&](i64 lo, i64 hi) -> i64 {
    i64 sum = 0;
    for (i64 id = 0; id < n_total; ++id) {
      if (!cur_ex[size_t(id)]) continue;
      const i64 t = cur_t[size_t(id)];
      if (t < lo || t >= hi) continue;
      sum += fac[size_t(cur_g[size_t(id)])] * cur_amt[size_t(id)];
    }
    return sum;
  };
  for (i32 v = 1; v < kStates; ++v) {
    const Edit& e = spec.edit[size_t(v)];
    switch (e.op) {
      case OP_REPLACE:
        if (e.id >= 0 && e.id < n_total) { cur_amt[size_t(e.id)] = e.val; cur_ex[size_t(e.id)] = 1; }
        break;
      case OP_INSERT:
      case OP_MOVE:
        if (e.id >= 0 && e.id < n_total) {
          cur_t[size_t(e.id)] = e.t;
          cur_g[size_t(e.id)] = e.g;
          cur_amt[size_t(e.id)] = e.val;
          cur_ex[size_t(e.id)] = 1;
        }
        break;
      case OP_DELETE:
        if (e.id >= 0 && e.id < n_total) cur_ex[size_t(e.id)] = 0;
        break;
      case OP_FACTOR:
        if (e.g >= 0 && e.g < spec.G) fac[size_t(e.g)] = e.val;
        break;
      default:
        break;
    }
    const bool want = ((v % stride) == 0) || (v == kStates - 1);
    if (want) {
      sampled[size_t(v - 1)] = 1;
      ++n_sampled;
      for (i32 k = 0; k < q; ++k) {
        const Query& qq = spec.query(v, k);
        answers[size_t(v - 1) * size_t(q) + size_t(k)] = scan_answer(qq.lo, qq.hi);
      }
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  FILE* out = std::fopen(answers_path.c_str(), "wb");
  if (!out) { std::fprintf(stderr, "cannot write answers\n"); return 4; }
  const bool ok = write_all(out, answers.data(), answers.size() * sizeof(i64));
  std::fclose(out);
  if (!ok) { std::fprintf(stderr, "short answer write\n"); return 4; }
  // sha256 of the sampled answers only (canonical for the receipt)
  std::vector<i64> picked;
  picked.reserve(size_t(n_sampled) * size_t(q));
  for (i64 s = 0; s < i64(kStates - 1); ++s)
    if (sampled[size_t(s)])
      for (i32 k = 0; k < q; ++k) picked.push_back(answers[size_t(s) * size_t(q) + size_t(k)]);
  const std::string sha = sha::of(picked.data(), picked.size() * sizeof(i64));
  const i64 oracle_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

  FILE* jf = std::fopen((json_path + ".tmp").c_str(), "w");
  if (!jf) { std::fprintf(stderr, "cannot write json\n"); return 4; }
  std::fprintf(jf, "{\n  \"arm\": \"oracle_bruteforce\",\n  \"family\": \"real_history\",\n");
  std::fprintf(jf, "  \"fixture\": \"%s\",\n  \"spec\": \"%s\",\n", fixture_path.c_str(), spec_path.c_str());
  std::fprintf(jf, "  \"repeat\": %d,\n  \"n_states\": %d,\n  \"q\": %d,\n  \"stride\": %d,\n",
               repeat, kStates, q, stride);
  std::fprintf(jf, "  \"sampled_states\": %" PRId64 ",\n  \"sampled_answers\": %zu,\n", n_sampled, picked.size());
  std::fprintf(jf, "  \"sampled_sha256\": \"%s\",\n  \"answers_sha256\": \"%s\",\n", sha.c_str(),
               sha::of(answers.data(), answers.size() * sizeof(i64)).c_str());
  std::fprintf(jf, "  \"oracle_ns\": %" PRId64 "\n}\n", oracle_ns);
  std::fclose(jf);
  if (std::rename((json_path + ".tmp").c_str(), json_path.c_str()) != 0) {
    std::fprintf(stderr, "cannot rename json\n");
    return 4;
  }
  std::printf("arm=oracle_bruteforce stride=%d sampled=%" PRId64 " full_ns=%" PRId64 "\n", stride,
              n_sampled, oracle_ns);
  return 0;
}
