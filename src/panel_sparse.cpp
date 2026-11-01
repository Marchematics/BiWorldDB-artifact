// panel_sparse.cpp — one timed worker for the counting-sort CSR sparse-base study
// (2026-10-31, sparse-csr).  New file; the shipped panel.cpp is untouched.
//
// The phase accounting, JSON contract, answer-vector checksum and peak-RSS read are
// the same as panel.cpp, so a run here is directly comparable with a shipped run.
// Two additions:
//   * `--evict` (1/0): before load_fixture(), open the fixture and evict its page
//     cache with posix_fadvise(POSIX_FADV_DONTNEED), so the run measures a cold
//     fixture (the worker itself still mmaps and faults the rows in the load/build
//     phases).  The benchmark driver records which mode was used.
//   * `--arm sparse`: the BWH_SPARSE_BASE=0|1|auto selector (SparseBaseArm); its JSON
//     says whether the run resolved to the CSR ("arm_mode":"sparse") or delegated to
//     the shipped dense HybridArm ("arm_mode":"hybrid"), plus the CSR byte/nnz stats.
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/resource.h>

#include "common.hpp"
#include "engine.hpp"
#include "sparse_engine.hpp"
#include "sha256.hpp"

using namespace bwh;

static i64 ns_between(std::chrono::steady_clock::time_point a,
                      std::chrono::steady_clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

static bool write_all(FILE* fp, const void* p, size_t n) { return std::fwrite(p, 1, n, fp) == n; }

static void evict_fixture(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return;
#ifdef POSIX_FADV_DONTNEED
  ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
  ::close(fd);
}

int main(int argc, char** argv) {
  std::string arm_name, fixture_path, base_id, answers_path, json_path;
  i32 topology = 0, seed = 7, factor_period = 4, q = 2, repeat = 0, evict = 0;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](const char* k) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k); std::exit(2); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--arm")) arm_name = next(argv[i]);
    else if (!std::strcmp(argv[i], "--fixture")) fixture_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--base-id")) base_id = next(argv[i]);
    else if (!std::strcmp(argv[i], "--topology")) topology = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--seed")) seed = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--factor-period")) factor_period = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--q")) q = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--repeat")) repeat = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--evict")) evict = std::atoi(next(argv[i]).c_str());
    else if (!std::strcmp(argv[i], "--answers")) answers_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--json")) json_path = next(argv[i]);
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  if (arm_name.empty() || fixture_path.empty() || answers_path.empty() || json_path.empty()) {
    std::fprintf(stderr, "usage: panel_sparse --arm sparse|hybrid|full_raw|full_projected "
                         "--fixture PATH --base-id ID --topology T --seed S --factor-period P "
                         "--q Q --repeat R [--evict 0|1] --answers A --json J\n");
    return 2;
  }
  if (q < 1 || q > kMaxQ) { std::fprintf(stderr, "unsupported q\n"); return 2; }

  const char* scope_env = std::getenv("BWH_RANGE_SCOPE");
  const bool per_state = scope_env && std::strcmp(scope_env, "per_state") == 0;
  if (evict) evict_fixture(fixture_path);
  const auto t0 = std::chrono::steady_clock::now();
  Fixture f;
  if (!load_fixture(fixture_path, f)) {
    std::fprintf(stderr, "cannot load fixture %s\n", fixture_path.c_str());
    return 3;
  }
  const auto t1 = std::chrono::steady_clock::now();

  Workload w = make_workload(f, topology, seed, factor_period, q, per_state);
  Arm* arm = make_sparse_arm(arm_name, f, w);
  if (!arm) { std::fprintf(stderr, "unknown arm %s\n", arm_name.c_str()); return 3; }
  arm->build();
  const auto t2 = std::chrono::steady_clock::now();

  // Depth-first traversal of the declared version tree; answers for states 1..kStates-1.
  std::vector<i64> answers(size_t(kStates - 1) * size_t(q), 0);
  std::vector<SavedRow> saved(static_cast<size_t>(kStates));
  std::vector<std::vector<i32>> kids(static_cast<size_t>(kStates));
  for (i32 v = 1; v < kStates; ++v) kids[size_t(w.parent[size_t(v)])].push_back(v);
  struct Ev { i32 v; u8 exit; };
  std::vector<Ev> st;
  st.reserve(size_t(kStates) + 8);
  st.push_back({0, 0});
  while (!st.empty()) {
    const Ev ev = st.back();
    st.pop_back();
    const i32 v = ev.v;
    if (ev.exit) {
      if (v > 0) arm->undo(w.edit[size_t(v)], saved[size_t(v)]);
      continue;
    }
    if (v > 0) {
      saved[size_t(v)] = arm->snapshot(w.edit[size_t(v)]);
      arm->apply(w.edit[size_t(v)]);
      arm->query_batch(v, &answers[size_t(v - 1) * size_t(q)], q);
    }
    st.push_back({v, 1});
    const auto& ch = kids[size_t(v)];
    for (auto it = ch.rbegin(); it != ch.rend(); ++it) st.push_back({*it, 0});
  }
  const auto t3 = std::chrono::steady_clock::now();

  FILE* out = std::fopen(answers_path.c_str(), "wb");
  if (!out) { std::fprintf(stderr, "cannot write answers\n"); return 4; }
  const bool ok = write_all(out, answers.data(), answers.size() * sizeof(i64));
  std::fclose(out);
  if (!ok) { std::fprintf(stderr, "short answer write\n"); return 4; }
  const u64 cks = fnv1a(answers.data(), answers.size() * sizeof(i64));
  const std::string sha = sha::of(answers.data(), answers.size() * sizeof(i64));
  const auto t4 = std::chrono::steady_clock::now();

  // capture the representation receipt before the arm is destroyed
  const std::string arm_mode = arm->name();
  SparseStats stats;
  bool has_stats = false;
  {
    SparseBaseArm* sb = dynamic_cast<SparseBaseArm*>(arm);
    if (sb) { stats = sb->st; has_stats = true; }
  }

  delete arm;
  f.release_rows();
  const auto t5 = std::chrono::steady_clock::now();

  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
  i64 peak_kb = 0;
#if defined(__linux__)
  {
    FILE* s = std::fopen("/proc/self/status", "r");
    if (s) {
      char line[256];
      while (std::fgets(line, sizeof(line), s)) {
        long v = 0;
        if (std::sscanf(line, "VmHWM: %ld kB", &v) == 1) { peak_kb = v; break; }
      }
      std::fclose(s);
    }
  }
#else
  peak_kb = i64(ru.ru_maxrss) / 1024;
#endif

  const i64 load_ns = ns_between(t0, t1);
  const i64 setup_ns = ns_between(t1, t2);
  const i64 exec_ns = ns_between(t2, t3);
  const i64 extract_ns = ns_between(t3, t4);
  const i64 cleanup_ns = ns_between(t4, t5);
  const i64 full_ns = load_ns + setup_ns + exec_ns + extract_ns + cleanup_ns;

  FILE* jf = std::fopen((json_path + ".tmp").c_str(), "w");
  if (!jf) { std::fprintf(stderr, "cannot write json\n"); return 4; }
  std::fprintf(jf, "{\n");
  std::fprintf(jf, "  \"arm\": \"%s\",\n  \"arm_mode\": \"%s\",\n  \"base_id\": \"%s\",\n",
               arm_name.c_str(), arm_mode.c_str(), base_id.c_str());
  std::fprintf(jf, "  \"fixture\": \"%s\",\n  \"topology\": %d,\n  \"seed\": %d, \"factor_period\": %d,\n",
               fixture_path.c_str(), topology, seed, factor_period);
  std::fprintf(jf, "  \"q\": %d,\n  \"repeat\": %d, \"n_states\": %d, \"n_answers\": %" PRId64 ",\n",
               q, repeat, kStates, i64(answers.size()));
  std::fprintf(jf, "  \"range_scope\": \"%s\",\n", per_state ? "per_state" : "shared");
  std::fprintf(jf, "  \"cold\": %s,\n  \"preload_fadvise_dontneed\": %s,\n",
               evict ? "true" : "false", evict ? "true" : "false");
  std::fprintf(jf, "  \"sparse_base_env\": \"%s\",\n", sparse_policy::env_name());
  std::fprintf(jf, "  \"answers_fn\": \"%016" PRIx64 "\",\n", cks);
  std::fprintf(jf, "  \"answers_sha256\": \"%s\",\n", sha.c_str());
  std::fprintf(jf, "  \"phases_ns\": {\"load\": %" PRId64 ", \"setup\": %" PRId64 ", \"execute\": %" PRId64
                   ", \"extract\": %" PRId64 ", \"cleanup\": %" PRId64 "},\n",
               load_ns, setup_ns, exec_ns, extract_ns, cleanup_ns);
  std::fprintf(jf, "  \"full_ns\": %" PRId64 ",\n  \"peak_rss_kb\": %" PRId64,
               full_ns, peak_kb);
  if (has_stats) {
    std::fprintf(jf,
                 ",\n  \"base_stats\": {\"mode\": \"%s\", \"n_active\": %d, \"classes\": %" PRId64
                 ", \"nnz\": %" PRId64 ", \"csr_bytes\": %" PRId64 ", \"dense_bytes\": %" PRId64
                 ", \"bitmap_bytes\": %" PRId64 ", \"lut_bytes\": %" PRId64
                 ", \"transient_bytes\": %" PRId64 ", \"est_nnz\": %" PRId64 ", \"build_us\": %" PRId64 "}",
                 stats.mode ? "sparse" : "dense", stats.n_active, stats.c_domain, stats.nnz,
                 stats.csr_bytes, stats.dense_bytes, stats.bitmap_bytes, stats.lut_bytes,
                 stats.transient_bytes, stats.est_nnz, stats.build_us);
  }
  std::fprintf(jf, "\n}\n");
  std::fclose(jf);
  if (std::rename((json_path + ".tmp").c_str(), json_path.c_str()) != 0) {
    std::fprintf(stderr, "cannot rename json\n");
    return 4;
  }
  std::printf("arm=%s mode=%s base=%s t%d s%d f%d q%d r%d cold=%d full_ns=%" PRId64
              " peak_kb=%" PRId64 "\n",
              arm_name.c_str(), arm_mode.c_str(), base_id.c_str(), topology, seed, factor_period, q,
              repeat, evict, full_ns, peak_kb);
  return 0;
}
