// panel_batch_gpu.cpp — one timed worker for the GPU (one RTX 4090) block-parallel v7
// executor.  New file (2026-11-01, GPU round); same CLI/phases/JSON contract as panel.cpp,
// plus a "gpu" object with upload/kernel/combine/timing details.
//
// CPU keeps the v7 setup (declared workload/query construction, offline plan, base scan,
// block pre-pass); the block inputs are uploaded once, one CUDA block per state block replays
// its local V/Fenwick exactly and writes local endpoint prefixes + block V-leaf deltas; the
// tiny serial carry pass and the final combine are done as in v7 (combine on the GPU).
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/resource.h>

#include "common.hpp"
#include "batch_engine_v7.hpp"
#include "sha_fast.hpp"
#include "gpu_v7.h"

#include <cuda_runtime.h>

using namespace bwh;

static i64 ns_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}
static bool write_all(FILE* fp, const void* p, size_t n) { return std::fwrite(p, 1, n, fp) == n; }
#define CUDA_CHECK(x)                                                                      \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      return 5;                                                                            \
    }                                                                                      \
  } while (0)

int main(int argc, char** argv) {
  std::string arm_name, fixture_path, base_id, answers_path, json_path;
  i32 topology = 0, seed = 7, factor_period = 4, q = 2, repeat = 0;
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
    else if (!std::strcmp(argv[i], "--answers")) answers_path = next(argv[i]);
    else if (!std::strcmp(argv[i], "--json")) json_path = next(argv[i]);
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  if (fixture_path.empty() || answers_path.empty() || json_path.empty()) {
    std::fprintf(stderr, "usage: panel_batch_gpu --fixture PATH --base-id ID --topology T --seed S "
                         "--factor-period P --q Q --repeat R --answers A --json J\n");
    return 2;
  }
  if (arm_name.empty()) arm_name = "hybrid";
  if (q < 1 || q > kMaxQ) { std::fprintf(stderr, "unsupported q\n"); return 2; }
  const char* scope_env = std::getenv("BWH_RANGE_SCOPE");
  const bool per_state = scope_env && std::strcmp(scope_env, "per_state") == 0;

  const auto t0 = std::chrono::steady_clock::now();
  Fixture f;
  if (!load_fixture(fixture_path, f)) { std::fprintf(stderr, "cannot load fixture\n"); return 3; }
  const auto t1 = std::chrono::steady_clock::now();

  // ---- CPU setup: v7 declared workload + offline plan + base scan + block pre-pass
  BatchWorkloadV7 bw = make_workload_fast_v7(f, topology, seed, factor_period, q, per_state);
  BatchEngineV7 eng(f, bw);
  const bool ok = eng.build();
  if (!ok) { std::fprintf(stderr, "v7 engine build failed (unsupported case)\n"); return 3; }

  // host-side block inputs (packing is charged to setup)
  const i32 B = eng.v7_blocks, Cpfx = eng.Cpfx, ncls = eng.ncls, G = eng.G;
  const i32 V = kStates - 1;
  const i32 qq = q;
  std::vector<GpuV7Node> nodes(static_cast<size_t>(kStates));
  std::memset(nodes.data(), 0, nodes.size() * sizeof(GpuV7Node));
  for (i32 v = 1; v < kStates; ++v) {
    const BatchEngineV7::NodePlan& p = eng.plan[size_t(v)];
    GpuV7Node& n = nodes[size_t(v)];
    n.kind = p.kind;
    n.g_old = p.g_old; n.c_old = p.c_old; n.ti_old = p.ti_old;
    n.g_new = p.g_new; n.c_new = p.c_new; n.ti_new = p.ti_new;
    n.amt_old = p.amt_old; n.amt_new = p.amt_new;
    n.fac_val = p.fac_val; n.fac_prev = p.fac_prev;
    n.ex_old = p.ex_old; n.ex_new = p.ex_new;
  }
  std::vector<i32> hv0(static_cast<size_t>(B)), hv1(static_cast<size_t>(B));
  for (i32 b = 0; b < B; ++b) { hv0[size_t(b)] = eng.v7_v0[size_t(b)]; hv1[size_t(b)] = eng.v7_v1[size_t(b)]; }
  std::vector<i64> fac_start(static_cast<size_t>(B) * size_t(G));
  for (i32 b = 0; b < B; ++b)
    for (i32 g = 0; g < G; ++g) fac_start[size_t(b) * size_t(G) + size_t(g)] = eng.v7_fac[size_t(b)][size_t(g)];
  std::vector<i64> need_off(static_cast<size_t>(B) * size_t(G), -1);
  for (i32 b = 0; b < B; ++b)
    for (size_t i = 0; i < eng.v7_groups[size_t(b)].size(); ++i)
      need_off[size_t(b) * size_t(G) + size_t(eng.v7_groups[size_t(b)][i])] = eng.v7_snap_off[size_t(b)][i];
  std::vector<i32> qlb(static_cast<size_t>(kStates) * size_t(qq)), qhb(static_cast<size_t>(kStates) * size_t(qq));
  for (size_t i = 0; i < qlb.size(); ++i) { qlb[i] = static_cast<i32>(eng.qlb[i]); qhb[i] = static_cast<i32>(eng.qhb[i]); }
  std::vector<i32> g2row(eng.g2row.begin(), eng.g2row.end());
  std::vector<i32> bidx(eng.v7_bidx.begin(), eng.v7_bidx.end());
  const size_t lvals_n = size_t(V) * size_t(2 * qq);
  const size_t dV_n = size_t(B) * size_t(Cpfx);
  const size_t carry_n = dV_n;
  const size_t ans_n = size_t(V) * size_t(qq);
  std::vector<i64> dV(dV_n, 0), carrypref(carry_n, 0);
  std::vector<i32> bidx_h = bidx;
  i32 threads = 256;
  if (const char* te = std::getenv("BWH_GPU_THREADS")) threads = std::atoi(te);
  if (threads < 32) threads = 32;
  if (threads > 1024) threads = 1024;
  const int shared_bytes = int((2 * Cpfx + G) * sizeof(i64));
  const auto t2 = std::chrono::steady_clock::now();

  // ---- GPU: allocate + upload once
  CUDA_CHECK(cudaSetDevice(0));
  GpuV7Node* d_nodes = nullptr; i32* d_v0 = nullptr; i32* d_v1 = nullptr;
  i64* d_fac_start = nullptr; i32* d_g2row = nullptr; i32* d_ms_off = nullptr;
  i64* d_ms_cls = nullptr; i64* d_ms_base = nullptr; i32* d_ms_dev = nullptr;
  i64* d_devbuf = nullptr; i64* d_need_off = nullptr; i32* d_qlb = nullptr; i32* d_qhb = nullptr;
  i64* d_lvals = nullptr; i64* d_dV = nullptr; i64* d_carry = nullptr; i32* d_bidx = nullptr;
  i64* d_answers = nullptr;
  // allocate the consolidated input buffer and the (small) output buffers before the upload
  // window: the first CUDA call also initialises the context, which is not a copy cost.
  void* gbuf = nullptr;
  {
    // reserve a context on the selected device before timing (first malloc includes init)
    CUDA_CHECK(cudaFree(nullptr));
  }
  auto tup0 = std::chrono::steady_clock::now();
  // consolidated upload: pack all inputs into one staging buffer, one H2D copy
  struct Up { void** dst; const void* src; size_t bytes; size_t off; };
  std::vector<Up> ups = {
      {(void**)&d_nodes, nodes.data(), nodes.size() * sizeof(GpuV7Node), 0},
      {(void**)&d_v0, hv0.data(), hv0.size() * sizeof(i32), 0},
      {(void**)&d_v1, hv1.data(), hv1.size() * sizeof(i32), 0},
      {(void**)&d_fac_start, fac_start.data(), fac_start.size() * sizeof(i64), 0},
      {(void**)&d_g2row, g2row.data(), g2row.size() * sizeof(i32), 0},
      {(void**)&d_ms_off, eng.ms_off.data(), eng.ms_off.size() * sizeof(i32), 0},
      {(void**)&d_ms_cls, eng.ms_cls.data(), eng.ms_cls.size() * sizeof(i64), 0},
      {(void**)&d_ms_base, eng.ms_base.data(), eng.ms_base.size() * sizeof(i64), 0},
      {(void**)&d_ms_dev, eng.ms_dev.data(), eng.ms_dev.size() * sizeof(i32), 0},
      {(void**)&d_devbuf, eng.v7_snap_data.data(), eng.v7_snap_data.size() * sizeof(i64), 0},
      {(void**)&d_need_off, need_off.data(), need_off.size() * sizeof(i64), 0},
      {(void**)&d_qlb, qlb.data(), qlb.size() * sizeof(i32), 0},
      {(void**)&d_qhb, qhb.data(), qhb.size() * sizeof(i32), 0},
      {(void**)&d_bidx, bidx_h.data(), bidx_h.size() * sizeof(i32), 0},
  };
  size_t total = 0;
  for (auto& u : ups) { total = (total + 255) & ~size_t(255); u.off = total; total += u.bytes; }
  std::vector<char> hbuf(total);
  CUDA_CHECK(cudaMalloc(&gbuf, total));
  CUDA_CHECK(cudaMalloc(&d_lvals, lvals_n * sizeof(i64)));
  CUDA_CHECK(cudaMalloc(&d_dV, dV_n * sizeof(i64)));
  CUDA_CHECK(cudaMalloc(&d_carry, carry_n * sizeof(i64)));
  CUDA_CHECK(cudaMalloc(&d_answers, ans_n * sizeof(i64)));
  for (auto& u : ups) std::memcpy(hbuf.data() + u.off, u.src, u.bytes);
  CUDA_CHECK(cudaMemcpy(gbuf, hbuf.data(), total, cudaMemcpyHostToDevice));
  for (auto& u : ups) *u.dst = static_cast<char*>(gbuf) + u.off;
  const auto tup1 = std::chrono::steady_clock::now();

  // ---- kernel 1: independent block-local replay
  GpuV7Args a{};
  a.B = B; a.G = G; a.Cpfx = Cpfx; a.ncls = ncls; a.q = qq;
  a.nodes = d_nodes; a.block_v0 = d_v0; a.block_v1 = d_v1; a.fac_start = d_fac_start;
  a.g2row = d_g2row; a.ms_off = d_ms_off; a.ms_cls = d_ms_cls; a.ms_base = d_ms_base;
  a.ms_dev = d_ms_dev; a.devbuf = d_devbuf; a.need_off = d_need_off;
  a.qlb = d_qlb; a.qhb = d_qhb; a.lvals = d_lvals; a.dV = d_dV;
  a.threads = threads; a.shared_bytes = shared_bytes;
  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));
  CUDA_CHECK(cudaEventRecord(ev0));
  int rc = gpu_v7_launch(&a);
  if (rc != 0) { std::fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString((cudaError_t)rc)); return 5; }
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float kernel_ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev0, ev1));
  CUDA_CHECK(cudaEventDestroy(ev0)); CUDA_CHECK(cudaEventDestroy(ev1));
  const auto tk = std::chrono::steady_clock::now();

  // ---- serial carry pass on the CPU (dense C)
  CUDA_CHECK(cudaMemcpy(dV.data(), d_dV, dV_n * sizeof(i64), cudaMemcpyDeviceToHost));
  {
    std::vector<i64> vstart = eng.v7_Vbase;
    for (i32 b = 0; b < B; ++b) {
      i64* cp = &carrypref[size_t(b) * size_t(Cpfx)];
      cp[0] = 0;
      for (i32 k = 1; k < Cpfx; ++k) cp[size_t(k)] = cp[size_t(k - 1)] + vstart[size_t(k)];
      const i64* dv = &dV[size_t(b) * size_t(Cpfx)];
      for (i32 k = 1; k < Cpfx; ++k) vstart[size_t(k)] += dv[size_t(k)];
    }
  }
  const auto tc = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaMemcpy(d_carry, carrypref.data(), carry_n * sizeof(i64), cudaMemcpyHostToDevice));

  // ---- combine kernel
  GpuV7CombineArgs ca{};
  ca.nstates = V; ca.q = qq; ca.Cpfx = Cpfx;
  ca.qlb = d_qlb; ca.qhb = d_qhb; ca.carrypref = d_carry; ca.bidx = d_bidx;
  ca.lvals = d_lvals; ca.answers = d_answers;
  cudaEvent_t ev2, ev3;
  CUDA_CHECK(cudaEventCreate(&ev2)); CUDA_CHECK(cudaEventCreate(&ev3));
  CUDA_CHECK(cudaEventRecord(ev2));
  rc = gpu_combine_launch(&ca);
  if (rc != 0) { std::fprintf(stderr, "combine launch failed: %s\n", cudaGetErrorString((cudaError_t)rc)); return 5; }
  CUDA_CHECK(cudaEventRecord(ev3));
  CUDA_CHECK(cudaEventSynchronize(ev3));
  float combine_ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&combine_ms, ev2, ev3));
  CUDA_CHECK(cudaEventDestroy(ev2)); CUDA_CHECK(cudaEventDestroy(ev3));
  const auto tco = std::chrono::steady_clock::now();

  std::vector<i64> answers(ans_n, 0);
  CUDA_CHECK(cudaMemcpy(answers.data(), d_answers, ans_n * sizeof(i64), cudaMemcpyDeviceToHost));
  const auto t3 = std::chrono::steady_clock::now();

  // ---- extract (identical to the CPU panels)
  FILE* out = std::fopen(answers_path.c_str(), "wb");
  if (!out) { std::fprintf(stderr, "cannot write answers\n"); return 4; }
  const bool wok = write_all(out, answers.data(), answers.size() * sizeof(i64));
  std::fclose(out);
  if (!wok) { std::fprintf(stderr, "short answer write\n"); return 4; }
  const u64 cks = fnv1a(answers.data(), answers.size() * sizeof(i64));
  const std::string sha = answers_sha256(answers.data(), answers.size() * sizeof(i64));
  const auto t4 = std::chrono::steady_clock::now();

  // ---- cleanup
  CUDA_CHECK(cudaFree(gbuf));
  CUDA_CHECK(cudaFree(d_lvals)); CUDA_CHECK(cudaFree(d_dV)); CUDA_CHECK(cudaFree(d_carry));
  CUDA_CHECK(cudaFree(d_answers));
  f.release_rows();
  const auto t5 = std::chrono::steady_clock::now();

  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
  i64 peak_kb = 0;
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
  const i64 load_ns = ns_between(t0, t1);
  const i64 setup_ns = ns_between(t1, t2);
  const i64 upload_ns = ns_between(tup0, tup1);
  const i64 kernel_ns = i64(kernel_ms * 1e6);
  const i64 carry_ns = ns_between(tk, tc);
  const i64 combine_ns = i64(combine_ms * 1e6) + ns_between(tc, tco) - i64(combine_ms * 1e6);
  const i64 download_ns = ns_between(tco, t3);
  const i64 execute_ns = upload_ns + kernel_ns + carry_ns + combine_ns + download_ns;
  const i64 extract_ns = ns_between(t3, t4);
  const i64 cleanup_ns = ns_between(t4, t5);
  const i64 full_ns = load_ns + setup_ns + execute_ns + extract_ns + cleanup_ns;
  char devname[128] = "n/a";
  {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) std::snprintf(devname, sizeof(devname), "%s", prop.name);
  }
  FILE* jf = std::fopen((json_path + ".tmp").c_str(), "w");
  if (!jf) { std::fprintf(stderr, "cannot write json\n"); return 4; }
  std::fprintf(jf, "{\n");
  std::fprintf(jf, "  \"arm\": \"gpu_v7\",\n  \"base_id\": \"%s\",\n", base_id.c_str());
  std::fprintf(jf, "  \"fixture\": \"%s\",\n  \"topology\": %d,\n  \"seed\": %d,\n  \"factor_period\": %d,\n",
               fixture_path.c_str(), topology, seed, factor_period);
  std::fprintf(jf, "  \"q\": %d,\n  \"repeat\": %d,\n  \"n_states\": %d,\n  \"n_answers\": %" PRId64 ",\n",
               q, repeat, kStates, i64(answers.size()));
  std::fprintf(jf, "  \"range_scope\": \"%s\",\n", per_state ? "per_state" : "shared");
  std::fprintf(jf, "  \"answers_fn\": \"%016" PRIx64 "\",\n  \"answers_sha256\": \"%s\",\n", cks, sha.c_str());
  std::fprintf(jf, "  \"phases_ns\": {\"load\": %" PRId64 ", \"setup\": %" PRId64 ", \"execute\": %" PRId64
                   ", \"extract\": %" PRId64 ", \"cleanup\": %" PRId64 "},\n",
               load_ns, setup_ns, execute_ns, extract_ns, cleanup_ns);
  std::fprintf(jf, "  \"gpu\": {\"device\": \"%s\", \"blocks\": %d, \"threads\": %d, \"shared_bytes\": %d, "
                   "\"upload_ns\": %" PRId64 ", \"kernel_ns\": %" PRId64 ", \"carry_ns\": %" PRId64
                   ", \"combine_ns\": %" PRId64 ", \"download_ns\": %" PRId64 "},\n",
               devname, B, threads, shared_bytes, upload_ns, kernel_ns, carry_ns, combine_ns, download_ns);
  std::fprintf(jf, "  \"full_ns\": %" PRId64 ",\n  \"peak_rss_kb\": %" PRId64 "\n}\n", full_ns, peak_kb);
  std::fclose(jf);
  if (std::rename((json_path + ".tmp").c_str(), json_path.c_str()) != 0) return 4;
  std::printf("arm=gpu_v7 base=%s t%d s%d f%d q%d r%d blocks=%d kernel_ms=%.3f full_ns=%" PRId64
              " peak_kb=%" PRId64 "\n",
              base_id.c_str(), topology, seed, factor_period, q, repeat, B, double(kernel_ns) / 1e6,
              full_ns, peak_kb);
  return 0;
}
