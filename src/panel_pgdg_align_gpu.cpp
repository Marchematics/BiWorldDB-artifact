// panel_pgdg_align_gpu.cpp — GPU block-parallel execute for the aligned PGD-G baseline
// (2026-11-01, GPU round; new file, shipped files untouched).  Same CLI/phases/JSON contract as
// panel.cpp plus a "gpu" object; one RTX 4090 (CUDA_VISIBLE_DEVICES selects it).
//
// FAIR-BASELINE STRUCTURE: the CPU setup is the *aligned PGD-G engine* (pgdg_align_gpu.hpp =
// fixed pgdg_align_omp.hpp + plan exporter: declared workload, offline chain plan, dense
// correction-prefix base scan).  The GPU execute is the generic block-parallel replay primitive
// (gpu_v7b.h/.cu, used unmodified): one CUDA block per version block replays its local
// unified weighted-mass V/Fenwick, writes local endpoint prefixes and its block V-leaf delta;
// a tiny CPU carry pass and the GPU combine produce the answers.  The algebra of the replay is
// exactly the aligned PGD-G identity V_v(c)=sum_g fac_v(g)*(baseLeaf_g(c)+dev_v(g,c)), so the
// answer vector must be bit-identical to the CPU aligned baseline.
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/resource.h>

#include "common.hpp"
#include "pgdg_align_gpu.hpp"
#include "sha_fast.hpp"
#include "gpu_v7b.h"

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

static const char* variant_name(int v) {
  switch (v) {
    case PgdAlignEngine::DENSE: return "pgdg_dense";
    case PgdAlignEngine::DENSE_CORR: return "pgdg_dense_corr";
    case PgdAlignEngine::SPARSE: return "pgdg_sparse";
    default: return "pgdg_sparse_corr";
  }
}

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

  // ---- CPU setup: aligned PGD-G engine (fixed dense boundary; OpenMP setup) + GPU plan export
  const char* blocks_env = std::getenv("BWH_GPU_BLOCKS");
  const i32 B = blocks_env ? std::atoi(blocks_env) : 32;
  BatchWorkload bw = make_workload_fast(f, topology, seed, factor_period, q, per_state);
  PgdAlignEngine eng(f, bw);
  PgdAutoChoice auto_choice = pgdg_align_auto(f, bw);
  const int variant = auto_choice.variant;
  if (!eng.build(static_cast<PgdAlignEngine::Variant>(variant))) {
    std::fprintf(stderr, "aligned PGD-G build unsupported\n");
    return 3;
  }
  PgdGpuExport ex;
  if (!eng.export_gpu(ex)) {
    std::fprintf(stderr, "GPU export unsupported (needs dense correction-prefix layout)\n");
    return 3;
  }
  if (ex.qn != q) { std::fprintf(stderr, "q mismatch\n"); return 3; }
  const i32 Cpfx = ex.Cpfx, ncls = ex.nclasses, G = ex.G;
  const i32 V = kStates - 1;
  const i32 qq = q;
  std::vector<GpuV7bNode> nodes(static_cast<size_t>(kStates));
  std::memset(nodes.data(), 0, nodes.size() * sizeof(GpuV7bNode));
  for (i32 v = 1; v < kStates; ++v) {
    GpuV7bNode& n = nodes[size_t(v)];
    const size_t i = static_cast<size_t>(v);
    n.kind = uint8_t((ex.kind[i] == 1 ? 1 : 2)
                     | 0);
    if (ex.kind[i] == 1) {
      n.kind = uint8_t(1);
      // existence bits are derived from the amounts (zero amount == absent, matches the engine)
      if (ex.ea[i] != 0) n.kind |= 4;
      if (ex.eb[i] != 0) n.kind |= 8;
    }
    n.g_old = int16_t(ex.go[i]); n.g_new = int16_t(ex.gn[i]);
    n.c_old = ex.co[i]; n.c_new = ex.cn[i];
    n.ti_old = ex.tio[i]; n.ti_new = ex.tin[i];
    n.a = ex.ea[i]; n.b = ex.eb[i];
  }
  std::vector<i32> hv0(static_cast<size_t>(B)), hv1(static_cast<size_t>(B));
  std::vector<i32> bidx(static_cast<size_t>(kStates), 0);
  for (i32 b = 0; b < B; ++b) {
    const i32 v0 = 1 + i32((i64(V) * b) / B);
    const i32 v1 = 1 + i32((i64(V) * (b + 1)) / B);
    hv0[size_t(b)] = v0; hv1[size_t(b)] = v1;
    for (i32 v = v0; v < v1; ++v) bidx[size_t(v)] = b;
  }
  // factor value at every block start (groups not factor-touched stay at s0)
  std::vector<i64> fac_start(static_cast<size_t>(B) * size_t(G), 0);
  {
    std::vector<i64> facv = f.base_factor;
    for (i32 b = 0; b < B; ++b) {
      for (i32 g = 0; g < G; ++g) fac_start[size_t(b) * size_t(G) + size_t(g)] = facv[size_t(g)];
      for (i32 v = hv0[size_t(b)]; v < hv1[size_t(b)]; ++v)
        if (ex.kind[size_t(v)] == 2) facv[size_t(ex.go[size_t(v)])] = ex.ea[size_t(v)];
    }
  }
  // merged support / g2row / qlb,qhb straight from the aligned plan
  const std::vector<u32> ms_off_u = ex.ms_off;
  std::vector<i32> ms_off_v(ms_off_u.begin(), ms_off_u.end());
  const std::vector<i64>& ms_cls_v = ex.ms_cls;
  const std::vector<i64>& ms_base_v = ex.ms_base;
  const std::vector<i32>& ms_dev_v = ex.ms_dev;
  const std::vector<i32>& g2row_v = ex.g2row;
  std::vector<i32> qlb = ex.qlb, qhb = ex.qhb;
  // per-(group,block) deviation snapshot regions + scatter lists (same contract as the native
  // block executor: a row delta is scattered into every later block that needs the group)
  std::vector<u64> gmask(static_cast<size_t>(G), 0);
  for (i32 v = 1; v < kStates; ++v)
    if (ex.kind[size_t(v)] == 2) gmask[size_t(ex.go[size_t(v)])] |= (u64(1) << bidx[size_t(v)]);
  std::vector<i32> gb_off(static_cast<size_t>(G) + 1, 0);
  std::vector<i32> gb_block;
  std::vector<i64> gb_reg;
  std::vector<i64> need_off(static_cast<size_t>(B) * size_t(G), -1);
  i64 devbuf_n = 0;
  for (i32 g = 0; g < G; ++g) {
    gb_off[size_t(g)] = i32(gb_block.size());
    const i32 row = g2row_v[size_t(g)];
    if (row < 0) continue;
    const u32 tsz = ex.tcoord_size[size_t(row)];
    for (i32 b = 0; b < B; ++b) {
      if ((gmask[size_t(g)] >> u64(b)) & 1ull) {
        need_off[size_t(b) * size_t(G) + size_t(g)] = devbuf_n;
        gb_block.push_back(b);
        gb_reg.push_back(devbuf_n);
        devbuf_n += i64(tsz);
      }
    }
  }
  gb_off[size_t(G)] = i32(gb_block.size());
  std::vector<i32> bidx_h = bidx;
  i32 threads = 256;
  if (const char* te = std::getenv("BWH_GPU_THREADS")) threads = std::atoi(te);
  if (threads < 32) threads = 32;
  if (threads > 1024) threads = 1024;
  const int shared_bytes = int((2 * Cpfx + G) * sizeof(i64));
  const size_t lvals_n = size_t(V) * size_t(2 * qq);
  const size_t dV_n = size_t(B) * size_t(Cpfx);
  const size_t carry_n = dV_n;
  const size_t ans_n = size_t(V) * size_t(qq);
  std::vector<i64> dV(dV_n, 0), carrypref(carry_n, 0);
  const auto t2 = std::chrono::steady_clock::now();

  // ---- GPU: allocate + upload once (compact 40 B nodes; snapshots are computed on-device)
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaFree(nullptr));
  GpuV7bNode* d_nodes = nullptr; i32* d_v0 = nullptr; i32* d_v1 = nullptr;
  i64* d_fac_start = nullptr; i32* d_g2row = nullptr; i32* d_ms_off = nullptr;
  i64* d_ms_cls = nullptr; i64* d_ms_base = nullptr; i32* d_ms_dev = nullptr;
  i64* d_need_off = nullptr; i32* d_qlb = nullptr; i32* d_qhb = nullptr; i32* d_bidx = nullptr;
  i32* d_gb_off = nullptr; i32* d_gb_block = nullptr; i64* d_gb_reg = nullptr;
  i64* d_lvals = nullptr; i64* d_dV = nullptr; i64* d_carry = nullptr; i64* d_answers = nullptr;
  i64* d_devbuf = nullptr;
  const auto tup0 = std::chrono::steady_clock::now();
  struct Up { size_t bytes; size_t off; };
  std::vector<Up> ups = {
      {nodes.size() * sizeof(GpuV7bNode), 0}, {hv0.size() * sizeof(i32), 0}, {hv1.size() * sizeof(i32), 0},
      {fac_start.size() * sizeof(i64), 0}, {g2row_v.size() * sizeof(i32), 0},
      {ms_off_v.size() * sizeof(i32), 0}, {ms_cls_v.size() * sizeof(i64), 0},
      {ms_base_v.size() * sizeof(i64), 0}, {ms_dev_v.size() * sizeof(i32), 0},
      {need_off.size() * sizeof(i64), 0}, {qlb.size() * sizeof(i32), 0}, {qhb.size() * sizeof(i32), 0},
      {bidx.size() * sizeof(i32), 0}, {gb_off.size() * sizeof(i32), 0},
      {gb_block.size() * sizeof(i32), 0}, {gb_reg.size() * sizeof(i64), 0},
  };
  size_t total = 0;
  for (auto& u : ups) { total = (total + 255) & ~size_t(255); u.off = total; total += u.bytes; }
  void* gbuf = nullptr;
  CUDA_CHECK(cudaMalloc(&gbuf, total));
  std::vector<char> hbuf(total);
  void* srcs[] = {(void*)nodes.data(), (void*)hv0.data(), (void*)hv1.data(),
                  (void*)fac_start.data(), (void*)g2row_v.data(), (void*)ms_off_v.data(),
                  (void*)ms_cls_v.data(), (void*)ms_base_v.data(), (void*)ms_dev_v.data(),
                  (void*)need_off.data(), (void*)qlb.data(), (void*)qhb.data(),
                  (void*)bidx.data(), (void*)gb_off.data(), (void*)gb_block.data(), (void*)gb_reg.data()};
  for (size_t i = 0; i < ups.size(); ++i) std::memcpy(hbuf.data() + ups[i].off, srcs[i], ups[i].bytes);
  CUDA_CHECK(cudaMemcpy(gbuf, hbuf.data(), total, cudaMemcpyHostToDevice));
  char* gb = static_cast<char*>(gbuf);
  d_nodes = reinterpret_cast<GpuV7bNode*>(gb + ups[0].off);
  d_v0 = reinterpret_cast<i32*>(gb + ups[1].off);
  d_v1 = reinterpret_cast<i32*>(gb + ups[2].off);
  d_fac_start = reinterpret_cast<i64*>(gb + ups[3].off);
  d_g2row = reinterpret_cast<i32*>(gb + ups[4].off);
  d_ms_off = reinterpret_cast<i32*>(gb + ups[5].off);
  d_ms_cls = reinterpret_cast<i64*>(gb + ups[6].off);
  d_ms_base = reinterpret_cast<i64*>(gb + ups[7].off);
  d_ms_dev = reinterpret_cast<i32*>(gb + ups[8].off);
  d_need_off = reinterpret_cast<i64*>(gb + ups[9].off);
  d_qlb = reinterpret_cast<i32*>(gb + ups[10].off);
  d_qhb = reinterpret_cast<i32*>(gb + ups[11].off);
  d_bidx = reinterpret_cast<i32*>(gb + ups[12].off);
  d_gb_off = reinterpret_cast<i32*>(gb + ups[13].off);
  d_gb_block = reinterpret_cast<i32*>(gb + ups[14].off);
  d_gb_reg = reinterpret_cast<i64*>(gb + ups[15].off);
  CUDA_CHECK(cudaMalloc(&d_devbuf, devbuf_n * sizeof(i64)));
  CUDA_CHECK(cudaMemset(d_devbuf, 0, devbuf_n * sizeof(i64)));
  CUDA_CHECK(cudaMalloc(&d_lvals, lvals_n * sizeof(i64)));
  CUDA_CHECK(cudaMalloc(&d_dV, dV_n * sizeof(i64)));
  CUDA_CHECK(cudaMalloc(&d_carry, carry_n * sizeof(i64)));
  CUDA_CHECK(cudaMalloc(&d_answers, ans_n * sizeof(i64)));
  const auto tup1 = std::chrono::steady_clock::now();

  // ---- GPU snapshot kernel: block-start sparse deviation snapshots (on-device, no upload)
  cudaEvent_t es0, es1;
  CUDA_CHECK(cudaEventCreate(&es0)); CUDA_CHECK(cudaEventCreate(&es1));
  GpuV7bSnapArgs sa{};
  sa.nstates = kStates; sa.nodes = d_nodes; sa.bidx = d_bidx;
  sa.gb_off = d_gb_off; sa.gb_block = d_gb_block; sa.gb_reg = d_gb_reg; sa.devbuf = d_devbuf;
  CUDA_CHECK(cudaEventRecord(es0));
  int rcs = gpu_v7b_snap_launch(&sa);
  if (rcs != 0) { std::fprintf(stderr, "snap kernel failed\n"); return 5; }
  CUDA_CHECK(cudaEventRecord(es1));
  CUDA_CHECK(cudaEventSynchronize(es1));
  float snap_ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&snap_ms, es0, es1));
  const auto tsnap = std::chrono::steady_clock::now();

  // ---- kernel 1: independent block-local replay
  GpuV7bArgs a{};
  a.B = B; a.G = G; a.Cpfx = Cpfx; a.ncls = ncls; a.q = qq;
  a.nodes = d_nodes; a.bv0 = d_v0; a.bv1 = d_v1; a.fac_start = d_fac_start;
  a.g2row = d_g2row; a.ms_off = d_ms_off; a.ms_cls = d_ms_cls; a.ms_base = d_ms_base;
  a.ms_dev = d_ms_dev; a.devbuf = d_devbuf; a.need_off = d_need_off;
  a.qlb = d_qlb; a.qhb = d_qhb; a.lvals = d_lvals; a.dV = d_dV;
  a.threads = threads; a.shared_bytes = shared_bytes;
  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));
  CUDA_CHECK(cudaEventRecord(ev0));
  int rc = gpu_v7b_launch(&a);
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
    std::vector<i64> vstart = ex.vbase;
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
  GpuV7bCombineArgs ca{};
  ca.nstates = V; ca.q = qq; ca.Cpfx = Cpfx;
  ca.qlb = d_qlb; ca.qhb = d_qhb; ca.carrypref = d_carry; ca.bidx = d_bidx;
  ca.lvals = d_lvals; ca.answers = d_answers;
  cudaEvent_t ev2, ev3;
  CUDA_CHECK(cudaEventCreate(&ev2)); CUDA_CHECK(cudaEventCreate(&ev3));
  CUDA_CHECK(cudaEventRecord(ev2));
  rc = gpu_v7b_combine_launch(&ca);
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
  CUDA_CHECK(cudaFree(d_devbuf));
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
  const i64 snap_ns = i64(snap_ms * 1e6) + ns_between(tup1, tsnap) - i64(snap_ms * 1e6);
  const i64 execute_ns = upload_ns + snap_ns + kernel_ns + carry_ns + combine_ns + download_ns;
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
  std::fprintf(jf, "  \"arm\": \"gpu_pgdg_align\",\n  \"family\": \"pgdg_align_gpu\",\n  \"mode\": \"%s\",\n  \"base_id\": \"%s\",\n", variant_name(variant), base_id.c_str());
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
                   "\"upload_ns\": %" PRId64 ", \"snap_ns\": %" PRId64 ", \"kernel_ns\": %" PRId64
                   ", \"carry_ns\": %" PRId64 ", \"combine_ns\": %" PRId64
                   ", \"download_ns\": %" PRId64 ", \"node_bytes\": %zu, \"devbuf_bytes\": %zu "
                   "},\n",
               devname, B, threads, shared_bytes, upload_ns, snap_ns, kernel_ns, carry_ns,
               combine_ns, download_ns, nodes.size() * sizeof(GpuV7bNode), devbuf_n * sizeof(i64));
  std::fprintf(jf, "  \"full_ns\": %" PRId64 ",\n  \"peak_rss_kb\": %" PRId64 "\n}\n", full_ns, peak_kb);
  std::fclose(jf);
  if (std::rename((json_path + ".tmp").c_str(), json_path.c_str()) != 0) return 4;
  std::printf("arm=gpu_pgdg_align base=%s t%d s%d f%d q%d r%d blocks=%d kernel_ms=%.3f full_ns=%" PRId64
              " peak_kb=%" PRId64 "\n",
              base_id.c_str(), topology, seed, factor_period, q, repeat, B, double(kernel_ns) / 1e6,
              full_ns, peak_kb);
  return 0;
}
