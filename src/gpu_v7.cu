// gpu_v7.cu — CUDA kernels for the v7 block-parallel chain execution (one RTX 4090).
// New file (2026-11-01, GPU round).  The CPU keeps the v7 setup/pre-pass; the kernels below
// execute one CUDA block per version-state block (independent local replay), writing each
// block's dense V-leaf delta and per-state local endpoint prefixes, and a small combine kernel.
#include <cuda_runtime.h>
#include <cstdint>
#include "gpu_v7.h"
#include <cstdio>

__global__ void __launch_bounds__(256) v7_block_kernel(GpuV7Args a) {
  extern __shared__ int64_t smem[];
  int64_t* fen = smem;              // Fenwick over classes (index 1..ncls)
  int64_t* leaf = smem + a.Cpfx;    // dense local V-leaf (index c+1)
  int64_t* fac = smem + 2 * a.Cpfx; // current factor per group
  const int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  const int Cpfx = a.Cpfx, ncls = a.ncls, G = a.G, q = a.q;
  for (int i = tid; i < Cpfx; i += nt) { fen[i] = 0; leaf[i] = 0; }
  for (int i = tid; i < G; i += nt) fac[i] = a.fac_start[(size_t)b * G + i];
  __syncthreads();
  const int v0 = a.block_v0[b], v1 = a.block_v1[b];
  for (int v = v0; v < v1; ++v) {
    const GpuV7Node p = a.nodes[v];
    if (p.kind == 1) {
      if (tid == 0) {
        if (p.ex_old && p.amt_old != 0) {
          const int64_t w = fac[p.g_old] * p.amt_old;
          leaf[p.c_old + 1] -= w;
          for (int i = p.c_old + 1; i <= ncls; i += i & (-i)) fen[i] -= w;
          const int64_t off = a.need_off[(size_t)b * G + p.g_old];
          if (off >= 0) a.devbuf[off + p.ti_old] -= p.amt_old;
        }
        if (p.ex_new && p.amt_new != 0) {
          const int64_t w = fac[p.g_new] * p.amt_new;
          leaf[p.c_new + 1] += w;
          for (int i = p.c_new + 1; i <= ncls; i += i & (-i)) fen[i] += w;
          const int64_t off = a.need_off[(size_t)b * G + p.g_new];
          if (off >= 0) a.devbuf[off + p.ti_new] += p.amt_new;
        }
      }
      __syncthreads();
    } else if (p.kind == 2) {
      const int g = p.g_old;
      const int row = a.g2row[g];
      const int64_t off = a.need_off[(size_t)b * G + g];
      const int64_t ds = p.fac_val - p.fac_prev;
      if (row >= 0 && off >= 0) {
        const int kb = a.ms_off[row], ke = a.ms_off[row + 1];
        for (int k = kb + tid; k < ke; k += nt) {
          const int di = a.ms_dev[k];
          const int64_t dev = (di >= 0) ? a.devbuf[off + di] : 0;
          const int64_t d = ds * (a.ms_base[k] + dev);
          if (d) leaf[(int)a.ms_cls[k] + 1] += d;
        }
        __syncthreads();
        // rebuild the local Fenwick from the dense leaf (level-parallel O(C) build)
        for (int i = tid; i < Cpfx; i += nt) fen[i] = leaf[i];
        __syncthreads();
        for (int s = 1; s < Cpfx; s <<= 1) {
          for (int idx = tid;; idx += nt) {
            const int i = s + 2 * s * idx;
            if (i + s >= Cpfx) break;
            fen[i + s] += fen[i];
          }
          __syncthreads();
        }
        if (tid == 0) fac[g] = p.fac_val;
        __syncthreads();
      } else if (tid == 0) {
        fac[g] = p.fac_val;
      }
    }
    for (int e = tid; e < 2 * q; e += nt) {
      const int kk = e >> 1;
      const int idx = (e & 1) ? a.qhb[v * q + kk] : a.qlb[v * q + kk];
      int64_t r = 0;
      for (int i = idx; i > 0; i -= i & (-i)) r += fen[i];
      a.lvals[(size_t)(v - 1) * (size_t)(2 * q) + e] = r;
    }
    __syncthreads();
  }
  for (int i = tid; i < Cpfx; i += nt) a.dV[(size_t)b * Cpfx + i] = leaf[i];
}

__global__ void gpu_combine_kernel(GpuV7CombineArgs a) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int v = idx + 1;
  if (v > a.nstates) return;
  const int b = a.bidx[v];
  const int64_t* cp = a.carrypref + (size_t)b * a.Cpfx;
  for (int k = 0; k < a.q; ++k) {
    const int i = a.qlb[v * a.q + k], j = a.qhb[v * a.q + k];
    const int64_t base = cp[j] - cp[i];
    const size_t lb = (size_t)(v - 1) * (size_t)(a.q) + (size_t)k;
    a.answers[lb] = base + (a.lvals[lb * 2 + 1] - a.lvals[lb * 2]);
  }
}

extern "C" int gpu_v7_launch(const GpuV7Args* a) {
  // dynamic shared memory above the 48 KiB default needs an explicit opt-in
  static bool attr_done = false;
  if (!attr_done) {
    cudaError_t e = cudaFuncSetAttribute(reinterpret_cast<const void*>(v7_block_kernel),
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    if (e != cudaSuccess) { std::fprintf(stderr, "gpu: func attr failed: %s\n", cudaGetErrorString(e)); return (int)e; }
    attr_done = true;
  }
  v7_block_kernel<<<a->B, a->threads, a->shared_bytes>>>(*a);
  cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess)
    std::fprintf(stderr, "gpu: launch B=%d thr=%d shmem=%d failed: %s\n", a->B, a->threads,
                 a->shared_bytes, cudaGetErrorString(le));
  return (int)le;
}

extern "C" int gpu_combine_launch(const GpuV7CombineArgs* a) {
  const int total = a->nstates;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  gpu_combine_kernel<<<blocks, threads>>>(*a);
  return (int)cudaGetLastError();
}
