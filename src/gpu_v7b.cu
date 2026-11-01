// gpu_v7b.cu — CUDA phase-2 kernels (one RTX 4090).  New file 2026-11-01.
//  * snap_kernel: builds the block-start sparse deviation snapshots on the device by scattering
//    every row-edit delta into every later block where its group is needed (no CPU pre-pass and
//    no snapshot upload);
//  * block kernel: identical to the v7 block replay but reading the compact 40-byte node
//    encoding; one CUDA block per state block, local Fenwick+leaf+factor in dynamic shared memory;
//  * combine kernel: carry prefix + stored local endpoint prefixes.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "gpu_v7b.h"

__global__ void snap_kernel(GpuV7bSnapArgs a) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (v >= a.nstates) return;
  const GpuV7bNode p = a.nodes[v];
  if ((p.kind & 3) != 1) return;  // row edits only
  const int b0 = a.bidx[v];
  auto scatter = [&](int g, int ti, int64_t delta) {
    for (int at = a.gb_off[g]; at < a.gb_off[g + 1]; ++at)
      if (a.gb_block[at] > b0) atomicAdd(reinterpret_cast<unsigned long long*>(&a.devbuf[a.gb_reg[at] + ti]), (unsigned long long)delta);
  };
  if (p.kind & 4) scatter(p.g_old, p.ti_old, -p.a);
  if (p.kind & 8) scatter(p.g_new, p.ti_new, p.b);
}

__global__ void block_kernel(GpuV7bArgs a) {
  extern __shared__ int64_t smem[];
  int64_t* fen = smem;
  int64_t* leaf = smem + a.Cpfx;
  int64_t* fac = smem + 2 * a.Cpfx;
  const int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  const int Cpfx = a.Cpfx, ncls = a.ncls, G = a.G, q = a.q;
  for (int i = tid; i < Cpfx; i += nt) { fen[i] = 0; leaf[i] = 0; }
  for (int i = tid; i < G; i += nt) fac[i] = a.fac_start[(size_t)b * G + i];
  __syncthreads();
  const int v0 = a.bv0[b], v1 = a.bv1[b];
  for (int v = v0; v < v1; ++v) {
    const GpuV7bNode p = a.nodes[v];
    const int kind = p.kind & 3;
    if (kind == 1) {
      if (tid == 0) {
        if ((p.kind & 4) && p.a != 0) {
          const int64_t w = fac[p.g_old] * p.a;
          leaf[p.c_old + 1] -= w;
          for (int i = p.c_old + 1; i <= ncls; i += i & (-i)) fen[i] -= w;
          const int64_t off = a.need_off[(size_t)b * G + p.g_old];
          if (off >= 0) a.devbuf[off + p.ti_old] -= p.a;
        }
        if ((p.kind & 8) && p.b != 0) {
          const int64_t w = fac[p.g_new] * p.b;
          leaf[p.c_new + 1] += w;
          for (int i = p.c_new + 1; i <= ncls; i += i & (-i)) fen[i] += w;
          const int64_t off = a.need_off[(size_t)b * G + p.g_new];
          if (off >= 0) a.devbuf[off + p.ti_new] += p.b;
        }
      }
      __syncthreads();
    } else if (kind == 2) {
      const int g = p.g_old;
      const int row = a.g2row[g];
      const int64_t off = a.need_off[(size_t)b * G + g];
      const int64_t ds = p.a - p.b;
      if (row >= 0 && off >= 0) {
        const int kb = a.ms_off[row], ke = a.ms_off[row + 1];
        for (int k = kb + tid; k < ke; k += nt) {
          const int di = a.ms_dev[k];
          const int64_t dev = (di >= 0) ? a.devbuf[off + di] : 0;
          const int64_t d = ds * (a.ms_base[k] + dev);
          if (d) leaf[(int)a.ms_cls[k] + 1] += d;
        }
        __syncthreads();
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
        if (tid == 0) fac[g] = p.a;
        __syncthreads();
      } else if (tid == 0) {
        fac[g] = p.a;
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

__global__ void combine_kernel(GpuV7bCombineArgs a) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int v = idx + 1;
  if (v > a.nstates) return;
  const int b = a.bidx[v];
  const int64_t* cp = a.carrypref + (size_t)b * a.Cpfx;
  for (int k = 0; k < a.q; ++k) {
    const int i = a.qlb[v * a.q + k], j = a.qhb[v * a.q + k];
    const size_t lb = (size_t)(v - 1) * (size_t)a.q + (size_t)k;
    a.answers[lb] = (cp[j] - cp[i]) + (a.lvals[lb * 2 + 1] - a.lvals[lb * 2]);
  }
}

extern "C" int gpu_v7b_snap_launch(const GpuV7bSnapArgs* a) {
  const int total = a->nstates - 1;
  const int threads = 256;
  snap_kernel<<<(total + threads - 1) / threads, threads>>>(*a);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) std::fprintf(stderr, "gpu: snap kernel: %s\n", cudaGetErrorString(e));
  return (int)e;
}

extern "C" int gpu_v7b_launch(const GpuV7bArgs* a) {
  static bool attr_done = false;
  if (!attr_done) {
    cudaError_t e = cudaFuncSetAttribute(reinterpret_cast<const void*>(block_kernel),
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    if (e != cudaSuccess) { std::fprintf(stderr, "gpu: attr: %s\n", cudaGetErrorString(e)); return (int)e; }
    attr_done = true;
  }
  block_kernel<<<a->B, a->threads, a->shared_bytes>>>(*a);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) std::fprintf(stderr, "gpu: block kernel: %s\n", cudaGetErrorString(e));
  return (int)e;
}

extern "C" int gpu_v7b_combine_launch(const GpuV7bCombineArgs* a) {
  const int threads = 256;
  combine_kernel<<<(a->nstates + threads - 1) / threads, threads>>>(*a);
  return (int)cudaGetLastError();
}
