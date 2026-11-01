// gpu_session.cu — session-resident, tiled-prefix chain replay kernels (new file, 2026-11-01).
// The class prefix is kept in GLOBAL memory per state block (Fenwick over the full class domain,
// tiled by the Fenwick stride), so there is no 6144-class shared-memory limit and Q8/deep frames
// work.  A session uploads nodes/metadata once; each query batch only reruns the replay kernel and
// the combine kernel with the batch's class bounds.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "gpu_session.h"

__global__ void sess_reset_kernel(GpuSessResetArgs a) {
  const long long n = (long long)a.B * a.Cpfx;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (long long)gridDim.x * blockDim.x) { a.fen[i] = 0; a.leaf[i] = 0; }
}

__global__ void sess_snap_kernel(GpuSessSnapArgs a) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (v >= a.nstates) return;
  const GpuSessNode p = a.nodes[v];
  if ((p.kind & 3) != 1) return;
  const int b0 = a.bidx[v];
  auto scatter = [&](int g, int ti, int64_t delta) {
    for (int at = a.gb_off[g]; at < a.gb_off[g + 1]; ++at)
      if (a.gb_block[at] > b0)
        atomicAdd(reinterpret_cast<unsigned long long*>(&a.devbuf[a.gb_reg[at] + ti]), (unsigned long long)delta);
  };
  if (p.kind & 4) scatter(p.g_old, p.ti_old, -p.a);
  if (p.kind & 8) scatter(p.g_new, p.ti_new, p.b);
}

__global__ void sess_run_kernel(GpuSessRunArgs a) {
  extern __shared__ int64_t sh[];
  int64_t* fac = sh;  // G
  const int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  const int Cpfx = a.Cpfx, ncls = a.ncls, G = a.G, q = a.q;
  int64_t* fen = a.fen + (size_t)b * Cpfx;
  int64_t* leaf = a.leaf + (size_t)b * Cpfx;
  for (int i = tid; i < G; i += nt) fac[i] = a.fac_start[(size_t)b * G + i];
  __syncthreads();
  const int v0 = a.bv0[b], v1 = a.bv1[b];
  for (int v = v0; v < v1; ++v) {
    const GpuSessNode p = a.nodes[v];
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
          if (d) {
            const int c1 = (int)a.ms_cls[k] + 1;
            leaf[c1] += d;   // distinct support classes: no conflicting writes
          }
        }
        __syncthreads();
        // rebuild the Fenwick tree from the dense leaf (level-parallel O(C), deterministic)
        for (int i = tid; i < Cpfx; i += nt) fen[i] = leaf[i];
        __syncthreads();
        for (int stride = 1; stride < Cpfx; stride <<= 1) {
          for (int idx = tid;; idx += nt) {
            const int i = stride + 2 * stride * idx;
            if (i + stride >= Cpfx) break;
            fen[i + stride] += fen[i];
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
}

__global__ void sess_combine_kernel(GpuSessCombineArgs a) {
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

extern "C" int gpu_sess_reset_launch(const GpuSessResetArgs* a) {
  const long long n = (long long)a->B * a->Cpfx;
  const int th = 256, bl = (int)((n + th - 1) / th > 65535 ? 65535 : (n + th - 1) / th);
  sess_reset_kernel<<<bl, th>>>(*a);
  return (int)cudaGetLastError();
}
extern "C" int gpu_sess_snap_launch(const GpuSessSnapArgs* a) {
  const int th = 256, total = a->nstates - 1;
  sess_snap_kernel<<<(total + th - 1) / th, th>>>(*a);
  return (int)cudaGetLastError();
}
extern "C" int gpu_sess_run_launch(const GpuSessRunArgs* a) {
  sess_run_kernel<<<a->B, a->threads, size_t(a->G) * sizeof(int64_t)>>>(*a);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) std::fprintf(stderr, "gpu_session: run kernel: %s\n", cudaGetErrorString(e));
  return (int)e;
}
extern "C" int gpu_sess_combine_launch(const GpuSessCombineArgs* a) {
  const int th = 256;
  sess_combine_kernel<<<(a->nstates + th - 1) / th, th>>>(*a);
  return (int)cudaGetLastError();
}
