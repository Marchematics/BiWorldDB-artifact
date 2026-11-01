// gpu_v7.h — POD interface between the CPU v7 setup and the CUDA block-parallel kernels
// (2026-11-01, GPU round).  New file; no CUDA types so it can be included from g++ and nvcc.
#pragma once
#include <cstdint>
#include <cstddef>

struct GpuV7Node {  // one edited state (node 0 is unused)
  int32_t kind;      // 1 row, 2 factor
  int32_t g_old, c_old, ti_old;
  int32_t g_new, c_new, ti_new;
  int64_t amt_old, amt_new, fac_val, fac_prev;
  uint8_t ex_old, ex_new;
  uint8_t pad[6];
};

struct GpuV7Args {
  int B, G, Cpfx, ncls, q;
  const GpuV7Node* nodes;
  const int32_t* block_v0;
  const int32_t* block_v1;
  const int64_t* fac_start;   // B*G
  const int32_t* g2row;       // G
  const int32_t* ms_off;      // n_active+1
  const int64_t* ms_cls;      // support classes
  const int64_t* ms_base;     // base mass at support classes
  const int32_t* ms_dev;      // deviation index or -1
  int64_t* devbuf;            // sparse deviation snapshots, block-local regions
  const int64_t* need_off;    // B*G absolute offsets into devbuf, -1 when not needed
  const int32_t* qlb;
  const int32_t* qhb;
  int64_t* lvals;             // (kStates-1)*2q local endpoint prefixes
  int64_t* dV;                // B*Cpfx dense block V-leaf deltas
  int threads;
  int shared_bytes;
};

struct GpuV7CombineArgs {
  int nstates;   // kStates-1
  int q;
  int Cpfx;
  const int32_t* qlb;
  const int32_t* qhb;
  const int64_t* carrypref;  // B*Cpfx
  const int32_t* bidx;       // kStates state -> block
  const int64_t* lvals;
  int64_t* answers;
};

extern "C" int gpu_v7_launch(const GpuV7Args* a);
extern "C" int gpu_combine_launch(const GpuV7CombineArgs* a);
