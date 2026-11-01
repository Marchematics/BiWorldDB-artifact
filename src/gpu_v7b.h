// gpu_v7b.h — POD interface for the GPU phase-2 kernels (compact v7 plan encoding, on-device
// block-start snapshots, block-parallel replay).  New file 2026-11-01; no CUDA types.
#pragma once
#include <cstdint>

struct GpuV7bNode {  // 40 bytes: kind bits (1=row, 2=factor, bit2=ex_old, bit3=ex_new)
  uint8_t kind;
  int16_t g_old, g_new;
  int32_t c_old, c_new;
  int32_t ti_old, ti_new;
  int64_t a, b;   // row: amt_old, amt_new ; factor: fac_val, fac_prev
};

struct GpuV7bSnapArgs {
  int nstates;               // kStates
  const GpuV7bNode* nodes;
  const int32_t* bidx;       // state -> block
  const int32_t* gb_off;     // G+1
  const int32_t* gb_block;   // per (group,block) occurrence
  const int64_t* gb_reg;     // devbuf offset of the group's region in that block
  int64_t* devbuf;
};

struct GpuV7bArgs {
  int B, G, Cpfx, ncls, q;
  const GpuV7bNode* nodes;
  const int32_t* bv0, *bv1;
  const int64_t* fac_start;   // B*G
  const int32_t* g2row;       // G
  const int32_t* ms_off;      // n_active+1
  const int64_t* ms_cls, *ms_base;
  const int32_t* ms_dev;
  int64_t* devbuf;
  const int64_t* need_off;    // B*G
  const int32_t* qlb, *qhb;
  int64_t* lvals, *dV;
  int threads;
  int shared_bytes;
};

struct GpuV7bCombineArgs {
  int nstates, q, Cpfx;
  const int32_t* qlb, *qhb;
  const int64_t* carrypref;   // B*Cpfx
  const int32_t* bidx;
  const int64_t* lvals;
  int64_t* answers;
};

extern "C" int gpu_v7b_snap_launch(const GpuV7bSnapArgs* a);
extern "C" int gpu_v7b_launch(const GpuV7bArgs* a);
extern "C" int gpu_v7b_combine_launch(const GpuV7bCombineArgs* a);
