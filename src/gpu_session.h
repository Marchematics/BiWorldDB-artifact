// gpu_session.h — POD contract for the GPU session-resident / tiled backend (new file, 2026-11-01).
#pragma once
#include <cstdint>
struct GpuSessNode { uint8_t kind; int16_t g_old, g_new; int32_t c_old, c_new, ti_old, ti_new; int64_t a, b; };
struct GpuSessResetArgs { int B, Cpfx; int64_t* fen; int64_t* leaf; };
struct GpuSessRunArgs {
  int B, G, Cpfx, ncls, q;
  const GpuSessNode* nodes;
  const int32_t* bv0, *bv1;
  const int64_t* fac_start;
  const int32_t* g2row;
  const int32_t* ms_off; const int64_t* ms_cls, *ms_base; const int32_t* ms_dev;
  int64_t* devbuf; const int64_t* need_off;
  const int32_t* qlb, *qhb;
  int64_t* lvals; int64_t* dV;
  int64_t* fen; int64_t* leaf;   // B * Cpfx each (global memory, tiled prefix domain)
  int threads;
};
struct GpuSessCombineArgs { int nstates,q,Cpfx; const int32_t* qlb,*qhb; const int64_t* carrypref; const int32_t* bidx; const int64_t* lvals; int64_t* answers; };
struct GpuSessSnapArgs { int nstates; const GpuSessNode* nodes; const int32_t* bidx; const int32_t* gb_off; const int32_t* gb_block; const int64_t* gb_reg; int64_t* devbuf; };
extern "C" int gpu_sess_reset_launch(const GpuSessResetArgs* a);
extern "C" int gpu_sess_run_launch(const GpuSessRunArgs* a);
extern "C" int gpu_sess_snap_launch(const GpuSessSnapArgs* a);
extern "C" int gpu_sess_combine_launch(const GpuSessCombineArgs* a);
