// panel_gpu_session.cpp — GPU session-resident/tiled backend (new file, 2026-11-01).
// One process = one session: fixture + declared workload + plan are built and uploaded once; every
// query batch then reruns only the global-memory (tiled-prefix) chain replay + combine kernels.
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "common.hpp"
#include "batch_engine_v7c.hpp"
#include "sha_fast.hpp"
#include "gpu_session.h"
using namespace bwh;
static i64 nsb(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}
#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){std::fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e_),__LINE__);return 5;} } while(0)
int main(int argc, char** argv) {
  std::string arm="gpu_session", fixture_path, base_id, answers_path, json_path;
  i32 topology=0, seed=7, factor_period=37, q=2, repeat=0, batches=10, blocks=-1;
  for (int i=1;i<argc;++i) {
    auto nx=[&](const char*k)->std::string{ if(i+1>=argc){std::fprintf(stderr,"missing %s\n",k);std::exit(2);} return argv[++i]; };
    if(!std::strcmp(argv[i],"--fixture")) fixture_path=nx(argv[i]);
    else if(!std::strcmp(argv[i],"--base-id")) base_id=nx(argv[i]);
    else if(!std::strcmp(argv[i],"--topology")) topology=std::atoi(nx(argv[i]).c_str());
    else if(!std::strcmp(argv[i],"--seed")) seed=std::atoi(nx(argv[i]).c_str());
    else if(!std::strcmp(argv[i],"--factor-period")) factor_period=std::atoi(nx(argv[i]).c_str());
    else if(!std::strcmp(argv[i],"--q")) q=std::atoi(nx(argv[i]).c_str());
    else if(!std::strcmp(argv[i],"--repeat")) repeat=std::atoi(nx(argv[i]).c_str());
    else if(!std::strcmp(argv[i],"--batches")) batches=std::atoi(nx(argv[i]).c_str());
    else if(!std::strcmp(argv[i],"--blocks")) blocks=std::atoi(nx(argv[i]).c_str());
    else if(!std::strcmp(argv[i],"--answers")) answers_path=nx(argv[i]);
    else if(!std::strcmp(argv[i],"--json")) json_path=nx(argv[i]);
    else { std::fprintf(stderr,"unknown arg %s\n",argv[i]); return 2; }
  }
  if (fixture_path.empty()||answers_path.empty()||json_path.empty()) { std::fprintf(stderr,"usage: panel_gpu_session --fixture F --base-id ID --topology T --seed S --factor-period P --q Q --batches N --answers A --json J\n"); return 2; }
  const char* scope=std::getenv("BWH_RANGE_SCOPE"); const bool per_state = scope && !std::strcmp(scope,"per_state");
  const auto t0=std::chrono::steady_clock::now();
  Fixture f; if(!load_fixture(fixture_path,f)){std::fprintf(stderr,"load failed\n");return 3;}
  const auto t1=std::chrono::steady_clock::now();
  setenv("BWH_V7_NOPREP","1",1);
  BatchWorkloadV7 bw=make_workload_fast_v7(f,topology,seed,factor_period,q,per_state);
  BatchEngineV7 eng(f,bw);
  const bool ok=eng.build();
  const auto t2=std::chrono::steady_clock::now();
  if(!ok){ std::fprintf(stderr,"session unsupported: engine gates rejected this frame (q=%d C=%d)\n",q,eng.ncls); return 3; }
  const i32 V=kStates-1, qq=q;
  const i32 Cpfx=eng.Cpfx, ncls=eng.ncls, G=eng.G;
  i32 B = blocks>0 ? blocks : eng.v7_blocks;
  if (B < 1) B = 1;
  const i32 threads=256;
  const size_t lvals_n=size_t(V)*size_t(2*qq), dV_n=size_t(B)*size_t(Cpfx), ans_n=size_t(V)*size_t(qq);
  // host session inputs
  std::vector<GpuSessNode> nodes(static_cast<size_t>(kStates)); std::memset(nodes.data(),0,nodes.size()*sizeof(GpuSessNode));
  for(i32 v=1;v<kStates;++v){ const BatchEngineV7::NodePlan& p=eng.plan[size_t(v)]; GpuSessNode& n=nodes[size_t(v)];
    n.kind=uint8_t((p.kind==1?1:2)|(p.ex_old?4:0)|(p.ex_new?8:0)); n.g_old=int16_t(p.g_old); n.g_new=int16_t(p.g_new);
    n.c_old=p.c_old; n.c_new=p.c_new; n.ti_old=p.ti_old; n.ti_new=p.ti_new;
    if(p.kind==1){n.a=p.amt_old;n.b=p.amt_new;} else {n.a=p.fac_val;n.b=p.fac_prev;} }
  std::vector<i32> hv0(static_cast<size_t>(B)),hv1(static_cast<size_t>(B));
  if (B == eng.v7_blocks) { for(i32 b=0;b<B;++b){hv0[size_t(b)]=eng.v7_v0[size_t(b)];hv1[size_t(b)]=eng.v7_v1[size_t(b)];} }
  else { const i32 VS=kStates-1; const i32 VB=eng.v7_blocks; for(i32 b=0;b<B;++b){ hv0[size_t(b)]=1+i32((i64(VS)*b)/B); hv1[size_t(b)]=(b==B-1)?kStates:1+i32((i64(VS)*(b+1))/B);} (void)VB; }
  std::vector<i64> fac_start(static_cast<size_t>(B)*static_cast<size_t>(G));
  if (B == eng.v7_blocks) { for(i32 b=0;b<B;++b) for(i32 g=0;g<G;++g) fac_start[size_t(b)*size_t(G)+g]=eng.v7_fac[size_t(b)][size_t(g)]; }
  else { std::vector<i64> fcur=eng.s0; for(i32 v=1;v<kStates;++v){ /* fallback: per-block factor vector computed below */ }
         // recompute factor vectors by scanning plan for this B
         std::vector<i64> fc=eng.s0; i32 bcur=0;
         for(i32 v=1;v<kStates;++v){ while(bcur+1<B && v>=hv1[size_t(bcur)]) ++bcur; if(v==hv0[size_t(bcur)]) for(i32 g=0;g<G;++g) fac_start[size_t(bcur)*size_t(G)+g]=fc[size_t(g)];
           const BatchEngineV7::NodePlan& p=eng.plan[size_t(v)]; if(p.kind==2) fc[size_t(p.g_old)]=p.fac_val; } }
  std::vector<i64> need_off(static_cast<size_t>(B)*static_cast<size_t>(G),-1);
  for(i32 b=0;b<B;++b) for(size_t i=0;i<eng.v7_groups[size_t(b)].size() && B==eng.v7_blocks;++i) need_off[size_t(b)*size_t(G)+size_t(eng.v7_groups[size_t(b)][i])]=eng.v7_snap_off[size_t(b)][i];
  if (B != eng.v7_blocks) { std::fprintf(stderr,"custom --blocks not supported in this build\n"); return 2; }
  std::vector<i32> gb_off=eng.v7_gb_off; std::vector<i32> gb_block=eng.v7_gb_block; std::vector<i64> gb_reg=eng.v7_gb_reg;
  const size_t devbuf_n=size_t(eng.v7_block_off[size_t(B)]);
  std::vector<i32> qlb(static_cast<size_t>(kStates)*static_cast<size_t>(qq)), qhb(static_cast<size_t>(kStates)*static_cast<size_t>(qq));
  for(size_t i=0;i<qlb.size();++i){qlb[i]=i32(eng.qlb[i]);qhb[i]=i32(eng.qhb[i]);}
  std::vector<i32> g2row(eng.g2row.begin(),eng.g2row.end()), bidx(eng.v7_bidx.begin(),eng.v7_bidx.end());
  const auto tu0=std::chrono::steady_clock::now();
  CK(cudaSetDevice(0)); CK(cudaFree(nullptr));
  GpuSessNode *d_nodes=nullptr; i32 *d_v0=nullptr,*d_v1=nullptr,*d_g2row=nullptr,*d_ms_off=nullptr,*d_ms_dev=nullptr;
  i64 *d_fac_start=nullptr,*d_ms_cls=nullptr,*d_ms_base=nullptr,*d_need_off=nullptr; i32 *d_qlb=nullptr,*d_qhb=nullptr,*d_bidx=nullptr,*d_gb_off=nullptr,*d_gb_block=nullptr; i64 *d_gb_reg=nullptr;
  i64 *d_devbuf=nullptr,*d_devbuf_bak=nullptr,*d_lvals=nullptr,*d_dV=nullptr,*d_carry=nullptr,*d_answers=nullptr,*d_fen=nullptr,*d_leaf=nullptr;
  struct Up{void** dst;const void* src;size_t bytes;size_t off;};
  std::vector<Up> ups={
    {(void**)&d_nodes,nodes.data(),nodes.size()*sizeof(GpuSessNode),0},
    {(void**)&d_v0,hv0.data(),hv0.size()*sizeof(i32),0},{(void**)&d_v1,hv1.data(),hv1.size()*sizeof(i32),0},
    {(void**)&d_fac_start,fac_start.data(),fac_start.size()*sizeof(i64),0},{(void**)&d_g2row,g2row.data(),g2row.size()*sizeof(i32),0},
    {(void**)&d_ms_off,eng.ms_off.data(),eng.ms_off.size()*sizeof(i32),0},{(void**)&d_ms_cls,eng.ms_cls.data(),eng.ms_cls.size()*sizeof(i64),0},
    {(void**)&d_ms_base,eng.ms_base.data(),eng.ms_base.size()*sizeof(i64),0},{(void**)&d_ms_dev,eng.ms_dev.data(),eng.ms_dev.size()*sizeof(i32),0},
    {(void**)&d_need_off,need_off.data(),need_off.size()*sizeof(i64),0},{(void**)&d_qlb,qlb.data(),qlb.size()*sizeof(i32),0},
    {(void**)&d_qhb,qhb.data(),qhb.size()*sizeof(i32),0},{(void**)&d_bidx,bidx.data(),bidx.size()*sizeof(i32),0},
    {(void**)&d_gb_off,gb_off.data(),gb_off.size()*sizeof(i32),0},{(void**)&d_gb_block,gb_block.data(),gb_block.size()*sizeof(i32),0},
    {(void**)&d_gb_reg,gb_reg.data(),gb_reg.size()*sizeof(i64),0}};
  size_t total=0; for(auto&u:ups){total=(total+255)&~size_t(255);u.off=total;total+=u.bytes;}
  void* gbuf=nullptr; CK(cudaMalloc(&gbuf,total)); std::vector<char> hbuf(total);
  for(auto&u:ups) std::memcpy(hbuf.data()+u.off,u.src,u.bytes);
  CK(cudaMemcpy(gbuf,hbuf.data(),total,cudaMemcpyHostToDevice));
  char* gb=static_cast<char*>(gbuf);
  d_nodes=(GpuSessNode*)(gb+ups[0].off); d_v0=(i32*)(gb+ups[1].off); d_v1=(i32*)(gb+ups[2].off);
  d_fac_start=(i64*)(gb+ups[3].off); d_g2row=(i32*)(gb+ups[4].off); d_ms_off=(i32*)(gb+ups[5].off);
  d_ms_cls=(i64*)(gb+ups[6].off); d_ms_base=(i64*)(gb+ups[7].off); d_ms_dev=(i32*)(gb+ups[8].off);
  d_need_off=(i64*)(gb+ups[9].off); d_qlb=(i32*)(gb+ups[10].off); d_qhb=(i32*)(gb+ups[11].off);
  d_bidx=(i32*)(gb+ups[12].off); d_gb_off=(i32*)(gb+ups[13].off); d_gb_block=(i32*)(gb+ups[14].off); d_gb_reg=(i64*)(gb+ups[15].off);
  CK(cudaMalloc(&d_devbuf,devbuf_n*sizeof(i64))); CK(cudaMemset(d_devbuf,0,devbuf_n*sizeof(i64)));
  CK(cudaMalloc(&d_lvals,lvals_n*sizeof(i64))); CK(cudaMalloc(&d_dV,dV_n*sizeof(i64))); CK(cudaMalloc(&d_carry,dV_n*sizeof(i64)));
  CK(cudaMalloc(&d_answers,ans_n*sizeof(i64))); CK(cudaMalloc(&d_fen,size_t(B)*size_t(Cpfx)*sizeof(i64))); CK(cudaMalloc(&d_leaf,size_t(B)*size_t(Cpfx)*sizeof(i64)));
  GpuSessSnapArgs sa{}; sa.nstates=kStates; sa.nodes=d_nodes; sa.bidx=d_bidx; sa.gb_off=d_gb_off; sa.gb_block=d_gb_block; sa.gb_reg=d_gb_reg; sa.devbuf=d_devbuf;
  int rc=gpu_sess_snap_launch(&sa); if(rc){std::fprintf(stderr,"snap failed\n");return 5;}
  CK(cudaMalloc(&d_devbuf_bak,devbuf_n*sizeof(i64)));
  CK(cudaMemcpy(d_devbuf_bak,d_devbuf,devbuf_n*sizeof(i64),cudaMemcpyDeviceToDevice));
  GpuSessResetArgs rea{}; rea.B=B; rea.Cpfx=Cpfx; rea.fen=d_fen; rea.leaf=d_leaf;
  GpuSessRunArgs ra{}; ra.B=B; ra.G=G; ra.Cpfx=Cpfx; ra.ncls=ncls; ra.q=qq; ra.nodes=d_nodes; ra.bv0=d_v0; ra.bv1=d_v1;
  ra.fac_start=d_fac_start; ra.g2row=d_g2row; ra.ms_off=d_ms_off; ra.ms_cls=d_ms_cls; ra.ms_base=d_ms_base; ra.ms_dev=d_ms_dev;
  ra.devbuf=d_devbuf; ra.need_off=d_need_off; ra.qlb=d_qlb; ra.qhb=d_qhb; ra.lvals=d_lvals; ra.dV=d_dV; ra.fen=d_fen; ra.leaf=d_leaf; ra.threads=threads;
  rc=gpu_sess_reset_launch(&rea); if(rc){std::fprintf(stderr,"reset failed\n");return 5;}
  rc=gpu_sess_run_launch(&ra); if(rc){std::fprintf(stderr,"warmup failed\n");return 5;}
  std::vector<i64> dV(dV_n,0), carrypref(dV_n,0);
  CK(cudaMemcpy(dV.data(),d_dV,dV_n*sizeof(i64),cudaMemcpyDeviceToHost));
  { std::vector<i64> vstart=eng.v7_Vbase;
    for(i32 b=0;b<B;++b){ i64* cp=&carrypref[size_t(b)*size_t(Cpfx)]; cp[0]=0;
      for(i32 k=1;k<Cpfx;++k) cp[size_t(k)]=cp[size_t(k-1)]+vstart[size_t(k)];
      const i64* dv=&dV[size_t(b)*size_t(Cpfx)]; for(i32 k=1;k<Cpfx;++k) vstart[size_t(k)]+=dv[size_t(k)]; } }
  CK(cudaMemcpy(d_carry,carrypref.data(),dV_n*sizeof(i64),cudaMemcpyHostToDevice));
  GpuSessCombineArgs ca{}; ca.nstates=V; ca.q=qq; ca.Cpfx=Cpfx; ca.qlb=d_qlb; ca.qhb=d_qhb; ca.carrypref=d_carry; ca.bidx=d_bidx; ca.lvals=d_lvals; ca.answers=d_answers;
  const auto tu1=std::chrono::steady_clock::now();
  std::vector<i64> answers(ans_n,0), batch_ns; float kernel_ms=0.f;
  for(i32 it=0; it<batches; ++it){
    const auto b0=std::chrono::steady_clock::now();
    CK(cudaMemcpy(d_devbuf,d_devbuf_bak,devbuf_n*sizeof(i64),cudaMemcpyDeviceToDevice));
    if(gpu_sess_reset_launch(&rea)){std::fprintf(stderr,"reset failed\n");return 5;}
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventRecord(e0));
    if(gpu_sess_run_launch(&ra)){std::fprintf(stderr,"run failed\n");return 5;}
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&kernel_ms,e0,e1));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    if(gpu_sess_combine_launch(&ca)){std::fprintf(stderr,"combine failed\n");return 5;}
    CK(cudaMemcpy(answers.data(),d_answers,ans_n*sizeof(i64),cudaMemcpyDeviceToHost));
    batch_ns.push_back(nsb(b0,std::chrono::steady_clock::now()));
  }
  const auto t3=std::chrono::steady_clock::now();
  FILE* out=std::fopen(answers_path.c_str(),"wb"); if(!out){std::fprintf(stderr,"write failed\n");return 4;}
  std::fwrite(answers.data(),sizeof(i64),answers.size(),out); std::fclose(out);
  const u64 cks=fnv1a(answers.data(),answers.size()*sizeof(i64));
  const std::string sha=answers_sha256(answers.data(),answers.size()*sizeof(i64));
  const auto t4=std::chrono::steady_clock::now();
  CK(cudaFree(gbuf)); CK(cudaFree(d_devbuf)); CK(cudaFree(d_lvals)); CK(cudaFree(d_dV)); CK(cudaFree(d_carry));
  CK(cudaFree(d_answers)); CK(cudaFree(d_fen)); CK(cudaFree(d_leaf)); CK(cudaFree(d_devbuf_bak)); f.release_rows();
  const auto t5=std::chrono::steady_clock::now();
  i64 peak_kb=0; { FILE* s=std::fopen("/proc/self/status","r"); if(s){char line[256]; while(std::fgets(line,sizeof(line),s)){long v=0; if(std::sscanf(line,"VmHWM: %ld kB",&v)==1){peak_kb=v;break;}} std::fclose(s);} }
  const i64 load_ns=nsb(t0,t1), build_ns=nsb(t1,t2), upload_ns=nsb(tu0,tu1);
  const i64 setup_ns=nsb(t0,tu1);
  i64 first=batch_ns.empty()?0:batch_ns[0], steady=0; for(size_t i=1;i<batch_ns.size();++i) steady+=batch_ns[i];
  const i64 steady_avg=batch_ns.size()>1?steady/i64(batch_ns.size()-1):0;
  const i64 extract_ns=nsb(t3,t4), cleanup_ns=nsb(t4,t5);
  const i64 amort1=setup_ns+first+extract_ns+cleanup_ns;
  const i64 amortN=setup_ns+((first+steady)/i64(batch_ns.size()))+extract_ns+cleanup_ns;
  FILE* jf=std::fopen((json_path+".tmp").c_str(),"w"); if(!jf){std::fprintf(stderr,"json failed\n");return 4;}
  std::fprintf(jf,"{\n  \"arm\": \"gpu_session\",\n  \"base_id\": \"%s\",\n  \"topology\": %d,\n  \"seed\": %d,\n  \"factor_period\": %d,\n  \"q\": %d,\n  \"repeat\": %d,\n  \"n_states\": %d,\n  \"batches\": %d,\n  \"range_scope\": \"%s\",\n  \"answers_fn\": \"%016" PRIx64 "\",\n  \"answers_sha256\": \"%s\",\n",
    base_id.c_str(),topology,seed,factor_period,qq,repeat,kStates,batches,per_state?"per_state":"shared",cks,sha.c_str());
  std::fprintf(jf,"  \"phases_ns\": {\"load\": %" PRId64 ", \"setup\": %" PRId64 ", \"execute\": %" PRId64 ", \"extract\": %" PRId64 ", \"cleanup\": %" PRId64 "},\n",
    load_ns,setup_ns,first,extract_ns,cleanup_ns);
  std::fprintf(jf,"  \"session\": {\"blocks\": %d, \"threads\": %d, \"Cpfx\": %d, \"ncls\": %d, \"kernel_ms\": %.3f, \"build_ns\": %" PRId64 ", \"upload_ns\": %" PRId64 ", \"first_batch_ns\": %" PRId64 ", \"steady_batch_ns\": %" PRId64 ", \"n_queries\": %zu, \"steady_per_query_ns\": %.2f, \"amortized_1_ns\": %" PRId64 ", \"amortized_N_ns\": %" PRId64 "},\n",
    B,threads,Cpfx,ncls,double(kernel_ms),build_ns,upload_ns,first,steady_avg,answers.size(),
    answers.empty()?0.0:double(steady_avg)/double(answers.size()),amort1,amortN);
  std::fprintf(jf,"  \"full_ns\": %" PRId64 ",\n  \"peak_rss_kb\": %" PRId64 "\n}\n",amort1,peak_kb);
  std::fclose(jf); std::rename((json_path+".tmp").c_str(),json_path.c_str());
  std::printf("gpu_session base=%s q=%d batches=%d C=%d setup_ms=%.1f first_batch_ms=%.2f steady_batch_ms=%.2f per_query_ns=%.1f kernel_ms=%.2f\n",
    base_id.c_str(),qq,batches,Cpfx,double(setup_ns)/1e6,double(first)/1e6,double(steady_avg)/1e6,
    answers.empty()?0.0:double(steady_avg)/double(answers.size()),double(kernel_ms));
  return 0;
}
