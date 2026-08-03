// GPU L1-bucket pipeline v2 — GLOBAL EVAL (working on NVIDIA where no AMD driver bug)
// Phase 1: scatter+sort+match ALL buckets batched (no per-bucket sync)
// Phase 2: ONE global eval using aCI + global LR indices
// scatter_2 reads from aCI at (x + bucket_offset) using GLOBAL access

#include "plot_pipeline.h"
#include "plot_config.h"
#include <fstream>
#include <algorithm>

namespace mmx {

static bool load_f2f9(GPUDevice& g,int ksize,
    cl_kernel& k2,cl_kernel& ko,cl_kernel& ks,cl_kernel& km,cl_kernel& ke)
{
    std::string kd;
    for(auto d:{"kernels","../kernels","../../kernels"})
        if(std::ifstream(std::string(d)+"/f2_f9.cl").good()){kd=d;break;}
    if(kd.empty())return 0;
    int l2=std::max(0,ksize-17);
    auto o="-cl-std=CL1.2 -DKSIZE="+std::to_string(ksize)
        +" -DLOGBUCKETS=8 -DLOGBUCKETS2="+std::to_string(l2)
        +" -DN_META="+std::to_string(N_META)+" -DN_META_OUT=12 -DN_TABLE=9"
        +" -DDSIZE_=5 -DPSIZE_="+std::to_string(ksize+1)
        +" -DPDSIZE="+std::to_string(5+ksize)
        +" -DX2SIZE="+std::to_string(2*ksize-1)+" -DXBITS="+std::to_string(ksize)
        +" -DHYBRID_SORT_LOG_THREADS=6 -DNUM_THREADS=64"
        +" -DKMASK="+std::to_string((1u<<ksize)-1)
        +" -DDMASK=31 -DMETA_BYTES="+std::to_string(N_META*4)+" -DMAX_LOCAL_SIZE=40";
    std::string src;
    for(auto f:{"/f2_f9.cl","/simple_sort.cl","/prefix_sum.cl"}){
        std::ifstream h(kd+f);if(!h.good())return 0;
        src+=std::string(std::istreambuf_iterator<char>(h),{})+"\n";
    }
    try{
        g.load_program("f2f9",src,o);
        k2=g.get_kernel("scatter_2");ko=g.get_kernel("calc_offset_sum");
        ks=g.get_kernel("simple_sort_y");km=g.get_kernel("match_p1");
        ke=g.get_kernel("eval_p1_tx");
        return k2&&ko&&ks&&km&&ke;
    }catch(...){return 0;}
}

bool run_gpu_l1_pipeline(GPUDevice& g,int ksize,uint32_t n,
    const std::vector<uint32_t>& Mf,const std::vector<uint32_t>& Xf,
    std::vector<PlotEntry>& entries,std::vector<uint32_t>& fy,
    std::vector<std::vector<PDEntry>>& pda,std::vector<uint32_t>& xp)
{
    cl_kernel k2=0,ko=0,ks=0,km=0,ke=0;
    if(!load_f2f9(g,ksize,k2,ko,ks,km,ke))return 0;
    const uint32_t KM=(1u<<ksize)-1,MB=std::max(1024u,(n/256)*3+256);
    const int NL=256,NS=std::max(1,1<<(ksize-17));
    const uint32_t M2=std::max(1024u,(uint32_t)(n/(NL*NS)*3+256));
    const size_t SZ=(size_t)NL*MB*N_META*4;
    size_t VA=0;clGetDeviceInfo(g.device,CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(VA),&VA,0);
    if(2*SZ+(size_t)NS*M2*8+(size_t)n*12>VA*9/10)return 0;
    cl_int e;int z=0;
    std::vector<cl_mem> B;
    auto A=[&](size_t s){cl_mem m=clCreateBuffer(g.context,CL_MEM_READ_WRITE,s,0,&e);
        GPUDevice::check(e);B.push_back(m);return m;};
    cl_mem CI=A(SZ),CO=A(SZ);
    cl_mem PY=A((size_t)NS*M2*8),SC=A((size_t)NS*4),SO=A((size_t)(NS+1)*4);
    cl_mem LR=A((size_t)n*8),PD=A((size_t)n*4);
    cl_mem NM=A(4),NB=A((size_t)NL*4),BC=A((size_t)NL*4),XB=A((size_t)MB*4);
    auto t0=std::chrono::high_resolution_clock::now();

    // Upload F1 into L1 buckets
    int sh=ksize-8;
    std::vector<uint32_t> f1c(NL,0);
    std::vector<std::vector<uint32_t>> l1m(NL),l1x(NL);
    for(size_t i=0;i<n;i++){
        uint32_t Y=0;
        for(int j=0;j<N_META;j++)Y^=Mf[i*N_META+j];
        Y&=KM;uint32_t b=Y>>sh;if(b>=256)b=255;
        f1c[b]++;for(int j=0;j<N_META;j++)l1m[b].push_back(Mf[i*N_META+j]);
        l1x[b].push_back(Xf.empty()?(uint32_t)i:Xf[i]);
    }
    for(int y=0;y<NL;y++){
        if(!f1c[y])continue;
        clEnqueueWriteBuffer(g.queue,CI,CL_FALSE,(size_t)y*MB*N_META*4,
            f1c[y]*N_META*4,l1m[y].data(),0,0,0);
        clEnqueueWriteBuffer(g.queue,XB,CL_FALSE,(size_t)y*MB*4,
            f1c[y]*4,l1x[y].data(),0,0,0);
    }
    clEnqueueWriteBuffer(g.queue,BC,CL_TRUE,0,NL*4,f1c.data(),0,0,0);
    g.finish();
    std::cout<<"[L1v2-NV] Upload: "
        <<std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count()<<"s\n";

    cl_mem aCI=CI,aCO=CO,aBC=BC;
    std::vector<std::vector<uint32_t>> lrT(10),pdT(10);

    for(int t=2;t<=9;t++){
        auto tt0=std::chrono::high_resolution_clock::now();
        clEnqueueFillBuffer(g.queue,NB,&z,4,0,NL*4,0,0,0);
        clEnqueueFillBuffer(g.queue,NM,&z,4,0,4,0,0,0);
        g.finish();

        std::vector<uint32_t> cnt(NL);
        clEnqueueReadBuffer(g.queue,aBC,CL_TRUE,0,NL*4,cnt.data(),0,0,0);

        // Phase 1: scatter+sort+match ALL buckets (batched, no per-bucket sync)
        for(int y=0;y<NL;y++){
            if(!cnt[y])continue;
            uint32_t cy=cnt[y],bo=y*MB;
            clEnqueueFillBuffer(g.queue,SC,&z,4,0,NS*4,0,0,0);
            uint32_t cyu=cy,m2=M2;
            // scatter_2 reads from aCI at (x + bucket_offset) * N_META + i
            clSetKernelArg(k2,0,8,&PY);clSetKernelArg(k2,1,8,&SC);
            clSetKernelArg(k2,2,8,0);clSetKernelArg(k2,3,8,&aCI);
            clSetKernelArg(k2,4,4,&cyu);clSetKernelArg(k2,5,4,&m2);
            clSetKernelArg(k2,6,4,&bo);
            size_t gs=((cy+63)/64)*64;
            clEnqueueNDRangeKernel(g.queue,k2,1,0,&gs,0,0,0,0);
            uint32_t nsu=(uint32_t)NS,wt=1;
            clSetKernelArg(ko,0,8,&SO);clSetKernelArg(ko,1,8,&SC);
            clSetKernelArg(ko,2,4,&nsu);clSetKernelArg(ko,3,4,&wt);
            size_t pl=std::min((size_t)NS,(size_t)1024);
            clEnqueueNDRangeKernel(g.queue,ko,1,0,&pl,&pl,0,0,0);
            clSetKernelArg(ks,0,8,&PY);clSetKernelArg(ks,1,8,&SC);
            clSetKernelArg(ks,2,4,&m2);clSetKernelArg(ks,3,4,&nsu);
            size_t sg[2]={256,(size_t)NS},sl[2]={256,1};
            clEnqueueNDRangeKernel(g.queue,ks,2,0,sg,sl,0,0,0);
            uint32_t mt=(uint32_t)n*4,wp=(t>=3)?1:0;
            clSetKernelArg(km,0,8,&LR);clSetKernelArg(km,1,8,&PD);
            clSetKernelArg(km,2,8,&NM);clSetKernelArg(km,3,8,&PY);
            clSetKernelArg(km,4,8,&SC);clSetKernelArg(km,5,8,&SO);
            clSetKernelArg(km,6,4,&nsu);clSetKernelArg(km,7,4,&m2);
            clSetKernelArg(km,8,4,&mt);clSetKernelArg(km,9,4,&wp);
            int gp=(M2+127)/128;
            size_t mg[2]={(size_t)(128*gp),(size_t)NS},ml[2]={128,1};
            clEnqueueNDRangeKernel(g.queue,km,2,0,mg,ml,0,0,0);
        }
        g.finish(); // Wait for Phase 1

        // Phase 2: ONE global eval using aCI + global LR indices
        uint32_t nmt=0;
        clEnqueueReadBuffer(g.queue,NM,CL_TRUE,0,4,&nmt,0,0,0);

        if(nmt>0){
            uint32_t mx=MB,tu=(uint32_t)t,wy=0,wc=1,hp=(t>=3),hx=(t==2);
            uint64_t pdo=0;uint32_t x2s=2*ksize-1;cl_mem n2=0;
            clSetKernelArg(ke,0,8,&n2);clSetKernelArg(ke,1,8,&aCO);
            clSetKernelArg(ke,2,8,&n2);clSetKernelArg(ke,3,8,&NB);
            clSetKernelArg(ke,4,8,&aCI); // GLOBAL C_in
            clSetKernelArg(ke,5,8,&n2);
            clSetKernelArg(ke,6,8,hx?&XB:&n2);
            clSetKernelArg(ke,7,8,&LR);clSetKernelArg(ke,8,8,&NM);
            clSetKernelArg(ke,9,8,&pdo);clSetKernelArg(ke,10,4,&mx);
            clSetKernelArg(ke,11,4,&x2s);clSetKernelArg(ke,12,4,&ksize);
            clSetKernelArg(ke,13,4,&tu);clSetKernelArg(ke,14,4,&wy);
            clSetKernelArg(ke,15,4,&wc);clSetKernelArg(ke,16,4,&hp);
            clSetKernelArg(ke,17,4,&hx);
            size_t hg=((nmt+63)/64)*64;
            clEnqueueNDRangeKernel(g.queue,ke,1,0,&hg,0,0,0,0);
            g.finish();
        }

        lrT[t].resize(nmt*2);pdT[t].resize(nmt);
        if(nmt>0){
            clEnqueueReadBuffer(g.queue,LR,CL_TRUE,0,nmt*8,lrT[t].data(),0,0,0);
            clEnqueueReadBuffer(g.queue,PD,CL_TRUE,0,nmt*4,pdT[t].data(),0,0,0);
        }
        std::swap(aCI,aCO);std::swap(aBC,NB);
        double te=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-tt0).count();
        std::cout<<"[T"<<t<<"] "<<nmt<<" matches ("<<te<<"s)\n";
    }

    // Download final results
    std::vector<uint32_t> fc(NL);
    clEnqueueReadBuffer(g.queue,aBC,CL_TRUE,0,NL*4,fc.data(),0,0,0);
    uint32_t tf=0;for(auto c:fc)tf+=c;
    entries.resize(tf);fy.resize(tf);
    size_t pos=0;
    for(int y=0;y<NL;y++){
        if(!fc[y])continue;
        uint32_t cnt_=fc[y];
        std::vector<uint32_t> meta(cnt_*N_META);
        size_t o=(size_t)y*MB*N_META*4;
        clEnqueueReadBuffer(g.queue,aCI,CL_TRUE,o,cnt_*N_META*4,meta.data(),0,0,0);
        for(uint32_t i=0;i<cnt_;i++,pos++){
            uint32_t Y=0;
            for(int j=0;j<N_META;j++){
                entries[pos].M[j]=meta[i*N_META+j];
                Y^=meta[i*N_META+j];
            }
            entries[pos].Y=Y&KM;entries[pos].orig_idx=(uint32_t)pos;
            fy[pos]=entries[pos].Y;
        }
    }
    pda.assign(10,{});
    for(int t_=3;t_<=9;t_++){
        pda[t_].resize(pdT[t_].size());
        for(size_t i=0;i<pdT[t_].size();i++)
            pda[t_][i]={pdT[t_][i]>>5,(uint16_t)(pdT[t_][i]&0x1F)};
    }
    xp.clear();
    for(size_t i=0;i<lrT[2].size();i+=2){xp.push_back(lrT[2][i]);xp.push_back(lrT[2][i+1]);}
    for(auto m:B)clReleaseMemObject(m);
    std::cout<<"[L1v2-NV] "<<std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now()-t0).count()<<"s\n";
    return 1;
}
} // namespace mmx
