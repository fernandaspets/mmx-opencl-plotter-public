#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

int main() {
    cl_platform_id p; cl_device_id d; cl_context ctx; cl_command_queue q; cl_int err;
    clGetPlatformIDs(1,&p,nullptr); clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,nullptr);
    ctx=clCreateContext(nullptr,1,&d,nullptr,nullptr,&err);
    q=clCreateCommandQueue(ctx,d,0,&err);
    
    // Build both kernels (f2_f9 for other kernels, simple_sort for sorting)
    std::ifstream f1("f2_f9.cl");
    std::string s1((std::istreambuf_iterator<char>(f1)),std::istreambuf_iterator<char>());
    std::ifstream f2("simple_sort.cl");
    std::string s2((std::istreambuf_iterator<char>(f2)),std::istreambuf_iterator<char>());
    std::string combined = s1 + "\n" + s2;
    const char* cs = combined.c_str();
    size_t len = combined.size();
    
    cl_program prog=clCreateProgramWithSource(ctx,1,&cs,&len,&err);
    std::string opts="-DKSIZE=20 -DLOGBUCKETS=6 -DLOGBUCKETS2=5 -DN_META=14 -DN_META_OUT=12 -DN_TABLE=9"
                     " -DDSIZE_=5 -DPSIZE_=21 -DPDSIZE=32 -DX2SIZE=39 -DXBITS=20"
                     " -DHYBRID_SORT_LOG_THREADS=6 -DNUM_THREADS=64 -DKMASK=1048575 -DDMASK=31"
                     " -DMETA_BYTES=56 -DMAX_LOCAL_SIZE=40";
    err=clBuildProgram(prog,1,&d,opts.c_str(),nullptr,nullptr);
    if(err!=CL_SUCCESS){char log[8192];clGetProgramBuildInfo(prog,d,CL_PROGRAM_BUILD_LOG,sizeof(log),log,nullptr);std::cerr<<log<<std::endl;return 1;}
    
    const int TOTAL_SUB=2048, MAX_BS=2048, N=5000;
    std::mt19937 rng(42);
    
    // Generate random Y values and scatter into sub-buckets
    std::vector<uint64_t> PY(TOTAL_SUB*MAX_BS, 0);
    std::vector<uint32_t> cnt(TOTAL_SUB, 0);
    for(int i=0;i<N;i++){
        uint32_t Y=rng()&((1u<<20)-1);
        uint32_t sub=Y>>9;
        uint32_t pos=cnt[sub]++;
        if(pos<MAX_BS) PY[sub*MAX_BS+pos]=((uint64_t)Y<<44)|(uint64_t)i;
    }
    
    // CPU reference sort
    std::vector<uint64_t> PY_ref=PY;
    int nonempty=0;
    for(int s=0;s<TOTAL_SUB;s++){
        if(cnt[s]>0) nonempty++;
        std::vector<uint64_t> tmp(PY_ref.begin()+s*MAX_BS, PY_ref.begin()+s*MAX_BS+cnt[s]);
        std::sort(tmp.begin(),tmp.end());
        for(int i=0;i<(int)cnt[s];i++) PY_ref[s*MAX_BS+i]=tmp[i];
    }
    std::cout<<"Non-empty sub-buckets: "<<nonempty<<", max bucket size: "<<*std::max_element(cnt.begin(),cnt.end())<<std::endl;
    
    // GPU simple sort
    cl_kernel k=clCreateKernel(prog,"simple_sort_y",&err);
    cl_mem PY_buf=clCreateBuffer(ctx,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,PY.size()*8,(void*)PY.data(),&err);
    cl_mem cnt_buf=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,cnt.size()*4,(void*)cnt.data(),&err);
    
    uint32_t maxbs=MAX_BS, num_sub=TOTAL_SUB;
    clSetKernelArg(k,0,sizeof(cl_mem),&PY_buf);
    clSetKernelArg(k,1,sizeof(cl_mem),&cnt_buf);
    clSetKernelArg(k,2,sizeof(uint32_t),&maxbs);
    clSetKernelArg(k,3,sizeof(uint32_t),&num_sub);
    
    // 256 threads per group, one group per sub-bucket
    size_t g[2]={256,(size_t)TOTAL_SUB}, l[2]={256,1};
    clEnqueueNDRangeKernel(q,k,2,nullptr,g,l,0,nullptr,nullptr);
    
    std::vector<uint64_t> PY_gpu(TOTAL_SUB*MAX_BS,0);
    clEnqueueReadBuffer(q,PY_buf,CL_TRUE,0,PY_gpu.size()*8,PY_gpu.data(),0,nullptr,nullptr);
    
    // Compare
    int mismatches=0;
    for(int s=0;s<TOTAL_SUB;s++){
        for(int i=0;i<cnt[s];i++){
            if(PY_ref[s*MAX_BS+i]!=PY_gpu[s*MAX_BS+i]){
                if(mismatches<5) std::cout<<"Mismatch s="<<s<<" i="<<i<<" CPU="<<PY_ref[s*MAX_BS+i]<<" GPU="<<PY_gpu[s*MAX_BS+i]<<std::endl;
                mismatches++;
            }
        }
    }
    std::cout<<(mismatches==0?"✅ simple_sort_y PASSED":"❌ simple_sort_y FAILED: "+std::to_string(mismatches)+" mismatches")<<std::endl;
    
    clReleaseMemObject(PY_buf);clReleaseMemObject(cnt_buf);clReleaseKernel(k);
    return 0;
}
