#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <openssl/sha.h>

const int KSIZE=20,LOGBUCKETS=6,LOGBUCKETS2=5,N_META=14;
const uint32_t KMASK=(1u<<KSIZE)-1;
const int TOTAL_SUB=2048,MAX_BS2=2048;

int main() {
    cl_platform_id p;cl_device_id d;cl_context ctx;cl_command_queue q;cl_int err;
    clGetPlatformIDs(1,&p,nullptr);clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,nullptr);
    ctx=clCreateContext(nullptr,1,&d,nullptr,nullptr,&err);
    q=clCreateCommandQueue(ctx,d,0,&err);
    
    std::ifstream f1("f2_f9.cl");std::string s1((std::istreambuf_iterator<char>(f1)),std::istreambuf_iterator<char>());
    std::ifstream f2("simple_sort.cl");std::string s2((std::istreambuf_iterator<char>(f2)),std::istreambuf_iterator<char>());
    std::string combined=s1+"\n"+s2;const char* cs=combined.c_str();size_t len=combined.size();
    cl_program prog=clCreateProgramWithSource(ctx,1,&cs,&len,&err);
    std::string opts="-DKSIZE=20 -DLOGBUCKETS=6 -DLOGBUCKETS2=5 -DN_META=14 -DN_META_OUT=12 -DN_TABLE=9"
                     " -DDSIZE_=5 -DPSIZE_=21 -DPDSIZE=32 -DX2SIZE=39 -DXBITS=20"
                     " -DHYBRID_SORT_LOG_THREADS=6 -DNUM_THREADS=64 -DKMASK=1048575 -DDMASK=31"
                     " -DMETA_BYTES=56 -DMAX_LOCAL_SIZE=40";
    err=clBuildProgram(prog,1,&d,opts.c_str(),nullptr,nullptr);
    if(err!=CL_SUCCESS){char log[8192];clGetProgramBuildInfo(prog,d,CL_PROGRAM_BUILD_LOG,sizeof(log),log,nullptr);std::cerr<<log<<std::endl;return 1;}
    
    const int N=5000;
    std::mt19937 rng(42);
    std::vector<uint32_t> Y_in(N),M_in(N*N_META);
    for(int i=0;i<N;i++){Y_in[i]=rng()&KMASK;for(int j=0;j<N_META;j++)M_in[i*N_META+j]=rng()&KMASK;}
    
    // CPU: sort by Y, find Y,Y+1 pairs
    std::vector<std::pair<uint32_t,uint32_t>> entries(N);
    for(int i=0;i<N;i++){uint32_t Y=0;for(int j=0;j<N_META;j++)Y^=M_in[i*N_META+j];Y&=KMASK;entries[i]={Y,(uint32_t)i};}
    
    std::sort(entries.begin(),entries.end());
    
    // Count CPU matches per sub-bucket pair
    std::map<int,int> cpu_matches_per_sub;
    int cpu_total=0;
    for(int i=0;i<N;i++){
        uint32_t YL=entries[i].first;
        int subL=YL>>9;
        for(int j=i+1;j<N;j++){
            uint32_t YR=entries[j].first;
            if(YR==YL+1){
                int subR=YR>>9;
                cpu_matches_per_sub[subL]++;
                cpu_total++;
            } else if(YR>YL+1) break;
        }
    }
    std::cout<<"CPU total: "<<cpu_total<<std::endl;
    
    // GPU: scatter + sort
    cl_kernel k_scatter=clCreateKernel(prog,"scatter_2",&err);
    cl_kernel k_sort=clCreateKernel(prog,"simple_sort_y",&err);
    
    cl_mem C_buf=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,N*N_META*4,(void*)M_in.data(),&err);
    cl_mem PY_buf=clCreateBuffer(ctx,CL_MEM_READ_WRITE,TOTAL_SUB*MAX_BS2*8,nullptr,&err);
    cl_mem cnt_buf=clCreateBuffer(ctx,CL_MEM_READ_WRITE,TOTAL_SUB*4,nullptr,&err);
    int zero=0;clEnqueueFillBuffer(q,cnt_buf,&zero,4,0,TOTAL_SUB*4,0,nullptr,nullptr);
    cl_mem null_mem=nullptr;
    uint32_t n32=N,maxbs=MAX_BS2;
    clSetKernelArg(k_scatter,0,sizeof(cl_mem),&PY_buf);
    clSetKernelArg(k_scatter,1,sizeof(cl_mem),&cnt_buf);
    clSetKernelArg(k_scatter,2,sizeof(cl_mem),&null_mem);
    clSetKernelArg(k_scatter,3,sizeof(cl_mem),&C_buf);
    clSetKernelArg(k_scatter,4,sizeof(uint32_t),&n32);
    clSetKernelArg(k_scatter,5,sizeof(uint32_t),&maxbs);
    size_t sg=N;if(sg%64)sg=((sg/64)+1)*64;
    clEnqueueNDRangeKernel(q,k_scatter,1,nullptr,&sg,nullptr,0,nullptr,nullptr);
    
    std::vector<uint32_t> cnt(TOTAL_SUB);
    clEnqueueReadBuffer(q,cnt_buf,CL_TRUE,0,TOTAL_SUB*4,cnt.data(),0,nullptr,nullptr);
    
    uint32_t maxbs_sort=MAX_BS2,num_sub=TOTAL_SUB;
    clSetKernelArg(k_sort,0,sizeof(cl_mem),&PY_buf);
    clSetKernelArg(k_sort,1,sizeof(cl_mem),&cnt_buf);
    clSetKernelArg(k_sort,2,sizeof(uint32_t),&maxbs_sort);
    clSetKernelArg(k_sort,3,sizeof(uint32_t),&num_sub);
    size_t sortg[2]={256,(size_t)TOTAL_SUB},sortl[2]={256,1};
    clEnqueueNDRangeKernel(q,k_sort,2,nullptr,sortg,sortl,0,nullptr,nullptr);
    
    // Read sorted PY
    std::vector<uint64_t> PY(TOTAL_SUB*MAX_BS2);
    clEnqueueReadBuffer(q,PY_buf,CL_TRUE,0,TOTAL_SUB*MAX_BS2*8,PY.data(),0,nullptr,nullptr);
    
    // Count Y,Y+1 pairs in GPU sorted data
    int gpu_intra=0,gpu_inter=0;
    std::map<int,int> gpu_matches_per_sub;
    for(int s=0;s<TOTAL_SUB;s++){
        if(cnt[s]==0)continue;
        // Intra sub-bucket
        for(int i=0;i<cnt[s];i++){
            uint32_t YL=PY[s*MAX_BS2+i]>>(64-KSIZE);
            for(int j=i+1;j<cnt[s];j++){
                uint32_t YR=PY[s*MAX_BS2+j]>>(64-KSIZE);
                if(YR==YL+1){gpu_intra++;gpu_matches_per_sub[s]++;}
                else if(YR>YL+1)break;
            }
        }
        // Inter sub-bucket (next)
        if(s+1<TOTAL_SUB&&cnt[s+1]>0){
            for(int i=0;i<cnt[s];i++){
                uint32_t YL=PY[s*MAX_BS2+i]>>(64-KSIZE);
                for(int j=0;j<cnt[s+1];j++){
                    uint32_t YR=PY[(s+1)*MAX_BS2+j]>>(64-KSIZE);
                    if(YR==YL+1){gpu_inter++;gpu_matches_per_sub[s]++;}
                    else if(YR>YL+1)break;
                }
            }
        }
    }
    std::cout<<"GPU intra="<<gpu_intra<<" inter="<<gpu_inter<<" total="<<gpu_intra+gpu_inter<<std::endl;
    
    // Compare per sub-bucket
    int lost=0;
    for(auto&[sub,cpu_count]:cpu_matches_per_sub){
        int gpu_count=gpu_matches_per_sub[sub];
        if(cpu_count!=gpu_count){
            std::cout<<"Sub "<<sub<<": CPU="<<cpu_count<<" GPU="<<gpu_count<<std::endl;
            lost+=cpu_count-gpu_count;
        }
    }
    std::cout<<"Lost matches: "<<lost<<std::endl;
    
    clReleaseMemObject(C_buf);clReleaseMemObject(PY_buf);clReleaseMemObject(cnt_buf);
    return 0;
}
