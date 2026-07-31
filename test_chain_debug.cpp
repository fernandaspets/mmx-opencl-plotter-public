#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>
#include <openssl/sha.h>

const int KSIZE=20, LOGBUCKETS=6, LOGBUCKETS2=5, N_META=14;
const uint32_t KMASK=(1u<<KSIZE)-1;
const int TOTAL_SUB=2048, MAX_BS2=2048;

int main() {
    cl_platform_id plat; cl_device_id dev; cl_context ctx; cl_command_queue q; cl_int err;
    clGetPlatformIDs(1,&plat,nullptr); clGetDeviceIDs(plat,CL_DEVICE_TYPE_GPU,1,&dev,nullptr);
    ctx=clCreateContext(nullptr,1,&dev,nullptr,nullptr,&err);
    q=clCreateCommandQueue(ctx,dev,0,&err);
    
    std::ifstream f("f2_f9.cl");
    std::string src((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>());
    const char* cs=src.c_str(); size_t len=src.size();
    cl_program prog=clCreateProgramWithSource(ctx,1,&cs,&len,&err);
    std::string opts="-DKSIZE=20 -DLOGBUCKETS=6 -DLOGBUCKETS2=5 -DN_META=14 -DN_META_OUT=12 -DN_TABLE=9"
                     " -DDSIZE_=5 -DPSIZE_=21 -DPDSIZE=32 -DX2SIZE=39 -DXBITS=20"
                     " -DHYBRID_SORT_LOG_THREADS=6 -DNUM_THREADS=64 -DKMASK=1048575 -DDMASK=31"
                     " -DMETA_BYTES=56 -DMAX_LOCAL_SIZE=40";
    err=clBuildProgram(prog,1,&dev,opts.c_str(),nullptr,nullptr);
    if(err!=CL_SUCCESS){char log[8192];clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,sizeof(log),log,nullptr);std::cerr<<log<<std::endl;return 1;}
    
    const int N=5000;
    std::mt19937 rng(42);
    std::vector<uint32_t> Y_in(N), M_in(N*N_META);
    for(int i=0;i<N;i++){Y_in[i]=rng()&KMASK; for(int j=0;j<N_META;j++) M_in[i*N_META+j]=rng()&KMASK;}
    
    // CPU: find all Y,Y+1 pairs for reference
    std::vector<std::pair<uint32_t,uint32_t>> entries(N);
    for(int i=0;i<N;i++) entries[i]={Y_in[i],(uint32_t)i};
    std::sort(entries.begin(),entries.end());
    int cpu_matches=0;
    for(int i=0;i<N;i++){
        for(int j=i+1;j<N;j++){
            if(entries[j].first==entries[i].first+1) cpu_matches++;
            else if(entries[j].first>entries[i].first+1) break;
        }
    }
    std::cout<<"CPU matches: "<<cpu_matches<<std::endl;
    
    // GPU scatter
    cl_kernel k_scatter=clCreateKernel(prog,"scatter_2",&err);
    cl_mem C_buf=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,N*N_META*4,(void*)M_in.data(),&err);
    cl_mem PY_buf=clCreateBuffer(ctx,CL_MEM_READ_WRITE,TOTAL_SUB*MAX_BS2*8,nullptr,&err);
    cl_mem cnt_buf=clCreateBuffer(ctx,CL_MEM_READ_WRITE,TOTAL_SUB*4,nullptr,&err);
    int zero=0;
    clEnqueueFillBuffer(q,cnt_buf,&zero,4,0,TOTAL_SUB*4,0,nullptr,nullptr);
    cl_mem null_mem=nullptr;
    uint32_t n32=N,maxbs=MAX_BS2;
    clSetKernelArg(k_scatter,0,sizeof(cl_mem),&PY_buf);
    clSetKernelArg(k_scatter,1,sizeof(cl_mem),&cnt_buf);
    clSetKernelArg(k_scatter,2,sizeof(cl_mem),&null_mem);
    clSetKernelArg(k_scatter,3,sizeof(cl_mem),&C_buf);
    clSetKernelArg(k_scatter,4,sizeof(uint32_t),&n32);
    clSetKernelArg(k_scatter,5,sizeof(uint32_t),&maxbs);
    size_t g=N; if(g%64)g=((g/64)+1)*64;
    clEnqueueNDRangeKernel(q,k_scatter,1,nullptr,&g,nullptr,0,nullptr,nullptr);
    
    // Read bucket counts
    std::vector<uint32_t> cnt(TOTAL_SUB);
    clEnqueueReadBuffer(q,cnt_buf,CL_TRUE,0,TOTAL_SUB*4,cnt.data(),0,nullptr,nullptr);
    
    // Read PY values for first few sub-buckets
    std::vector<uint64_t> PY(TOTAL_SUB*MAX_BS2);
    clEnqueueReadBuffer(q,PY_buf,CL_TRUE,0,TOTAL_SUB*MAX_BS2*8,PY.data(),0,nullptr,nullptr);
    
    // Check which sub-buckets have entries
    int nonempty=0;
    for(int i=0;i<TOTAL_SUB;i++) if(cnt[i]>0) nonempty++;
    std::cout<<"Non-empty sub-buckets: "<<nonempty<<std::endl;
    
    // Find a sub-bucket with entries and check if Y values are in range
    for(int s=0;s<TOTAL_SUB;s++){
        if(cnt[s]>0){
            std::cout<<"\nSub-bucket "<<s<<" (count="<<cnt[s]<<"):"<<std::endl;
            std::cout<<"  Expected Y range: ["<<(s<<(KSIZE-LOGBUCKETS-LOGBUCKETS2))
                     <<", "<<((s+1)<<(KSIZE-LOGBUCKETS-LOGBUCKETS2))-1<<"]"<<std::endl;
            // Print first 5 Y values
            for(int i=0;i<std::min(5,(int)cnt[s]);i++){
                uint32_t Y=PY[s*MAX_BS2+i]>>(64-KSIZE);
                std::cout<<"  Y["<<i<<"] = "<<Y<<std::endl;
            }
            // Check if sorted
            bool sorted=true;
            for(int i=1;i<cnt[s];i++){
                if(PY[s*MAX_BS2+i-1]>PY[s*MAX_BS2+i]){sorted=false;break;}
            }
            std::cout<<"  Sorted? "<<(sorted?"YES":"NO")<<std::endl;
            break;
        }
    }
    
    // GPU sort
    cl_kernel k_sort=clCreateKernel(prog,"hybrid_sort_y",&err);
    uint32_t maxbs_sort=MAX_BS2, num_sub=TOTAL_SUB;
    clSetKernelArg(k_sort,0,sizeof(cl_mem),&PY_buf);
    clSetKernelArg(k_sort,1,sizeof(cl_mem),&cnt_buf);
    clSetKernelArg(k_sort,2,sizeof(uint32_t),&maxbs_sort);
    clSetKernelArg(k_sort,3,sizeof(uint32_t),&num_sub);
    size_t sg[2]={64,(size_t)TOTAL_SUB}, sl[2]={64,1};
    clEnqueueNDRangeKernel(q,k_sort,2,nullptr,sg,sl,0,nullptr,nullptr);
    
    // Read sorted PY
    clEnqueueReadBuffer(q,PY_buf,CL_TRUE,0,TOTAL_SUB*MAX_BS2*8,PY.data(),0,nullptr,nullptr);
    
    // Check sorted again
    int total_yy1_intra=0, total_yy1_inter=0;
    for(int s=0;s<TOTAL_SUB;s++){
        if(cnt[s]==0) continue;
        // Check sorted
        bool sorted=true;
        for(int i=1;i<cnt[s];i++){
            if(PY[s*MAX_BS2+i-1]>PY[s*MAX_BS2+i]){sorted=false;break;}
        }
        if(!sorted) std::cout<<"Sub-bucket "<<s<<" NOT sorted after GPU sort!"<<std::endl;
        
        // Count Y,Y+1 within same sub-bucket
        for(int i=0;i<cnt[s];i++){
            uint32_t YL=PY[s*MAX_BS2+i]>>(64-KSIZE);
            for(int j=i+1;j<cnt[s];j++){
                uint32_t YR=PY[s*MAX_BS2+j]>>(64-KSIZE);
                if(YR==YL+1) total_yy1_intra++;
                else if(YR>YL+1) break;
            }
        }
        // Count Y,Y+1 across to next sub-bucket
        if(s+1<TOTAL_SUB && cnt[s+1]>0){
            for(int i=0;i<cnt[s];i++){
                uint32_t YL=PY[s*MAX_BS2+i]>>(64-KSIZE);
                for(int j=0;j<cnt[s+1];j++){
                    uint32_t YR=PY[(s+1)*MAX_BS2+j]>>(64-KSIZE);
                    if(YR==YL+1) total_yy1_inter++;
                    else if(YR>YL+1) break;
                }
            }
        }
    }
    std::cout<<"\nY,Y+1 pairs: intra="<<total_yy1_intra<<" inter="<<total_yy1_inter<<" total="<<total_yy1_intra+total_yy1_inter<<std::endl;
    std::cout<<"CPU total: "<<cpu_matches<<std::endl;
    
    clReleaseMemObject(C_buf);clReleaseMemObject(PY_buf);clReleaseMemObject(cnt_buf);
    return 0;
}
