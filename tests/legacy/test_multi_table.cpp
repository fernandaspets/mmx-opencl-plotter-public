// test_multi_table.cpp — Chain all 8 tables (F2-F9) with GPU hashing
// CPU: sort + match. GPU: SHA-512 hash (eval_p1_tx).
// Compare final output with pure CPU reference.
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>
#include <openssl/sha.h>
#include <array>
#include <chrono>

const int KSIZE=20,LOGBUCKETS=6,N_META=14,N_TABLE=9;
const uint32_t KMASK=(1u<<KSIZE)-1;
const int DSIZE_=5,PSIZE_=KSIZE+1,PDBYTES=(PSIZE_+DSIZE_+7)/8,PDSIZE=PDBYTES*8;
const int XBITS=KSIZE,X2SIZE=2*XBITS-1;

cl_platform_id platform;cl_device_id device;cl_context context;
cl_command_queue queue;cl_program prog;cl_int err;
cl_kernel k_eval;

void cl_check(cl_int e,const char*m){if(e!=CL_SUCCESS){std::cerr<<"CL error: "<<m<<" code="<<e<<std::endl;exit(1);}}

void init_opencl(){
    clGetPlatformIDs(1,&platform,nullptr);
    clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,1,&device,nullptr);
    context=clCreateContext(nullptr,1,&device,nullptr,nullptr,&err);cl_check(err,"ctx");
    queue=clCreateCommandQueue(context,device,0,&err);cl_check(err,"queue");
    
    std::ifstream f("f2_f9.cl");
    std::string src((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>());
    const char* cs=src.c_str();size_t len=src.size();
    prog=clCreateProgramWithSource(context,1,&cs,&len,&err);
    std::string opts="-DKSIZE=20 -DLOGBUCKETS=6 -DLOGBUCKETS2=5 -DN_META=14 -DN_META_OUT=12 -DN_TABLE=9"
                     " -DDSIZE_=5 -DPSIZE_=21 -DPDSIZE=32 -DX2SIZE=39 -DXBITS=20"
                     " -DHYBRID_SORT_LOG_THREADS=6 -DNUM_THREADS=64 -DKMASK=1048575 -DDMASK=31"
                     " -DMETA_BYTES=56 -DMAX_LOCAL_SIZE=40";
    err=clBuildProgram(prog,1,&device,opts.c_str(),nullptr,nullptr);
    if(err!=CL_SUCCESS){char log[8192];clGetProgramBuildInfo(prog,device,CL_PROGRAM_BUILD_LOG,sizeof(log),log,nullptr);std::cerr<<log<<std::endl;exit(1);}
    k_eval=clCreateKernel(prog,"eval_p1_tx",&err);
    std::cout<<"[OCL] eval_p1_tx loaded"<<std::endl;
}

// CPU: compute Y from metadata
uint32_t compute_Y(const std::vector<uint32_t>& M,size_t idx){
    uint32_t Y=0;for(int j=0;j<N_META;j++)Y^=M[idx*N_META+j];return Y&KMASK;
}

// CPU: sort + match, returns LR pairs
void cpu_sort_match(const std::vector<uint32_t>& Y,const std::vector<uint32_t>& M,
                    std::vector<std::pair<uint32_t,uint32_t>>& LR){
    size_t n=Y.size();
    std::vector<std::pair<uint32_t,uint32_t>> entries(n);
    for(size_t i=0;i<n;i++)entries[i]={Y[i],(uint32_t)i};
    std::sort(entries.begin(),entries.end());
    
    // Tie-break by metadata (same as CPU pipeline)
    auto sort_func=[&M](const auto&L,const auto&R){
        if(L.first==R.first){
            for(int j=0;j<N_META;j++){
                uint32_t ml=M[L.second*N_META+j],mr=M[R.second*N_META+j];
                if(ml!=mr)return ml<mr;
            }
        }
        return L<R;
    };
    std::sort(entries.begin(),entries.end(),sort_func);
    
    for(size_t i=0;i<n;i++){
        uint32_t YL=entries[i].first;
        for(size_t j=i+1;j<n;j++){
            uint32_t YR=entries[j].first;
            if(YR==YL+1)LR.push_back({entries[i].second,entries[j].second});
            else if(YR>YL+1)break;
        }
    }
}

// GPU: hash LR pairs, produce new Y and M
void gpu_hash_table(const std::vector<uint32_t>& M_in,
                    const std::vector<std::pair<uint32_t,uint32_t>>& LR,
                    const uint32_t* PD_in,int table,
                    std::vector<uint32_t>& Y_out,std::vector<uint32_t>& M_out){
    size_t num=LR.size();
    if(num==0){Y_out.clear();M_out.clear();return;}
    
    // Prepare LR as flat array
    std::vector<uint32_t> LR_flat(num*2);
    for(size_t i=0;i<num;i++){LR_flat[i*2]=LR[i].first;LR_flat[i*2+1]=LR[i].second;}
    
    cl_mem null_mem=nullptr;
    int zero=0;
    int num_out_buckets=1<<LOGBUCKETS;
    int max_bs_out=16384;  // generous
    
    cl_mem C_in_buf=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        M_in.size()*4,(void*)M_in.data(),&err);cl_check(err,"C_in");
    cl_mem LR_buf=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        LR_flat.size()*4,(void*)LR_flat.data(),&err);cl_check(err,"LR");
    uint32_t num32=num;
    cl_mem num_buf=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        4,(void*)&num32,&err);cl_check(err,"num");
    
    cl_mem Y_buf=clCreateBuffer(context,CL_MEM_WRITE_ONLY,num_out_buckets*max_bs_out*4,nullptr,&err);
    cl_mem C_out_buf=clCreateBuffer(context,CL_MEM_WRITE_ONLY,num_out_buckets*max_bs_out*N_META*4,nullptr,&err);
    cl_mem cnt_buf=clCreateBuffer(context,CL_MEM_READ_WRITE,num_out_buckets*4,nullptr,&err);
    cl_mem PD_out_buf=clCreateBuffer(context,CL_MEM_WRITE_ONLY,num_out_buckets*max_bs_out*PDSIZE,nullptr,&err);
    cl_mem PD_in_buf=nullptr;
    if(PD_in){
        PD_in_buf=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
            num*4,(void*)PD_in,&err);cl_check(err,"PD_in");
    }
    clEnqueueFillBuffer(queue,cnt_buf,&zero,4,0,num_out_buckets*4,0,nullptr,nullptr);
    clEnqueueFillBuffer(queue,PD_out_buf,&zero,4,0,num_out_buckets*max_bs_out*PDSIZE,0,nullptr,nullptr);
    
    ulong PD_0=0;uint max_bs=max_bs_out,x2size=X2SIZE,xbits=XBITS;
    uint write_y=(table==N_TABLE)?1:0;
    uint write_c=1;
    uint has_pd_in=PD_in?1:0;
    uint has_x_in=0;
    
    clSetKernelArg(k_eval,0,sizeof(cl_mem),&Y_buf);
    clSetKernelArg(k_eval,1,sizeof(cl_mem),&C_out_buf);
    clSetKernelArg(k_eval,2,sizeof(cl_mem),&PD_out_buf);
    clSetKernelArg(k_eval,3,sizeof(cl_mem),&cnt_buf);
    clSetKernelArg(k_eval,4,sizeof(cl_mem),&C_in_buf);
    clSetKernelArg(k_eval,5,sizeof(cl_mem),PD_in?&PD_in_buf:&null_mem);
    clSetKernelArg(k_eval,6,sizeof(cl_mem),&null_mem);  // X_in
    clSetKernelArg(k_eval,7,sizeof(cl_mem),&LR_buf);
    clSetKernelArg(k_eval,8,sizeof(cl_mem),&num_buf);
    clSetKernelArg(k_eval,9,sizeof(ulong),&PD_0);
    clSetKernelArg(k_eval,10,sizeof(uint),&max_bs);
    clSetKernelArg(k_eval,11,sizeof(uint),&x2size);
    clSetKernelArg(k_eval,12,sizeof(uint),&xbits);
    clSetKernelArg(k_eval,13,sizeof(uint),(uint*)&table);
    clSetKernelArg(k_eval,14,sizeof(uint),&write_y);
    clSetKernelArg(k_eval,15,sizeof(uint),&write_c);
    clSetKernelArg(k_eval,16,sizeof(uint),&has_pd_in);
    clSetKernelArg(k_eval,17,sizeof(uint),&has_x_in);
    
    size_t g=num;if(g%64)g=((g/64)+1)*64;
    clEnqueueNDRangeKernel(queue,k_eval,1,nullptr,&g,nullptr,0,nullptr,nullptr);
    
    // Read back
    std::vector<uint32_t> cnt(num_out_buckets);
    clEnqueueReadBuffer(queue,cnt_buf,CL_TRUE,0,num_out_buckets*4,cnt.data(),0,nullptr,nullptr);
    std::vector<uint32_t> Y_gpu(num_out_buckets*max_bs_out,0),C_gpu(num_out_buckets*max_bs_out*N_META,0);
    clEnqueueReadBuffer(queue,Y_buf,CL_TRUE,0,num_out_buckets*max_bs_out*4,Y_gpu.data(),0,nullptr,nullptr);
    clEnqueueReadBuffer(queue,C_out_buf,CL_TRUE,0,num_out_buckets*max_bs_out*N_META*4,C_gpu.data(),0,nullptr,nullptr);
    
    // Collect
    Y_out.clear();M_out.clear();
    for(int b=0;b<num_out_buckets;b++)
        for(int j=0;j<(int)cnt[b];j++){
            if(write_y)Y_out.push_back(Y_gpu[b*max_bs_out+j]);
            else Y_out.push_back(0);  // will compute from M_out
            for(int m=0;m<N_META;m++)M_out.push_back(C_gpu[(b*max_bs_out+j)*N_META+m]);
        }
    
    // If Y wasn't written (table<N_TABLE), compute from metadata
    if(!write_y){
        for(size_t i=0;i<Y_out.size();i++){
            uint32_t Y=0;
            for(int m=0;m<N_META;m++)Y^=M_out[i*N_META+m];
            Y_out[i]=Y&KMASK;
        }
    }
    
    clReleaseMemObject(C_in_buf);clReleaseMemObject(LR_buf);clReleaseMemObject(num_buf);
    clReleaseMemObject(Y_buf);clReleaseMemObject(C_out_buf);clReleaseMemObject(cnt_buf);
    clReleaseMemObject(PD_out_buf);
    if(PD_in_buf)clReleaseMemObject(PD_in_buf);
}

// Pure CPU reference (for comparison)
void cpu_hash_table(const std::vector<uint32_t>& M_in,
                    const std::vector<std::pair<uint32_t,uint32_t>>& LR,
                    std::vector<uint32_t>& Y_out,std::vector<uint32_t>& M_out){
    Y_out.clear();M_out.clear();
    for(auto&lr:LR){
        uint8_t msg[2*N_META*4];
        for(int j=0;j<N_META;j++){
            memcpy(msg+j*4,&M_in[lr.first*N_META+j],4);
            memcpy(msg+(N_META+j)*4,&M_in[lr.second*N_META+j],4);
        }
        uint8_t hash[64];SHA512(msg,2*N_META*4,hash);
        uint32_t Y=0;
        for(int j=0;j<N_META;j++){uint32_t h32;memcpy(&h32,hash+j*4,4);Y^=h32;}
        Y&=KMASK;Y_out.push_back(Y);
        for(int j=0;j<N_META;j++){uint32_t h32;memcpy(&h32,hash+j*4,4);M_out.push_back(h32&KMASK);}
    }
}

int main(){
    init_opencl();
    
    const int N=100000;
    std::mt19937 rng(42);
    
    // Generate F1 output: metadata for N entries
    std::vector<uint32_t> M_curr(N*N_META);
    for(auto&v:M_curr)v=rng()&KMASK;
    std::vector<uint32_t> Y_curr(N);
    for(int i=0;i<N;i++)Y_curr[i]=compute_Y(M_curr,i);
    
    // Run 8 tables (2..9) on GPU pipeline
    auto t0=std::chrono::steady_clock::now();
    std::vector<uint32_t> M_gpu=M_curr,Y_gpu=Y_curr;
    for(int t=2;t<=N_TABLE;t++){
        std::vector<std::pair<uint32_t,uint32_t>> LR;
        cpu_sort_match(Y_gpu,M_gpu,LR);
        std::cout<<"[T"<<t<<"] "<<Y_gpu.size()<<" entries → "<<LR.size()<<" matches"<<std::endl;
        
        std::vector<uint32_t> Y_new,M_new;
        gpu_hash_table(M_gpu,LR,nullptr,t,Y_new,M_new);
        Y_gpu=Y_new;M_gpu=M_new;
    }
    auto t1=std::chrono::steady_clock::now();
    double gpu_time=std::chrono::duration<double>(t1-t0).count();
    
    // Run 8 tables on CPU for reference
    std::vector<uint32_t> M_cpu=M_curr,Y_cpu=Y_curr;
    for(int t=2;t<=N_TABLE;t++){
        std::vector<std::pair<uint32_t,uint32_t>> LR;
        cpu_sort_match(Y_cpu,M_cpu,LR);
        std::vector<uint32_t> Y_new,M_new;
        cpu_hash_table(M_cpu,LR,Y_new,M_new);
        Y_cpu=Y_new;M_cpu=M_new;
    }
    
    // Compare final output
    std::vector<std::pair<uint32_t,std::array<uint32_t,14>>> ce,ge;
    for(size_t i=0;i<Y_cpu.size();i++){std::array<uint32_t,14>m;for(int j=0;j<14;j++)m[j]=M_cpu[i*14+j];ce.push_back({Y_cpu[i],m});}
    for(size_t i=0;i<Y_gpu.size();i++){std::array<uint32_t,14>m;for(int j=0;j<14;j++)m[j]=M_gpu[i*14+j];ge.push_back({Y_gpu[i],m});}
    std::sort(ce.begin(),ce.end());std::sort(ge.begin(),ge.end());
    
    int mm=0;
    if(ce.size()!=ge.size()){std::cout<<"❌ Count: CPU="<<ce.size()<<" GPU="<<ge.size()<<std::endl;mm=1;}
    else{
        for(size_t i=0;i<ce.size();i++){
            if(ce[i].first!=ge[i].first){if(mm<3)std::cout<<"Y mismatch "<<i<<": CPU="<<ce[i].first<<" GPU="<<ge[i].first<<std::endl;mm++;}
            else{for(int j=0;j<14;j++)if(ce[i].second[j]!=ge[i].second[j]){if(mm<3)std::cout<<"M["<<j<<"] mismatch "<<i<<std::endl;mm++;break;}}
        }
    }
    
    std::cout<<"\nGPU time: "<<gpu_time<<" sec"<<std::endl;
    std::cout<<(mm==0?"✅ Multi-table chain PASSED (8 tables)":"❌ Multi-table chain FAILED")<<std::endl;
    return 0;
}
