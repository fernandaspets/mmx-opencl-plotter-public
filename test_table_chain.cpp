// test_table_chain.cpp — Chain all GPU kernels for one complete table
// scatter_2 → simple_sort_y → CPU prefix sum → match_p1 → eval_p1_tx
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

const int KSIZE=20, LOGBUCKETS=6, LOGBUCKETS2=5, N_META=14;
const uint32_t KMASK=(1u<<KSIZE)-1;
const int TOTAL_SUB=1<<(LOGBUCKETS+LOGBUCKETS2);  // 2048
const int MAX_BS2=2048, MAX_BS_OUT=4096;

cl_platform_id platform; cl_device_id device; cl_context context;
cl_command_queue queue; cl_program prog; cl_int err;
cl_kernel k_scatter, k_sort, k_match, k_eval;

void cl_check(cl_int e, const char* msg) {
    if(e!=CL_SUCCESS){std::cerr<<"CL error: "<<msg<<" code="<<e<<std::endl;exit(1);}
}

void init_opencl() {
    clGetPlatformIDs(1,&platform,nullptr);
    clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,1,&device,nullptr);
    context=clCreateContext(nullptr,1,&device,nullptr,nullptr,&err); cl_check(err,"ctx");
    queue=clCreateCommandQueue(context,device,0,&err); cl_check(err,"queue");
    
    // Combine f2_f9.cl + simple_sort.cl
    std::ifstream f1("f2_f9.cl");
    std::string s1((std::istreambuf_iterator<char>(f1)),std::istreambuf_iterator<char>());
    std::ifstream f2("simple_sort.cl");
    std::string s2((std::istreambuf_iterator<char>(f2)),std::istreambuf_iterator<char>());
    std::string combined=s1+"\n"+s2;
    const char* cs=combined.c_str(); size_t len=combined.size();
    
    prog=clCreateProgramWithSource(context,1,&cs,&len,&err);
    std::string opts="-DKSIZE=20 -DLOGBUCKETS=6 -DLOGBUCKETS2=5 -DN_META=14 -DN_META_OUT=12 -DN_TABLE=9"
                     " -DDSIZE_=5 -DPSIZE_=21 -DPDSIZE=32 -DX2SIZE=39 -DXBITS=20"
                     " -DHYBRID_SORT_LOG_THREADS=6 -DNUM_THREADS=64 -DKMASK=1048575 -DDMASK=31"
                     " -DMETA_BYTES=56 -DMAX_LOCAL_SIZE=40";
    err=clBuildProgram(prog,1,&device,opts.c_str(),nullptr,nullptr);
    if(err!=CL_SUCCESS){char log[8192];clGetProgramBuildInfo(prog,device,CL_PROGRAM_BUILD_LOG,sizeof(log),log,nullptr);std::cerr<<log<<std::endl;exit(1);}
    
    k_scatter=clCreateKernel(prog,"scatter_2",&err);
    k_sort=clCreateKernel(prog,"simple_sort_y",&err);
    k_match=clCreateKernel(prog,"match_p1",&err);
    k_eval=clCreateKernel(prog,"eval_p1_tx",&err);
    std::cout<<"[OCL] Kernels loaded"<<std::endl;
}

void cpu_table(const std::vector<uint32_t>& Y_in, const std::vector<uint32_t>& M_in,
               std::vector<uint32_t>& Y_out, std::vector<uint32_t>& M_out) {
    size_t n=Y_in.size();
    std::vector<std::pair<uint32_t,uint32_t>> entries(n);
    for(size_t i=0;i<n;i++){uint32_t Y=0;for(int j=0;j<N_META;j++)Y^=M_in[i*N_META+j];Y&=KMASK;entries[i]={Y,(uint32_t)i};}

    std::sort(entries.begin(),entries.end());
    
    std::vector<std::pair<uint32_t,uint32_t>> LR;
    for(size_t i=0;i<n;i++){
        uint32_t YL=entries[i].first;
        for(size_t j=i+1;j<n;j++){
            uint32_t YR=entries[j].first;
            if(YR==YL+1) LR.push_back({entries[i].second,entries[j].second});
            else if(YR>YL) break;
        }
    }
    
    Y_out.clear(); M_out.clear();
    for(auto& lr:LR){
        uint8_t msg[2*N_META*4];
        for(int j=0;j<N_META;j++){
            memcpy(msg+j*4,&M_in[lr.first*N_META+j],4);
            memcpy(msg+(N_META+j)*4,&M_in[lr.second*N_META+j],4);
        }
        uint8_t hash[64]; SHA512(msg,2*N_META*4,hash);
        uint32_t Y_new=0;
        for(int j=0;j<N_META;j++){uint32_t h32;memcpy(&h32,hash+j*4,4);Y_new^=h32;}
        Y_new&=KMASK; Y_out.push_back(Y_new);
        for(int j=0;j<N_META;j++){uint32_t h32;memcpy(&h32,hash+j*4,4);M_out.push_back(h32&KMASK);}
    }
}

int main() {
    init_opencl();
    
    const int N=5000;
    std::mt19937 rng(42);
    std::vector<uint32_t> Y_in(N), M_in(N*N_META);
    for(int i=0;i<N;i++){Y_in[i]=rng()&KMASK; for(int j=0;j<N_META;j++) M_in[i*N_META+j]=rng()&KMASK;}
    
    // CPU reference
    std::vector<uint32_t> cpu_Y, cpu_M;
    cpu_table(Y_in,M_in,cpu_Y,cpu_M);
    std::cout<<"CPU: "<<N<<" entries → "<<cpu_Y.size()<<" matches"<<std::endl;
    
    cl_mem null_mem=nullptr;
    int zero=0;
    
    // Step 1: scatter_2
    cl_mem C_in_buf=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,N*N_META*4,(void*)M_in.data(),&err);
    cl_mem PY_buf=clCreateBuffer(context,CL_MEM_READ_WRITE,TOTAL_SUB*MAX_BS2*8,nullptr,&err);
    cl_mem sub_cnt_buf=clCreateBuffer(context,CL_MEM_READ_WRITE,TOTAL_SUB*4,nullptr,&err);
    clEnqueueFillBuffer(queue,sub_cnt_buf,&zero,4,0,TOTAL_SUB*4,0,nullptr,nullptr);
    
    uint32_t n32=N, maxbs2=MAX_BS2;
    clSetKernelArg(k_scatter,0,sizeof(cl_mem),&PY_buf);
    clSetKernelArg(k_scatter,1,sizeof(cl_mem),&sub_cnt_buf);
    clSetKernelArg(k_scatter,2,sizeof(cl_mem),&null_mem);
    clSetKernelArg(k_scatter,3,sizeof(cl_mem),&C_in_buf);
    clSetKernelArg(k_scatter,4,sizeof(uint32_t),&n32);
    clSetKernelArg(k_scatter,5,sizeof(uint32_t),&maxbs2);
    size_t sg=N; if(sg%64)sg=((sg/64)+1)*64;
    clEnqueueNDRangeKernel(queue,k_scatter,1,nullptr,&sg,nullptr,0,nullptr,nullptr);
    
    // Step 2: CPU prefix sum
    std::vector<uint32_t> sub_cnt(TOTAL_SUB);
    clEnqueueReadBuffer(queue,sub_cnt_buf,CL_TRUE,0,TOTAL_SUB*4,sub_cnt.data(),0,nullptr,nullptr);
    std::vector<uint32_t> sub_off(TOTAL_SUB+1,0);
    for(int i=0;i<TOTAL_SUB;i++) sub_off[i+1]=sub_off[i]+sub_cnt[i];
    uint32_t total_scattered=sub_off[TOTAL_SUB];
    std::cout<<"GPU scatter: "<<total_scattered<<" entries placed"<<std::endl;
    
    cl_mem sub_off_buf=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(TOTAL_SUB+1)*4,(void*)sub_off.data(),&err);
    
    // Step 3: simple_sort_y
    uint32_t maxbs_sort=MAX_BS2, num_sub=TOTAL_SUB;
    clSetKernelArg(k_sort,0,sizeof(cl_mem),&PY_buf);
    clSetKernelArg(k_sort,1,sizeof(cl_mem),&sub_cnt_buf);
    clSetKernelArg(k_sort,2,sizeof(uint32_t),&maxbs_sort);
    clSetKernelArg(k_sort,3,sizeof(uint32_t),&num_sub);
    size_t sort_g[2]={256,(size_t)TOTAL_SUB}, sort_l[2]={256,1};
    clEnqueueNDRangeKernel(queue,k_sort,2,nullptr,sort_g,sort_l,0,nullptr,nullptr);
    
    // Step 4: match_p1
    cl_mem LR_buf=clCreateBuffer(context,CL_MEM_WRITE_ONLY,N*8,nullptr,&err);
    cl_mem PD_match_buf=clCreateBuffer(context,CL_MEM_WRITE_ONLY,N*4,nullptr,&err);
    cl_mem num_matches_buf=clCreateBuffer(context,CL_MEM_READ_WRITE,4,nullptr,&err);
    clEnqueueFillBuffer(queue,num_matches_buf,&zero,4,0,4,0,nullptr,nullptr);
    
    uint32_t max_total=N, write_pd=0;
    clSetKernelArg(k_match,0,sizeof(cl_mem),&LR_buf);
    clSetKernelArg(k_match,1,sizeof(cl_mem),&PD_match_buf);
    clSetKernelArg(k_match,2,sizeof(cl_mem),&num_matches_buf);
    clSetKernelArg(k_match,3,sizeof(cl_mem),&PY_buf);
    clSetKernelArg(k_match,4,sizeof(cl_mem),&sub_cnt_buf);
    clSetKernelArg(k_match,5,sizeof(cl_mem),&sub_off_buf);
    clSetKernelArg(k_match,6,sizeof(uint32_t),&num_sub);
    clSetKernelArg(k_match,7,sizeof(uint32_t),&maxbs_sort);
    clSetKernelArg(k_match,8,sizeof(uint32_t),&max_total);
    clSetKernelArg(k_match,9,sizeof(uint32_t),&write_pd);
    
    int groups_per_sub=(MAX_BS2+127)/128;
    size_t match_g[2]={(size_t)(128*groups_per_sub),(size_t)TOTAL_SUB}, match_l[2]={128,1};
    clEnqueueNDRangeKernel(queue,k_match,2,nullptr,match_g,match_l,0,nullptr,nullptr);
    
    uint32_t gpu_matches=0;
    clEnqueueReadBuffer(queue,num_matches_buf,CL_TRUE,0,4,&gpu_matches,0,nullptr,nullptr);
    std::cout<<"GPU match: "<<gpu_matches<<" matches"<<std::endl;
    
    // Step 5: eval_p1_tx
    int num_out_buckets=1<<LOGBUCKETS;  // 64
    cl_mem Y_out_buf=clCreateBuffer(context,CL_MEM_WRITE_ONLY,num_out_buckets*MAX_BS_OUT*4,nullptr,&err);
    cl_mem C_out_buf=clCreateBuffer(context,CL_MEM_WRITE_ONLY,num_out_buckets*MAX_BS_OUT*N_META*4,nullptr,&err);
    cl_mem out_cnt_buf=clCreateBuffer(context,CL_MEM_READ_WRITE,num_out_buckets*4,nullptr,&err);
    cl_mem PD_out_buf=clCreateBuffer(context,CL_MEM_WRITE_ONLY,num_out_buckets*MAX_BS_OUT*32,nullptr,&err);
    cl_mem num_found_buf=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,4,(void*)&gpu_matches,&err);
    clEnqueueFillBuffer(queue,out_cnt_buf,&zero,4,0,num_out_buckets*4,0,nullptr,nullptr);
    
    ulong PD_0=0; uint max_bs_out=MAX_BS_OUT, x2size=39, xbits=20, table=2;
    uint write_y=1, write_c=1, has_pd_in=0, has_x_in=0;
    clSetKernelArg(k_eval,0,sizeof(cl_mem),&Y_out_buf);
    clSetKernelArg(k_eval,1,sizeof(cl_mem),&C_out_buf);
    clSetKernelArg(k_eval,2,sizeof(cl_mem),&PD_out_buf);
    clSetKernelArg(k_eval,3,sizeof(cl_mem),&out_cnt_buf);
    clSetKernelArg(k_eval,4,sizeof(cl_mem),&C_in_buf);
    clSetKernelArg(k_eval,5,sizeof(cl_mem),&null_mem);
    clSetKernelArg(k_eval,6,sizeof(cl_mem),&null_mem);
    clSetKernelArg(k_eval,7,sizeof(cl_mem),&LR_buf);
    clSetKernelArg(k_eval,8,sizeof(cl_mem),&num_found_buf);
    clSetKernelArg(k_eval,9,sizeof(ulong),&PD_0);
    clSetKernelArg(k_eval,10,sizeof(uint),&max_bs_out);
    clSetKernelArg(k_eval,11,sizeof(uint),&x2size);
    clSetKernelArg(k_eval,12,sizeof(uint),&xbits);
    clSetKernelArg(k_eval,13,sizeof(uint),&table);
    clSetKernelArg(k_eval,14,sizeof(uint),&write_y);
    clSetKernelArg(k_eval,15,sizeof(uint),&write_c);
    clSetKernelArg(k_eval,16,sizeof(uint),&has_pd_in);
    clSetKernelArg(k_eval,17,sizeof(uint),&has_x_in);
    size_t eg=gpu_matches; if(eg%64)eg=((eg/64)+1)*64;
    clEnqueueNDRangeKernel(queue,k_eval,1,nullptr,&eg,nullptr,0,nullptr,nullptr);
    
    // Read back
    std::vector<uint32_t> out_cnt(num_out_buckets);
    clEnqueueReadBuffer(queue,out_cnt_buf,CL_TRUE,0,num_out_buckets*4,out_cnt.data(),0,nullptr,nullptr);
    std::vector<uint32_t> gpu_Y(num_out_buckets*MAX_BS_OUT,0), gpu_M(num_out_buckets*MAX_BS_OUT*N_META,0);
    clEnqueueReadBuffer(queue,Y_out_buf,CL_TRUE,0,num_out_buckets*MAX_BS_OUT*4,gpu_Y.data(),0,nullptr,nullptr);
    clEnqueueReadBuffer(queue,C_out_buf,CL_TRUE,0,num_out_buckets*MAX_BS_OUT*N_META*4,gpu_M.data(),0,nullptr,nullptr);
    
    std::vector<uint32_t> gpu_Y_all, gpu_M_all;
    for(int b=0;b<num_out_buckets;b++)
        for(int j=0;j<(int)out_cnt[b];j++){
            gpu_Y_all.push_back(gpu_Y[b*MAX_BS_OUT+j]);
            for(int m=0;m<N_META;m++) gpu_M_all.push_back(gpu_M[(b*MAX_BS_OUT+j)*N_META+m]);
        }
    std::cout<<"GPU eval: "<<gpu_Y_all.size()<<" entries output"<<std::endl;
    
    // Compare
    std::vector<std::pair<uint32_t,std::array<uint32_t,14>>> ce,ge;
    for(size_t i=0;i<cpu_Y.size();i++){std::array<uint32_t,14>m;for(int j=0;j<14;j++)m[j]=cpu_M[i*14+j];ce.push_back({cpu_Y[i],m});}
    for(size_t i=0;i<gpu_Y_all.size();i++){std::array<uint32_t,14>m;for(int j=0;j<14;j++)m[j]=gpu_M_all[i*14+j];ge.push_back({gpu_Y_all[i],m});}
    std::sort(ce.begin(),ce.end());std::sort(ge.begin(),ge.end());
    
    int mm=0;
    if(ce.size()!=ge.size()){std::cout<<"❌ Count mismatch: CPU="<<ce.size()<<" GPU="<<ge.size()<<std::endl;mm=1;}
    else{
        for(size_t i=0;i<ce.size();i++){
            if(ce[i].first!=ge[i].first){if(mm<3)std::cout<<"Y mismatch at "<<i<<": CPU="<<ce[i].first<<" GPU="<<ge[i].first<<std::endl;mm++;}
            else{for(int j=0;j<14;j++)if(ce[i].second[j]!=ge[i].second[j]){if(mm<3)std::cout<<"M["<<j<<"] mismatch at "<<i<<std::endl;mm++;break;}}
        }
    }
    std::cout<<(mm==0?"✅ Full table chain PASSED":"❌ Full table chain FAILED")<<std::endl;
    
    clReleaseMemObject(C_in_buf);clReleaseMemObject(PY_buf);clReleaseMemObject(sub_cnt_buf);
    clReleaseMemObject(sub_off_buf);clReleaseMemObject(LR_buf);clReleaseMemObject(PD_match_buf);
    clReleaseMemObject(num_matches_buf);clReleaseMemObject(num_found_buf);
    clReleaseMemObject(Y_out_buf);clReleaseMemObject(C_out_buf);clReleaseMemObject(out_cnt_buf);clReleaseMemObject(PD_out_buf);
    return 0;
}
