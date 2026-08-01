#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

int main() {
    cl_platform_id p; cl_device_id d; cl_context ctx; cl_command_queue q; cl_int err;
    clGetPlatformIDs(1,&p,nullptr); clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,nullptr);
    ctx=clCreateContext(nullptr,1,&d,nullptr,nullptr,&err);
    q=clCreateCommandQueue(ctx,d,0,&err);
    
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
    
    // Test with 1 sub-bucket, 5 entries (unsorted)
    const int MAX_BS=2048;
    std::vector<uint64_t> PY(MAX_BS, 0);
    std::vector<uint32_t> cnt(1, 5);
    
    // 5 unsorted values
    PY[0] = 500ULL << 44;  // Y=500
    PY[1] = 100ULL << 44;  // Y=100
    PY[2] = 300ULL << 44;  // Y=300
    PY[3] = 200ULL << 44;  // Y=200
    PY[4] = 400ULL << 44;  // Y=400
    
    std::cout << "Before sort:" << std::endl;
    for(int i=0;i<5;i++) std::cout << "  PY[" << i << "] = " << PY[i] << " (Y=" << (PY[i]>>44) << ")" << std::endl;
    
    cl_kernel k=clCreateKernel(prog,"simple_sort_y",&err);
    cl_mem PY_buf=clCreateBuffer(ctx,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,PY.size()*8,(void*)PY.data(),&err);
    cl_mem cnt_buf=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,4,(void*)cnt.data(),&err);
    
    uint32_t maxbs=MAX_BS, num_sub=1;
    clSetKernelArg(k,0,sizeof(cl_mem),&PY_buf);
    clSetKernelArg(k,1,sizeof(cl_mem),&cnt_buf);
    clSetKernelArg(k,2,sizeof(uint32_t),&maxbs);
    clSetKernelArg(k,3,sizeof(uint32_t),&num_sub);
    
    // Use 8 threads (more than enough for 5 elements)
    size_t g[2]={8,1}, l[2]={8,1};
    clEnqueueNDRangeKernel(q,k,2,nullptr,g,l,0,nullptr,nullptr);
    
    std::vector<uint64_t> PY_gpu(MAX_BS,0);
    clEnqueueReadBuffer(q,PY_buf,CL_TRUE,0,PY_gpu.size()*8,PY_gpu.data(),0,nullptr,nullptr);
    
    std::cout << "\nAfter GPU sort:" << std::endl;
    for(int i=0;i<5;i++) std::cout << "  PY[" << i << "] = " << PY_gpu[i] << " (Y=" << (PY_gpu[i]>>44) << ")" << std::endl;
    
    // CPU reference
    std::vector<uint64_t> cpu(PY.begin(),PY.begin()+5);
    std::sort(cpu.begin(),cpu.end());
    std::cout << "\nCPU sorted:" << std::endl;
    for(int i=0;i<5;i++) std::cout << "  PY[" << i << "] = " << cpu[i] << " (Y=" << (cpu[i]>>44) << ")" << std::endl;
    
    bool ok=true;
    for(int i=0;i<5;i++) if(cpu[i]!=PY_gpu[i]) ok=false;
    std::cout << "\n" << (ok?"✅ PASS":"❌ FAIL") << std::endl;
    
    clReleaseMemObject(PY_buf);clReleaseMemObject(cnt_buf);clReleaseKernel(k);
    return 0;
}
