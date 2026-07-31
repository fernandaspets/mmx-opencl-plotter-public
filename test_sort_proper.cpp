#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

int main() {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_int err;
    
    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    
    std::ifstream f("f2_f9.cl");
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const char* cstr = src.c_str();
    size_t len = src.size();
    
    cl_program prog = clCreateProgramWithSource(context, 1, &cstr, &len, &err);
    std::string opts = "-DKSIZE=20 -DLOGBUCKETS=6 -DLOGBUCKETS2=5 -DN_META=14 -DN_META_OUT=12 -DN_TABLE=9"
                       " -DDSIZE_=5 -DPSIZE_=21 -DPDSIZE=40 -DX2SIZE=39 -DXBITS=20"
                       " -DHYBRID_SORT_LOG_THREADS=6 -DNUM_THREADS=64"
                       " -DKMASK=1048575 -DDMASK=31 -DMETA_BYTES=56"
                       " -DMAX_LOCAL_SIZE=40";
    err = clBuildProgram(prog, 1, &device, opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) { char log[8192]; clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr); std::cerr << log << std::endl; return 1; }
    
    const int KSIZE = 20, LOGBUCKETS = 6, LOGBUCKETS2 = 5;
    const int MAX_BS = 2048, NUM_SUB = 1;
    
    // All entries in same sub-bucket: same top (LOGBUCKETS+LOGBUCKETS2)=11 bits of Y
    // Y >> 9 = same value for all. So Y = (sub_bucket_id << 9) | lower_9_bits
    const uint32_t SUB_BUCKET_ID = 42;
    std::mt19937_64 rng64(42);
    
    std::vector<uint64_t> PY(MAX_BS, 0);
    const int N = 100;
    std::vector<uint32_t> bs(1, N);
    
    for (int i = 0; i < N; i++) {
        // Y has same top 11 bits, random lower 9 bits
        uint32_t lower = rng64() & 0x1FF;  // 9 bits
        uint32_t Y = (SUB_BUCKET_ID << 9) | lower;
        PY[i] = ((uint64_t)Y << (64 - KSIZE)) | (uint64_t)i;
    }
    
    // CPU full sort
    std::vector<uint64_t> cpu_sorted(PY.begin(), PY.begin() + N);
    std::sort(cpu_sorted.begin(), cpu_sorted.end());
    
    // GPU sort
    cl_kernel sort_k = clCreateKernel(prog, "hybrid_sort_y", &err);
    cl_mem PY_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, PY.size() * 8, PY.data(), &err);
    cl_mem size_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4, bs.data(), &err);
    
    uint32_t max_bs = MAX_BS, num_sub = NUM_SUB;
    clSetKernelArg(sort_k, 0, sizeof(cl_mem), &PY_buf);
    clSetKernelArg(sort_k, 1, sizeof(cl_mem), &size_buf);
    clSetKernelArg(sort_k, 2, sizeof(uint32_t), &max_bs);
    clSetKernelArg(sort_k, 3, sizeof(uint32_t), &num_sub);
    
    size_t global = 64, local = 64;
    clEnqueueNDRangeKernel(queue, sort_k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    
    std::vector<uint64_t> PY_gpu(MAX_BS, 0);
    clEnqueueReadBuffer(queue, PY_buf, CL_TRUE, 0, PY_gpu.size() * 8, PY_gpu.data(), 0, nullptr, nullptr);
    
    // Compare (should be full Y sort since all in same sub-bucket)
    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        if (cpu_sorted[i] != PY_gpu[i]) {
            if (mismatches < 5) std::cout << "Mismatch " << i << ": CPU=" << cpu_sorted[i] << " GPU=" << PY_gpu[i] << std::endl;
            mismatches++;
        }
    }
    
    // Also check if GPU output is sorted by Y
    bool gpu_sorted = true;
    for (int i = 1; i < N; i++) {
        if ((PY_gpu[i-1] >> 44) > (PY_gpu[i] >> 44)) { gpu_sorted = false; break; }
    }
    
    std::cout << "Mismatches: " << mismatches << "/" << N << std::endl;
    std::cout << "GPU Y-sorted? " << (gpu_sorted ? "YES" : "NO") << std::endl;
    std::cout << (mismatches == 0 && gpu_sorted ? "✅ hybrid_sort_y PASSED" : "❌ FAILED") << std::endl;
    
    clReleaseMemObject(PY_buf); clReleaseMemObject(size_buf); clReleaseKernel(sort_k);
    return 0;
}
