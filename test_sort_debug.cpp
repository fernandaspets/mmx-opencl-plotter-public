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
    if (err != CL_SUCCESS) {
        char log[8192]; clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << "Build failed:\n" << log << std::endl; return 1;
    }
    
    // Use scatter_2 to put entries into buckets, then sort.
    // Test with 1 sub-bucket, 10 entries
    const int MAX_BS = 2048;
    
    std::vector<uint64_t> PY(MAX_BS, 0);
    std::vector<uint32_t> bs(1, 10);
    
    std::mt19937_64 rng64(42);
    for (int i = 0; i < 10; i++) {
        uint32_t Y = rng64() & ((1u << 20) - 1);
        PY[i] = ((uint64_t)Y << 44) | (uint64_t)i;
    }
    
    // Compute thread bins on CPU
    std::cout << "Thread bin assignments:" << std::endl;
    for (int i = 0; i < 10; i++) {
        uint32_t bin = (PY[i] >> (64 - 6 - 5 - 6)) & 63;
        std::cout << "  PY[" << i << "] bin=" << bin << " val=" << PY[i] << std::endl;
    }
    
    // CPU reference: group by bin, sort within bin, concatenate
    std::vector<std::vector<uint64_t>> bins(64);
    for (int i = 0; i < 10; i++) {
        uint32_t bin = (PY[i] >> (64 - 6 - 5 - 6)) & 63;
        bins[bin].push_back(PY[i]);
    }
    std::vector<uint64_t> cpu_expected;
    for (int b = 0; b < 64; b++) {
        std::sort(bins[b].begin(), bins[b].end());
        for (auto v : bins[b]) cpu_expected.push_back(v);
    }
    std::cout << "\nCPU expected (bin-sorted):" << std::endl;
    for (int i = 0; i < 10; i++) std::cout << "  " << cpu_expected[i] << std::endl;
    
    // GPU sort
    cl_kernel sort_k = clCreateKernel(prog, "hybrid_sort_y", &err);
    cl_mem PY_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, PY.size() * 8, PY.data(), &err);
    cl_mem size_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4, bs.data(), &err);
    
    uint32_t max_bs = MAX_BS;
    uint32_t num_sub = 1;
    clSetKernelArg(sort_k, 0, sizeof(cl_mem), &PY_buf);
    clSetKernelArg(sort_k, 1, sizeof(cl_mem), &size_buf);
    clSetKernelArg(sort_k, 2, sizeof(uint32_t), &max_bs);
    clSetKernelArg(sort_k, 3, sizeof(uint32_t), &num_sub);
    
    size_t global = 1 * 64;
    size_t local = 64;
    clEnqueueNDRangeKernel(queue, sort_k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    
    std::vector<uint64_t> PY_gpu(MAX_BS, 0);
    clEnqueueReadBuffer(queue, PY_buf, CL_TRUE, 0, PY_gpu.size() * 8, PY_gpu.data(), 0, nullptr, nullptr);
    
    std::cout << "\nGPU output:" << std::endl;
    for (int i = 0; i < 10; i++) std::cout << "  " << PY_gpu[i] << std::endl;
    
    // Compare
    bool match = true;
    for (int i = 0; i < 10; i++) {
        if (cpu_expected[i] != PY_gpu[i]) {
            std::cout << "\nMismatch at " << i << ": CPU=" << cpu_expected[i] << " GPU=" << PY_gpu[i] << std::endl;
            match = false;
        }
    }
    std::cout << "\n" << (match ? "✅ PASS" : "❌ FAIL") << std::endl;
    
    clReleaseMemObject(PY_buf);
    clReleaseMemObject(size_buf);
    clReleaseKernel(sort_k);
    return 0;
}
