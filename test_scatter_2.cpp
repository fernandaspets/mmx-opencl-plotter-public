// test_scatter_2.cpp — Test scatter_2 kernel against CPU reference
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>

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
    
    // Build f2_f9.cl
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
        char log[8192];
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << "Build failed:\n" << log << std::endl;
        return 1;
    }
    std::cout << "Build OK" << std::endl;
    
    // Constants for k20
    const int KSIZE = 20;
    const int LOGBUCKETS = 6;
    const int LOGBUCKETS2 = 5;
    const int N_META = 14;
    const uint32_t KMASK = (1u << KSIZE) - 1;  // 1048575
    const int NUM_BUCKETS = 1 << (LOGBUCKETS + LOGBUCKETS2);  // 2^11 = 2048
    const int MAX_BUCKET_SIZE = 2048;  // generous
    
    // Generate random test data: C_in (metadata) for N entries
    const int N = 10000;
    std::vector<uint32_t> C_in(N * N_META);
    std::mt19937 rng(42);
    for (auto& v : C_in) v = rng() & KMASK;
    
    // CPU reference: compute Y_i for each entry, then bucket
    std::vector<uint32_t> Y_ref(N);
    for (int i = 0; i < N; i++) {
        uint32_t Y = 0;
        for (int j = 0; j < N_META; j++) Y ^= C_in[i * N_META + j];
        Y &= KMASK;
        Y_ref[i] = Y;
    }
    
    // CPU reference scatter: PY_out[bucket * MAX_BUCKET_SIZE + pos] = (Y << (64-KSIZE)) | local_pos
    std::vector<uint64_t> PY_ref(NUM_BUCKETS * MAX_BUCKET_SIZE, 0);
    std::vector<uint32_t> bucket_count_ref(NUM_BUCKETS, 0);
    for (int i = 0; i < N; i++) {
        uint32_t bucket = Y_ref[i] >> (KSIZE - LOGBUCKETS - LOGBUCKETS2);
        uint32_t pos = bucket_count_ref[bucket]++;
        if (pos < MAX_BUCKET_SIZE) {
            PY_ref[bucket * MAX_BUCKET_SIZE + pos] = ((uint64_t)Y_ref[i] << (64 - KSIZE)) | (uint64_t)i;
        }
    }
    
    // GPU: run scatter_2
    cl_kernel scatter_k = clCreateKernel(prog, "scatter_2", &err);
    
    cl_mem PY_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, NUM_BUCKETS * MAX_BUCKET_SIZE * 8, nullptr, &err);
    cl_mem count_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_BUCKETS * 4, nullptr, &err);
    int zero = 0;
    clEnqueueFillBuffer(queue, count_buf, &zero, 4, 0, NUM_BUCKETS * 4, 0, nullptr, nullptr);
    cl_mem C_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, C_in.size() * 4, C_in.data(), &err);
    
    // Pass NULL for Y_in (not used for tables 2-9 where Y is computed from metadata)
    cl_mem Y_null = nullptr;  // pass 0
    uint32_t bucket_size_in = N;
    uint32_t max_bs2 = MAX_BUCKET_SIZE;
    
    clSetKernelArg(scatter_k, 0, sizeof(cl_mem), &PY_buf);
    clSetKernelArg(scatter_k, 1, sizeof(cl_mem), &count_buf);
    clSetKernelArg(scatter_k, 2, sizeof(cl_mem), &Y_null);  // NULL = compute Y from C_in
    clSetKernelArg(scatter_k, 3, sizeof(cl_mem), &C_buf);
    clSetKernelArg(scatter_k, 4, sizeof(uint32_t), &bucket_size_in);
    clSetKernelArg(scatter_k, 5, sizeof(uint32_t), &max_bs2);
    
    size_t global = N;
    if (global % 64) global = ((global / 64) + 1) * 64;
    clEnqueueNDRangeKernel(queue, scatter_k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    
    // Read back
    std::vector<uint64_t> PY_gpu(NUM_BUCKETS * MAX_BUCKET_SIZE, 0);
    std::vector<uint32_t> bucket_count_gpu(NUM_BUCKETS, 0);
    clEnqueueReadBuffer(queue, PY_buf, CL_TRUE, 0, PY_gpu.size() * 8, PY_gpu.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, count_buf, CL_TRUE, 0, bucket_count_gpu.size() * 4, bucket_count_gpu.data(), 0, nullptr, nullptr);
    
    // Compare
    int mismatches = 0;
    for (int b = 0; b < NUM_BUCKETS; b++) {
        if (bucket_count_ref[b] != bucket_count_gpu[b]) {
            if (mismatches < 5) {
                std::cout << "Bucket count mismatch at " << b << ": CPU=" << bucket_count_ref[b] 
                          << " GPU=" << bucket_count_gpu[b] << std::endl;
            }
            mismatches++;
        }
    }
    
    // Compare PY values (only for entries that were placed)
    int py_mismatches = 0;
    for (int b = 0; b < NUM_BUCKETS; b++) {
        int count = std::min(bucket_count_ref[b], bucket_count_gpu[b]);
        // The positions may differ because atomics are non-deterministic in order
        // But the SET of PY values should match
        std::vector<uint64_t> cpu_vals(PY_ref.begin() + b * MAX_BUCKET_SIZE, 
                                        PY_ref.begin() + b * MAX_BUCKET_SIZE + count);
        std::vector<uint64_t> gpu_vals(PY_gpu.begin() + b * MAX_BUCKET_SIZE,
                                       PY_gpu.begin() + b * MAX_BUCKET_SIZE + count);
        std::sort(cpu_vals.begin(), cpu_vals.end());
        std::sort(gpu_vals.begin(), gpu_vals.end());
        for (int i = 0; i < count; i++) {
            if (cpu_vals[i] != gpu_vals[i]) {
                if (py_mismatches < 5) {
                    std::cout << "PY mismatch at bucket " << b << " idx " << i 
                              << ": CPU=" << cpu_vals[i] << " GPU=" << gpu_vals[i] << std::endl;
                }
                py_mismatches++;
            }
        }
    }
    
    if (mismatches == 0 && py_mismatches == 0) {
        std::cout << "✅ scatter_2 PASSED — bucket counts and PY values match" << std::endl;
    } else {
        std::cout << "❌ scatter_2 FAILED — " << mismatches << " count mismatches, " 
                  << py_mismatches << " PY mismatches" << std::endl;
    }
    
    clReleaseMemObject(PY_buf);
    clReleaseMemObject(count_buf);
    clReleaseMemObject(C_buf);
    clReleaseKernel(scatter_k);
    return 0;
}
