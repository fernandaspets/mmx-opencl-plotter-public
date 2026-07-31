// test_sort.cpp — Test hybrid_sort_y kernel
// Generates random PY values, scatters into buckets, sorts on GPU, compares with CPU sort
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
    
    std::ifstream f("f2_f9.cl");
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const char* cstr = src.c_str();
    size_t len = src.size();
    
    cl_program prog = clCreateProgramWithSource(context, 1, &cstr, &len, &err);
    // k20: LOGBUCKETS=6, LOGBUCKETS2=5 → sub-buckets have avg 2^(20-6-5)=512 entries
    // But we use smaller test: LOGBUCKETS2=4 → avg 2^10=1024 entries per sub-bucket
    // Actually let's match k20 exactly: LOGBUCKETS2 = 20-6-9 = 5
    // sub-buckets per first-level: 2^5 = 32. avg entries = 2^(20-6-5) = 512
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
    
    const int KSIZE = 20;
    const int LOGBUCKETS = 6;
    const int LOGBUCKETS2 = 5;
    const int NUM_SUB_BUCKETS = 1 << LOGBUCKETS2;  // 32
    const int MAX_BUCKET_SIZE = 2048;  // generous for test
    
    // Generate random PY values for a single first-level bucket
    // PY = (Y << (64 - KSIZE)) | local_pos
    // Y is k20 bits, local_pos is lower bits
    const int N = 500;  // entries in this bucket (less than avg 512)
    std::mt19937_64 rng64(42);
    
    std::vector<uint64_t> PY(NUM_SUB_BUCKETS * MAX_BUCKET_SIZE, 0);
    std::vector<uint32_t> bucket_size(NUM_SUB_BUCKETS, 0);
    
    // Randomly assign entries to sub-buckets
    for (int i = 0; i < N; i++) {
        uint32_t sub = rng64() % NUM_SUB_BUCKETS;
        uint32_t pos = bucket_size[sub]++;
        if (pos < MAX_BUCKET_SIZE) {
            uint32_t Y = rng64() & ((1u << KSIZE) - 1);
            PY[sub * MAX_BUCKET_SIZE + pos] = ((uint64_t)Y << (64 - KSIZE)) | (uint64_t)i;
        }
    }
    
    // CPU reference: sort each sub-bucket by Y (the high bits of PY)
    std::vector<uint64_t> PY_sorted = PY;
    for (int b = 0; b < NUM_SUB_BUCKETS; b++) {
        std::vector<uint64_t> tmp(PY_sorted.begin() + b * MAX_BUCKET_SIZE,
                                  PY_sorted.begin() + b * MAX_BUCKET_SIZE + bucket_size[b]);
        std::sort(tmp.begin(), tmp.end(), [](uint64_t a, uint64_t b) {
            return a > b ? false : a < b;  // ascending (higher Y bits = higher value)
        });
        // Actually PY = (Y << (64-KSIZE)) | pos. Higher Y → higher PY. Sort ascending = sort by Y ascending.
        std::sort(tmp.begin(), tmp.end());
        for (int i = 0; i < (int)tmp.size(); i++) {
            PY_sorted[b * MAX_BUCKET_SIZE + i] = tmp[i];
        }
    }
    
    // GPU: run hybrid_sort_y
    cl_kernel sort_k = clCreateKernel(prog, "hybrid_sort_y", &err);
    
    cl_mem PY_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, PY.size() * 8, PY.data(), &err);
    cl_mem size_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bucket_size.size() * 4, bucket_size.data(), &err);
    
    uint32_t max_bs = MAX_BUCKET_SIZE;
    uint32_t num_sub = NUM_SUB_BUCKETS;
    
    clSetKernelArg(sort_k, 0, sizeof(cl_mem), &PY_buf);
    clSetKernelArg(sort_k, 1, sizeof(cl_mem), &size_buf);
    clSetKernelArg(sort_k, 2, sizeof(uint32_t), &max_bs);
    clSetKernelArg(sort_k, 3, sizeof(uint32_t), &num_sub);
    
    // One work-group per sub-bucket, 64 threads per group
    size_t global = NUM_SUB_BUCKETS * 64;
    size_t local = 64;
    clEnqueueNDRangeKernel(queue, sort_k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    
    std::vector<uint64_t> PY_gpu(NUM_SUB_BUCKETS * MAX_BUCKET_SIZE, 0);
    clEnqueueReadBuffer(queue, PY_buf, CL_TRUE, 0, PY_gpu.size() * 8, PY_gpu.data(), 0, nullptr, nullptr);
    
    // Compare: each sub-bucket should be sorted by PY (which means sorted by Y)
    int mismatches = 0;
    for (int b = 0; b < NUM_SUB_BUCKETS; b++) {
        int count = bucket_size[b];
        for (int i = 0; i < count; i++) {
            uint64_t cpu_val = PY_sorted[b * MAX_BUCKET_SIZE + i];
            uint64_t gpu_val = PY_gpu[b * MAX_BUCKET_SIZE + i];
            if (cpu_val != gpu_val) {
                if (mismatches < 10) {
                    std::cout << "Mismatch bucket " << b << " idx " << i 
                              << ": CPU=" << cpu_val << " GPU=" << gpu_val << std::endl;
                }
                mismatches++;
            }
        }
    }
    
    if (mismatches == 0) {
        std::cout << "✅ hybrid_sort_y PASSED — all sub-buckets sorted correctly" << std::endl;
    } else {
        std::cout << "❌ hybrid_sort_y FAILED — " << mismatches << " mismatches" << std::endl;
        // Debug: print first bucket
        int b = 0;
        std::cout << "\nBucket 0 (size=" << bucket_size[b] << "):" << std::endl;
        for (int i = 0; i < std::min(10, (int)bucket_size[b]); i++) {
            std::cout << "  CPU[" << i << "] = " << PY_sorted[b * MAX_BUCKET_SIZE + i]
                      << "  GPU[" << i << "] = " << PY_gpu[b * MAX_BUCKET_SIZE + i] << std::endl;
        }
    }
    
    clReleaseMemObject(PY_buf);
    clReleaseMemObject(size_buf);
    clReleaseKernel(sort_k);
    return 0;
}
