/*
 * Test GPU table hash vs CPU table hash
 */
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <mmx/hash_512_t.hpp>
#include <mmx/pos/config.h>
#include <mmx/utils.h>

#include <cstdio>
#include <cstdint>
#include <vector>
#include <array>
#include <cstring>
#include <iostream>

using namespace mmx;

int main() {
    printf("=== GPU Table Hash Test ===\n");
    
    const int N = 1000;
    const int N_META = 14;
    const int KSIZE = 26;
    const uint32_t kmask = (1u << KSIZE) - 1;
    
    // Generate random L_meta and R_meta
    std::vector<uint32_t> L_meta(N * N_META), R_meta(N * N_META);
    for(int i = 0; i < N * N_META; i++) {
        L_meta[i] = rand() & kmask;
        R_meta[i] = rand() & kmask;
    }
    
    // CPU hash
    printf("Computing CPU hash...\n");
    std::vector<uint32_t> cpu_Y(N), cpu_M(N * N_META);
    for(int i = 0; i < N; i++) {
        uint32_t msg[N_META * 2];
        for(int j = 0; j < N_META; j++) {
            msg[j] = L_meta[i * N_META + j];
            msg[N_META + j] = R_meta[i * N_META + j];
        }
        const hash_512_t tmp(&msg, sizeof(msg));
        uint32_t hash[16];
        std::memcpy(hash, tmp.data(), tmp.size());
        
        uint32_t Y = 0;
        for(int j = 0; j < N_META; j++) Y ^= hash[j];
        Y &= kmask;
        cpu_Y[i] = Y;
        for(int j = 0; j < N_META; j++) cpu_M[i * N_META + j] = hash[j] & kmask;
    }
    
    // GPU hash
    printf("Computing GPU hash...\n");
    FILE* f = fopen("/home/ubman/mmx-app/opencl-recompute/table_hash.cl", "r");
    if(!f) { printf("Cannot open kernel\n"); return 1; }
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* src = new char[sz+1]; size_t rd = fread(src, 1, sz, f); src[sz] = 0; fclose(f);
    
    cl_int err;
    cl_platform_id plat; clGetPlatformIDs(1, &plat, nullptr);
    cl_device_id dev; clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, &sz, &err);
    delete[] src;
    err = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);
    if(err) {
        char log[4096]; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        printf("Build err: %s\n", log); return 1;
    }
    cl_kernel k = clCreateKernel(prog, "hash_table_entries", &err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    
    cl_mem Lb = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, N*N_META*4, L_meta.data(), &err);
    cl_mem Rb = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, N*N_META*4, R_meta.data(), &err);
    cl_mem Yb = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, N*4, nullptr, &err);
    cl_mem Mb = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, N*N_META*4, nullptr, &err);
    
    uint32_t num = N;
    clSetKernelArg(k,0,sizeof(cl_mem),&Lb);
    clSetKernelArg(k,1,sizeof(cl_mem),&Rb);
    clSetKernelArg(k,2,sizeof(cl_mem),&Yb);
    clSetKernelArg(k,3,sizeof(cl_mem),&Mb);
    clSetKernelArg(k,4,sizeof(uint32_t),&kmask);
    clSetKernelArg(k,5,sizeof(uint32_t),&num);
    
    size_t global = N, local = 64;
    if(global % local) global = ((global/local)+1)*local;
    clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    
    std::vector<uint32_t> gpu_Y(N), gpu_M(N * N_META);
    clEnqueueReadBuffer(q, Yb, CL_TRUE, 0, N*4, gpu_Y.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(q, Mb, CL_TRUE, 0, N*N_META*4, gpu_M.data(), 0, nullptr, nullptr);
    
    // Compare
    int y_mismatch = 0, m_mismatch = 0;
    for(int i = 0; i < N; i++) {
        if(cpu_Y[i] != gpu_Y[i]) {
            if(y_mismatch < 5) printf("Y mismatch at %d: cpu=%u gpu=%u\n", i, cpu_Y[i], gpu_Y[i]);
            y_mismatch++;
        }
        for(int j = 0; j < N_META; j++) {
            if(cpu_M[i*N_META+j] != gpu_M[i*N_META+j]) {
                if(m_mismatch < 5) printf("M mismatch at [%d][%d]: cpu=%u gpu=%u\n", i, j, cpu_M[i*N_META+j], gpu_M[i*N_META+j]);
                m_mismatch++;
            }
        }
    }
    
    printf("Y mismatches: %d / %d → %s\n", y_mismatch, N, y_mismatch==0?"✅":"❌");
    printf("M mismatches: %d / %d → %s\n", m_mismatch, N*N_META, m_mismatch==0?"✅":"❌");
    
    clReleaseMemObject(Lb); clReleaseMemObject(Rb);
    clReleaseMemObject(Yb); clReleaseMemObject(Mb);
    clReleaseKernel(k); clReleaseProgram(prog); clReleaseContext(ctx);
    
    return y_mismatch + m_mismatch;
}
