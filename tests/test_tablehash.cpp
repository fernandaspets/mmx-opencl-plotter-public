// test_tablehash.cpp — Compare SHA-512(L_meta || R_meta) between CPU and GPU
// This is the F2-F9 table hash.

#include <CL/cl.h>
#include <cstdint>
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <mmx/hash_512_t.hpp>

int main() {
    int num_test = 20;
    int ksize = 18;
    uint32_t kmask = ((uint64_t(1) << ksize) - 1);
    
    // Generate random L_meta and R_meta
    std::vector<uint32_t> L_meta(num_test * 14), R_meta(num_test * 14);
    for(int i = 0; i < num_test * 14; i++) {
        L_meta[i] = rand() & kmask;
        R_meta[i] = rand() & kmask;
    }
    
    // CPU: hash L_meta || R_meta = 28 uint32 = 112 bytes
    std::cout << "=== Table Hash CPU vs GPU ===" << std::endl;
    int cpu_mismatches = 0;
    
    for(int t = 0; t < num_test; t++) {
        uint32_t msg[28];
        for(int i = 0; i < 14; i++) msg[i] = L_meta[t * 14 + i];
        for(int i = 0; i < 14; i++) msg[14 + i] = R_meta[t * 14 + i];
        
        mmx::hash_512_t hash(&msg, sizeof(msg));
        
        uint32_t hash32[16];
        std::memcpy(hash32, hash.data(), 64);
        
        uint32_t Y = 0;
        for(int i = 0; i < 14; i++) Y ^= hash32[i];
        Y &= kmask;
        
        if(t < 5) {
            printf("CPU T%d: Y=%08x M[0]=%08x M[1]=%08x\n", t, Y, hash32[0] & kmask, hash32[1] & kmask);
        }
    }
    
    // GPU: load table_hash.cl and run hash_table_entries kernel
    std::ifstream f("table_hash.cl");
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);
    
    const char* cstr = src.c_str();
    size_t len = src.size();
    cl_program program = clCreateProgramWithSource(context, 1, &cstr, &len, nullptr);
    cl_int err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
    if(err != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << "Build failed: " << log << std::endl;
        return 1;
    }
    
    cl_kernel kernel = clCreateKernel(program, "hash_table_entries", &err);
    if(err != CL_SUCCESS) {
        std::cerr << "Kernel not found: " << err << std::endl;
        return 1;
    }
    
    cl_mem Lb = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        num_test * 14 * 4, L_meta.data(), nullptr);
    cl_mem Rb = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        num_test * 14 * 4, R_meta.data(), nullptr);
    cl_mem Yb = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        num_test * 4, nullptr, nullptr);
    cl_mem Mb = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        num_test * 14 * 4, nullptr, nullptr);
    
    uint32_t num_u32 = num_test;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &Lb);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &Rb);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &Yb);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &Mb);
    clSetKernelArg(kernel, 4, sizeof(uint32_t), &kmask);
    clSetKernelArg(kernel, 5, sizeof(uint32_t), &num_u32);
    
    size_t global = num_test;
    if(global % 64) global = ((global/64)+1)*64;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    
    std::vector<uint32_t> gpu_Y(num_test), gpu_M(num_test * 14);
    clEnqueueReadBuffer(queue, Yb, CL_TRUE, 0, num_test * 4, gpu_Y.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, Mb, CL_TRUE, 0, num_test * 14 * 4, gpu_M.data(), 0, nullptr, nullptr);
    
    // Compare
    int mismatches = 0;
    for(int t = 0; t < num_test; t++) {
        uint32_t msg[28];
        for(int i = 0; i < 14; i++) msg[i] = L_meta[t * 14 + i];
        for(int i = 0; i < 14; i++) msg[14 + i] = R_meta[t * 14 + i];
        
        mmx::hash_512_t hash(&msg, sizeof(msg));
        uint32_t hash32[16];
        std::memcpy(hash32, hash.data(), 64);
        
        uint32_t cpu_Y = 0;
        for(int i = 0; i < 14; i++) cpu_Y ^= hash32[i];
        cpu_Y &= kmask;
        
        bool y_match = (gpu_Y[t] == cpu_Y);
        bool m_match = true;
        for(int i = 0; i < 14; i++) {
            if(gpu_M[t * 14 + i] != (hash32[i] & kmask)) { m_match = false; break; }
        }
        
        if(!y_match || !m_match) {
            mismatches++;
            if(mismatches <= 3) {
                printf("MISMATCH T%d: CPU Y=%08x GPU Y=%08x\n", t, cpu_Y, gpu_Y[t]);
            }
        }
        
        if(t < 5) {
            printf("GPU T%d: Y=%08x M[0]=%08x M[1]=%08x\n", t, gpu_Y[t], gpu_M[t*14], gpu_M[t*14+1]);
        }
    }
    
    std::cout << std::endl << "=== " << (num_test - mismatches) << "/" << num_test 
              << " match (" << mismatches << " mismatches) ===" << std::endl;
    
    clReleaseMemObject(Lb); clReleaseMemObject(Rb);
    clReleaseMemObject(Yb); clReleaseMemObject(Mb);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);
    
    return mismatches > 0 ? 1 : 0;
}
