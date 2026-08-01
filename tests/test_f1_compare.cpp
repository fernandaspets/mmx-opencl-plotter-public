// test_f1_compare.cpp — Compare F1 output: GPU vs CPU byte-for-byte
//
// This test computes F1 for X=0..N-1 on both CPU (mmx-node reference) and GPU (our OpenCL kernel),
// then compares Y and M values. Any mismatch indicates a bug in our GPU implementation.
//
// Build: see build_test.sh
// Run: ./test_f1_compare

#include <CL/cl.h>
#include <cstdint>
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>

// MMX includes
#include <mmx/hash_512_t.hpp>
#include <mmx/pos/mem_hash.h>
#include <mmx/pos/util.h>
#include <mmx/hash_t.hpp>

// Constants (must match plotter)
static const int N_META = 14;
static const int MEM_HASH_ITER = 11337;  // from mmx-node pos config
static const int MEM_SIZE = 32 * 32;     // 1024

// ============================================================================
// CPU F1 computation (exact copy of mmx-node/src/pos/verify.cpp compute_f1)
// ============================================================================
struct F1Result {
    uint32_t Y;
    uint32_t M[N_META];
};

F1Result cpu_compute_f1(uint32_t X_i, const uint8_t* plot_id, int ksize) {
    uint32_t kmask = ((uint64_t(1) << ksize) - 1);
    
    uint32_t msg[9] = {};
    msg[0] = X_i;
    std::memcpy(msg + 1, plot_id, 32);
    
    mmx::hash_512_t key(&msg, sizeof(msg));
    
    std::vector<uint32_t> mem_buf(MEM_SIZE);
    mmx::pos::gen_mem_array(mem_buf.data(), key.data(), MEM_SIZE);
    
    uint8_t mem_hash[64 + 128] = {};
    std::memcpy(mem_hash, key.data(), key.size());
    mmx::pos::calc_mem_hash(mem_buf.data(), mem_hash + 64, MEM_HASH_ITER);
    
    mmx::hash_512_t final_hash(mem_hash, sizeof(mem_hash));
    
    uint32_t hash[16] = {};
    std::memcpy(hash, final_hash.data(), final_hash.size());
    
    F1Result result;
    result.Y = 0;
    for(int i = 0; i < N_META; i++) {
        result.Y ^= hash[i];
        result.M[i] = hash[i] & kmask;
    }
    result.Y &= kmask;
    return result;
}

// ============================================================================
// GPU F1 computation (calls our OpenCL kernel)
// ============================================================================
bool gpu_compute_f1_batch(const std::vector<uint32_t>& X_values, 
                          const uint8_t* plot_id,
                          int ksize,
                          std::vector<uint32_t>& Y_out,
                          std::vector<uint32_t>& M_out)
{
    // Load and compile the F1 kernel
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);
    
    // Load kernel source
    std::ifstream f("pos_recompute.cl");
    if(!f.good()) {
        std::cerr << "Cannot open pos_recompute.cl" << std::endl;
        return false;
    }
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    
    const char* cstr = src.c_str();
    size_t len = src.size();
    cl_program program = clCreateProgramWithSource(context, 1, &cstr, &len, nullptr);
    
    std::string opts = "-cl-std=CL1.2 -DN_META=14 -DMEM_HASH_ITER=11337";
    cl_int err = clBuildProgram(program, 1, &device, opts.c_str(), nullptr, nullptr);
    if(err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << "Build failed: " << log << std::endl;
        return false;
    }
    
    cl_kernel kernel = clCreateKernel(program, "compute_f1_kernel", &err);
    if(err != CL_SUCCESS) {
        std::cerr << "Kernel not found: " << err << std::endl;
        return false;
    }
    
    // Prepare inputs
    std::vector<uint32_t> id_u32(8);
    std::memcpy(id_u32.data(), plot_id, 32);
    
    uint32_t kmask = ((uint64_t(1) << ksize) - 1);
    uint32_t num_x = X_values.size();
    uint32_t xbits = ksize;  // uncompressed
    
    cl_mem X_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        num_x * sizeof(uint32_t), (void*)X_values.data(), nullptr);
    cl_mem ID_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        8 * sizeof(uint32_t), (void*)id_u32.data(), nullptr);
    cl_mem Y_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        num_x * sizeof(uint32_t), nullptr, nullptr);
    cl_mem M_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        num_x * N_META * sizeof(uint32_t), nullptr, nullptr);
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &X_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &ID_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &Y_buf);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &M_buf);
    clSetKernelArg(kernel, 4, sizeof(uint32_t), &kmask);
    clSetKernelArg(kernel, 5, sizeof(uint32_t), &xbits);
    clSetKernelArg(kernel, 6, sizeof(uint32_t), &num_x);
    
    size_t global = num_x;
    if(global % 64) global = ((global / 64) + 1) * 64;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    
    Y_out.resize(num_x);
    M_out.resize(num_x * N_META);
    clEnqueueReadBuffer(queue, Y_buf, CL_TRUE, 0, num_x * sizeof(uint32_t), Y_out.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, M_buf, CL_TRUE, 0, num_x * N_META * sizeof(uint32_t), M_out.data(), 0, nullptr, nullptr);
    
    clReleaseMemObject(X_buf);
    clReleaseMemObject(ID_buf);
    clReleaseMemObject(Y_buf);
    clReleaseMemObject(M_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return true;
}

// ============================================================================
// Main: compare CPU vs GPU F1 for N entries
// ============================================================================
int main(int argc, char** argv) {
    int ksize = 18;
    int num_test = 100;  // test first 100 X values
    
    // Use the plot_id that gave 0% pass rate
    uint8_t plot_id[32];
    // Parse the "bad" plot ID: 89AFB708A213123203000F47C904ED89794A989399C690126FD609E019320E26
    std::string pid_hex = "89AFB708A213123203000F47C904ED89794A989399C690126FD609E019320E26";
    for(int i = 0; i < 32; i++) {
        plot_id[i] = std::stoul(pid_hex.substr(i*2, 2), nullptr, 16);
    }
    
    std::cout << "=== F1 CPU vs GPU Comparison ===" << std::endl;
    std::cout << "Plot ID: " << pid_hex << std::endl;
    std::cout << "K: " << ksize << ", Testing " << num_test << " entries" << std::endl;
    std::cout << std::endl;
    
    // CPU computation
    std::cout << "Computing F1 on CPU..." << std::endl;
    std::vector<F1Result> cpu_results(num_test);
    for(int i = 0; i < num_test; i++) {
        cpu_results[i] = cpu_compute_f1(i, plot_id, ksize);
    }
    
    // GPU computation
    std::cout << "Computing F1 on GPU..." << std::endl;
    std::vector<uint32_t> X_values(num_test);
    for(int i = 0; i < num_test; i++) X_values[i] = i;
    
    std::vector<uint32_t> gpu_Y, gpu_M;
    if(!gpu_compute_f1_batch(X_values, plot_id, ksize, gpu_Y, gpu_M)) {
        std::cerr << "GPU computation failed!" << std::endl;
        return 1;
    }
    
    // Compare
    std::cout << std::endl << "Comparing results..." << std::endl;
    int mismatches = 0;
    for(int i = 0; i < num_test; i++) {
        bool y_match = (gpu_Y[i] == cpu_results[i].Y);
        bool m_match = true;
        for(int j = 0; j < N_META; j++) {
            if(gpu_M[i * N_META + j] != cpu_results[i].M[j]) {
                m_match = false;
                break;
            }
        }
        
        if(!y_match || !m_match) {
            mismatches++;
            if(mismatches <= 10) {
                std::cout << "  MISMATCH at X=" << i << ":" << std::endl;
                if(!y_match) {
                    std::cout << "    Y: CPU=" << std::hex << cpu_results[i].Y 
                              << " GPU=" << gpu_Y[i] << std::dec << std::endl;
                }
                if(!m_match) {
                    for(int j = 0; j < N_META; j++) {
                        if(gpu_M[i * N_META + j] != cpu_results[i].M[j]) {
                            std::cout << "    M[" << j << "]: CPU=" << std::hex << cpu_results[i].M[j]
                                      << " GPU=" << gpu_M[i * N_META + j] << std::dec << std::endl;
                        }
                    }
                }
            }
        }
    }
    
    std::cout << std::endl;
    std::cout << "=== Results: " << (num_test - mismatches) << "/" << num_test 
              << " match (" << mismatches << " mismatches) ===" << std::endl;
    
    if(mismatches == 0) {
        std::cout << "PASS: F1 GPU output matches CPU byte-for-byte" << std::endl;
        return 0;
    } else {
        std::cout << "FAIL: F1 GPU output differs from CPU" << std::endl;
        return 1;
    }
}
