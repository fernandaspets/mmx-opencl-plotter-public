// test_sha512_gpu.cpp — Compare SHA-512(X || plot_id) between CPU and GPU
// Isolates the SHA-512 step from the memory-hard function.

#include <CL/cl.h>
#include <cstdint>
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <mmx/hash_512_t.hpp>

// Read the SHA-512 functions from pos_recompute.cl by extracting just the needed parts
std::string load_sha512_kernel() {
    // Read pos_recompute.cl
    std::ifstream f("pos_recompute.cl");
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    
    // Extract everything up to compute_f1_kernel (the SHA-512 + helper functions)
    size_t end = src.find("void compute_f1_kernel");
    if(end == std::string::npos) end = src.size();
    
    std::string sha_code = src.substr(0, end);
    
    // Add our test kernel
    sha_code += R"(
__kernel void test_sha512_key(
    __global const uint* X_in,
    __global const uint* ID_in,
    __global uint* key_out,
    const uint num_x)
{
    const uint gid = get_global_id(0);
    if(gid >= num_x) return;
    
    const uint X_i = X_in[gid];
    
    uint msg32[18];
    for(int i = 0; i < 18; i++) msg32[i] = 0;
    msg32[0] = X_i;
    for(int i = 0; i < 8; i++) msg32[1 + i] = ID_in[i];
    
    ulong msg64[16];
    for(int i = 0; i < 16; i++) msg64[i] = 0;
    pack_uint32_to_be_ulong(msg32, 9, msg64, 16);
    
    ulong key_state[8];
    sha512_hash(msg64, 36, key_state);
    
    uint key32[16];
    extract_uint32_from_sha512(key_state, key32);
    
    for(int i = 0; i < 16; i++) key_out[gid * 16 + i] = key32[i];
}
)";
    return sha_code;
}

int main() {
    uint8_t plot_id[32];
    std::string pid_hex = "89AFB708A213123203000F47C904ED89794A989399C690126FD609E019320E26";
    for(int i = 0; i < 32; i++) plot_id[i] = std::stoul(pid_hex.substr(i*2, 2), nullptr, 16);
    
    int num_test = 10;
    
    std::cout << "=== SHA-512 Key Comparison (CPU vs GPU) ===" << std::endl;
    std::cout << "Plot ID: " << pid_hex << std::endl;
    std::cout << "Testing X=0.." << num_test-1 << std::endl << std::endl;
    
    // CPU: compute SHA-512 key for each X
    std::vector<std::array<uint32_t, 16>> cpu_keys(num_test);
    for(int x = 0; x < num_test; x++) {
        uint32_t msg[9] = {};
        msg[0] = x;
        std::memcpy(msg + 1, plot_id, 32);
        mmx::hash_512_t key(&msg, sizeof(msg));
        std::memcpy(cpu_keys[x].data(), key.data(), 64);
    }
    
    // GPU: compute SHA-512 key
    std::string kernel_src = load_sha512_kernel();
    
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);
    
    const char* cstr = kernel_src.c_str();
    size_t len = kernel_src.size();
    cl_program program = clCreateProgramWithSource(context, 1, &cstr, &len, nullptr);
    cl_int err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
    if(err != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << "Build failed: " << log << std::endl;
        return 1;
    }
    
    cl_kernel kernel = clCreateKernel(program, "test_sha512_key", &err);
    
    std::vector<uint32_t> X_values(num_test);
    for(int i = 0; i < num_test; i++) X_values[i] = i;
    
    std::vector<uint32_t> id_u32(8);
    std::memcpy(id_u32.data(), plot_id, 32);
    
    cl_mem X_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        num_test * 4, X_values.data(), nullptr);
    cl_mem ID_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        32, id_u32.data(), nullptr);
    cl_mem key_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        num_test * 64, nullptr, nullptr);
    
    uint32_t num_x = num_test;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &X_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &ID_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &key_buf);
    clSetKernelArg(kernel, 3, sizeof(uint32_t), &num_x);
    
    size_t global = num_test;
    if(global % 64) global = ((global/64)+1)*64;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    
    std::vector<uint32_t> gpu_keys(num_test * 16);
    clEnqueueReadBuffer(queue, key_buf, CL_TRUE, 0, num_test * 64, gpu_keys.data(), 0, nullptr, nullptr);
    
    // Compare
    int mismatches = 0;
    for(int x = 0; x < num_test; x++) {
        bool match = true;
        for(int i = 0; i < 16; i++) {
            if(gpu_keys[x * 16 + i] != cpu_keys[x][i]) {
                match = false;
                break;
            }
        }
        if(!match) {
            mismatches++;
            std::cout << "MISMATCH at X=" << x << ":" << std::endl;
            std::cout << "  CPU: ";
            for(int i = 0; i < 16; i++) printf("%08x ", cpu_keys[x][i]);
            std::cout << std::endl;
            std::cout << "  GPU: ";
            for(int i = 0; i < 16; i++) printf("%08x ", gpu_keys[x * 16 + i]);
            std::cout << std::endl;
        } else {
            std::cout << "MATCH at X=" << x << std::endl;
        }
    }
    
    std::cout << std::endl << "=== " << (num_test - mismatches) << "/" << num_test 
              << " match (" << mismatches << " mismatches) ===" << std::endl;
    
    clReleaseMemObject(X_buf);
    clReleaseMemObject(ID_buf);
    clReleaseMemObject(key_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return mismatches > 0 ? 1 : 0;
}
