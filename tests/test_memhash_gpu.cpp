// test_memhash_gpu.cpp — Compare gen_mem_array + calc_mem_hash between CPU and GPU
// SHA-512 key is correct (verified separately), so the bug is in the memory-hard function.

#include <CL/cl.h>
#include <cstdint>
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <mmx/hash_512_t.hpp>
#include <mmx/pos/mem_hash.h>

int main() {
    uint8_t plot_id[32];
    std::string pid_hex = "89AFB708A213123203000F47C904ED89794A989399C690126FD609E019320E26";
    for(int i = 0; i < 32; i++) plot_id[i] = std::stoul(pid_hex.substr(i*2, 2), nullptr, 16);
    
    int num_test = 5;
    int ksize = 18;
    int MEM_HASH_ITER = 11337;
    int MEM_SIZE = 1024;
    
    std::cout << "=== gen_mem_array + calc_mem_hash Comparison ===" << std::endl;
    
    // CPU: compute key, gen_mem_array, calc_mem_hash for X=0
    for(int x = 0; x < num_test; x++) {
        uint32_t X_i = x;
        
        // Step 1: SHA-512 key
        uint32_t msg[9] = {};
        msg[0] = X_i;
        std::memcpy(msg + 1, plot_id, 32);
        mmx::hash_512_t key(&msg, sizeof(msg));
        
        // Step 2: gen_mem_array
        std::vector<uint32_t> mem(MEM_SIZE);
        mmx::pos::gen_mem_array(mem.data(), key.data(), MEM_SIZE);
        
        // Print first 32 uint32 of mem (first "row")
        std::cout << "X=" << x << " CPU mem[0..31]:" << std::endl;
        for(int i = 0; i < 32; i++) printf("%08x ", mem[i]);
        std::cout << std::endl;
        
        // Print last 32 uint32 of mem (last "row" = used as calc_mem_hash initial state)
        std::cout << "X=" << x << " CPU mem[992..1023]:" << std::endl;
        for(int i = 992; i < 1024; i++) printf("%08x ", mem[i]);
        std::cout << std::endl;
        
        // Step 3: calc_mem_hash
        uint8_t mem_hash[64 + 128] = {};
        std::memcpy(mem_hash, key.data(), 64);  // key first
        mmx::pos::calc_mem_hash(mem.data(), mem_hash + 64, MEM_HASH_ITER);
        
        // Print hash_state (32 uint32 = 128 bytes)
        uint32_t* hash_state = (uint32_t*)(mem_hash + 64);
        std::cout << "X=" << x << " CPU hash_state[0..31]:" << std::endl;
        for(int i = 0; i < 32; i++) printf("%08x ", hash_state[i]);
        std::cout << std::endl;
        
        // Step 4: final SHA-512(key || hash_state)
        mmx::hash_512_t final_hash(mem_hash, sizeof(mem_hash));
        uint32_t hash32[16];
        std::memcpy(hash32, final_hash.data(), 64);
        
        uint32_t kmask = ((uint64_t(1) << ksize) - 1);
        uint32_t Y = 0;
        for(int i = 0; i < 14; i++) Y ^= hash32[i];
        Y &= kmask;
        
        std::cout << "X=" << x << " CPU Y=" << std::hex << Y << std::dec << std::endl;
        std::cout << "X=" << x << " CPU M[0..13]: ";
        for(int i = 0; i < 14; i++) printf("%08x ", hash32[i] & kmask);
        std::cout << std::endl << std::endl;
    }
    
    // Now compute on GPU and compare
    // Load pos_recompute.cl and add a kernel that outputs mem[0..31] and hash_state
    std::ifstream f("pos_recompute.cl");
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    
    // Extract everything up to compute_f1_kernel
    size_t end = src.find("void compute_f1_kernel");
    std::string helper_code = src.substr(0, end);
    
    // Add test kernel that outputs intermediate results
    std::string kernel_code = helper_code + R"(

__kernel void test_memhash(
    __global const uint* X_in,
    __global const uint* ID_in,
    __global uint* mem_out,         // [num_x * 1024] mem array
    __global uint* hash_state_out,  // [num_x * 32] hash state after calc_mem_hash
    __global uint* Y_out,           // [num_x] Y value
    __global uint* M_out,           // [num_x * 14] M values
    const uint kmask,
    const uint xbits,
    const uint num_x)
{
    const uint gid = get_global_id(0);
    if(gid >= num_x) return;
    
    const uint X_i = X_in[gid];
    
    // Step 1: key = SHA-512(X_i || plot_id)
    uint msg32_key[18];
    for(int i = 0; i < 18; i++) msg32_key[i] = 0;
    msg32_key[0] = X_i;
    for(int i = 0; i < 8; i++) msg32_key[1 + i] = ID_in[i];
    
    ulong msg64_key[16];
    for(int i = 0; i < 16; i++) msg64_key[i] = 0;
    pack_uint32_to_be_ulong(msg32_key, 9, msg64_key, 16);
    
    ulong key_state[8];
    sha512_hash(msg64_key, 36, key_state);
    
    uint key32[16];
    extract_uint32_from_sha512(key_state, key32);
    
    // Step 2: gen_mem_array
    uint state[32];
    for(int i = 0; i < 16; i++) state[i] = key32[i];
    for(int i = 0; i < 16; i++) state[16 + i] = MEM_HASH_INIT[i];
    
    uint b = 0, c = 0;
    uint mem[1024];
    
    for(uint i = 0; i < 32; i++) {
        for(int j = 0; j < 4; j++) {
            for(int k = 0; k < 16; k++) {
                MMXPOS_HASHROUND(state[k], b, c, state[16 + k]);
            }
        }
        for(int k = 0; k < 32; k++) mem[i * 32 + k] = state[k];
    }
    
    // Output mem first and last rows
    for(int i = 0; i < 32; i++) mem_out[gid * 1024 + i] = mem[i];
    for(int i = 992; i < 1024; i++) mem_out[gid * 1024 + i] = mem[i];
    
    // Step 3: calc_mem_hash
    uint hash_state[32];
    for(int i = 0; i < 32; i++) hash_state[i] = mem[31 * 32 + i];
    
    for(int iter = 0; iter < MEM_HASH_ITER; iter++) {
        uint sum = 0;
        for(int i = 0; i < 32; i++) {
            sum += rotl32(hash_state[i], i % 32);
        }
        uint dir = sum + (sum << 11) + (sum << 22);
        uint bits = (dir >> 22) % 32;
        uint offset = (dir >> 27);
        
        for(int i = 0; i < 32; i++) {
            hash_state[i] += rotl32(mem[offset * 32 + (iter + i) % 32], bits) ^ sum;
        }
        for(int i = 0; i < 32; i++) {
            mem[offset * 32 + i] ^= hash_state[i];
        }
    }
    
    // Output hash_state
    for(int i = 0; i < 32; i++) hash_state_out[gid * 32 + i] = hash_state[i];
    
    // Step 4: final SHA-512(key || hash_state)
    uint msg32_final[48];
    for(int i = 0; i < 16; i++) msg32_final[i] = key32[i];
    for(int i = 0; i < 32; i++) msg32_final[16 + i] = hash_state[i];
    
    ulong msg64_final[32];
    for(int i = 0; i < 32; i++) msg64_final[i] = 0;
    pack_uint32_to_be_ulong(msg32_final, 48, msg64_final, 32);
    
    ulong final_state[8];
    sha512_hash(msg64_final, 192, final_state);
    
    uint hash32[16];
    extract_uint32_from_sha512(final_state, hash32);
    
    uint Y = 0;
    for(int i = 0; i < 14; i++) Y ^= hash32[i];
    Y &= kmask;
    
    Y_out[gid] = Y;
    for(int i = 0; i < 14; i++) M_out[gid * 14 + i] = hash32[i] & kmask;
}
)";
    
    // Setup OpenCL
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);
    
    const char* cstr = kernel_code.c_str();
    size_t len = kernel_code.size();
    cl_program program = clCreateProgramWithSource(context, 1, &cstr, &len, nullptr);
    cl_int err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2 -DMEM_HASH_ITER=11337", nullptr, nullptr);
    if(err != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << "Build failed: " << log << std::endl;
        return 1;
    }
    
    cl_kernel kernel = clCreateKernel(program, "test_memhash", &err);
    
    std::vector<uint32_t> X_values(num_test);
    for(int i = 0; i < num_test; i++) X_values[i] = i;
    std::vector<uint32_t> id_u32(8);
    std::memcpy(id_u32.data(), plot_id, 32);
    
    uint32_t kmask = ((uint64_t(1) << ksize) - 1);
    uint32_t num_x = num_test;
    uint32_t xbits = ksize;
    
    cl_mem X_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        num_test * 4, X_values.data(), nullptr);
    cl_mem ID_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        32, id_u32.data(), nullptr);
    cl_mem mem_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        num_test * 1024 * 4, nullptr, nullptr);
    cl_mem hs_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        num_test * 32 * 4, nullptr, nullptr);
    cl_mem Y_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        num_test * 4, nullptr, nullptr);
    cl_mem M_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        num_test * 14 * 4, nullptr, nullptr);
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &X_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &ID_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_buf);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &hs_buf);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &Y_buf);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &M_buf);
    clSetKernelArg(kernel, 6, sizeof(uint32_t), &kmask);
    clSetKernelArg(kernel, 7, sizeof(uint32_t), &xbits);
    clSetKernelArg(kernel, 8, sizeof(uint32_t), &num_x);
    
    size_t global = num_test;
    if(global % 64) global = ((global/64)+1)*64;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    
    std::vector<uint32_t> gpu_mem(num_test * 1024);
    std::vector<uint32_t> gpu_hs(num_test * 32);
    std::vector<uint32_t> gpu_Y(num_test);
    std::vector<uint32_t> gpu_M(num_test * 14);
    
    clEnqueueReadBuffer(queue, mem_buf, CL_TRUE, 0, num_test * 1024 * 4, gpu_mem.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, hs_buf, CL_TRUE, 0, num_test * 32 * 4, gpu_hs.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, Y_buf, CL_TRUE, 0, num_test * 4, gpu_Y.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, M_buf, CL_TRUE, 0, num_test * 14 * 4, gpu_M.data(), 0, nullptr, nullptr);
    
    std::cout << std::endl << "=== GPU Results ===" << std::endl;
    for(int x = 0; x < num_test; x++) {
        std::cout << "X=" << x << " GPU mem[0..31]:" << std::endl;
        for(int i = 0; i < 32; i++) printf("%08x ", gpu_mem[x * 1024 + i]);
        std::cout << std::endl;
        
        std::cout << "X=" << x << " GPU mem[992..1023]:" << std::endl;
        for(int i = 992; i < 1024; i++) printf("%08x ", gpu_mem[x * 1024 + i]);
        std::cout << std::endl;
        
        std::cout << "X=" << x << " GPU hash_state[0..31]:" << std::endl;
        for(int i = 0; i < 32; i++) printf("%08x ", gpu_hs[x * 32 + i]);
        std::cout << std::endl;
        
        std::cout << "X=" << x << " GPU Y=" << std::hex << gpu_Y[x] << std::dec << std::endl;
        std::cout << "X=" << x << " GPU M[0..13]: ";
        for(int i = 0; i < 14; i++) printf("%08x ", gpu_M[x * 14 + i]);
        std::cout << std::endl << std::endl;
    }
    
    // Cleanup
    clReleaseMemObject(X_buf); clReleaseMemObject(ID_buf);
    clReleaseMemObject(mem_buf); clReleaseMemObject(hs_buf);
    clReleaseMemObject(Y_buf); clReleaseMemObject(M_buf);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);
    
    return 0;
}
