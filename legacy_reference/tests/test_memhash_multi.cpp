// test_memhash_multi.cpp — Run calc_mem_hash with multiple work-items
// to check for race conditions in private memory.

#include <CL/cl.h>
#include <cstdint>
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <mmx/hash_512_t.hpp>
#include <mmx/pos/mem_hash.h>

int main() {
    uint8_t plot_id[32];
    std::string pid_hex = "89AFB708A213123203000F47C904ED89794A989399C690126FD609E019320E26";
    for(int i = 0; i < 32; i++) plot_id[i] = std::stoul(pid_hex.substr(i*2, 2), nullptr, 16);
    
    int num_test = 10;
    int num_iter = 11337;
    
    // CPU reference
    std::cout << "=== CPU ===" << std::endl;
    for(int x = 0; x < num_test; x++) {
        uint32_t msg[9] = {};
        msg[0] = x;
        std::memcpy(msg + 1, plot_id, 32);
        mmx::hash_512_t key(&msg, sizeof(msg));
        
        std::vector<uint32_t> mem(1024);
        mmx::pos::gen_mem_array(mem.data(), key.data(), 1024);
        
        uint32_t state[32];
        for(int i = 0; i < 32; i++) state[i] = mem[31*32 + i];
        
        for(int iter = 0; iter < num_iter; iter++) {
            uint32_t sum = 0;
            for(int i = 0; i < 32; i++) sum += mmx::pos::rotl_32(state[i], i % 32);
            uint32_t dir = sum + (sum << 11) + (sum << 22);
            uint32_t bits = (dir >> 22) % 32u;
            uint32_t offset = (dir >> 27);
            for(int i = 0; i < 32; i++)
                state[i] += mmx::pos::rotl_32(mem[offset * 32 + (iter + i) % 32], bits) ^ sum;
            for(int i = 0; i < 32; i++)
                mem[offset * 32 + i] ^= state[i];
        }
        
        printf("X=%d CPU: %08x %08x %08x %08x\n", x, state[0], state[1], state[2], state[3]);
    }
    
    // GPU with multiple work-items
    std::ifstream f("pos_recompute.cl");
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t end = src.find("void compute_f1_kernel");
    std::string helper = src.substr(0, end);
    
    std::string kernel_code = helper + R"(
__kernel void test_multi(
    __global const uint* X_in,
    __global const uint* ID_in,
    __global uint* state_out,
    const uint num_iter,
    const uint num_x)
{
    const uint gid = get_global_id(0);
    if(gid >= num_x) return;
    
    const uint X_i = X_in[gid];
    
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
    
    uint hash_state[32];
    for(int i = 0; i < 32; i++) hash_state[i] = mem[31 * 32 + i];
    
    for(uint iter = 0; iter < num_iter; iter++) {
        uint sum = 0;
        for(int i = 0; i < 32; i++) sum += rotl32(hash_state[i], i % 32);
        uint dir = sum + (sum << 11) + (sum << 22);
        uint bits = (dir >> 22) % 32;
        uint offset = (dir >> 27);
        for(int i = 0; i < 32; i++)
            hash_state[i] += rotl32(mem[offset * 32 + (iter + i) % 32], bits) ^ sum;
        for(int i = 0; i < 32; i++)
            mem[offset * 32 + i] ^= hash_state[i];
    }
    
    for(int i = 0; i < 32; i++) state_out[gid * 32 + i] = hash_state[i];
}
)";
    
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);
    
    const char* cstr = kernel_code.c_str();
    size_t len = kernel_code.size();
    cl_program program = clCreateProgramWithSource(context, 1, &cstr, &len, nullptr);
    cl_int err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
    if(err != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << "Build failed: " << log << std::endl;
        return 1;
    }
    
    cl_kernel kernel = clCreateKernel(program, "test_multi", &err);
    
    std::vector<uint32_t> X_values(num_test);
    for(int i = 0; i < num_test; i++) X_values[i] = i;
    std::vector<uint32_t> id_u32(8);
    std::memcpy(id_u32.data(), plot_id, 32);
    
    cl_mem X_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        num_test * 4, X_values.data(), nullptr);
    cl_mem ID_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        32, id_u32.data(), nullptr);
    cl_mem state_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        num_test * 32 * 4, nullptr, nullptr);
    
    uint32_t ni = num_iter;
    uint32_t nx = num_test;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &X_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &ID_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &state_buf);
    clSetKernelArg(kernel, 3, sizeof(uint32_t), &ni);
    clSetKernelArg(kernel, 4, sizeof(uint32_t), &nx);
    
    // Test with different work-group sizes
    for(int wg_size : {1, 4, 10, 64}) {
        size_t global = num_test;
        if(global % wg_size) global = ((global / wg_size) + 1) * wg_size;
        size_t local = wg_size;
        
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        
        std::vector<uint32_t> gpu_state(num_test * 32);
        clEnqueueReadBuffer(queue, state_buf, CL_TRUE, 0, num_test * 32 * 4, gpu_state.data(), 0, nullptr, nullptr);
        
        std::cout << std::endl << "=== GPU (work-group=" << wg_size << ") ===" << std::endl;
        int mismatches = 0;
        for(int x = 0; x < num_test; x++) {
            printf("X=%d GPU: %08x %08x %08x %08x\n", x, 
                   gpu_state[x*32], gpu_state[x*32+1], gpu_state[x*32+2], gpu_state[x*32+3]);
        }
    }
    
    clReleaseMemObject(X_buf);
    clReleaseMemObject(ID_buf);
    clReleaseMemObject(state_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return 0;
}
