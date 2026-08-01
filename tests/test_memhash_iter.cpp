// test_memhash_iter.cpp — Compare calc_mem_hash after N iterations (1, 10, 100)
// to find exactly when the divergence starts.

#include <CL/cl.h>
#include <cstdint>
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <mmx/hash_512_t.hpp>
#include <mmx/pos/mem_hash.h>

// CPU calc_mem_hash modified to output state after N iterations
void cpu_calc_mem_hash_n(uint32_t* mem, uint32_t* state_out, int num_iter) {
    const int N = 32;
    uint32_t state[N];
    for(int i = 0; i < N; i++) state[i] = mem[(N-1)*N + i];
    
    for(int iter = 0; iter < num_iter; iter++) {
        uint32_t sum = 0;
        for(int i = 0; i < N; i++) sum += mmx::pos::rotl_32(state[i], i % 32);
        
        uint32_t dir = sum + (sum << 11) + (sum << 22);
        uint32_t bits = (dir >> 22) % 32u;
        uint32_t offset = (dir >> 27);
        
        for(int i = 0; i < N; i++)
            state[i] += mmx::pos::rotl_32(mem[offset * N + (iter + i) % N], bits) ^ sum;
        for(int i = 0; i < N; i++)
            mem[offset * N + i] ^= state[i];
    }
    
    std::memcpy(state_out, state, N * 4);
}

int main() {
    uint8_t plot_id[32];
    std::string pid_hex = "89AFB708A213123203000F47C904ED89794A989399C690126FD609E019320E26";
    for(int i = 0; i < 32; i++) plot_id[i] = std::stoul(pid_hex.substr(i*2, 2), nullptr, 16);
    
    // CPU: compute key and mem for X=0
    uint32_t X_i = 0;
    uint32_t msg[9] = {};
    msg[0] = X_i;
    std::memcpy(msg + 1, plot_id, 32);
    mmx::hash_512_t key(&msg, sizeof(msg));
    
    const int MEM_SIZE = 1024;
    
    // Test for 1, 10, 100, 1000 iterations
    for(int num_iter : {1, 10, 100, 1000, 11337}) {
        // Fresh mem for each test
        std::vector<uint32_t> mem(MEM_SIZE);
        mmx::pos::gen_mem_array(mem.data(), key.data(), MEM_SIZE);
        
        uint32_t state[32];
        cpu_calc_mem_hash_n(mem.data(), state, num_iter);
        
        std::cout << "CPU iter=" << num_iter << " state[0..7]: ";
        for(int i = 0; i < 8; i++) printf("%08x ", state[i]);
        std::cout << std::endl;
    }
    
    // Now GPU: load kernel, run with different iteration counts
    // We need a kernel that takes num_iter as a parameter
    std::ifstream f("pos_recompute.cl");
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t end = src.find("void compute_f1_kernel");
    std::string helper = src.substr(0, end);
    
    // Create a test kernel with num_iter as parameter
    std::string kernel_code = helper + R"(
__kernel void test_memhash_iter(
    __global const uint* ID_in,
    __global uint* state_out,
    const uint num_iter,
    const uint x_value)
{
    const uint X_i = x_value;
    
    // SHA-512 key
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
    
    // gen_mem_array
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
    
    // calc_mem_hash with variable iterations
    uint hash_state[32];
    for(int i = 0; i < 32; i++) hash_state[i] = mem[31 * 32 + i];
    
    for(uint iter = 0; iter < num_iter; iter++) {
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
    
    for(int i = 0; i < 32; i++) state_out[i] = hash_state[i];
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
    
    cl_kernel kernel = clCreateKernel(program, "test_memhash_iter", &err);
    
    std::vector<uint32_t> id_u32(8);
    std::memcpy(id_u32.data(), plot_id, 32);
    
    cl_mem ID_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        32, id_u32.data(), nullptr);
    cl_mem state_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        32 * 4, nullptr, nullptr);
    
    std::cout << std::endl;
    for(int num_iter : {1, 10, 100, 1000, 11337}) {
        uint32_t ni = num_iter;
        uint32_t xv = 0;
        
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &ID_buf);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &state_buf);
        clSetKernelArg(kernel, 2, sizeof(uint32_t), &ni);
        clSetKernelArg(kernel, 3, sizeof(uint32_t), &xv);
        
        size_t global = 1;
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        
        uint32_t gpu_state[32];
        clEnqueueReadBuffer(queue, state_buf, CL_TRUE, 0, 32 * 4, gpu_state, 0, nullptr, nullptr);
        
        std::cout << "GPU iter=" << num_iter << " state[0..7]: ";
        for(int i = 0; i < 8; i++) printf("%08x ", gpu_state[i]);
        std::cout << std::endl;
    }
    
    clReleaseMemObject(ID_buf);
    clReleaseMemObject(state_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return 0;
}
