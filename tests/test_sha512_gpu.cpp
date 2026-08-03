#include "../src/gpu_device.h"
#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>
#include <openssl/sha.h>

// Inline SHA-512 kernel for testing
// We copy exactly what table_hash.cl and f1.cl use
const char* test_kernel_src = R"(
__constant ulong SHA512_K[80] = {
    0x428a2f98d728ae22UL, 0x7137449123ef65cdUL, 0xb5c0fbcfec4d3b2fUL, 0xe9b5dba58189dbbcUL,
    0x3956c25bf348b538UL, 0x59f111f1b605d019UL, 0x923f82a4af194f9bUL, 0xab1c5ed5da6d8118UL,
    0xd807aa98a3030242UL, 0x12835b0145706fbeUL, 0x243185be4ee4b28cUL, 0x550c7dc3d5ffb4e2UL,
    0x72be5d74f27b896fUL, 0x80deb1fe3b1696b1UL, 0x9bdc06a725c71235UL, 0xc19bf174cf692694UL,
    0xe49b69c19ef14ad2UL, 0xefbe4786384f25e3UL, 0x0fc19dc68b8cd5b5UL, 0x240ca1cc77ac9c65UL,
    0x2de92c6f592b0275UL, 0x4a7484aa6ea6e483UL, 0x5cb0a9dcbd41fbd4UL, 0x76f988da831153b5UL,
    0x983e5152ee66dfabUL, 0xa831c66d2db43210UL, 0xb00327c898fb213fUL, 0xbf597fc7beef0ee4UL,
    0xc6e00bf33da88fc2UL, 0xd5a79147930aa725UL, 0x06ca6351e003826fUL, 0x142929670a0e6e70UL,
    0x27b70a8546d22ffcUL, 0x2e1b21385c26c926UL, 0x4d2c6dfc5ac42aedUL, 0x53380d139d95b3dfUL,
    0x650a73548baf63deUL, 0x766a0abb3c77b2a8UL, 0x81c2c92e47edaee6UL, 0x92722c851482353bUL,
    0xa2bfe8a14cf10364UL, 0xa81a664bbc423001UL, 0xc24b8b70d0f89791UL, 0xc76c51a30654be30UL,
    0xd192e819d6ef5218UL, 0xd69906245565a910UL, 0xf40e35855771202aUL, 0x106aa07032bbd1b8UL,
    0x19a4c116b8d2d0c8UL, 0x1e376c085141ab53UL, 0x2748774cdf8eeb99UL, 0x34b0bcb5e19b48a8UL,
    0x391c0cb3c5c95a63UL, 0x4ed8aa4ae3418acbUL, 0x5b9cca4f7763e373UL, 0x682e6ff3d6b2b8a3UL,
    0x748f82ee5defb2fcUL, 0x78a5636f43172f60UL, 0x84c87814a1f0ab72UL, 0x8cc702081a6439ecUL,
    0x90befffa23631e28UL, 0xa4506cebde82bde9UL, 0xbef9a3f7b2c67915UL, 0xc67178f2e372532bUL,
    0xca273eceea26619cUL, 0xd186b8c721c0c207UL, 0xeada7dd6cde0eb1eUL, 0xf57d4f7fee6ed178UL,
    0x06f067aa72176fbaUL, 0x0a637dc5a2c898a6UL, 0x113f9804bef90daeUL, 0x1b710b35131c471bUL,
    0x28db77f523047d84UL, 0x32caab7b40c72493UL, 0x3c9ebe0a15c9bebcUL, 0x431d67c49c100d4cUL,
    0x4cc5d4becb3e42b6UL, 0x597f299cfc657e2aUL, 0x5fcb6fab3ad6faecUL, 0x6c44198c4a475817UL
};
__constant ulong SHA512_INIT[8] = {
    0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL, 0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
    0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL, 0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL
};
ulong rotr64(ulong x, int n) { return (x >> n) | (x << (64 - n)); }
void sha512_block(__private ulong* msg, __private ulong* state) {
    __private ulong w[80];
    for(int i = 0; i < 16; i++) w[i] = msg[i];
    for(int i = 16; i < 80; i++) {
        ulong s0 = rotr64(w[i-15], 1) ^ rotr64(w[i-15], 8) ^ (w[i-15] >> 7);
        ulong s1 = rotr64(w[i-2], 19) ^ rotr64(w[i-2], 61) ^ (w[i-2] >> 6);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    ulong a=state[0], b=state[1], c=state[2], d=state[3];
    ulong e=state[4], f=state[5], g=state[6], h=state[7];
    for(int i = 0; i < 80; i++) {
        ulong S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
        ulong ch = (e & f) ^ ((~e) & g);
        ulong t1 = h + S1 + ch + SHA512_K[i] + w[i];
        ulong S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
        ulong maj = (a & b) ^ (a & c) ^ (b & c);
        ulong t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

// Direct SHA-512 test: hash a simple 112-byte message (same as table hash)
__kernel void hash_112bytes(
    __global const ulong* data_in,   // 14 ulong = 112 bytes (BE representation)
    __global ulong* state_out)       // 8 ulong output state
{
    __private ulong msg[16];
    // msg[0..13] = data (14 big-endian ulong = 112 bytes)
    for(int i = 0; i < 14; i++) msg[i] = data_in[i];
    // msg[14] = padding start: 0x80 followed by zeros
    msg[14] = 0x8000000000000000UL;
    msg[15] = 0;
    
    __private ulong state[8];
    for(int i = 0; i < 8; i++) state[i] = SHA512_INIT[i];
    sha512_block(msg, state);
    
    // Block 2: zeros + bit length at end
    __private ulong block2[16];
    for(int i = 0; i < 16; i++) block2[i] = 0;
    block2[15] = (ulong)112 * 8;  // bit length = 896
    
    sha512_block(block2, state);
    
    for(int i = 0; i < 8; i++) state_out[i] = state[i];
}
)";

int main() {
    mmx::GPUDevice gpu;
    gpu.init(0);
    
    // Create a 112-byte message like the table hash uses:
    // 28 uint32s: L_meta[0..13] = 0,1,2,...,13; R_meta[0..13] = 100..113
    uint32_t msg32[28];
    for(int i = 0; i < 14; i++) msg32[i] = i;
    for(int i = 0; i < 14; i++) msg32[14+i] = 100 + i;
    
    // Convert to big-endian ulong (same as kernel: two LE uint32 → one BE ulong)
    uint64_t msg_be[14];
    for(int i = 0; i < 14; i++) {
        uint64_t lo = (uint64_t)msg32[i*2];
        uint64_t hi = (uint64_t)msg32[i*2+1];
        uint64_t le_val = lo | (hi << 32);
        // byte-swap
        msg_be[i] = ((le_val & 0xFF00000000000000ULL) >> 56)
                  | ((le_val & 0x00FF000000000000ULL) >> 40)
                  | ((le_val & 0x0000FF0000000000ULL) >> 24)
                  | ((le_val & 0x000000FF00000000ULL) >> 8)
                  | ((le_val & 0x00000000FF000000ULL) << 8)
                  | ((le_val & 0x0000000000FF0000ULL) << 24)
                  | ((le_val & 0x000000000000FF00ULL) << 40)
                  | ((le_val & 0x00000000000000FFULL) << 56);
    }
    
    // CPU reference
    uint8_t cpu_hash[64];
    SHA512((uint8_t*)msg32, 112, cpu_hash);
    
    std::cout << "CPU SHA-512 of 112-byte message:" << std::endl;
    for(int i = 0; i < 8; i++) {
        uint64_t cpu_state = 0;
        for(int j = 0; j < 8; j++) cpu_state = (cpu_state << 8) | cpu_hash[i*8 + j];
        std::cout << "  state[" << i << "] = 0x" << std::hex << cpu_state << std::dec << std::endl;
    }
    
    // GPU test
    cl_int err;
    cl_program prog = clCreateProgramWithSource(gpu.context, 1, &test_kernel_src, nullptr, &err);
    mmx::GPUDevice::check(err, "prog");
    err = clBuildProgram(prog, 1, &gpu.device, "", nullptr, nullptr);
    if(err != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(prog, gpu.device, CL_PROGRAM_BUILD_LOG, 8192, log, nullptr);
        std::cerr << "Build: " << log << std::endl;
        return 1;
    }
    cl_kernel ker = clCreateKernel(prog, "hash_112bytes", &err);
    mmx::GPUDevice::check(err, "kernel");
    
    cl_mem msg_buf = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 14*8, msg_be, &err);
    mmx::GPUDevice::check(err, "msg buf");
    cl_mem state_buf = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY, 8*8, nullptr, &err);
    mmx::GPUDevice::check(err, "state buf");
    
    clSetKernelArg(ker, 0, sizeof(cl_mem), &msg_buf);
    clSetKernelArg(ker, 1, sizeof(cl_mem), &state_buf);
    
    size_t gs = 1;
    err = clEnqueueNDRangeKernel(gpu.queue, ker, 1, nullptr, &gs, nullptr, 0, nullptr, nullptr);
    mmx::GPUDevice::check(err, "enq");
    gpu.finish();
    
    uint64_t gpu_state[8];
    clEnqueueReadBuffer(gpu.queue, state_buf, CL_TRUE, 0, 8*8, gpu_state, 0, nullptr, nullptr);
    
    std::cout << "\nGPU SHA-512 state:" << std::endl;
    for(int i = 0; i < 8; i++) {
        std::cout << "  state[" << i << "] = 0x" << std::hex << gpu_state[i] << std::dec << std::endl;
    }
    
    clReleaseMemObject(msg_buf);
    clReleaseMemObject(state_buf);
    clReleaseKernel(ker);
    clReleaseProgram(prog);
    return 0;
}
