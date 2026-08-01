// test_hash_cpu_vs_gpu.cpp — Verify GPU SHA-512 hash matches CPU
// Build: g++ -o test_hash test_hash_cpu_vs_gpu.cpp -lOpenCL -I~/mmx-node/include
// Run: ./test_hash

#include <CL/cl.h>
#include <cstdint>
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include "../../../mmx-node/include/mmx/pos/encoding.h"
#include "../../../mmx-node/include/mmx/hash_512_t.h"

// CPU reference: hash L_meta || R_meta using MMX SHA-512
void cpu_hash(const uint32_t L[14], const uint32_t R[14], uint32_t& Y, uint32_t M[14], uint32_t kmask) {
    uint32_t msg[28];
    for(int i = 0; i < 14; i++) { msg[i] = L[i]; msg[14+i] = R[i]; }
    
    mmx::hash_512_t hash(&msg, sizeof(msg));
    
    uint32_t hash32[16];
    for(int i = 0; i < 16; i++) {
        memcpy(&hash32[i], hash.data() + i * 4, 4);
    }
    
    Y = 0;
    for(int i = 0; i < 14; i++) Y ^= hash32[i];
    Y &= kmask;
    
    for(int i = 0; i < 14; i++) M[i] = hash32[i] & kmask;
}

int main() {
    // Test data: two entries with known metadata
    uint32_t kmask = (1u << 18) - 1; // k18
    uint32_t L[14] = {0x12345, 0xABCDE, 0x111, 0x222, 0x333, 0x444, 0x555, 0x666, 0x777, 0x888, 0x999, 0xAAA, 0xBBB, 0xCCC};
    uint32_t R[14] = {0x77777, 0x22222, 0x333, 0x444, 0x555, 0x666, 0x777, 0x888, 0x999, 0xAAA, 0xBBB, 0xCCC, 0xDDD, 0xEEE};
    
    // CPU reference
    uint32_t cpu_Y, cpu_M[14];
    cpu_hash(L, R, cpu_Y, cpu_M, kmask);
    
    std::cout << "CPU Y: " << std::hex << cpu_Y << std::endl;
    std::cout << "CPU M[0]: " << cpu_M[0] << " M[1]: " << cpu_M[1] << std::endl;
    
    // TODO: Run GPU kernel and compare
    std::cout << "\nGPU comparison not implemented yet." << std::endl;
    std::cout << "This test verifies the CPU hash reference." << std::endl;
    
    return 0;
}
