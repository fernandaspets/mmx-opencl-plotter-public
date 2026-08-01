// test_sha512_step.cpp — Compare SHA-512(X_i || plot_id) between CPU and GPU
// This isolates whether the SHA-512 implementation itself is correct.

#include <CL/cl.h>
#include <cstdint>
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <mmx/hash_512_t.hpp>

int main() {
    // Test plot ID
    uint8_t plot_id[32];
    std::string pid_hex = "89AFB708A213123203000F47C904ED89794A989399C690126FD609E019320E26";
    for(int i = 0; i < 32; i++) plot_id[i] = std::stoul(pid_hex.substr(i*2, 2), nullptr, 16);
    
    uint32_t X_i = 0;
    
    // CPU: key = SHA-512(X_i || plot_id) = SHA-512 of 36 bytes
    uint32_t msg[9] = {};
    msg[0] = X_i;
    std::memcpy(msg + 1, plot_id, 32);
    
    mmx::hash_512_t cpu_key(&msg, sizeof(msg));
    
    std::cout << "=== CPU SHA-512 key for X=0 ===" << std::endl;
    std::cout << "Input bytes (hex): ";
    uint8_t* p = (uint8_t*)msg;
    for(int i = 0; i < 36; i++) printf("%02x", p[i]);
    std::cout << std::endl;
    
    std::cout << "Key (hex): ";
    for(int i = 0; i < 64; i++) printf("%02x", cpu_key.data()[i]);
    std::cout << std::endl;
    
    // Print as uint32 array (how the kernel sees it)
    uint32_t key32[16];
    std::memcpy(key32, cpu_key.data(), 64);
    std::cout << "Key as uint32: ";
    for(int i = 0; i < 16; i++) printf("%08x ", key32[i]);
    std::cout << std::endl;
    
    // Now compute on GPU using our kernel's SHA-512
    // We need a minimal kernel that just does SHA-512 and outputs the key
    // For now, let me check the byte packing
    
    std::cout << std::endl;
    std::cout << "=== Byte layout analysis ===" << std::endl;
    std::cout << "msg[0] (X_i) = " << std::hex << msg[0] << std::dec << std::endl;
    std::cout << "msg bytes [0..3] (X_i LE): ";
    for(int i = 0; i < 4; i++) printf("%02x ", p[i]);
    std::cout << std::endl;
    std::cout << "msg bytes [4..35] (plot_id): ";
    for(int i = 4; i < 36; i++) printf("%02x", p[i]);
    std::cout << std::endl;
    
    // Our GPU kernel does:
    // msg32_key[0] = X_i; msg32_key[1..8] = ID_in[0..7]
    // ID_in[i] = uint32 from memcpy of plot_id
    // pack_uint32_to_be_ulong(msg32_key, 9, msg64_key, 16)
    
    std::vector<uint32_t> id_u32(8);
    std::memcpy(id_u32.data(), plot_id, 32);
    std::cout << std::endl;
    std::cout << "GPU ID_in uint32: ";
    for(int i = 0; i < 8; i++) printf("%08x ", id_u32[i]);
    std::cout << std::endl;
    
    // msg32_key = [X_i, id_u32[0], ..., id_u32[7]] = same as CPU msg
    // So the input should be the same. The issue is in pack_uint32_to_be_ulong
    // or in the SHA-512 implementation itself.
    
    return 0;
}
