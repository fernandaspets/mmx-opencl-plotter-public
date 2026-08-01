// test_f1_cpu_vs_gpu.cpp — Compare F1 output: GPU vs CPU reference
// Build: manually compile and link with mmx-node pos library
#include <cstdint>
#include <vector>
#include <cstring>
#include <iostream>

// We'll test by running the plotter with a known plot_id and dumping F1[0]
// Then compute F1[0] on CPU using mmx-node's compute_f1

int main() {
    // Test plot_id (random but fixed)
    uint8_t plot_id_bytes[32] = {
        0x5b, 0xfa, 0x94, 0x60, 0xe1, 0xc8, 0xf4, 0x15,
        0x8a, 0x71, 0x1f, 0xe8, 0x30, 0xb7, 0x75, 0x9f,
        0x0a, 0x37, 0xa5, 0x9c, 0xf6, 0xca, 0x5c, 0xaa,
        0x57, 0x51, 0x56, 0x93, 0x60, 0xfc, 0xe4, 0x79
    };
    
    // X_i = 0, ksize = 18, xbits = 18
    uint32_t X_i = 0;
    int ksize = 18;
    int xbits = 18;
    
    // CPU F1 computation (from mmx-node/src/pos/verify.cpp compute_f1)
    uint32_t kmask = ((uint64_t(1) << ksize) - 1);
    
    uint32_t msg[9] = {};
    msg[0] = X_i;
    std::memcpy(msg + 1, plot_id_bytes, 32);
    
    // Need SHA-512 and mem_hash from mmx-node
    // For now, just print the msg to verify byte layout
    std::cout << "msg bytes (hex): ";
    for(int i = 0; i < 9; i++) {
        uint8_t* p = (uint8_t*)&msg[i];
        for(int b = 0; b < 4; b++) printf("%02x", p[b]);
        std::cout << " ";
    }
    std::cout << std::endl;
    
    // Expected: 00000000 (X_i=0 LE) || plot_id bytes LE
    std::cout << "plot_id bytes (hex): ";
    for(int i = 0; i < 32; i++) printf("%02x", plot_id_bytes[i]);
    std::cout << std::endl;
    
    return 0;
}
