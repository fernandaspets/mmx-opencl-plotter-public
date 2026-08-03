// PID screening: test if k18 proof count predicts k25 proof count
#include <CL/cl.h>
#include <vnx/vnx.h>
#include <mmx/PlotHeader.hxx>
#include <mmx/hash_t.hpp>
#include <mmx/hash_512_t.hpp>
#include <mmx/pos/encoding.h>
#include <mmx/pos/config.h>
#include <mmx/pos/util.h>
#include <mmx/utils.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <random>
#include <algorithm>

// We need to call the plotter functions. Let's just use the binary directly
// and parse the output for proof counts.

int main(int argc, char** argv) {
    // Generate N random seeds, derive PIDs for k18 and k25, run plotter, count proofs
    int N = 20; // number of seeds to test
    if(argc > 1) N = atoi(argv[1]);
    
    std::string farmer_key_hex = "02292cd11aa18e5f64344cbe6c580249364dfe5a3683adc25446aadcc1b38555d7";
    // Parse farmer key
    std::vector<uint8_t> farmer_key(48);
    for(int i = 0; i < 48; i++) {
        farmer_key[i] = std::stoul(farmer_key_hex.substr(i*2, 2), nullptr, 16);
    }
    
    std::cout << "Generating " << N << " random seeds and testing correlation..." << std::endl;
    std::cout << "seed_hex,k18_pid,k18_proofs,k25_pid,k25_proofs" << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    
    for(int n = 0; n < N; n++) {
        // Generate random seed
        std::array<uint8_t, 32> seed;
        for(int i = 0; i < 32; i++) seed[i] = dis(gen);
        
        // Derive k18 PID
        {
            uint8_t buf[1024] = {};
            uint32_t offset = 0;
            const std::string tag("MMX/PLOTID/OG");
            std::memcpy(buf + offset, tag.data(), tag.size()); offset += tag.size();
            const uint8_t ksize = 18;
            std::memcpy(buf + offset, &ksize, 1); offset += 1;
            std::memcpy(buf + offset, seed.data(), seed.size()); offset += seed.size();
            std::memcpy(buf + offset, farmer_key.data(), farmer_key.size()); offset += farmer_key.size();
            auto hash = mmx::hash_t(buf, offset);
            std::string pid_hex = vnx::to_hex_string(hash.data(), hash.size());
            std::cout << vnx::to_hex_string(seed.data(), seed.size()) << "," << pid_hex;
        }
        
        // Derive k25 PID
        {
            uint8_t buf[1024] = {};
            uint32_t offset = 0;
            const std::string tag("MMX/PLOTID/OG");
            std::memcpy(buf + offset, tag.data(), tag.size()); offset += tag.size();
            const uint8_t ksize = 25;
            std::memcpy(buf + offset, &ksize, 1); offset += 1;
            std::memcpy(buf + offset, seed.data(), seed.size()); offset += seed.size();
            std::memcpy(buf + offset, farmer_key.data(), farmer_key.size()); offset += farmer_key.size();
            auto hash = mmx::hash_t(buf, offset);
            std::string pid_hex = vnx::to_hex_string(hash.data(), hash.size());
            std::cout << "," << pid_hex;
        }
        std::cout << std::endl;
    }
    return 0;
}
