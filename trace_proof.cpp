#include <mmx/pos/Prover.h>
#include <mmx/pos/verify.h>
#include <mmx/pos/encoding.h>
#include <mmx/PlotHeader.hxx>
#include <vnx/vnx.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>

using namespace mmx;
using namespace mmx::pos;

int main(int argc, char** argv) {
    if(argc < 2) return 1;
    vnx::init("trace", argc, argv);
    mmx::secp256k1_init();
    
    auto prover = std::make_shared<pos::Prover>(argv[1]);
    auto header = prover->get_header();
    
    // Get qualities for a known challenge
    hash_t challenge(header->plot_id + "0");
    auto qualities = prover->get_qualities(challenge, 4);
    
    std::cout << "Found " << qualities.size() << " qualities" << std::endl;
    
    for(const auto& q : qualities) {
        if(!q.valid) {
            std::cout << "  entry " << q.index << ": invalid (" << q.error_msg << ")" << std::endl;
            continue;
        }
        std::cout << "  entry " << q.index << ": valid, meta=" << q.meta.to_string().substr(0,16) << "..." << std::endl;
        
        // Get full proof
        try {
            auto proof = prover->get_full_proof(q.index);
            std::cout << "    proof valid: " << proof.valid << std::endl;
            if(!proof.proof.empty()) {
                // Check for duplicates
                std::set<uint32_t> unique_x(proof.proof.begin(), proof.proof.end());
                std::cout << "    X values: " << proof.proof.size() 
                          << ", unique: " << unique_x.size() << std::endl;
                
                // Try to verify
                try {
                    pos::verify(proof.proof, challenge, header->plot_id, 4, 0, header->ksize, true);
                    std::cout << "    VERIFY: PASS" << std::endl;
                } catch(const std::exception& ex) {
                    std::cout << "    VERIFY: FAIL - " << ex.what() << std::endl;
                }
                
                // Show first few X values
                std::cout << "    First 16 X: ";
                for(size_t i = 0; i < 16; i++) std::cout << proof.proof[i] << " ";
                std::cout << std::endl;
            }
        } catch(const std::exception& ex) {
            std::cout << "    get_full_proof error: " << ex.what() << std::endl;
        }
    }
    
    vnx::close();
    mmx::secp256k1_free();
    return 0;
}
