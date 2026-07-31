// Verify PD tree by tracing entry 0 and checking X values
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
static uint64_t cdiv(uint64_t a, uint64_t b) { return (a+b-1)/b; }

using namespace mmx;
using namespace mmx::pos;

static uint64_t read_bits(const void* data, uint64_t bit_offset, uint64_t num_bits) {
    uint64_t result = 0;
    const uint8_t* bytes = (const uint8_t*)data;
    for(uint64_t i = 0; i < num_bits; i++) {
        uint64_t bp = bit_offset + i;
        if(bytes[bp / 8] & (1 << (bp % 8)))
            result |= (uint64_t(1) << i);
    }
    return result;
}

int main(int argc, char** argv) {
    if(argc < 2) return 1;
    vnx::init("verify_pd", argc, argv);
    mmx::secp256k1_init();
    
    std::string file_path = argv[1];
    auto prover = std::make_shared<pos::Prover>(file_path);
    auto header = prover->get_header();
    
    std::ifstream file(file_path, std::ios::binary);
    
    const int KSIZE = header->ksize;
    const int PARK_SIZE_PD = header->park_size_pd;
    const int PARK_BYTES_PD = header->park_bytes_pd;
    
    // Trace PD tree for entry 0
    std::vector<uint64_t> pointers = {0};
    std::vector<uint64_t> pd_park(cdiv(PARK_BYTES_PD, 8));
    
    int table = 9;
    for(const auto pd_offset : header->table_offset_pd) {
        std::vector<uint64_t> new_pointers;
        for(const auto index : pointers) {
            uint64_t park_index = index / PARK_SIZE_PD;
            uint32_t park_offset = index % PARK_SIZE_PD;
            file.seekg(pd_offset + park_index * PARK_BYTES_PD);
            file.read((char*)pd_park.data(), PARK_BYTES_PD);
            
            uint64_t position = read_bits(pd_park.data(), park_offset * KSIZE, KSIZE);
            auto offsets = mmx::pos::decode(pd_park, park_offset + 1, PARK_SIZE_PD * KSIZE);
            uint64_t P2 = position + offsets.back();
            
            std::cout << "T" << table << "->" << (table-1) << ": index=" << index
                      << " pos=" << position << " delta=" << offsets.back() 
                      << " P2=" << P2 << std::endl;
            new_pointers.push_back(position);
            new_pointers.push_back(P2);
        }
        pointers = new_pointers;
        table--;
    }
    
    // Read X values from X table
    std::cout << "\nFinal pointers (into X table): " << pointers.size() << std::endl;
    std::vector<uint32_t> X_values;
    std::vector<uint64_t> x_park(cdiv(header->park_bytes_x, 8));
    for(const auto index : pointers) {
        uint64_t park_index = index / header->park_size_x;
        uint32_t park_offset = index % header->park_size_x;
        file.seekg(header->table_offset_x + park_index * header->park_bytes_x);
        file.read((char*)x_park.data(), header->park_bytes_x);
        uint64_t LP = read_bits(x_park.data(), park_offset * header->entry_bits_x, header->entry_bits_x);
        
        auto pair = (header->xbits < header->ksize) ? 
            mmx::pos::LinePointToSquare2(LP) : mmx::pos::LinePointToSquare(LP);
        X_values.push_back(pair.first);
        X_values.push_back(pair.second);
    }
    
    std::cout << "X values: " << X_values.size() << std::endl;
    std::set<uint32_t> unique_x(X_values.begin(), X_values.end());
    std::cout << "Unique X values: " << unique_x.size() << std::endl;
    std::cout << "First 16 X: ";
    for(size_t i = 0; i < 16 && i < X_values.size(); i++) std::cout << X_values[i] << " ";
    std::cout << std::endl;
    
    // Try to verify
    try {
        auto res = pos::compute(X_values, nullptr, header->plot_id, header->ksize, header->ksize - header->xbits);
        std::cout << "\ncompute() returned " << res.size() << " entries" << std::endl;
        if(!res.empty()) {
            std::cout << "Y[0] = " << res[0].first << std::endl;
            std::cout << "Proof VALID!" << std::endl;
        }
    } catch(const std::exception& ex) {
        std::cout << "\ncompute() error: " << ex.what() << std::endl;
    }
    
    vnx::close();
    mmx::secp256k1_free();
    return 0;
}
