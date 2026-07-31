#include <mmx/pos/Prover.h>
#include <mmx/pos/verify.h>
#include <mmx/pos/encoding.h>
#include <mmx/PlotHeader.hxx>
#include <vnx/vnx.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
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
    if(argc < 2) { std::cerr << "Usage: " << argv[0] << " <plot.file>" << std::endl; return 1; }
    vnx::init("debug_plot", argc, argv);
    mmx::secp256k1_init();
    
    std::string file_path = argv[1];
    auto prover = std::make_shared<pos::Prover>(file_path);
    auto header = prover->get_header();
    
    std::ifstream file(file_path, std::ios::binary);
    
    // Read first few Y parks to check Y values
    std::cout << "=== Y Table ===" << std::endl;
    for(int p = 0; p < 3; p++) {
        file.seekg(header->table_offset_y + p * header->park_bytes_y);
        uint64_t tmp = 0;
        file.read((char*)&tmp, 4);
        uint32_t Y_first = read_bits(&tmp, 0, header->ksize);
        std::cout << "Y park " << p << ": first Y = " << Y_first << std::endl;
    }
    
    // Read first few X parks to check line points
    std::cout << "\n=== X Table ===" << std::endl;
    int num_x_entries = header->num_entries_y;  // WRONG? should be table 2 entries
    std::cout << "(header says num_entries_y = " << header->num_entries_y << ")" << std::endl;
    
    std::vector<uint64_t> x_park(cdiv(header->park_bytes_x, 8));
    for(int p = 0; p < 3; p++) {
        file.seekg(header->table_offset_x + p * header->park_bytes_x);
        file.read((char*)x_park.data(), header->park_bytes_x);
        for(int i = 0; i < 4; i++) {
            uint64_t LP = read_bits(x_park.data(), i * header->entry_bits_x, header->entry_bits_x);
            auto pair = (header->xbits < header->ksize) ? LinePointToSquare2(LP) : LinePointToSquare(LP);
            std::cout << "X park " << p << " entry " << i << ": LP=" << LP 
                      << " → (" << pair.first << ", " << pair.second << ")" << std::endl;
        }
    }
    
    // Read PD table 0 (for table 9→8) first park
    std::cout << "\n=== PD Table 0 (table 9→8) ===" << std::endl;
    std::vector<uint64_t> pd_park(cdiv(header->park_bytes_pd, 8));
    file.seekg(header->table_offset_pd[0]);
    file.read((char*)pd_park.data(), header->park_bytes_pd);
    for(int i = 0; i < 4; i++) {
        uint64_t pos = read_bits(pd_park.data(), i * header->ksize, header->ksize);
        std::cout << "PD entry " << i << ": position = " << pos << std::endl;
    }
    
    // Read last PD table (for table 3→2) first park
    std::cout << "\n=== PD Table 6 (table 3→2) ===" << std::endl;
    file.seekg(header->table_offset_pd[6]);
    file.read((char*)pd_park.data(), header->park_bytes_pd);
    for(int i = 0; i < 4; i++) {
        uint64_t pos = read_bits(pd_park.data(), i * header->ksize, header->ksize);
        std::cout << "PD entry " << i << ": position = " << pos << std::endl;
    }
    
    // Try to trace full proof for entry 0
    std::cout << "\n=== Full Proof Trace for entry 0 ===" << std::endl;
    try {
        auto proof = prover->get_full_proof(0);
        std::cout << "Proof valid: " << proof.valid << std::endl;
        std::cout << "X values: ";
        for(size_t i = 0; i < std::min(proof.proof.size(), (size_t)16); i++)
            std::cout << proof.proof[i] << " ";
        std::cout << "..." << std::endl;
    } catch(const std::exception& ex) {
        std::cout << "Error: " << ex.what() << std::endl;
    }
    
    vnx::close();
    mmx::secp256k1_free();
    return 0;
}
