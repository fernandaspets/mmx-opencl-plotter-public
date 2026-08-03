// Test: pid_derive.h — seed generation and plot_id derivation
#include "../src/pid_derive.h"
#include <cassert>
#include <iostream>
#include <mmx/utils.h>

int main() {
    std::cout << "=== test_pid_derive ===" << std::endl;

    mmx::secp256k1_init();
    { int argc = 0; char** argv = nullptr; vnx::init("test_pid_derive", argc, argv); }

    // Test 1: Derive a plot_id with known inputs
    mmx::hash_t seed;
    memset(seed.data(), 0x42, seed.size());  // all 0x42

    mmx::pubkey_t farmer_key;
    const std::string fk_hex = "02292cd11aa18e5f64344cbe6c580249364dfe5a3683adc25446aadcc1b38555d7";
    farmer_key.from_string(fk_hex);

    // OG plot (not NFT)
    auto pid_og = mmx::derive_plot_id(seed, 29, farmer_key, false);
    std::cout << "  OG plot_id (k29): " << pid_og.to_string() << std::endl;
    assert(pid_og != mmx::hash_t());

    // NFT plot with contract
    mmx::addr_t contract;
    const std::string contract_str = "mmx18rcdx8nhh56twmr2gq3h22kwj00slsn23ejan8qp00rqqw8yl4jq6ccysq";
    contract.from_string(contract_str);
    auto pid_nft = mmx::derive_plot_id(seed, 29, farmer_key, true, contract);
    std::cout << "  NFT plot_id (k29): " << pid_nft.to_string() << std::endl;
    assert(pid_nft != mmx::hash_t());
    assert(pid_nft != pid_og);  // NFT and OG should differ

    // Different k-size should give different plot_id
    auto pid_k30 = mmx::derive_plot_id(seed, 30, farmer_key, false);
    assert(pid_k30 != pid_og);  // k29 vs k30 should differ
    std::cout << "  k29 vs k30 differ: PASS" << std::endl;

    // Different seed should give different plot_id
    mmx::hash_t seed2;
    memset(seed2.data(), 0x99, seed2.size());
    auto pid_seed2 = mmx::derive_plot_id(seed2, 29, farmer_key, false);
    assert(pid_seed2 != pid_og);
    std::cout << "  different seeds differ: PASS" << std::endl;

    // Test 2: Verify PID derivation matches reference formula
    // SHA256("MMX/PLOTID/OG" || ksize_byte || seed || farmer_key)
    {
        uint8_t buf[1024] = {};
        uint32_t offset = 0;
        std::string tag = "MMX/PLOTID/OG";
        memcpy(buf + offset, tag.data(), tag.size()); offset += tag.size();
        uint8_t k = 29;
        memcpy(buf + offset, &k, 1); offset += 1;
        memcpy(buf + offset, seed.data(), seed.size()); offset += seed.size();
        memcpy(buf + offset, farmer_key.data(), farmer_key.size()); offset += farmer_key.size();
        mmx::hash_t expected(buf, offset);
        assert(pid_og == expected);
        std::cout << "  PID matches reference formula: PASS" << std::endl;
    }

    // Test 3: Generate plot identity fills in params correctly
    mmx::PlotParams params;
    params.ksize = 29;
    params.clevel = 5;
    params.ssd_mode = false;
    params.farmer_key = farmer_key;
    params.init_derived();
    mmx::generate_plot_identity(params);
    assert(params.seed != mmx::hash_t());  // seed should be random
    assert(params.plot_id != mmx::hash_t());  // plot_id should be derived
    assert(params.xbits == 24);  // k29 - C5 = 24
    assert(params.is_compressed());
    std::cout << "  generate_plot_identity: PASS" << std::endl;
    std::cout << "  C5 k29: xbits=" << params.xbits << ", compressed=" << params.is_compressed() << std::endl;

    // Test 4: C0 (no compression)
    mmx::PlotParams params_c0;
    params_c0.ksize = 29;
    params_c0.clevel = 0;
    params_c0.farmer_key = farmer_key;
    params_c0.init_derived();
    assert(params_c0.xbits == 29);  // k29 - C0 = 29 = full
    assert(!params_c0.is_compressed());
    assert(params_c0.entry_bits_x == 2 * 29 - 1);  // full line point
    std::cout << "  C0 k29: xbits=" << params_c0.xbits << ", entry_bits_x=" << params_c0.entry_bits_x << std::endl;

    // Test 5: C15 (max compression)
    mmx::PlotParams params_c15;
    params_c15.ksize = 29;
    params_c15.clevel = 15;
    params_c15.farmer_key = farmer_key;
    params_c15.init_derived();
    assert(params_c15.xbits == 14);  // k29 - C15 = 14
    assert(params_c15.is_compressed());
    assert(params_c15.entry_bits_x == 2 * 14 - 1);  // compressed line point
    std::cout << "  C15 k29: xbits=" << params_c15.xbits << ", entry_bits_x=" << params_c15.entry_bits_x << std::endl;

    // Test 6: SSD mode
    mmx::PlotParams params_ssd;
    params_ssd.ksize = 29;
    params_ssd.ssd_mode = true;
    params_ssd.init_derived();
    assert(!params_ssd.is_hdd_plot);
    std::cout << "  SSD mode: is_hdd_plot=" << params_ssd.is_hdd_plot << std::endl;

    // Note: vnx::close() and secp256k1_free() can cause double-free in unit tests
    // In production, main.cpp handles cleanup. For tests, just return.
    // vnx::close();
    // mmx::secp256k1_free();

    std::cout << "=== ALL TESTS PASSED ===" << std::endl;
    _exit(0);  // Avoid vnx static cleanup segfault
    return 0;
}
