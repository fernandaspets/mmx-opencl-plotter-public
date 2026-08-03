// Test: plot_codec.h — encode_symbol, decode_symbol, line_point roundtrip
#include "../src/plot_codec.h"
#include <cassert>
#include <iostream>
#include <random>

int main() {
    std::cout << "=== test_plot_codec ===" << std::endl;

    // Test encode_symbol / decode_symbol roundtrip for all valid values (0-47)
    for(uint32_t sym = 0; sym <= 47; ++sym) {
        auto [encoded, nbits] = mmx::encode_symbol((uint8_t)sym);
        auto [decoded, consumed] = mmx::decode_symbol(encoded);
        assert(decoded == sym);
        assert(consumed == nbits);
    }
    std::cout << "  encode/decode roundtrip: PASS (0-47)" << std::endl;

    // Test that encode_symbol throws for values > 47
    bool threw = false;
    try {
        mmx::encode_symbol(48);
    } catch(const std::logic_error&) {
        threw = true;
    }
    assert(threw);
    std::cout << "  encode_symbol(48) throws: PASS" << std::endl;

    // Test encode_deltas / decode_deltas roundtrip
    std::vector<uint8_t> symbols = {0, 1, 2, 3, 5, 10, 15, 20, 30, 47, 0, 1, 2};
    uint64_t total_bits = 0;
    auto encoded = mmx::encode_deltas(symbols, total_bits);
    auto decoded = mmx::decode_deltas(encoded, symbols.size(), 0);
    assert(decoded == symbols);
    std::cout << "  encode/decode deltas roundtrip: PASS" << std::endl;

    // Test line_point roundtrip
    std::mt19937 rng(42);
    for(int i = 0; i < 1000; ++i) {
        uint32_t x = rng() % 100000;
        uint32_t y = rng() % 100000;
        uint64_t lp = mmx::calc_line_point(x, y);
        auto [dx, dy] = mmx::line_point_to_square(lp);
        // calc_line_point normalizes so that y >= x
        uint32_t ex = std::min(x, y);
        uint32_t ey = std::max(x, y);
        assert(dx == ex);
        assert(dy == ey);
    }
    std::cout << "  line_point roundtrip: PASS (1000 random)" << std::endl;

    // Test line_point2 roundtrip (compressed, with +1 offset)
    for(int i = 0; i < 1000; ++i) {
        uint32_t x = rng() % 100000;
        uint32_t y = rng() % 100000;
        uint64_t lp = mmx::calc_line_point2(x, y);
        auto [dx, dy] = mmx::line_point_to_square2(lp);
        assert(dx == x);
        assert(dy == y);
    }
    std::cout << "  line_point2 roundtrip: PASS (1000 random)" << std::endl;

    // Test write_bits / read_bits
    std::vector<uint64_t> buf(100, 0);
    uint64_t offset = 0;
    offset = mmx::write_bits(buf, 0x1234, offset, 16);
    offset = mmx::write_bits(buf, 0xABCD, offset, 16);
    offset = mmx::write_bits(buf, 0xDEADBEEF, offset, 32);
    assert(mmx::read_bits(buf.data(), 0, 16) == 0x1234);
    assert(mmx::read_bits(buf.data(), 16, 16) == 0xABCD);
    assert(mmx::read_bits(buf.data(), 32, 32) == 0xDEADBEEF);
    std::cout << "  write/read bits: PASS" << std::endl;

    std::cout << "=== ALL TESTS PASSED ===" << std::endl;
    return 0;
}
