#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>

namespace mmx {

// Bit-level read/write operations for park encoding.
// Matches reference cuda_encoding.h / mmx pos::encoding

// Write bits into a uint64 buffer at a given bit offset.
// Returns the new bit offset.
inline uint64_t write_bits(std::vector<uint64_t>& dst, uint64_t value,
                           uint64_t bit_offset, uint32_t num_bits) {
    if(num_bits < 64) {
        value &= ((uint64_t(1) << num_bits) - 1);
    }
    const uint32_t shift = bit_offset % 64;
    const uint32_t free_bits = 64 - shift;

    dst[bit_offset / 64] |= (value << shift);

    if(free_bits < num_bits) {
        dst[bit_offset / 64 + 1] |= (value >> free_bits);
    }
    return bit_offset + num_bits;
}

// Read bits from a uint64 buffer at a given bit offset.
inline uint64_t read_bits(const uint64_t* src, uint64_t bit_offset, uint32_t num_bits) {
    uint32_t count = 0;
    uint64_t offset = bit_offset;
    uint64_t result = 0;
    while(count < num_bits) {
        const uint32_t shift = offset % 64;
        const uint32_t bits = std::min(num_bits - count, 64u - shift);
        const uint64_t value = src[offset / 64] >> shift;
        result |= value << count;
        count += bits;
        offset += bits;
    }
    if(num_bits < 64) {
        result &= ((uint64_t(1) << num_bits) - 1);
    }
    return result;
}

// Encode a delta value into a variable-length symbol.
// Returns (encoded_bits, num_bits).
// Matches mmx::pos::encode_symbol exactly.
inline std::pair<uint32_t, uint32_t> encode_symbol(uint8_t sym) {
    switch(sym) {
        case 0: return {0, 2};
        case 1: return {1, 2};
        case 2: return {2, 2};
    }
    const uint32_t index = sym / 3;
    const uint32_t mod = sym % 3;

    if(index > 15) {
        throw std::logic_error("encode_symbol: symbol out of range: " + std::to_string(sym)
            + " (full plot required for small deltas)");
    }
    uint32_t out = uint32_t(-1) >> (32 - 2 * index);
    out |= mod << (2 * index);
    return {out, 2 * index + 2};
}

// Decode a variable-length symbol from bits.
// Returns (decoded_value, num_bits_consumed).
// Matches mmx::pos::decode_symbol exactly.
inline std::pair<uint32_t, uint32_t> decode_symbol(uint32_t bits) {
    switch(bits & 3) {
        case 0: return {0, 2};
        case 1: return {1, 2};
        case 2: return {2, 2};
    }
    uint32_t shift = bits;
    for(uint32_t index = 0; index < 16; ++index) {
        const auto mod = shift & 3;
        if(mod == 3) {
            shift >>= 2;
        } else {
            return {3 * index + mod, 2 * index + 2};
        }
    }
    return {48, 32};
}

// Encode an array of delta symbols into a bit stream.
// Returns the encoded uint64 vector and sets total_bits.
inline std::vector<uint64_t> encode_deltas(const std::vector<uint8_t>& symbols, uint64_t& total_bits) {
    std::vector<uint64_t> out;
    total_bits = 0;
    uint32_t offset = 0;
    uint64_t buffer = 0;

    for(const auto sym : symbols) {
        const auto [bits, nbits] = encode_symbol(sym);
        buffer |= uint64_t(bits) << offset;

        const auto end = offset + nbits;
        if(end >= 64) {
            out.push_back(buffer);
            buffer = 0;
        }
        if(end > 64) {
            buffer = bits >> (64 - offset);
        }
        offset = end % 64;
        total_bits += nbits;
    }
    if(offset) {
        out.push_back(buffer);
    }
    return out;
}

// Decode delta symbols from a bit stream.
inline std::vector<uint8_t> decode_deltas(const std::vector<uint64_t>& bit_stream,
                                           uint64_t num_symbols, uint64_t bit_offset) {
    std::vector<uint8_t> out;
    out.reserve(num_symbols);

    uint32_t bits = 0;
    uint64_t offset = bit_offset;
    uint64_t buffer = 0;

    while(out.size() < num_symbols) {
        if(bits < 32) {
            const auto index = offset / 64;
            if(index < bit_stream.size()) {
                const uint32_t count = std::min<uint32_t>(64 - (offset % 64), 64 - bits);
                buffer |= (bit_stream[index] >> (offset % 64)) << bits;
                offset += count;
                bits += count;
            } else if(bits == 0) {
                throw std::logic_error("bit stream underflow");
            }
        }
        const auto [sym, consumed] = decode_symbol(buffer);
        out.push_back(sym);

        if(consumed > bits) {
            throw std::logic_error("symbol decode error");
        }
        buffer >>= consumed;
        bits -= consumed;
    }
    return out;
}

// Convert (x, y) to a line point.
// line_point = max(x,y)^2 + max(x,y) + min(x,y) = max(x,y)*(max(x,y)+1) + min(x,y)
// Actually the reference uses: get_x_enc(max(x,y)) + min(x,y)
// where get_x_enc(n) = n*(n+1)/2 (triangular number)
inline uint64_t calc_line_point(uint32_t x, uint32_t y) {
    if(x > y) std::swap(x, y);
    // triangular number: y*(y+1)/2 + x
    return uint64_t(y) * (y + 1) / 2 + x;
}

// Convert (x, y) to a line point with +1 offset (for compressed plots).
// Used when xbits < ksize: calc_line_point(x+1, y+1)
inline uint64_t calc_line_point2(uint32_t x, uint32_t y) {
    return calc_line_point(x + 1, y + 1);
}

// Convert a line point back to (x, y).
// Finds n such that n*(n+1)/2 <= lp < (n+1)*(n+2)/2, then x = lp - n*(n+1)/2, y = n
inline std::pair<uint32_t, uint32_t> line_point_to_square(uint64_t line_point) {
    // Binary search for the square root of 2*line_point
    uint64_t lo = 0, hi = 1ULL << 32;
    while(lo < hi) {
        uint64_t mid = (lo + hi + 1) / 2;
        if(mid * (mid + 1) / 2 <= line_point) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    uint64_t y = lo;
    uint64_t x = line_point - y * (y + 1) / 2;
    return {(uint32_t)x, (uint32_t)y};
}

// Convert a line point back to (x, y) for compressed plots (reverse of calc_line_point2).
inline std::pair<uint32_t, uint32_t> line_point_to_square2(uint64_t line_point) {
    auto [x, y] = line_point_to_square(line_point);
    return {x - 1, y - 1};  // reverse the +1 offset
}

} // namespace mmx
