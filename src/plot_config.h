#pragma once

#include <cstdint>
#include <stdexcept>

namespace mmx {

// MMX Proof of Space constants (from mmx-node/include/mmx/pos/config.h)
constexpr int N_META = 14;        // metadata uint32 values per entry (input)
constexpr int N_META_OUT = 12;    // metadata uint32 values per entry (output)
constexpr int N_TABLE = 9;        // total tables (F1..F9)
constexpr int MEM_HASH_N = 32;    // gen_mem_array state size
constexpr int MEM_HASH_ITER = 256; // gen_mem_array iterations

// Plot file constants (from reference config.h)
constexpr uint64_t FILE_ALIGNMENT = 4096;
constexpr double MAX_AVG_OFFSET_BITS = 2.65;
constexpr double MAX_AVG_YDELTA_BITS = 2.25;

// PD entry encoding (from reference config.h)
// PSIZE = ksize + 1 (position bits), DSIZE = 5 (delta bits)
// PDBYTES = cdiv(PSIZE + DSIZE, 8)
// These are computed at runtime since ksize is a parameter

// Park sizes (from reference config.h)
constexpr int PARK_SIZE_X = 2048;
constexpr int PARK_SIZE_Y = 8192;
constexpr int PARK_SIZE_PD = 2048;
constexpr int PARK_SIZE_META = 256;

// Plot file header magic
constexpr uint32_t PLOT_MAGIC = 0x1337FFFF;  // vnx magic

// Compression level limits
constexpr int MIN_CLEVEL = 0;
constexpr int MAX_CLEVEL = 15;

// K-size limits
constexpr int MIN_KSIZE = 18;
constexpr int MAX_KSIZE = 32;

// Mainnet minimum
constexpr int MAINNET_MIN_KSIZE = 29;

inline uint32_t kmask_for(int ksize) {
    return (uint64_t(1) << ksize) - 1;
}

inline int logbuckets_for(int ksize) {
    // Reference uses LOGBUCKETS=6 for small k, 8 for large k
    if(ksize <= 26) return 6;
    return 8;
}

inline int logbuckets2_for(int ksize, int logbuckets) {
    return ksize - logbuckets - 9;
}

inline uint32_t cdiv(uint64_t a, uint64_t b) {
    return (a + b - 1) / b;
}

} // namespace mmx
