#pragma once
#include <cstdint>
#include <vector>
#include <omp.h>

// Parallel prefix sum for uint32_t array
// input: counts[0..n-1], output: offsets[0..n-1] where offsets[i] = sum(counts[0..i-1])
inline void parallel_prefix_sum(const uint32_t* counts, uint32_t* offsets, uint32_t n) {
    if(n == 0) return;
    int num_threads = omp_get_max_threads();
    if(n < 16384 || num_threads <= 1) {
        // Sequential for small arrays
        uint32_t sum = 0;
        for(uint32_t i = 0; i < n; i++) { offsets[i] = sum; sum += counts[i]; }
        return;
    }
    // Two-pass parallel prefix sum
    uint32_t chunk = (n + num_threads - 1) / num_threads;
    std::vector<uint32_t> chunk_sums(num_threads, 0);
    
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        uint32_t start = tid * chunk;
        uint32_t end = std::min(start + chunk, n);
        uint32_t local_sum = 0;
        for(uint32_t i = start; i < end; i++) {
            offsets[i] = local_sum;
            local_sum += counts[i];
        }
        chunk_sums[tid] = local_sum;
    }
    
    // Sequential prefix sum of chunk sums (small: num_threads elements)
    uint32_t prefix = 0;
    for(int t = 0; t < num_threads; t++) {
        uint32_t cs = chunk_sums[t];
        chunk_sums[t] = prefix;
        prefix += cs;
    }
    
    // Add chunk offsets
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        uint32_t start = tid * chunk;
        uint32_t end = std::min(start + chunk, n);
        uint32_t chunk_offset = chunk_sums[tid];
        for(uint32_t i = start; i < end; i++) {
            offsets[i] += chunk_offset;
        }
    }
}
