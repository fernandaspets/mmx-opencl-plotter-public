// Bitmap-based matching: eliminates sort + sorted scan + lr_flat_build
// Uses counting sort (via position map) instead of radix sort
// Produces LR pairs in ORIGINAL position order (ready for GPU hash)

#pragma once
#include <vector>
#include <utility>
#include <cstdint>
#include <omp.h>
#include <cstring>

// Bitmap matching: given unsorted entries (Y, orig_pos), find all Y,Y+1 pairs
// Outputs:
//   lr_pairs: (orig_L, orig_R) — ready for GPU hash (no conversion needed)
//   sorted_entries: (Y, orig_pos) sorted by Y — for sort_matches and build_pd
//   inv_map: orig_pos → sorted_pos — for converting lr_pairs to sorted positions for PD
static inline void bitmap_match(
    const std::vector<std::pair<uint32_t, uint32_t>>& entries,
    std::vector<std::pair<uint32_t, uint32_t>>& lr_pairs,
    std::vector<std::pair<uint32_t, uint32_t>>& sorted_entries,
    std::vector<uint32_t>& inv_map,
    uint32_t ksize)
{
    const size_t n = entries.size();
    const uint32_t kmask = (1u << ksize) - 1;
    const uint32_t yrange = 1u << ksize;
    
    // Step 1: Build position map (counting sort)
    // pos_map[Y] = index of FIRST entry with that Y (in entries array)
    // next_same[i] = index of next entry with same Y as entries[i]
    std::vector<uint32_t> pos_map(yrange, UINT32_MAX);
    std::vector<uint32_t> next_same(n, UINT32_MAX);
    
    // Build chain: for each Y, maintain a tail pointer for O(1) append
    std::vector<uint32_t> tail_map(yrange, UINT32_MAX);
    
    for(size_t i = 0; i < n; i++) {
        uint32_t Y = entries[i].first & kmask;
        if(pos_map[Y] == UINT32_MAX) {
            pos_map[Y] = (uint32_t)i;
            tail_map[Y] = (uint32_t)i;
        } else {
            next_same[tail_map[Y]] = (uint32_t)i;
            tail_map[Y] = (uint32_t)i;
        }
    }
    
    // Step 2: Build sorted entries by iterating Y in order (counting sort output)
    // Parallelized: each thread handles a range of Y values
    sorted_entries.resize(n);
    inv_map.resize(n);
    
    // First, compute prefix sum of counts per Y to get global offsets
    // Use parallel prefix sum over non-empty Y values
    // Step 2a: compute count per Y (already have via chain lengths)
    // Step 2b: prefix sum to get offsets
    // Step 2c: scatter entries to sorted positions
    
    // For parallelism, divide Y range into chunks
    const int nthreads = omp_get_max_threads();
    std::vector<uint32_t> thread_start(nthreads + 1, 0);
    
    // Compute entries per Y-chunk for load balancing
    uint32_t y_per_thread = (yrange + nthreads - 1) / nthreads;
    
    // Count entries per thread's Y range
    std::vector<uint64_t> thread_counts(nthreads, 0);
    #pragma omp parallel for schedule(static)
    for(int ti = 0; ti < nthreads; ti++) {
        uint32_t y_start = ti * y_per_thread;
        uint32_t y_end = std::min(y_start + y_per_thread, yrange);
        uint64_t cnt = 0;
        for(uint32_t Y = y_start; Y < y_end; Y++) {
            uint32_t pos = pos_map[Y];
            while(pos != UINT32_MAX) { cnt++; pos = next_same[pos]; }
        }
        thread_counts[ti] = cnt;
    }
    
    // Prefix sum for thread offsets
    thread_start[0] = 0;
    for(int ti = 0; ti < nthreads; ti++) {
        thread_start[ti + 1] = thread_start[ti] + (uint32_t)thread_counts[ti];
    }
    
    // Scatter entries to sorted positions (parallel)
    #pragma omp parallel for schedule(static)
    for(int ti = 0; ti < nthreads; ti++) {
        uint32_t y_start = ti * y_per_thread;
        uint32_t y_end = std::min(y_start + y_per_thread, yrange);
        uint32_t sorted_pos = thread_start[ti];
        for(uint32_t Y = y_start; Y < y_end; Y++) {
            uint32_t pos = pos_map[Y];
            while(pos != UINT32_MAX) {
                uint32_t orig_pos = entries[pos].second;
                sorted_entries[sorted_pos] = {Y, orig_pos};
                inv_map[orig_pos] = sorted_pos;
                sorted_pos++;
                pos = next_same[pos];
            }
        }
    }
    
    // Step 3: Match — for each entry, check if Y+1 exists (parallel)
    // Each thread processes a range of entries and collects its own lr_pairs
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> thread_lr(nthreads);
    
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < n; i++) {
        int ti = omp_get_thread_num();
        uint32_t Y = entries[i].first & kmask;
        if(Y == kmask) continue;
        
        uint32_t pos_Y1 = pos_map[Y + 1];
        if(pos_Y1 == UINT32_MAX) continue;
        
        uint32_t orig_L = entries[i].second;
        uint32_t cur = pos_Y1;
        while(cur != UINT32_MAX) {
            uint32_t orig_R = entries[cur].second;
            thread_lr[ti].emplace_back(orig_L, orig_R);
            cur = next_same[cur];
        }
    }
    
    // Flatten thread_lr into lr_pairs
    size_t total = 0;
    for(int ti = 0; ti < nthreads; ti++) total += thread_lr[ti].size();
    lr_pairs.clear();
    lr_pairs.reserve(total);
    for(int ti = 0; ti < nthreads; ti++) {
        for(const auto& p : thread_lr[ti]) lr_pairs.push_back(p);
    }
}
