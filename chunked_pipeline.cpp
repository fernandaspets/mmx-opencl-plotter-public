// chunked_pipeline.cpp — Per-bucket chunked pipeline for large k-sizes
// Processes one first-level bucket at a time to limit memory usage.
// Uses in-memory bucket store (requires ~66 GB RAM for k29).

#include "bucket_store.h"
#include <omp.h>
#include <openssl/sha.h>
#include <algorithm>
#include <cstring>
#include <chrono>

// External globals from plotter.cpp
extern int KSIZE, LOGBUCKETS, MY_N_META, MY_N_META_OUT, MY_N_TABLE, DSIZE_, PDSIZE, XBITS;
extern uint32_t KMASK;

// In-memory bucket store: vectors of metadata, one per bucket
struct MemBucketStore {
    int num_buckets;
    int max_bucket_size;
    int n_meta;
    std::vector<std::vector<uint32_t>> buckets;  // buckets[y] = [count * n_meta] flat metadata
    std::vector<uint32_t> counts;  // counts[y] = number of entries in bucket y
    
    MemBucketStore(int nb, int mbs, int nm)
        : num_buckets(nb), max_bucket_size(mbs), n_meta(nm)
    {
        buckets.resize(nb);
        counts.resize(nb, 0);
    }
    
    void clear() {
        for(auto& b : buckets) b.clear();
        std::fill(counts.begin(), counts.end(), 0);
    }
    
    // Append entries to a bucket
    void append(int bucket, const uint32_t* meta, uint32_t count) {
        if(bucket < 0 || bucket >= num_buckets) return;
        if(counts[bucket] + count > (uint32_t)max_bucket_size) {
            count = max_bucket_size - counts[bucket];  // clamp
        }
        if(count == 0) return;
        auto& b = buckets[bucket];
        b.insert(b.end(), meta, meta + count * n_meta);
        counts[bucket] += count;
    }
    
    // Get metadata for a bucket (returns pointer and count)
    const uint32_t* get(int bucket, uint32_t& count) const {
        count = counts[bucket];
        if(count == 0) return nullptr;
        return buckets[bucket].data();
    }
};

// Compute F1 in batches and write to in-memory bucket store
void compute_f1_chunked(
    OCL_Plotter& plotter,
    const hash_t& plot_id,
    MemBucketStore& store,
    int batch_size)
{
    const uint64_t total_entries = (uint64_t)1 << KSIZE;
    const int num_buckets = store.num_buckets;
    const int n_meta = store.n_meta;
    const int shift = KSIZE - LOGBUCKETS;
    
    const uint32_t num_batches = (total_entries + batch_size - 1) / batch_size;
    std::cout << "[F1] Computing " << total_entries << " entries in " << num_batches << " batches..." << std::endl;
    auto t0 = my_time_ms();
    
    std::vector<uint32_t> X_batch(batch_size), Y_batch(batch_size), M_batch(batch_size * n_meta);
    
    // Per-batch bucket buffers
    std::vector<std::vector<uint32_t>> batch_buckets(num_buckets);
    
    for(uint32_t b = 0; b < num_batches; b++) {
        const uint32_t start = b * batch_size;
        const uint32_t count = std::min((uint32_t)batch_size, (uint32_t)(total_entries - start));
        
        for(uint32_t i = 0; i < count; i++) X_batch[i] = start + i;
        
        plotter.compute_f1_batch(X_batch, plot_id, Y_batch, M_batch);
        
        // Clear batch bucket buffers
        for(int i = 0; i < num_buckets; i++) batch_buckets[i].clear();
        
        // Bucket by Y's top bits
        for(uint32_t i = 0; i < count; i++) {
            uint32_t Y = Y_batch[i];
            uint32_t bucket = Y >> shift;
            if(bucket >= (uint32_t)num_buckets) bucket = num_buckets - 1;
            for(int j = 0; j < n_meta; j++) {
                batch_buckets[bucket].push_back(M_batch[i * n_meta + j]);
            }
        }
        
        // Append to store
        for(int y = 0; y < num_buckets; y++) {
            if(batch_buckets[y].empty()) continue;
            uint32_t cnt = batch_buckets[y].size() / n_meta;
            store.append(y, batch_buckets[y].data(), cnt);
        }
        
        if(b % 16 == 0 || b == num_batches - 1) {
            std::cerr << "\r[F1] Batch " << (b+1) << "/" << num_batches
                      << " (" << (b+1)*100/num_batches << "%) "
                      << (my_time_ms() - t0) / 1000.0 << "s" << std::flush;
        }
    }
    
    std::cout << "\n[F1] Done in " << (my_time_ms() - t0) / 1000.0 << " sec" << std::endl;
    
    uint64_t total = 0;
    for(int i = 0; i < num_buckets; i++) total += store.counts[i];
    std::cout << "[F1] Total entries: " << total << std::endl;
}

// Process one table for one first-level bucket
void process_bucket_chunk(
    OCL_Plotter& plotter,
    MemBucketStore& src,
    MemBucketStore& dst,
    int y,
    int table)
{
    const int n_meta = src.n_meta;
    const int shift = KSIZE - LOGBUCKETS;
    
    uint32_t count_y = src.counts[y];
    if(count_y == 0) return;
    
    const uint32_t* meta_y = src.buckets[y].data();
    
    // Read bucket y+1 for cross-boundary matching
    uint32_t count_ynext = 0;
    const uint32_t* meta_ynext = nullptr;
    if(y + 1 < src.num_buckets) {
        count_ynext = src.counts[y + 1];
        if(count_ynext > 0) meta_ynext = src.buckets[y + 1].data();
    }
    
    uint32_t total = count_y + count_ynext;
    
    // Compute Y values
    std::vector<uint32_t> Y_all(total);
    for(uint32_t i = 0; i < count_y; i++) {
        uint32_t Y = 0;
        for(int j = 0; j < n_meta; j++) Y ^= meta_y[i * n_meta + j];
        Y_all[i] = Y & KMASK;
    }
    for(uint32_t i = 0; i < count_ynext; i++) {
        uint32_t Y = 0;
        for(int j = 0; j < n_meta; j++) Y ^= meta_ynext[i * n_meta + j];
        Y_all[count_y + i] = Y & KMASK;
    }
    
    // Build entries: (Y, index) where index 0..count_y-1 is bucket y
    std::vector<std::pair<uint32_t, uint32_t>> entries(total);
    for(uint32_t i = 0; i < total; i++) entries[i] = {Y_all[i], i};
    
    // Sort by Y
    std::sort(entries.begin(), entries.end());
    
    // Match Y,Y+1 — only LEFT in bucket y (index < count_y)
    std::vector<uint32_t> LR_flat;
    for(size_t i = 0; i < entries.size(); i++) {
        if(entries[i].second >= count_y) continue;  // LEFT must be in bucket y
        uint32_t YL = entries[i].first;
        for(size_t j = i + 1; j < entries.size(); j++) {
            uint32_t YR = entries[j].first;
            if(YR == YL + 1) {
                LR_flat.push_back(entries[i].second);
                LR_flat.push_back(entries[j].second);
            } else if(YR > YL + 1) break;
        }
    }
    
    if(LR_flat.empty()) return;
    
    // Build combined metadata array for GPU hashing
    std::vector<uint32_t> M_combined(total * n_meta);
    std::memcpy(M_combined.data(), meta_y, count_y * n_meta * 4);
    if(count_ynext > 0) {
        std::memcpy(M_combined.data() + count_y * n_meta, meta_ynext, count_ynext * n_meta * 4);
    }
    
    // GPU hash
    std::vector<uint32_t> Y_out, M_out;
    plotter.gpu_hash_table_lr(M_combined, LR_flat, Y_out, M_out, KMASK);
    
    // Bucket new metadata by new Y's top bits → dst store
    std::vector<std::vector<uint32_t>> dst_batch(dst.num_buckets);
    for(size_t i = 0; i < Y_out.size(); i++) {
        uint32_t bucket = Y_out[i] >> shift;
        if(bucket >= (uint32_t)dst.num_buckets) bucket = dst.num_buckets - 1;
        for(int j = 0; j < n_meta; j++) {
            dst_batch[bucket].push_back(M_out[i * n_meta + j]);
        }
    }
    
    for(int b = 0; b < dst.num_buckets; b++) {
        if(dst_batch[b].empty()) continue;
        uint32_t cnt = dst_batch[b].size() / n_meta;
        dst.append(b, dst_batch[b].data(), cnt);
    }
}

// Full chunked F2-F9 pipeline
void compute_f2_f9_chunked(
    OCL_Plotter& plotter,
    MemBucketStore& store,
    int num_buckets,
    int max_bucket_size,
    int n_meta)
{
    // Ping-pong stores
    MemBucketStore src(num_buckets, max_bucket_size, n_meta);
    MemBucketStore dst(num_buckets, max_bucket_size, n_meta);
    
    // Copy F1 output from store to src
    for(int y = 0; y < num_buckets; y++) {
        if(store.counts[y] > 0) {
            src.append(y, store.buckets[y].data(), store.counts[y]);
        }
    }
    
    auto t0 = my_time_ms();
    
    for(int t = 2; t <= MY_N_TABLE; t++) {
        auto tt0 = my_time_ms();
        dst.clear();
        
        for(int y = 0; y < num_buckets; y++) {
            if(src.counts[y] == 0) continue;
            process_bucket_chunk(plotter, src, dst, y, t);
            
            if(y % 32 == 0 || y == num_buckets - 1) {
                std::cerr << "\r[T" << t << "] Bucket " << (y+1) << "/" << num_buckets
                          << " (" << (y+1)*100/num_buckets << "%) "
                          << (my_time_ms() - tt0) / 1000.0 << "s" << std::flush;
            }
        }
        
        double elapsed = (my_time_ms() - tt0) / 1000.0;
        uint64_t total = 0;
        for(int y = 0; y < num_buckets; y++) total += dst.counts[y];
        std::cout << "\n[T" << t << "] " << total << " entries (" << elapsed << " sec)" << std::endl;
        
        // Swap: src = dst, dst = cleared
        std::swap(src.buckets, dst.buckets);
        std::swap(src.counts, dst.counts);
        dst.clear();
    }
    
    // Copy final result back to store
    store.clear();
    for(int y = 0; y < num_buckets; y++) {
        if(src.counts[y] > 0) {
            store.append(y, src.buckets[y].data(), src.counts[y]);
        }
    }
    
    std::cout << "[CPU] F2-F9 done in " << (my_time_ms() - t0) / 1000.0 << " sec" << std::endl;
}
