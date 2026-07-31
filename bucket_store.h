#ifndef BUCKET_STORE_H
#define BUCKET_STORE_H

#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <iostream>

/*
 * BucketStore: stores metadata for all entries, organized by first-level bucket.
 * Used for chunked plotting — process one bucket at a time to limit memory.
 *
 * Layout: fixed-size slots per bucket. Each slot holds max_bucket_size entries
 * of N_META uint32s each.
 *
 * File format:
 *   Header: num_buckets (uint32), max_bucket_size (uint32), n_meta (uint32)
 *   Data:   for each bucket: bucket_size (uint32), then max_bucket_size * n_meta uint32s
 *   Total size: header(12) + num_buckets * (4 + max_bucket_size * n_meta * 4)
 */

class BucketStore {
public:
    const int num_buckets;
    const int max_bucket_size;
    const int n_meta;
    std::string path;
    
    // In-memory mode: store all buckets in RAM (for smaller k-sizes)
    std::vector<std::vector<uint32_t>> buckets;
    
    // File-backed mode
    std::fstream file;
    bool in_memory;
    
    // Slot size in bytes: max_bucket_size * n_meta * 4
    const size_t slot_bytes;
    // Bucket record size: 4 (size) + slot_bytes
    const size_t record_bytes;
    
    BucketStore(const std::string& path, int num_buckets, int max_bucket_size, int n_meta, bool in_memory = false)
        : num_buckets(num_buckets), max_bucket_size(max_bucket_size), n_meta(n_meta), path(path),
          in_memory(in_memory),
          slot_bytes((size_t)max_bucket_size * n_meta * 4),
          record_bytes(4 + slot_bytes)
    {
        if(in_memory) {
            buckets.resize(num_buckets);
        } else {
            // Create file with pre-allocated space
            file.open(path, std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
            if(!file.good()) throw std::runtime_error("Cannot create bucket store: " + path);
            
            // Write header
            uint32_t nb = num_buckets, mbs = max_bucket_size, nm = n_meta;
            file.write((char*)&nb, 4);
            file.write((char*)&mbs, 4);
            file.write((char*)&nm, 4);
            
            // Pre-allocate space for all buckets (zero-filled)
            // Each record: 4 bytes (size) + slot_bytes (data)
            size_t total = (size_t)num_buckets * record_bytes;
            std::vector<char> zeros(total, 0);
            file.write(zeros.data(), total);
            file.flush();
            
            std::cout << "[Store] Created " << path << ": " << (12 + total) / 1e9 << " GB"
                      << " (" << num_buckets << " buckets × " << max_bucket_size << " entries × " << n_meta << " uint32s)" << std::endl;
        }
    }
    
    ~BucketStore() {
        if(!in_memory && file.is_open()) {
            file.close();
        }
    }
    
    // Write a bucket's metadata
    void write_bucket(int bucket_idx, const std::vector<uint32_t>& data, uint32_t count) {
        if(in_memory) {
            buckets[bucket_idx] = data;
            buckets[bucket_idx].resize(count * n_meta);
            return;
        }
        
        // Seek to bucket record
        size_t offset = 12 + (size_t)bucket_idx * record_bytes;
        file.seekp(offset);
        file.write((char*)&count, 4);
        if(count > 0 && data.size() >= (size_t)count * n_meta) {
            file.write((char*)data.data(), count * n_meta * 4);
        }
        // Fill rest with zeros (in case of previous larger data)
        if(count < (size_t)max_bucket_size) {
            size_t remaining = ((size_t)max_bucket_size - count) * n_meta * 4;
            std::vector<char> zeros(remaining, 0);
            file.write(zeros.data(), remaining);
        }
    }
    
    // Read a bucket's metadata
    void read_bucket(int bucket_idx, std::vector<uint32_t>& data, uint32_t& count) {
        if(in_memory) {
            count = buckets[bucket_idx].size() / n_meta;
            data = buckets[bucket_idx];
            return;
        }
        
        size_t offset = 12 + (size_t)bucket_idx * record_bytes;
        file.seekg(offset);
        file.read((char*)&count, 4);
        if(count > (size_t)max_bucket_size) count = max_bucket_size;  // safety
        data.resize((size_t)max_bucket_size * n_meta);
        if(count > 0) {
            file.read((char*)data.data(), count * n_meta * 4);
        }
        // Skip remaining
        if(count < (size_t)max_bucket_size) {
            size_t remaining = ((size_t)max_bucket_size - count) * n_meta * 4;
            file.seekg(offset + 4 + count * n_meta * 4 + remaining);
        }
    }
    
    // Get bucket count (number of entries in a bucket)
    uint32_t get_count(int bucket_idx) {
        if(in_memory) {
            return buckets[bucket_idx].size() / n_meta;
        }
        
        size_t offset = 12 + (size_t)bucket_idx * record_bytes;
        file.seekg(offset);
        uint32_t count = 0;
        file.read((char*)&count, 4);
        if(count > (size_t)max_bucket_size) count = max_bucket_size;
        return count;
    }
    
    void flush() {
        if(!in_memory && file.is_open()) {
            file.flush();
        }
    }
    
    size_t total_size_bytes() const {
        return 12 + (size_t)num_buckets * record_bytes;
    }
};

#endif
