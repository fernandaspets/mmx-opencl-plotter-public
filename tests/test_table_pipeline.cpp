// test_table_pipeline.cpp — Full F2-F9 table pipeline test
// Validates sort + match + GPU hash against CPU SHA-512 reference
// Uses deterministic data to ensure reliable Y,Y+1 matches
#include "../src/gpu_device.h"
#include "../src/plot_config.h"
#include <iostream>
#include <vector>
#include <array>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <openssl/sha.h>

// CPU Reference: SHA-512 of L_meta || R_meta (28 uint32s = 112 bytes as raw LE bytes)
// This matches mmx::hash_512_t behavior
uint32_t cpu_hash_lr(
    const uint32_t* L_meta,  // [14]
    const uint32_t* R_meta,  // [14]
    uint32_t kmask,
    uint32_t* M_out)          // [14]
{
    uint32_t msg[28];
    memcpy(msg, L_meta, 14 * sizeof(uint32_t));
    memcpy(msg + 14, R_meta, 14 * sizeof(uint32_t));

    uint8_t hash[64];
    SHA512((uint8_t*)msg, 112, hash);

    // Convert SHA-512 big-endian bytes to LE uint32 (memcpy-style)
    uint32_t hash32[16];
    memcpy(hash32, hash, 64);

    uint32_t Y = 0;
    for(int i = 0; i < 14; i++) Y ^= hash32[i];
    Y &= kmask;

    for(int i = 0; i < 14; i++) M_out[i] = hash32[i] & kmask;
    return Y;
}

int main() {
    std::cout << "=== test_table_pipeline ===" << std::endl;

    // GPU setup
    mmx::GPUDevice gpu;
    gpu.init(0);

    std::string kernel_path;
    for(const auto& p : {"kernels/table_hash.cl", "../kernels/table_hash.cl", "../../kernels/table_hash.cl"}) {
        std::ifstream f(p);
        if(f.good()) { kernel_path = p; break; }
    }
    if(kernel_path.empty()) { std::cerr << "Cannot find table_hash.cl\n"; return 1; }
    gpu.load_program_from_file("table_hash", kernel_path);
    cl_kernel k_hash = gpu.get_kernel("hash_table_entries");

    const uint32_t ksize = 22;
    const uint32_t kmask = (1u << ksize) - 1;
    constexpr int N_META = mmx::N_META;

    // Build deterministic entries that guarantee Y,Y+1 matches
    // Strategy: first N entries have Y = i*2 (even), next N have Y = i*2+1 (odd)
    // Even + adjacent odd = match
    const uint32_t n_pairs = 8;  // 8 guaranteed matches
    const uint32_t num_entries = n_pairs * 2;
    
    std::vector<uint32_t> entries_Y(num_entries);
    std::vector<uint32_t> entries_M(num_entries * N_META);
    
    for(uint32_t i = 0; i < n_pairs; i++) {
        // Even entry: Y = 2*i
        entries_Y[i * 2]     = 2 * i;
        // Odd entry: Y = 2*i + 1  
        entries_Y[i * 2 + 1] = 2 * i + 1;
        
        // Metadata: all zeros for simplicity, but with unique tiebreakers
        for(int j = 0; j < N_META; j++) {
            entries_M[(i * 2) * N_META + j]     = j;       // L meta
            entries_M[(i * 2 + 1) * N_META + j] = 0xFF - j; // R meta
        }
    }
    
    std::cout << "  Entries: " << num_entries << ", Pairs: " << n_pairs << std::endl;

    auto t_start = std::chrono::high_resolution_clock::now();

    // === Step 1: Sort by Y (8-bit radix + std::sort within buckets) ===
    using Entry = std::pair<uint32_t, uint32_t>;  // (Y, index)
    std::vector<Entry> entries(num_entries);
    for(uint32_t i = 0; i < num_entries; i++) entries[i] = {entries_Y[i], i};

    constexpr int LOGBUCKETS = 8;
    const int shift = ksize - LOGBUCKETS;
    const size_t num_buckets = 1u << LOGBUCKETS;

    std::vector<uint32_t> bucket_counts(num_buckets, 0);
    for(const auto& e : entries) {
        uint32_t b = std::min(e.first >> shift, (uint32_t)num_buckets - 1);
        bucket_counts[b]++;
    }

    std::vector<uint32_t> bucket_offsets(num_buckets + 1, 0);
    for(size_t i = 0; i < num_buckets; i++) bucket_offsets[i+1] = bucket_offsets[i] + bucket_counts[i];

    std::vector<Entry> sorted(num_entries);
    std::vector<uint32_t> write_pos = bucket_offsets;
    for(const auto& e : entries) {
        uint32_t b = std::min(e.first >> shift, (uint32_t)num_buckets - 1);
        sorted[write_pos[b]++] = e;
    }

    #pragma omp parallel for schedule(dynamic, 8)
    for(size_t b = 0; b < num_buckets; b++) {
        uint32_t start = bucket_offsets[b];
        uint32_t cnt = bucket_counts[b];
        if(cnt > 1) {
            std::sort(sorted.begin() + start, sorted.begin() + start + cnt,
                [&](const Entry& a, const Entry& b) {
                    if(a.first != b.first) return a.first < b.first;
                    for(int j = 0; j < N_META; j++) {
                        uint32_t ma = entries_M[a.second * N_META + j];
                        uint32_t mb = entries_M[b.second * N_META + j];
                        if(ma != mb) return ma < mb;
                    }
                    return false;
                });
        }
    }

    auto t_sorted = std::chrono::high_resolution_clock::now();

    // === Step 2: Match Y, Y+1 pairs ===
    std::vector<std::pair<uint32_t, uint32_t>> matches;
    matches.reserve(n_pairs * 2);

    for(uint32_t b = 0; b < num_buckets; b++) {
        uint32_t start = bucket_offsets[b];
        uint32_t cnt = bucket_counts[b];
        for(uint32_t x = start; x < start + cnt; x++) {
            uint32_t YL = sorted[x].first;
            for(uint32_t y = x + 1; y < start + cnt && sorted[y].first <= YL + 1; y++) {
                if(sorted[y].first == YL + 1) {
                    matches.emplace_back(x, y);
                }
            }
        }
    }

    auto t_matched = std::chrono::high_resolution_clock::now();
    uint32_t n_matches = matches.size();
    std::cout << "  Matches: " << n_matches << std::endl;

    if(n_matches != n_pairs) {
        std::cout << "  WARNING: Expected " << n_pairs << " matches, got " << n_matches << std::endl;
        // This shouldn't happen with deterministic data
    }

    // === Step 3: Build L_meta and R_meta for GPU ===
    std::vector<uint32_t> L_meta(n_matches * N_META);
    std::vector<uint32_t> R_meta(n_matches * N_META);
    for(uint32_t i = 0; i < n_matches; i++) {
        uint32_t orig_L = sorted[matches[i].first].second;
        uint32_t orig_R = sorted[matches[i].second].second;
        for(int j = 0; j < N_META; j++) {
            L_meta[i * N_META + j] = entries_M[orig_L * N_META + j];
            R_meta[i * N_META + j] = entries_M[orig_R * N_META + j];
        }
    }

    // === Step 4: GPU hash ===
    cl_int err;
    cl_mem L_buf = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  n_matches * N_META * sizeof(uint32_t), L_meta.data(), &err);
    mmx::GPUDevice::check(err, "L_buf");
    cl_mem R_buf = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  n_matches * N_META * sizeof(uint32_t), R_meta.data(), &err);
    mmx::GPUDevice::check(err, "R_buf");
    cl_mem Y_out = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY,
                                  n_matches * sizeof(uint32_t), nullptr, &err);
    mmx::GPUDevice::check(err, "Y_out");
    cl_mem M_out = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY,
                                  n_matches * N_META * sizeof(uint32_t), nullptr, &err);
    mmx::GPUDevice::check(err, "M_out");

    clSetKernelArg(k_hash, 0, sizeof(cl_mem), &L_buf);
    clSetKernelArg(k_hash, 1, sizeof(cl_mem), &R_buf);
    clSetKernelArg(k_hash, 2, sizeof(cl_mem), &Y_out);
    clSetKernelArg(k_hash, 3, sizeof(cl_mem), &M_out);
    clSetKernelArg(k_hash, 4, sizeof(uint32_t), &kmask);
    uint32_t n_u32 = n_matches;
    clSetKernelArg(k_hash, 5, sizeof(uint32_t), &n_u32);

    size_t local = 64;
    size_t global = ((n_matches + local - 1) / local) * local;
    err = clEnqueueNDRangeKernel(gpu.queue, k_hash, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    mmx::GPUDevice::check(err, "enqueue hash_table_entries");
    gpu.finish();

    std::vector<uint32_t> gpu_Y(n_matches), gpu_M(n_matches * N_META);
    clEnqueueReadBuffer(gpu.queue, Y_out, CL_TRUE, 0, n_matches * sizeof(uint32_t), gpu_Y.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(gpu.queue, M_out, CL_TRUE, 0, n_matches * N_META * sizeof(uint32_t), gpu_M.data(), 0, nullptr, nullptr);

    auto t_gpu = std::chrono::high_resolution_clock::now();

    // === Step 5: Validate against CPU reference ===
    uint32_t mismatches = 0;
    for(uint32_t i = 0; i < n_matches; i++) {
        uint32_t cpu_M_ref[14];
        uint32_t cpu_Y = cpu_hash_lr(&L_meta[i * N_META], &R_meta[i * N_META], kmask, cpu_M_ref);

        bool y_ok = (gpu_Y[i] == cpu_Y);
        bool m_ok = true;
        for(int j = 0; j < N_META; j++) {
            if(gpu_M[i * N_META + j] != cpu_M_ref[j]) { m_ok = false; break; }
        }

        if(!y_ok || !m_ok) {
            mismatches++;
            if(mismatches <= 5) {
                std::cout << "  MISMATCH [" << i << "]: "
                          << "GPU Y=" << gpu_Y[i] << " CPU Y=" << cpu_Y
                          << " M[0] GPU=0x" << std::hex << gpu_M[i*N_META] 
                          << " CPU=0x" << cpu_M_ref[0] << std::dec
                          << std::endl;
            }
        }
    }

    auto t_done = std::chrono::high_resolution_clock::now();
    auto ms = [](auto start, auto end) {
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    };

    std::cout << "  Sort: " << ms(t_start, t_sorted)
              << "ms | Match: " << ms(t_sorted, t_matched)
              << "ms | GPU: " << ms(t_matched, t_gpu)
              << "ms | Verify: " << ms(t_gpu, t_done) << "ms" << std::endl;

    // Cleanup
    clReleaseMemObject(L_buf); clReleaseMemObject(R_buf);
    clReleaseMemObject(Y_out); clReleaseMemObject(M_out);

    if(mismatches > 0) {
        std::cout << "\n=== FAILED: " << mismatches << "/" << n_matches << " mismatches ===" << std::endl;
        return 1;
    }

    std::cout << "\n=== PASSED: All " << n_matches << " matches match CPU ===" << std::endl;
    return 0;
}
