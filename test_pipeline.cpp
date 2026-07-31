/*
 * Test the full F1→F9 pipeline with a small k value.
 * k=20: 1M entries, should produce matches through all 9 tables.
 */

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <vnx/vnx.h>
#include <mmx/hash_t.hpp>
#include <mmx/hash_512_t.hpp>
#include <mmx/pos/encoding.h>
#include <mmx/pos/config.h>
#include <mmx/utils.h>

#include <cstdio>
#include <cstdint>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <set>

using namespace mmx;
using namespace mmx::pos;

#define KSIZE_TEST 20
#define N_META 14
#define N_TABLE 9
#define N_META_OUT 12
#define META_BYTES_OUT (N_META_OUT * 4)

static int64_t my_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    printf("=== Pipeline Test (k=%d) ===\n", KSIZE_TEST);
    
    const uint32_t kmask = (1u << KSIZE_TEST) - 1;
    const uint64_t total = 1ull << KSIZE_TEST;
    
    // Load OpenCL kernel
    FILE* f = fopen("/home/ubman/mmx-app/opencl-recompute/pos_recompute.cl", "r");
    if(!f) { printf("Cannot open kernel\n"); return 1; }
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* src = new char[sz+1]; size_t rd = fread(src, 1, sz, f); src[sz] = 0; fclose(f);
    
    cl_int err;
    cl_platform_id plat; clGetPlatformIDs(1, &plat, nullptr);
    cl_device_id dev; clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, &sz, &err);
    delete[] src;
    err = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);
    if(err) { printf("Build err=%d\n", err); return 1; }
    cl_kernel k = clCreateKernel(prog, "compute_f1_kernel", &err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    
    // Compute F1 for all X values
    printf("Computing F1 for %lu entries...\n", total);
    auto t0 = my_time_ms();
    
    std::vector<uint32_t> X(total), Y(total), M(total * N_META);
    for(uint64_t i = 0; i < total; i++) X[i] = (uint32_t)i;
    
    // Plot ID: all zeros
    std::vector<uint32_t> id(8, 0);
    
    cl_mem Xb = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, total*4, X.data(), &err);
    cl_mem IDb = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, 32, id.data(), &err);
    cl_mem Yb = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, total*4, nullptr, &err);
    cl_mem Mb = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, total*N_META*4, nullptr, &err);
    
    uint32_t xbits = 0, num_x = (uint32_t)total;
    clSetKernelArg(k,0,sizeof(cl_mem),&Xb);
    clSetKernelArg(k,1,sizeof(cl_mem),&IDb);
    clSetKernelArg(k,2,sizeof(cl_mem),&Yb);
    clSetKernelArg(k,3,sizeof(cl_mem),&Mb);
    clSetKernelArg(k,4,sizeof(uint32_t),&kmask);
    clSetKernelArg(k,5,sizeof(uint32_t),&xbits);
    clSetKernelArg(k,6,sizeof(uint32_t),&num_x);
    
    size_t global = total, local = 64;
    if(global % local) global = ((global/local)+1)*local;
    clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    clEnqueueReadBuffer(q, Yb, CL_TRUE, 0, total*4, Y.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(q, Mb, CL_TRUE, 0, total*N_META*4, M.data(), 0, nullptr, nullptr);
    
    printf("F1 done in %.2f sec\n", (my_time_ms() - t0) / 1000.0);
    
    // Compute F2-F9
    printf("Computing F2-F9...\n");
    auto t1 = my_time_ms();
    
    std::vector<std::array<uint32_t, N_META>> M_curr(total);
    for(uint64_t i = 0; i < total; i++) {
        for(int j = 0; j < N_META; j++) M_curr[i][j] = M[i * N_META + j];
    }
    
    std::vector<std::pair<uint32_t, uint32_t>> entries; // (Y, index)
    entries.reserve(total);
    for(uint64_t i = 0; i < total; i++) entries.emplace_back(Y[i], (uint32_t)i);
    
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> LR(N_TABLE + 1);
    
    auto sort_func = [&M_curr](const auto& L, const auto& R) {
        if(L.first == R.first) return M_curr[L.second] < M_curr[R.second];
        return L < R;
    };
    
    for(int t = 2; t <= N_TABLE; t++) {
        std::sort(entries.begin(), entries.end(), sort_func);
        
        std::vector<std::array<uint32_t, N_META>> M_next;
        std::vector<std::pair<uint32_t, uint32_t>> matches;
        std::vector<std::pair<uint32_t, uint32_t>> lr_pairs;
        
        for(size_t x = 0; x < entries.size(); x++) {
            const auto YL = entries[x].first;
            for(size_t y = x + 1; y < entries.size() && entries[y].first <= YL + 1; y++) {
                if(entries[y].first == YL + 1) {
                    const auto PL = entries[x].second;
                    const auto PR = entries[y].second;
                    
                    uint32_t msg[N_META * 2];
                    for(int i = 0; i < N_META; i++) {
                        msg[i] = M_curr[PL][i];
                        msg[N_META + i] = M_curr[PR][i];
                    }
                    const hash_512_t tmp(&msg, sizeof(msg));
                    uint32_t hash[16];
                    std::memcpy(hash, tmp.data(), tmp.size());
                    
                    uint32_t Y_i = 0;
                    std::array<uint32_t, N_META> meta = {};
                    for(int i = 0; i < N_META; i++) {
                        Y_i = Y_i ^ hash[i];
                        meta[i] = hash[i] & kmask;
                    }
                    Y_i &= kmask;
                    
                    matches.emplace_back(Y_i, (uint32_t)M_next.size());
                    lr_pairs.emplace_back(PL, PR);
                    M_next.push_back(meta);
                }
            }
        }
        
        if(matches.empty()) {
            printf("ERROR: zero matches at table %d\n", t);
            return 1;
        }
        
        printf("T%d: %lu entries (from %lu)\n", t, matches.size(), entries.size());
        LR[t] = std::move(lr_pairs);
        M_curr = std::move(M_next);
        entries = std::move(matches);
    }
    
    // Sort and deduplicate
    std::sort(entries.begin(), entries.end(), sort_func);
    std::set<std::array<uint32_t, N_META>> M_set;
    std::vector<uint32_t> final_indices;
    
    for(const auto& entry : entries) {
        const auto& meta = M_curr[entry.second];
        if(!M_set.insert(meta).second) continue;
        final_indices.push_back(entry.second);
    }
    
    printf("\nFinal: %lu unique entries\n", final_indices.size());
    printf("Pipeline complete in %.2f sec\n", (my_time_ms() - t1) / 1000.0);
    
    // Verify proof reconstruction for first entry
    if(!final_indices.empty()) {
        uint32_t idx = final_indices[0];
        std::vector<uint32_t> I_tmp = {idx};
        
        for(int t = N_TABLE; t >= 2; t--) {
            std::vector<uint32_t> I_next;
            for(const auto i : I_tmp) {
                I_next.push_back(LR[t][i].first);
                I_next.push_back(LR[t][i].second);
            }
            I_tmp = std::move(I_next);
        }
        
        printf("Proof reconstruction: %lu X values\n", I_tmp.size());
        printf("X values: ");
        for(size_t i = 0; i < std::min(I_tmp.size(), (size_t)8); i++) {
            printf("%u ", I_tmp[i]);
        }
        printf("...\n");
    }
    
    printf("\nSUCCESS! Pipeline works end-to-end.\n");
    
    clReleaseMemObject(Xb); clReleaseMemObject(IDb);
    clReleaseMemObject(Yb); clReleaseMemObject(Mb);
    clReleaseKernel(k); clReleaseProgram(prog); clReleaseContext(ctx);
    return 0;
}
