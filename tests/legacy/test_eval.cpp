// test_eval.cpp — Test eval_p1_tx kernel
// Takes LR pairs + metadata, hashes with SHA-512, produces new Y + metadata
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <array>
#include <cstring>

// CPU SHA-512 reference (simplified — just check Y_new matches)
// We'll use the GPU table_hash kernel result as reference since it's already verified.
// Actually, let's compute SHA-512 on CPU using OpenSSL or a simple implementation.

// Simple CPU SHA-512 for testing
#include <openssl/sha.h>

int main() {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_int err;
    
    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    
    std::ifstream f("f2_f9.cl");
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const char* cstr = src.c_str();
    size_t len = src.size();
    
    cl_program prog = clCreateProgramWithSource(context, 1, &cstr, &len, &err);
    std::string opts = "-DKSIZE=20 -DLOGBUCKETS=6 -DLOGBUCKETS2=5 -DN_META=14 -DN_META_OUT=12 -DN_TABLE=9"
                       " -DDSIZE_=5 -DPSIZE_=21 -DPDSIZE=40 -DX2SIZE=39 -DXBITS=20"
                       " -DHYBRID_SORT_LOG_THREADS=6 -DNUM_THREADS=64"
                       " -DKMASK=1048575 -DDMASK=31 -DMETA_BYTES=56"
                       " -DMAX_LOCAL_SIZE=40";
    err = clBuildProgram(prog, 1, &device, opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) { char log[8192]; clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr); std::cerr << log << std::endl; return 1; }
    
    const int KSIZE = 20, N_META = 14, LOGBUCKETS = 6;
    const uint32_t KMASK = (1u << KSIZE) - 1;
    const int NUM_BUCKETS = 1 << LOGBUCKETS;  // 64
    const int MAX_BUCKET_SIZE = 1024;
    
    // Generate random metadata for 100 entries
    const int N = 100;
    std::mt19937 rng(42);
    std::vector<uint32_t> C_in(N * N_META);
    for (auto& v : C_in) v = rng() & KMASK;
    
    // Generate random LR pairs (matching entry indices)
    std::vector<uint32_t> LR_data(N * 2);
    for (int i = 0; i < N; i++) {
        LR_data[i * 2] = rng() % N;      // P_1
        LR_data[i * 2 + 1] = rng() % N;  // P_2
    }
    
    // CPU reference: hash each pair with SHA-512, extract Y_new
    std::vector<uint32_t> cpu_Y(N), cpu_M(N * N_META);
    for (int i = 0; i < N; i++) {
        uint32_t P1 = LR_data[i * 2];
        uint32_t P2 = LR_data[i * 2 + 1];
        
        // SHA-512 of C_in[P1] || C_in[P2] (each N_META uint32s = 4 bytes each)
        uint8_t msg[2 * N_META * 4];
        for (int j = 0; j < N_META; j++) {
            memcpy(msg + j * 4, &C_in[P1 * N_META + j], 4);
            memcpy(msg + (N_META + j) * 4, &C_in[P2 * N_META + j], 4);
        }
        
        uint8_t hash[64];
        SHA512(msg, 2 * N_META * 4, hash);
        
        // Extract Y_new: XOR of lower 32 bits of each uint64 in hash
        uint32_t Y_new = 0;
        for (int j = 0; j < N_META; j++) {
            uint32_t h32;
            memcpy(&h32, hash + j * 4, 4);  // lower 32 bits of each uint64
            Y_new ^= h32;
        }
        Y_new &= KMASK;
        cpu_Y[i] = Y_new;
        
        for (int j = 0; j < N_META; j++) {
            uint32_t h32;
            memcpy(&h32, hash + j * 4, 4);
            cpu_M[i * N_META + j] = h32 & KMASK;
        }
    }
    
    std::cout << "CPU computed " << N << " hashes" << std::endl;
    
    // GPU eval_p1_tx
    cl_kernel eval_k = clCreateKernel(prog, "eval_p1_tx", &err);
    
    cl_mem C_in_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, C_in.size() * 4, (void*)C_in.data(), &err);
    cl_mem LR_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, LR_data.size() * 4, (void*)LR_data.data(), &err);
    cl_mem num_found_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4, (void*)&N, &err);
    cl_mem C_out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, NUM_BUCKETS * MAX_BUCKET_SIZE * N_META * 4, nullptr, &err);
    cl_mem Y_out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, NUM_BUCKETS * MAX_BUCKET_SIZE * 4, nullptr, &err);
    cl_mem bucket_size_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_BUCKETS * 4, nullptr, &err);
    cl_mem PD_out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, NUM_BUCKETS * MAX_BUCKET_SIZE * 40, nullptr, &err);  // PDSIZE=40
    
    int zero = 0;
    clEnqueueFillBuffer(queue, bucket_size_buf, &zero, 4, 0, NUM_BUCKETS * 4, 0, nullptr, nullptr);
    clEnqueueFillBuffer(queue, PD_out_buf, &zero, 4, 0, NUM_BUCKETS * MAX_BUCKET_SIZE * 40, 0, nullptr, nullptr);
    
    cl_mem null_mem = nullptr;
    ulong PD_0 = 0;
    uint max_bs = MAX_BUCKET_SIZE;
    uint x2size = 39, xbits = 20, table = 2;
    uint write_y = 1, write_c = 1, has_pd_in = 0, has_x_in = 0;
    
    clSetKernelArg(eval_k, 0, sizeof(cl_mem), &Y_out_buf);
    clSetKernelArg(eval_k, 1, sizeof(cl_mem), &C_out_buf);
    clSetKernelArg(eval_k, 2, sizeof(cl_mem), &PD_out_buf);
    clSetKernelArg(eval_k, 3, sizeof(cl_mem), &bucket_size_buf);
    clSetKernelArg(eval_k, 4, sizeof(cl_mem), &C_in_buf);
    clSetKernelArg(eval_k, 5, sizeof(cl_mem), &null_mem);  // PD_in = null
    clSetKernelArg(eval_k, 6, sizeof(cl_mem), &null_mem);  // X_in = null
    clSetKernelArg(eval_k, 7, sizeof(cl_mem), &LR_buf);
    clSetKernelArg(eval_k, 8, sizeof(cl_mem), &num_found_buf);
    clSetKernelArg(eval_k, 9, sizeof(ulong), &PD_0);
    clSetKernelArg(eval_k, 10, sizeof(uint), &max_bs);
    clSetKernelArg(eval_k, 11, sizeof(uint), &x2size);
    clSetKernelArg(eval_k, 12, sizeof(uint), &xbits);
    clSetKernelArg(eval_k, 13, sizeof(uint), &table);
    clSetKernelArg(eval_k, 14, sizeof(uint), &write_y);
    clSetKernelArg(eval_k, 15, sizeof(uint), &write_c);
    clSetKernelArg(eval_k, 16, sizeof(uint), &has_pd_in);
    clSetKernelArg(eval_k, 17, sizeof(uint), &has_x_in);
    
    size_t global = N;
    if (global % 64) global = ((global / 64) + 1) * 64;
    clEnqueueNDRangeKernel(queue, eval_k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    
    // Read back bucket sizes and Y/C values
    std::vector<uint32_t> gpu_bucket_size(NUM_BUCKETS);
    clEnqueueReadBuffer(queue, bucket_size_buf, CL_TRUE, 0, NUM_BUCKETS * 4, gpu_bucket_size.data(), 0, nullptr, nullptr);
    
    std::vector<uint32_t> gpu_Y(NUM_BUCKETS * MAX_BUCKET_SIZE, 0);
    std::vector<uint32_t> gpu_C(NUM_BUCKETS * MAX_BUCKET_SIZE * N_META, 0);
    clEnqueueReadBuffer(queue, Y_out_buf, CL_TRUE, 0, NUM_BUCKETS * MAX_BUCKET_SIZE * 4, gpu_Y.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, C_out_buf, CL_TRUE, 0, NUM_BUCKETS * MAX_BUCKET_SIZE * N_META * 4, gpu_C.data(), 0, nullptr, nullptr);
    
    // Collect GPU Y and M values (sorted by bucket)
    std::vector<uint32_t> gpu_Y_all, gpu_M_all;
    for (int b = 0; b < NUM_BUCKETS; b++) {
        for (int j = 0; j < (int)gpu_bucket_size[b]; j++) {
            gpu_Y_all.push_back(gpu_Y[b * MAX_BUCKET_SIZE + j]);
            for (int m = 0; m < N_META; m++) {
                gpu_M_all.push_back(gpu_C[b * MAX_BUCKET_SIZE * N_META + j * N_META + m]);
            }
        }
    }
    
    // Compare: sort both CPU and GPU by Y value, then compare
    // Note: the bucket assignment depends on Y_new >> (KSIZE - LOGBUCKETS), so same Y → same bucket.
    // But within bucket, order depends on atomic_add (non-deterministic). Sort to compare.
    
    // Create sortable entries: (Y, M[0..13])
    std::vector<std::pair<uint32_t, std::array<uint32_t, 14>>> cpu_entries, gpu_entries;
    for (int i = 0; i < N; i++) {
        std::array<uint32_t, 14> m;
        for (int j = 0; j < 14; j++) m[j] = cpu_M[i * 14 + j];
        cpu_entries.push_back({cpu_Y[i], m});
    }
    for (size_t i = 0; i < gpu_Y_all.size(); i++) {
        std::array<uint32_t, 14> m;
        for (int j = 0; j < 14; j++) m[j] = gpu_M_all[i * 14 + j];
        gpu_entries.push_back({gpu_Y_all[i], m});
    }
    
    std::sort(cpu_entries.begin(), cpu_entries.end());
    std::sort(gpu_entries.begin(), gpu_entries.end());
    
    std::cout << "CPU entries: " << cpu_entries.size() << ", GPU entries: " << gpu_entries.size() << std::endl;
    
    int mismatches = 0;
    if (cpu_entries.size() != gpu_entries.size()) {
        std::cout << "❌ Count mismatch!" << std::endl;
        mismatches = 1;
    } else {
        for (size_t i = 0; i < cpu_entries.size(); i++) {
            if (cpu_entries[i].first != gpu_entries[i].first) {
                if (mismatches < 3) std::cout << "Y mismatch at " << i << ": CPU=" << cpu_entries[i].first << " GPU=" << gpu_entries[i].first << std::endl;
                mismatches++;
            } else {
                for (int j = 0; j < 14; j++) {
                    if (cpu_entries[i].second[j] != gpu_entries[i].second[j]) {
                        if (mismatches < 3) std::cout << "M[" << j << "] mismatch at " << i << ": CPU=" << cpu_entries[i].second[j] << " GPU=" << gpu_entries[i].second[j] << std::endl;
                        mismatches++;
                        break;
                    }
                }
            }
        }
    }
    
    std::cout << (mismatches == 0 ? "✅ eval_p1_tx PASSED" : "❌ eval_p1_tx FAILED") << std::endl;
    
    clReleaseMemObject(C_in_buf); clReleaseMemObject(LR_buf); clReleaseMemObject(num_found_buf);
    clReleaseMemObject(C_out_buf); clReleaseMemObject(Y_out_buf); clReleaseMemObject(bucket_size_buf);
    clReleaseMemObject(PD_out_buf);
    clReleaseKernel(eval_k);
    return 0;
}
