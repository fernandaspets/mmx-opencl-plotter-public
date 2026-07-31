// test_match.cpp — Test match_p1 kernel
// Generate sorted PY values with some Y,Y+1 pairs, run match_p1 on GPU, compare with CPU
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <set>

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
    
    const int KSIZE = 20;
    const int MAX_BS = 2048;
    const int NUM_SUB = 4;  // 4 sub-buckets
    
    // Generate sorted PY values with Y,Y+1 pairs
    // All entries in same first-level bucket. Distribute across sub-buckets.
    // Sub-bucket index = Y >> (KSIZE - LOGBUCKETS - LOGBUCKETS2) = Y >> 9
    // For entries to be in same first-level bucket: Y >> (KSIZE - LOGBUCKETS) = same
    // = Y >> 14 = same. So Y < 2^14 * (bucket_id+1)
    
    std::mt19937_64 rng64(42);
    std::vector<uint64_t> PY(NUM_SUB * MAX_BS, 0);
    std::vector<uint32_t> bucket_size(NUM_SUB, 0);
    std::vector<uint32_t> bucket_offset(NUM_SUB, 0);
    
    // Generate entries: for each sub-bucket, create sorted Y values with some Y,Y+1 pairs
    const int ENTRIES_PER_SUB = 200;
    for (int s = 0; s < NUM_SUB; s++) {
        bucket_offset[s] = s * MAX_BS;
        
        // Y values for this sub-bucket: Y >> 9 = s (sub-bucket index)
        // So Y = (s << 9) | lower_9_bits
        std::vector<uint32_t> Y_vals;
        for (int i = 0; i < ENTRIES_PER_SUB; i++) {
            uint32_t lower = rng64() & 0x1FF;
            uint32_t Y = (s << 9) | lower;
            Y_vals.push_back(Y);
        }
        // Sort Y values
        std::sort(Y_vals.begin(), Y_vals.end());
        
        // Store as PY (Y << 44 | position)
        for (int i = 0; i < ENTRIES_PER_SUB; i++) {
            PY[s * MAX_BS + i] = ((uint64_t)Y_vals[i] << (64 - KSIZE)) | (uint64_t)i;
        }
        bucket_size[s] = ENTRIES_PER_SUB;
    }
    
    // CPU reference: find Y,Y+1 pairs
    std::vector<std::pair<uint32_t, uint32_t>> cpu_LR;
    for (int s = 0; s < NUM_SUB; s++) {
        int count = bucket_size[s];
        int next_count = (s + 1 < NUM_SUB) ? bucket_size[s + 1] : 0;
        
        for (int x = 0; x < count; x++) {
            uint32_t YL = PY[s * MAX_BS + x] >> (64 - KSIZE);
            uint32_t PL = (uint32_t)PY[s * MAX_BS + x];
            
            for (int i = x + 1; ; i++) {
                bool is_next = (i >= count);
                int j = is_next ? i - count : i;
                if (is_next && j >= next_count) break;
                
                int src_sub = is_next ? s + 1 : s;
                uint32_t YR = PY[src_sub * MAX_BS + j] >> (64 - KSIZE);
                uint32_t PR = (uint32_t)PY[src_sub * MAX_BS + j];
                
                if (YR == YL + 1) {
                    cpu_LR.push_back({PL, PR});
                } else if (YR > YL) break;
            }
        }
    }
    std::cout << "CPU found " << cpu_LR.size() << " matches" << std::endl;
    
    // GPU match_p1
    cl_kernel match_k = clCreateKernel(prog, "match_p1", &err);
    
    cl_mem PY_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, PY.size() * 8, PY.data(), &err);
    cl_mem size_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bucket_size.size() * 4, bucket_size.data(), &err);
    cl_mem offset_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bucket_offset.size() * 4, bucket_offset.data(), &err);
    cl_mem LR_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 10000 * 8, nullptr, &err);  // generous
    cl_mem PD_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 10000 * 4, nullptr, &err);
    cl_mem num_matches_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 4, nullptr, &err);
    
    int zero = 0;
    clEnqueueFillBuffer(queue, num_matches_buf, &zero, 4, 0, 4, 0, nullptr, nullptr);
    
    uint32_t num_sub = NUM_SUB;
    uint32_t max_bs = MAX_BS;
    uint32_t max_total = 10000;
    uint32_t write_pd = 0;  // table 2, no PD
    
    clSetKernelArg(match_k, 0, sizeof(cl_mem), &LR_buf);
    clSetKernelArg(match_k, 1, sizeof(cl_mem), &PD_buf);
    clSetKernelArg(match_k, 2, sizeof(cl_mem), &num_matches_buf);
    clSetKernelArg(match_k, 3, sizeof(cl_mem), &PY_buf);
    clSetKernelArg(match_k, 4, sizeof(cl_mem), &size_buf);
    clSetKernelArg(match_k, 5, sizeof(cl_mem), &offset_buf);
    clSetKernelArg(match_k, 6, sizeof(uint32_t), &num_sub);
    clSetKernelArg(match_k, 7, sizeof(uint32_t), &max_bs);
    clSetKernelArg(match_k, 8, sizeof(uint32_t), &max_total);
    clSetKernelArg(match_k, 9, sizeof(uint32_t), &write_pd);
    
    // 1 work-group per sub-bucket, 128 threads
    int entries_rounded = ((ENTRIES_PER_SUB + 127) / 128) * 128;
    size_t global[2] = {(size_t)entries_rounded, (size_t)NUM_SUB};
    size_t local[2] = {128, 1};
    clEnqueueNDRangeKernel(queue, match_k, 2, nullptr, global, local, 0, nullptr, nullptr);
    
    uint32_t gpu_num_matches = 0;
    clEnqueueReadBuffer(queue, num_matches_buf, CL_TRUE, 0, 4, &gpu_num_matches, 0, nullptr, nullptr);
    
    std::vector<uint32_t> LR_gpu(10000 * 2, 0);
    clEnqueueReadBuffer(queue, LR_buf, CL_TRUE, 0, 10000 * 8, LR_gpu.data(), 0, nullptr, nullptr);
    
    std::cout << "GPU found " << gpu_num_matches << " matches" << std::endl;
    
    // Compare: extract GPU LR pairs
    std::vector<std::pair<uint32_t, uint32_t>> gpu_LR;
    for (uint32_t i = 0; i < gpu_num_matches; i++) {
        gpu_LR.push_back({LR_gpu[i * 2], LR_gpu[i * 2 + 1]});
    }
    
    // Sort both for comparison (order may differ due to atomics)
    std::sort(cpu_LR.begin(), cpu_LR.end());
    std::sort(gpu_LR.begin(), gpu_LR.end());
    
    int mismatches = 0;
    if (cpu_LR.size() != gpu_LR.size()) {
        std::cout << "Count mismatch: CPU=" << cpu_LR.size() << " GPU=" << gpu_LR.size() << std::endl;
        mismatches++;
    } else {
        for (size_t i = 0; i < cpu_LR.size(); i++) {
            if (cpu_LR[i] != gpu_LR[i]) {
                if (mismatches < 5) std::cout << "Mismatch " << i << ": CPU=(" << cpu_LR[i].first << "," << cpu_LR[i].second 
                    << ") GPU=(" << gpu_LR[i].first << "," << gpu_LR[i].second << ")" << std::endl;
                mismatches++;
            }
        }
    }
    
    std::cout << (mismatches == 0 ? "✅ match_p1 PASSED" : "❌ match_p1 FAILED") << std::endl;
    
    clReleaseMemObject(PY_buf); clReleaseMemObject(size_buf); clReleaseMemObject(offset_buf);
    clReleaseMemObject(LR_buf); clReleaseMemObject(PD_buf); clReleaseMemObject(num_matches_buf);
    clReleaseKernel(match_k);
    return 0;
}
