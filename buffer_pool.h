// ============================================================================
// BufferPool — pre-allocated OpenCL buffers for reuse across buckets (Module C)
// ============================================================================
// Instead of creating+destroying 7 buffers per bucket (14,336 alloc/free cycles),
// we allocate max-size buffers once and reuse them with clEnqueueWriteBuffer.
// This eliminates clCreateBuffer/clReleaseMemObject driver overhead (~500us/bucket).
// ============================================================================

#pragma once
#include <CL/cl.h>
#include <iostream>

struct BufferPool {
    cl_context context;
    cl_command_queue queue;
    
    // Pre-allocated GPU buffers (max size for given k)
    cl_mem C_in_buf;        // [max_bucket_size * N_META] metadata input
    cl_mem PY_buf;           // [num_sub * max_bs2 * 2] scatter output (Y + pos)
    cl_mem sub_cnt_buf;      // [num_sub] sub-bucket counts
    cl_mem sub_off_buf;      // [num_sub + 1] sub-bucket offsets
    cl_mem LR_buf;           // [max_total * 2] match pairs (P1, P2)
    cl_mem PD_match_buf;     // [max_total] PD per match
    cl_mem num_matches_buf;  // [1] match count
    
    // Module B buffers
    cl_mem L_gathered;       // [max_matches * N_META]
    cl_mem R_gathered;       // [max_matches * N_META]
    cl_mem Y_hash_buf;       // [max_matches] hash Y output
    cl_mem M_hash_buf;       // [max_matches * N_META] hash metadata output
    
    bool initialized = false;
    
    // Module F: Mapped host pointers for zero-copy access
    void* mapped_C_in = nullptr;
    void* mapped_LR = nullptr;
    void* mapped_Y = nullptr;
    void* mapped_M = nullptr;
    
    void init(cl_context ctx, cl_command_queue q, 
              int max_bucket_size, int n_meta, int num_sub, int max_bs2) {
        context = ctx;
        queue = q;
        cl_int err;
        
        // Max entries per bucket = max_bucket_size
        // Max matches = max_bucket_size * 4 (match kernel allows 4x entries)
        size_t max_total = (size_t)max_bucket_size;
        size_t max_matches = (size_t)max_bucket_size * 4;
        
        C_in_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
            max_bucket_size * n_meta * 4, nullptr, &err);
        PY_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
            (size_t)num_sub * max_bs2 * 8, nullptr, &err);
        sub_cnt_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
            num_sub * 4, nullptr, &err);
        sub_off_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
            (num_sub + 1) * 4, nullptr, &err);
        LR_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            max_total * 4 * 8, nullptr, &err);  // *4 for match 4x factor
        PD_match_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            max_total * 4 * 4, nullptr, &err);
        num_matches_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
            4, nullptr, &err);
        
        // Module B buffers
        L_gathered = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            max_matches * n_meta * 4, nullptr, &err);
        R_gathered = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            max_matches * n_meta * 4, nullptr, &err);
        Y_hash_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            max_matches * 4, nullptr, &err);
        M_hash_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            max_matches * n_meta * 4, nullptr, &err);
        
        initialized = true;
        std::cout << "[OCL] Buffer pool initialized (reusable buffers)" << std::endl;
    }
    
    void cleanup() {
        if(!initialized) return;
        clReleaseMemObject(C_in_buf);
        clReleaseMemObject(PY_buf);
        clReleaseMemObject(sub_cnt_buf);
        clReleaseMemObject(sub_off_buf);
        clReleaseMemObject(LR_buf);
        clReleaseMemObject(PD_match_buf);
        clReleaseMemObject(num_matches_buf);
        clReleaseMemObject(L_gathered);
        clReleaseMemObject(R_gathered);
        clReleaseMemObject(Y_hash_buf);
        clReleaseMemObject(M_hash_buf);
        initialized = false;
    }
};
