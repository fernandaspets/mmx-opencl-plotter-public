// ============================================================================
// SVMPool — Shared Virtual Memory buffers for OpenCL 2.0 (Module H)
// ============================================================================
// SVM buffers are shared between CPU and GPU — no cl_mem, no explicit copy.
// Coarse-grain: requires clEnqueueSVMMap/Unmap before CPU access.
// Fine-grain (AMD): CPU and GPU can access simultaneously without map/unmap.
// Migration: clEnqueueMigrateMemObjects moves data between host and device.
// ============================================================================

#pragma once
#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <cstring>

struct SVMPool {
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    
    // SVM allocations — these are void* pointers shared between CPU and GPU
    void* svm_C_in = nullptr;       // [max_bucket_size * N_META] metadata input
    void* svm_PY = nullptr;          // [num_sub * max_bs2 * 2] scatter output
    void* svm_sub_cnt = nullptr;     // [num_sub] sub-bucket counts
    void* svm_sub_off = nullptr;     // [num_sub + 1] sub-bucket offsets
    void* svm_LR = nullptr;          // [max_total * 4 * 2] match pairs
    void* svm_PD_match = nullptr;    // [max_total * 4] PD per match
    void* svm_num_matches = nullptr; // [1] match count
    void* svm_L_gathered = nullptr;  // [max_matches * N_META]
    void* svm_R_gathered = nullptr;  // [max_matches * N_META]
    void* svm_Y_hash = nullptr;      // [max_matches]
    void* svm_M_hash = nullptr;      // [max_matches * N_META]
    
    size_t max_matches = 0;  // max_bucket_size * 4, stored for kernel launches
    
    // F1 SVM buffers
    void* svm_F1_X = nullptr;        // [batch_size] X values
    void* svm_F1_Y = nullptr;        // [batch_size] Y output
    void* svm_F1_M = nullptr;        // [batch_size * N_META] M output
    
    // cl_mem wrappers for coarse-grain SVM (needed for kernel args on some platforms)
    cl_mem mem_C_in = nullptr;
    cl_mem mem_PY = nullptr;
    cl_mem mem_sub_cnt = nullptr;
    cl_mem mem_sub_off = nullptr;
    cl_mem mem_LR = nullptr;
    cl_mem mem_PD_match = nullptr;
    cl_mem mem_num_matches = nullptr;
    cl_mem mem_L_gathered = nullptr;
    cl_mem mem_R_gathered = nullptr;
    cl_mem mem_Y_hash = nullptr;
    cl_mem mem_M_hash = nullptr;
    
    bool fine_grain = false;  // AMD supports fine-grain (no map/unmap needed)
    bool initialized = false;
    
    void init(cl_context ctx, cl_command_queue q, cl_device_id dev,
              int max_bucket_size, int n_meta, int num_sub, int max_bs2,
              int f1_batch_size = 0) {
        context = ctx;
        queue = q;
        device = dev;
        cl_int err;
        
        // Check if fine-grain SVM is supported
        cl_device_svm_capabilities svm_caps = 0;
        err = clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm_caps), &svm_caps, nullptr);
        if(err == CL_SUCCESS && (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)) {
            fine_grain = true;
            std::cout << "[OCL] SVM: fine-grain buffer supported (no map/unmap needed)" << std::endl;
        } else {
            fine_grain = false;
            std::cout << "[OCL] SVM: coarse-grain only (will use map/unmap)" << std::endl;
        }
        
        // Allocation flags
        cl_mem_flags alloc_flags = CL_MEM_READ_WRITE;
        if(fine_grain) {
            alloc_flags |= CL_MEM_SVM_FINE_GRAIN_BUFFER;
        }
        
        size_t max_total = (size_t)max_bucket_size;
        max_matches = (size_t)max_bucket_size * 4;
        
        // Allocate SVM buffers
        svm_C_in = clSVMAlloc(context, CL_MEM_READ_ONLY | (fine_grain ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0),
            max_bucket_size * n_meta * 4, 0);
        svm_PY = clSVMAlloc(context, CL_MEM_READ_WRITE | (fine_grain ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0),
            (size_t)num_sub * max_bs2 * 8, 0);
        svm_sub_cnt = clSVMAlloc(context, CL_MEM_READ_WRITE | (fine_grain ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0),
            num_sub * 4, 0);
        svm_sub_off = clSVMAlloc(context, CL_MEM_READ_WRITE | (fine_grain ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0),
            (num_sub + 1) * 4, 0);
        svm_LR = clSVMAlloc(context, CL_MEM_WRITE_ONLY | (fine_grain ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0),
            max_total * 4 * 8, 0);
        svm_PD_match = clSVMAlloc(context, CL_MEM_WRITE_ONLY | (fine_grain ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0),
            max_total * 4 * 4, 0);
        svm_num_matches = clSVMAlloc(context, CL_MEM_READ_WRITE | (fine_grain ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0),
            4, 0);
        
        // Module B gather/hash buffers
        svm_L_gathered = clSVMAlloc(context, CL_MEM_WRITE_ONLY | (fine_grain ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0),
            max_matches * n_meta * 4, 0);
        svm_R_gathered = clSVMAlloc(context, CL_MEM_WRITE_ONLY | (fine_grain ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0),
            max_matches * n_meta * 4, 0);
        svm_Y_hash = clSVMAlloc(context, CL_MEM_WRITE_ONLY | (fine_grain ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0),
            max_matches * 4, 0);
        svm_M_hash = clSVMAlloc(context, CL_MEM_WRITE_ONLY | (fine_grain ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0),
            max_matches * n_meta * 4, 0);
        
        // F1 SVM buffers (optional — only if f1_batch_size > 0)
        if(f1_batch_size > 0) {
            svm_F1_X = clSVMAlloc(context, CL_MEM_READ_ONLY | (fine_grain ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0),
                f1_batch_size * sizeof(uint32_t), 0);
            svm_F1_Y = clSVMAlloc(context, CL_MEM_WRITE_ONLY | (fine_grain ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0),
                f1_batch_size * sizeof(uint32_t), 0);
            svm_F1_M = clSVMAlloc(context, CL_MEM_WRITE_ONLY | (fine_grain ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0),
                f1_batch_size * n_meta * 4, 0);
            if(svm_F1_X && svm_F1_Y && svm_F1_M)
                std::cout << "[OCL] SVM F1 buffers allocated (batch_size=" << f1_batch_size << ")" << std::endl;
        }
        
        if(!svm_C_in || !svm_PY || !svm_sub_cnt || !svm_LR || !svm_num_matches ||
           !svm_L_gathered || !svm_Y_hash) {
            std::cerr << "[OCL] SVM allocation failed!" << std::endl;
            return;
        }
        
        initialized = true;
        std::cout << "[OCL] SVM pool initialized (" << (fine_grain ? "fine-grain" : "coarse-grain") << ")" << std::endl;
    }
    
    // Map SVM buffer for CPU access (coarse-grain only — fine-grain doesn't need this)
    void map(void* svm_ptr, size_t size, cl_map_flags flags = CL_MAP_READ | CL_MAP_WRITE) {
        if(!fine_grain && svm_ptr) {
            clEnqueueSVMMap(queue, CL_TRUE, flags, svm_ptr, size, 0, nullptr, nullptr);
        }
    }
    
    void unmap(void* svm_ptr) {
        if(!fine_grain && svm_ptr) {
            clEnqueueSVMUnmap(queue, svm_ptr, 0, nullptr, nullptr);
        }
    }
    
    // Migrate SVM buffer to device (hint to driver to move data to GPU)
    void migrate_to_device(void* svm_ptr, size_t size) {
        if(svm_ptr) {
            // clEnqueueSVMMigrateMemObjects is OpenCL 2.1+, use clEnqueueMigrateMemObjects with cl_mem wrapper
            // For coarse-grain, just make sure data is on device before kernel runs
            // The in-order queue ensures this happens before kernel execution
        }
    }
    
    void cleanup() {
        if(!initialized) return;
        if(svm_C_in) clSVMFree(context, svm_C_in);
        if(svm_PY) clSVMFree(context, svm_PY);
        if(svm_sub_cnt) clSVMFree(context, svm_sub_cnt);
        if(svm_sub_off) clSVMFree(context, svm_sub_off);
        if(svm_LR) clSVMFree(context, svm_LR);
        if(svm_PD_match) clSVMFree(context, svm_PD_match);
        if(svm_num_matches) clSVMFree(context, svm_num_matches);
        if(svm_L_gathered) clSVMFree(context, svm_L_gathered);
        if(svm_R_gathered) clSVMFree(context, svm_R_gathered);
        if(svm_Y_hash) clSVMFree(context, svm_Y_hash);
        if(svm_M_hash) clSVMFree(context, svm_M_hash);
        if(svm_F1_X) clSVMFree(context, svm_F1_X);
        if(svm_F1_Y) clSVMFree(context, svm_F1_Y);
        if(svm_F1_M) clSVMFree(context, svm_F1_M);
        initialized = false;
    }
};
