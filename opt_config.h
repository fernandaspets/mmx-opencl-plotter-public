// ============================================================================
// Optimization Configuration — modular flags for pipeline optimization
// ============================================================================
// Each flag enables/disables an independent optimization module.
// Modules can be combined in any way to test different configurations.
//
//   --opt-gpu-prefix-sum   Module A: GPU prefix sum (eliminates readback #1)
//   --opt-gpu-meta         Module B: GPU metadata extraction (eliminates readback #2,3 + CPU stall)
//   --opt-async            Module D: Async PCIe transfers (non-blocking)
//   --opt-queues N         Module E: Multi-queue pipelining (N parallel buckets)
//   --opt-fuse             Module F: Kernel fusion (scatter+sort+match in one launch)
//
// Default: all off (original sequential pipeline)
// ============================================================================

#pragma once
#include <string>
#include <iostream>

struct OptConfig {
    // Module A: GPU prefix sum instead of CPU
    bool gpu_prefix_sum = false;
    
    // Module B: GPU metadata extraction (use hash_table_lr kernel directly)
    // Eliminates: LR download, CPU meta extraction, L/R meta re-upload
    bool gpu_meta_extract = false;
    
    // Module D: Async (non-blocking) PCIe transfers
    bool async_transfers = false;
    
    // Module E: Multi-queue pipelining (number of parallel queues)
    int num_queues = 1;
    
    // Module F: Kernel fusion (combine scatter+sort+match)
    bool fuse_kernels = false;
    
    // Module C: Pre-allocated buffer pool (reuse GPU buffers across buckets)
    bool bufpool = false;
    
    // Module F: Zero-copy / pinned memory (CL_MEM_ALLOC_HOST_PTR + map/unmap)
    bool zero_copy = false;
    
    // Module G: Pinned memory (CL_MEM_ALLOC_HOST_PTR) for async DMA
    bool pinned = false;
    
    // Module H: SVM (Shared Virtual Memory) — OpenCL 2.0
    bool svm = false;
    
    // Helper: any optimization enabled?
    bool any_enabled() const {
        return gpu_prefix_sum || gpu_meta_extract || async_transfers || num_queues > 1 || fuse_kernels || bufpool || zero_copy || pinned || svm;
    }
    
    // Print current config
    void print() const {
        std::cout << "[Opt] Config:" << std::endl;
        std::cout << "  gpu_prefix_sum:  " << (gpu_prefix_sum ? "ON" : "off") << std::endl;
        std::cout << "  gpu_meta_extract:" << (gpu_meta_extract ? "ON" : "off") << std::endl;
        std::cout << "  async_transfers: " << (async_transfers ? "ON" : "off") << std::endl;
        std::cout << "  num_queues:      " << num_queues << std::endl;
        std::cout << "  fuse_kernels:   " << (fuse_kernels ? "ON" : "off") << std::endl;
        std::cout << "  bufpool:        " << (bufpool ? "ON" : "off") << std::endl;
        std::cout << "  zero_copy:     " << (zero_copy ? "ON" : "off") << std::endl;
        std::cout << "  pinned:        " << (pinned ? "ON" : "off") << std::endl;
        std::cout << "  svm:           " << (svm ? "ON" : "off") << std::endl;
    }
};

// Global config (set from command line args)
inline OptConfig& get_opt_config() {
    static OptConfig cfg;
    return cfg;
}
