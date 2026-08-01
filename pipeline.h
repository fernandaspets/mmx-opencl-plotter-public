// ============================================================================
// Pipeline — 3-phase bucket processing for multi-GPU overlap (Module E v2)
// ============================================================================
// Split process_bucket_gpu into submit → submit_hash → collect phases.
// All GPU work is enqueued non-blocking with events. CPU only blocks when
// it needs the data. This allows 2 GPUs to process buckets concurrently
// without threads — just 2 command queues with event-based sync.
// ============================================================================

#pragma once
#include <CL/cl.h>
#include <vector>
#include <cstring>
#include <chrono>
#include <iostream>
#include "svm_pool.h"

// Forward declarations
class OCL_Plotter;
struct BufferPool;
struct MemBucketStore;

// Pending state for one bucket being processed through the pipeline
struct BucketPending {
    bool skip = true;              // true until submit populates it
    bool zero_matches = false;     // true if gpu_matches came back 0

    int y = -1, table = -1;
    OCL_Plotter* active_plotter = nullptr;
    BufferPool* active_pool = nullptr;
    SVMPool* active_svm = nullptr;  // Module H: SVM pool
    bool using_svm = false;
    bool using_pool = false;
    cl_command_queue q = nullptr;

    int n_meta = 0, shift = 0, num_sub = 0, max_bs2 = 0;
    uint32_t count_y = 0, total = 0, bucket_offset = 0;
    uint32_t max_bs_sort = 0, num_sub_u32 = 0;

    std::vector<uint32_t> C_combined;

    cl_mem C_in_buf = nullptr, PY_buf = nullptr, sub_cnt_buf = nullptr, sub_off_buf = nullptr;
    cl_mem LR_buf = nullptr, PD_match_buf = nullptr, num_matches_buf = nullptr;

    uint32_t gpu_matches = 0;
    std::vector<uint32_t> LR_filtered;
    std::vector<uint32_t> pd_data;
    uint32_t num_filt = 0;

    std::vector<uint32_t> xp_flat;

    cl_mem L_gathered = nullptr, R_gathered = nullptr, Yb = nullptr, Mb = nullptr;
    std::vector<uint32_t> Y_out, M_out;
    size_t num_matches = 0;

    cl_event ev_match_done = nullptr;
    cl_event ev_hash_done = nullptr;
    
    // SVM pointers (Module H) — when using_svm, these replace cl_mem
    void* svm_C_in = nullptr;
    void* svm_PY = nullptr;
    void* svm_sub_cnt = nullptr;
    void* svm_sub_off = nullptr;
    void* svm_LR = nullptr;
    void* svm_PD_match = nullptr;
    void* svm_num_matches = nullptr;
    void* svm_L_gathered = nullptr;
    void* svm_R_gathered = nullptr;
    void* svm_Y_hash = nullptr;
    void* svm_M_hash = nullptr;
};
