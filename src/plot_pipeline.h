#ifndef MMX_PLOT_PIPELINE_H
#define MMX_PLOT_PIPELINE_H

#include "gpu_device.h"
#include "plot_config.h"
#include <vector>
#include <array>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <numeric>
#include <omp.h>

namespace mmx {

// One table entry: (Y, metadata[0..N_META-1], orig_idx)
struct PlotEntry {
    uint32_t Y;
    uint32_t orig_idx;  // original index in the PREVIOUS table
    std::array<uint32_t, N_META> M;
};

// Matched pair: (sorted_L_index, sorted_R_index)
using MatchPair = std::pair<uint32_t, uint32_t>;

// PD entry: (original_sorted_pos, delta)
using PDEntry = std::pair<uint32_t, uint16_t>;

// Timing info for one table
struct TableTiming {
    double sort_ms = 0;
    double match_ms = 0;
    double extract_ms = 0;
    double hash_ms = 0;
    double build_ms = 0;
    uint32_t n_entries = 0;
    uint32_t n_matches = 0;
};

// Full plot data collected during pipeline execution
struct PlotData {
    // Final Y values (table 9 entries = proof targets)
    std::vector<uint32_t> final_Y;
    
    // Per-table entries (table_entries[0] = F1, ..., table_entries[7] = F9)
    std::vector<std::vector<PlotEntry>> table_entries;
    
    // PD data: PD[table_idx] = vector of PD entries for that table
    // PD[0] corresponds to table 2→1 (from sorted positions in table 1)
    // PD[t-2] corresponds to table t→t-1 (matching table t to its predecessor)
    std::vector<std::vector<PDEntry>> pd_data;
    
    // X pairs for table 2 (original X values for proof reconstruction)
    std::vector<uint32_t> x_pairs;
    
    // Per-table timings
    std::vector<TableTiming> timings;
};

class PlotPipeline {
public:
    PlotPipeline(GPUDevice& gpu, uint32_t ksize = 26);
    ~PlotPipeline();

    // Disable copy
    PlotPipeline(const PlotPipeline&) = delete;
    PlotPipeline& operator=(const PlotPipeline&) = delete;

    // Initialize kernels
    void init();

    // Expose table hash kernel for multi-GPU use
    cl_kernel get_table_hash_kernel() { return k_table_hash; }

    // Compute F1 with batch X values
    void compute_f1(
        const std::vector<uint32_t>& X_values,
        const uint32_t* plot_id,
        std::vector<uint32_t>& Y_out,
        std::vector<uint32_t>& M_out);

    // Process one table (F2-F9): sort → match → hash → build next
    // entries: in/out — sorted in place, replaced with next table entries
    // pd_out: output PD data (one entry per matched pair)
    // x_pairs_out: for table 2, original X values (empty otherwise)
    // Returns timing info
    TableTiming process_table(
        std::vector<PlotEntry>& entries,
        const std::vector<uint32_t>* x_values_orig,
        std::vector<PDEntry>* pd_out,
        std::vector<MatchPair>* LR_pairs = nullptr);

    // Run full pipeline with PD and X-pair collection
    void run_full_pipeline(
        const std::vector<uint32_t>& X_values,
        const uint32_t* plot_id,
        PlotData& result);

    // Public sort/match for multi-GPU distribution
    void sort_entries_by_y(std::vector<PlotEntry>& entries);
    std::vector<MatchPair> match_entries(const std::vector<PlotEntry>& entries);

private:
    GPUDevice& gpu;
    uint32_t ksize;
    uint32_t kmask;

    // Kernels
    cl_kernel k_f1 = nullptr;
    cl_kernel k_gen_mem = nullptr;
    cl_kernel k_calc_memhash = nullptr;
    cl_kernel k_scatter_f1 = nullptr;
    bool f1_warp_loaded = false;
    cl_kernel k_table_hash = nullptr;

    // GPU buffers (reused)
    cl_mem f1_X_buf = nullptr;
    cl_mem f1_ID_buf = nullptr;
    cl_mem f1_Y_buf = nullptr;
    cl_mem f1_M_buf = nullptr;

    cl_mem hash_L_buf = nullptr;
    cl_mem hash_R_buf = nullptr;
    cl_mem hash_Y_buf = nullptr;
    cl_mem hash_M_buf = nullptr;

    // GPU-resident M buffer management
    cl_mem M_curr_gpu = nullptr;  // current metadata (read for hash)
    cl_mem M_out_gpu = nullptr;   // output metadata (written by hash)
    cl_kernel k_table_hash_lr = nullptr;
    bool use_gpu_resident = false;
    int g_hash_local = 64;
    size_t gpu_resident_capacity = 0;

    size_t f1_buf_size = 0;
    size_t hash_buf_size = 0;

    void ensure_f1_buffers(size_t num_x);
    void ensure_hash_buffers(size_t num_matches);
    void ensure_gpu_resident_buffers(size_t num_entries);
    bool init_hash_lr_kernel();
    void load_kernels();
};

} // namespace mmx

#endif // MMX_PLOT_PIPELINE_H
