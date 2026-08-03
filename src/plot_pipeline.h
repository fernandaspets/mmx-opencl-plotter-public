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

// One table entry: (Y, metadata[0..N_META-1])
struct PlotEntry {
    uint32_t Y;
    std::array<uint32_t, N_META> M;
};

// Matched pair: (sorted_L_index, sorted_R_index)
using MatchPair = std::pair<uint32_t, uint32_t>;

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

class PlotPipeline {
public:
    PlotPipeline(GPUDevice& gpu, uint32_t ksize = 26);
    ~PlotPipeline();

    // Disable copy
    PlotPipeline(const PlotPipeline&) = delete;
    PlotPipeline& operator=(const PlotPipeline&) = delete;

    // Initialize kernels
    void init();

    // Compute F1 for all X values (Phase 1)
    void compute_f1(
        const std::vector<uint32_t>& X_values,
        const uint32_t* plot_id,
        std::vector<uint32_t>& Y_out,
        std::vector<uint32_t>& M_out);

    // Compute one table (F2-F9): sort → match → hash → build next
    // entries: in/out — sorted in place, replaced with next table entries
    // Returns match pairs and timing info
    TableTiming process_table(
        std::vector<PlotEntry>& entries);

    // Full pipeline: F1 → F2 → ... → F9
    // X_values: input X values
    // plot_id: 32-byte plot identifier
    // Run full F1→F9 pipeline
    // Returns per-table entries and timing info
    void run_full_pipeline(
        const std::vector<uint32_t>& X_values,
        const uint32_t* plot_id,
        std::vector<std::vector<PlotEntry>>& table_entries,
        std::vector<std::vector<MatchPair>>& table_matches,
        std::vector<TableTiming>& timings);

private:
    GPUDevice& gpu;
    uint32_t ksize;
    uint32_t kmask;

    // Kernels
    cl_kernel k_f1 = nullptr;
    cl_kernel k_table_hash = nullptr;

    // GPU buffers (reused across tables)
    cl_mem f1_X_buf = nullptr;
    cl_mem f1_ID_buf = nullptr;
    cl_mem f1_Y_buf = nullptr;
    cl_mem f1_M_buf = nullptr;

    cl_mem hash_L_buf = nullptr;
    cl_mem hash_R_buf = nullptr;
    cl_mem hash_Y_buf = nullptr;
    cl_mem hash_M_buf = nullptr;

    size_t f1_buf_size = 0;
    size_t hash_buf_size = 0;

    void ensure_f1_buffers(size_t num_x);
    void ensure_hash_buffers(size_t num_matches);
    void load_kernels();

    // Sort entries by Y (8-bit radix + std::sort within buckets)
    // Uses entries[n] = {(Y, idx)}
    // On output: entries are sorted by Y with metadata tiebreaker
    void sort_entries_by_y(std::vector<PlotEntry>& entries);

    // Match Y,Y+1 pairs within sorted entries
    // Returns (sorted_L_idx, sorted_R_idx)
    std::vector<MatchPair> match_entries(
        const std::vector<PlotEntry>& entries);
};

} // namespace mmx

#endif // MMX_PLOT_PIPELINE_H
