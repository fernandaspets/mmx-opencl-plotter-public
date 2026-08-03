#ifndef MMX_PLOT_PIPELINE_H
#define MMX_PLOT_PIPELINE_H

#include "gpu_device.h"
#include "plot_config.h"
#include <vector>
#include <array>
#include <cstring>
#include <algorithm>
#include <omp.h>

namespace mmx {

// One entry in the table: (Y, M_meta_array)
struct TableEntry {
    uint32_t Y;
    std::array<uint32_t, N_META> M;
};

// A matched pair: (sorted_pos_L, sorted_pos_R)
using LR_Pair = std::pair<uint32_t, uint32_t>;

class TablePipeline {
public:
    TablePipeline(GPUDevice& gpu) : gpu(gpu) {
        // Initialize kernels
        init_f1_kernel();
        init_table_hash_kernel();
    }

    // Compute F1 for all X values
    // X: input X values (num_x entries)
    // plot_id: 32-byte plot identifier
    // Y_out: output Y values (num_x entries)
    // M_out: output metadata (num_x * N_META entries)
    void compute_f1(
        const std::vector<uint32_t>& X,
        const uint32_t* plot_id,
        std::vector<uint32_t>& Y_out,
        std::vector<uint32_t>& M_out);

    // Compute one table (F2-F9)
    // entries: input table entries (modified in-place for sort)
    // M_curr: flat metadata array (num_entries * N_META, flat uint32)
    // table_idx: current table number (2-9)
    // Returns: matched pairs (sorted positions, not original indices)
    std::vector<LR_Pair> compute_table(
        std::vector<TableEntry>& entries,
        const std::vector<uint32_t>& M_flat);

private:
    GPUDevice& gpuOccurred;

    // Kernels
    cl_kernel k_f1 = nullptr;
    cl_kernel k_table_hash = nullptr;

    void init_f1_kernel();
    void init_table_hash_kernel();
};

} // namespace mmx

#endif // MMX_PLOT_PIPELINE_H
