// GPU Bulk Pipeline: all F2-F9 steps on GPU, no PCIe transfers between tables
// 
// Per table:
// 1. GPU bucket sort Y_in → (Y_sorted, pos_sorted)
// 2. GPU match: scan sorted Y for Y+1 pairs → LR pairs (orig_L, orig_R, sorted_L, sorted_R)
// 3. GPU hash: hash LR pairs → Y_out, M_out
// 4. Swap: Y_in = Y_out, M_curr = M_out
// 5. Save LR pairs for PD build (download or keep on GPU)
//
// After all tables: download final Y and M, build PD and X_pairs

#include "pipeline.h"
#include <vector>
#include <iostream>
#include <chrono>

// Forward declarations from plotter.cpp
extern OCL_Plotter g_plotter;

void compute_gpu_bulk(
    const std::vector<uint32_t>& Y_all,
    const std::vector<uint32_t>& M_all,
    PlotData& plot,
    OCL_Plotter& gpu_plotter,
    const hash_t& plot_id)
{
    // TODO: implement
    std::cout << "[GPU-Bulk] Not yet implemented" << std::endl;
}
