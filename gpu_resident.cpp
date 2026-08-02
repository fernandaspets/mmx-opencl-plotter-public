// ============================================================================
// GPU-Resident Pipeline: F2-F9 entirely on GPU, no CPU↔GPU transfers per table.
// Reuses chunked kernels (scatter_2, hybrid_sort_y, match_p1, eval_p1_tx).
// Processes all L1 buckets per table, keeping data on GPU between tables.
//
// Key optimization: NO PCIe transfers between tables.
// Only transfers: initial F1 upload + final download.
// Eliminates ~18s of CPU overhead for k26 (match, flatten, sort, PD).
// ============================================================================

#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <cstring>

struct PlotData;

// External refs to plotter types
struct OCL_Plotter;

static void compute_gpu_resident(
    const std::vector<uint32_t>& Y_all,
    const std::vector<uint32_t>& M_all,
    struct PlotData& plot,
    struct OCL_Plotter& gpu_plotter_ref,
    const class hash_t& plot_id);
