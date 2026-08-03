#pragma once

#include "gpu_manager.h"
#include "plot_pipeline.h"
#include <vector>
#include <memory>

namespace mmx {

// Multi-GPU pipeline wrapper.
// Distributes hashing work across multiple GPUs while keeping CPU sort+match unified.
class MultiPipeline {
public:
    MultiPipeline(GPUManager& gpu_mgr, uint32_t ksize = 26)
        : gpu_mgr(gpu_mgr), ksize(ksize), kmask((1u << ksize) - 1) {}

    void init() {
        // Create a pipeline for each GPU
        for(size_t i = 0; i < gpu_mgr.size(); i++) {
            auto pipe = std::make_unique<PlotPipeline>(gpu_mgr[i], ksize);
            pipe->init();
            pipelines.push_back(std::move(pipe));
        }
    }

    // Compute F1 across all GPUs (split X values)
    void compute_f1(
        const std::vector<uint32_t>& X_values,
        const uint32_t* plot_id,
        std::vector<uint32_t>& Y_out,
        std::vector<uint32_t>& M_out);

    // Process one table using all GPUs for hash
    // Returns timing info aggregated across all GPUs
    TableTiming process_table(
        std::vector<PlotEntry>& entries,
        const std::vector<uint32_t>* x_values_orig = nullptr,
        std::vector<PDEntry>* pd_out = nullptr);

    // Run full pipeline
    void run_full_pipeline(
        const std::vector<uint32_t>& X_values,
        const uint32_t* plot_id,
        PlotData& result);

    // Access underlying pipelines
    PlotPipeline& operator[](size_t i) { return *pipelines[i]; }

private:
    GPUManager& gpu_mgr;
    uint32_t ksize;
    uint32_t kmask;
    std::vector<std::unique_ptr<PlotPipeline>> pipelines;

    // Per-GPU buffer handles (reusable)
    struct GPUBufs {
        cl_mem L_buf = nullptr;
        cl_mem R_buf = nullptr;
        cl_mem Y_buf = nullptr;
        cl_mem M_buf = nullptr;
        size_t capacity = 0;
    };
    std::vector<GPUBufs> gpu_bufs;
    void ensure_gpu_bufs(int gpu_idx, size_t n_matches);
};

} // namespace mmx
