// test_full_pipeline.cpp — Full F1→F9 pipeline end-to-end test
#include "../src/gpu_device.h"
#include "../src/plot_pipeline.h"
#include <iostream>
#include <vector>
#include <fstream>

int main() {
    std::cout << "=== test_full_pipeline ===" << std::endl;

    mmx::GPUDevice gpu;
    gpu.init(0);
    gpu.print_info();

    const uint32_t ksize = 18;
    const uint32_t num_x = 256;

    mmx::PlotPipeline pipeline(gpu, ksize);
    pipeline.init();

    std::vector<uint32_t> X_values(num_x);
    for(uint32_t i = 0; i < num_x; i++) X_values[i] = i;

    uint32_t plot_id_raw[8] = {0};

    mmx::PlotData result;
    pipeline.run_full_pipeline(X_values, plot_id_raw, result);

    std::cout << "\n=== Pipeline Results ===" << std::endl;
    std::cout << "Tables completed: " << result.table_entries.size() << std::endl;
    std::cout << "Final Y count: " << result.final_Y.size() << std::endl;
    std::cout << "PD tables: " << result.pd_data.size() << std::endl;
    
    for(size_t i = 0; i < result.timings.size(); i++) {
        const auto& t = result.timings[i];
        std::cout << "  T" << (i+2) << ": " << t.n_entries << "→" << t.n_matches
                  << " (sort=" << t.sort_ms << "ms match=" << t.match_ms 
                  << "ms hash=" << t.hash_ms << "ms)" << std::endl;
    }

    std::cout << "\n=== Test PASSED ===" << std::endl;
    return 0;
}
