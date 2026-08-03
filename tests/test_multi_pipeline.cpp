// test_multi_pipeline.cpp — Dual-GPU pipeline test
#include "../src/gpu_manager.h"
#include "../src/multi_pipeline.h"
#include <iostream>

int main() {
    std::cout << "=== test_multi_pipeline ===" << std::endl;
    
    mmx::GPUManager gpu_mgr;
    gpu_mgr.init(0, 0);  // all available GPUs
    gpu_mgr.print_info();
    
    const uint32_t ksize = 18;
    const uint32_t num_x = 10000;
    
    mmx::MultiPipeline pipeline(gpu_mgr, ksize);
    pipeline.init();
    
    std::vector<uint32_t> X_values(num_x);
    for(uint32_t i = 0; i < num_x; i++) X_values[i] = i;
    uint32_t plot_id[8] = {0};
    
    mmx::PlotData result;
    pipeline.run_full_pipeline(X_values, plot_id, result);
    
    std::cout << "\nFinal entries: " << result.final_Y.size() << std::endl;
    std::cout << "=== TEST PASSED ===" << std::endl;
    return 0;
}
