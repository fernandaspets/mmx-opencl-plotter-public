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

    // Small test: k18, 256 X values
    const uint32_t ksize = 18;
    const uint32_t num_x = 256;
    const uint32_t num_x_total = 1u << ksize;  // full range test

    mmx::PlotPipeline pipeline(gpu, ksize);
    pipeline.init();

    // Generate X values: first 256 values (0..255)
    std::vector<uint32_t> X_values(num_x);
    for(uint32_t i = 0; i < num_x; i++) X_values[i] = i;

    // Zero plot_id for test
    std::vector<uint32_t> plot_id(8, 0);
    uint32_t plot_id_raw[8] = {0};
    memcpy(plot_id_raw, plot_id.data(), 32);

    std::cout << "Running F1... " << std::flush;
    std::vector<uint32_t> Y_out, M_out;
    pipeline.compute_f1(X_values, plot_id_raw, Y_out, M_out);
    std::cout << Y_out.size() << " results" << std::endl;

    // Print first few Y values
    for(int i = 0; i < std::min((size_t)8, Y_out.size()); i++) {
        std::cout << "  F1 X=" << X_values[i] << " → Y=" << Y_out[i] << std::endl;
    }

    // Build table 1 entries
    std::vector<mmx::PlotEntry> entries(num_x);
    for(size_t i = 0; i < num_x; i++) {
        entries[i].Y = Y_out[i];
        for(int j = 0; j < mmx::N_META; j++) {
            entries[i].M[j] = M_out[i * mmx::N_META + j];
        }
    }

    // Run tables F2-F9
    for(int t = 2; t <= 9; t++) {
        std::cout << "[T" << t << "] " << entries.size() << " entries... " << std::flush;
        auto timing = pipeline.process_table(entries);
        std::cout << timing.n_matches << " matches "
                  << "(sort=" << timing.sort_ms << "ms "
                  << "match=" << timing.match_ms << "ms "
                  << "hash=" << timing.hash_ms << "ms)" << std::endl;

        if(entries.empty()) {
            std::cout << "  (no entries — stopping)" << std::endl;
            break;
        }
    }

    std::cout << "\n=== Pipeline completed successfully ===" << std::endl;
    return 0;
}
