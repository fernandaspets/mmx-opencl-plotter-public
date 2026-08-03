// mmx_opencl_plotter — MMX OpenCL Plotter
// Usage: mmx_opencl_plotter <plot_id_hex> <farmer_key_hex> [output_dir] [options]
#include "gpu_device.h"
#include "plot_pipeline.h"
#include "plot_writer.h"
#include <mmx/PlotHeader.hxx>
#include <mmx/hash_t.hpp>
#include <mmx/pubkey_t.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <fstream>
#include <filesystem>

int main(int argc, char** argv) {
    if(argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <plot_id_hex> <farmer_key_hex> [output_dir] [options]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --k N           Set plot k-size (default: 26, range " << mmx::MIN_KSIZE << "-" << mmx::MAX_KSIZE << ")" << std::endl;
        std::cerr << "  --limit N       Limit entries (test mode)" << std::endl;
        std::cerr << "  --device N      GPU device index (default: 0)" << std::endl;
        std::cerr << "  --ramdisk DIR   Use tmpfs at DIR for fast plotting" << std::endl;
        std::cerr << "  --no-meta       SSD mode (no metadata table, ~55% smaller)" << std::endl;
        std::cerr << "  --clevel N      Compression level 0-15 (default: 0)" << std::endl;
        return 1;
    }

    std::string pid_str = argv[1];
    std::string fk_str = argv[2];
    std::string output_dir = "./";
    std::string ramdisk_dir;
    uint32_t ksize = 26;
    uint32_t clevel = 0;  // compression level C0
    uint64_t limit = 0;
    int device_id = 0;
    bool has_meta = true;

    for(int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if(arg == "--k" && i+1 < argc) ksize = std::stoi(argv[++i]);
        else if(arg == "--limit" && i+1 < argc) limit = std::stoull(argv[++i]);
        else if(arg == "--device" && i+1 < argc) device_id = std::stoi(argv[++i]);
        else if(arg == "--ramdisk" && i+1 < argc) ramdisk_dir = argv[++i];
        else if(arg == "--no-meta") has_meta = false;
        else if(arg == "--clevel" && i+1 < argc) clevel = std::stoi(argv[++i]);
        else output_dir = arg;
    }

    if(ksize < mmx::MIN_KSIZE || ksize > mmx::MAX_KSIZE) {
        std::cerr << "Error: ksize must be " << mmx::MIN_KSIZE << "-" << mmx::MAX_KSIZE << std::endl;
        return 1;
    }

    // Parse hex keys
    mmx::hash_t plot_id;
    mmx::pubkey_t farmer_key;
    try {
        plot_id.from_string(pid_str);
        farmer_key.from_string(fk_str);
    } catch(const std::exception& e) {
        std::cerr << "Error parsing keys: " << e.what() << std::endl;
        return 1;
    }

    uint64_t num_x = uint64_t(1) << ksize;
    if(limit > 0 && limit < num_x) num_x = limit;

    std::string plot_name = "mmx-" + plot_id.to_string() + "-k" + std::to_string(ksize)
        + (has_meta ? "" : "-ssd") + ".plot";

    std::string output_path = ramdisk_dir.empty()
        ? output_dir + "/" + plot_name
        : ramdisk_dir + "/" + plot_name;

    std::cout << "=== MMX OpenCL Plotter ===" << std::endl;
    std::cout << "  Plot ID: " << plot_id.to_string() << std::endl;
    std::cout << "  Farmer Key: " << farmer_key.to_string() << std::endl;
    std::cout << "  K-Size: " << ksize << std::endl;
    std::cout << "  Compression: C" << clevel << " (xbits=" << (ksize - clevel) << ")" << std::endl;
    std::cout << "  Entries: " << num_x << " (2^" << ksize << ")" << std::endl;
    std::cout << "  Output: " << output_path << std::endl;
    std::cout << "  Mode: " << (has_meta ? "HDD" : "SSD") << std::endl;

    // GPU init
    mmx::GPUDevice gpu;
    try { gpu.init(device_id); gpu.print_info(); }
    catch(const std::exception& e) {
        std::cerr << "GPU init failed: " << e.what() << std::endl;
        return 1;
    }

    // Pipeline
    mmx::PlotPipeline pipeline(gpu, ksize);
    pipeline.init();

    // Generate X values (0, 1, 2, ..., num_x-1)
    std::cout << "[F1] Generating " << num_x << " X values..." << std::endl;
    std::vector<uint32_t> X_values(num_x);
    #pragma omp parallel for schedule(static)
    for(uint64_t i = 0; i < num_x; i++) X_values[i] = (uint32_t)i;

    uint32_t plot_id_raw[8];
    std::memcpy(plot_id_raw, plot_id.data(), 32);

    auto t0 = std::chrono::high_resolution_clock::now();

    // Run pipeline
    mmx::PlotData result;
    try { pipeline.run_full_pipeline(X_values, plot_id_raw, result); }
    catch(const std::exception& e) {
        std::cerr << "Pipeline failed: " << e.what() << std::endl;
        return 1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double pipeline_sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n=== Pipeline Complete ===" << std::endl;
    std::cout << "  Time: " << pipeline_sec << "s" << std::endl;
    std::cout << "  Final entries: " << result.final_Y.size() << std::endl;

    if(result.final_Y.empty()) {
        std::cout << "No matches found — plot would be empty." << std::endl;
        return 1;
    }

    // Write plot file
    uint32_t xbits_val = ksize - clevel;  // bits kept = ksize - C
    std::cout << "\n[Plot] Writing (C" << clevel << ", xbits=" << xbits_val << ")..." << std::endl;
    mmx::PlotWriter writer(ksize, xbits_val, has_meta);

    uint8_t fk_raw[33];
    std::memset(fk_raw, 0, 33);
    std::memcpy(fk_raw, farmer_key.data(), std::min((size_t)32, farmer_key.size()));

    try {
        writer.write_file(output_path,
            (const uint8_t*)plot_id.data(),
            fk_raw,
            result);
    } catch(const std::exception& e) {
        std::cerr << "Write failed: " << e.what() << std::endl;
        return 1;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t2 - t0).count();

    // File size
    std::ifstream f(output_path, std::ios::binary | std::ios::ate);
    uint64_t file_size = f.tellg();

    std::cout << "\n=== Plot Complete ===" << std::endl;
    std::cout << "  File: " << output_path << std::endl;
    std::cout << "  Size: " << (file_size / 1e6) << " MB" << std::endl;
    std::cout << "  Time: " << total_sec << "s" << std::endl;

    // Copy from ramdisk if needed
    if(!ramdisk_dir.empty() && output_dir != "./") {
        std::string final_path = output_dir + "/" + plot_name;
        std::cout << "[Plot] Copying to " << final_path << "..." << std::endl;
        std::filesystem::copy(output_path, final_path,
            std::filesystem::copy_options::overwrite_existing);
        std::filesystem::remove(output_path);
        std::cout << "[Plot] Copied" << std::endl;
    }

    return 0;
}
