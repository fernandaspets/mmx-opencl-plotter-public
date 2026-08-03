// mmx_opencl_plotter — MMX OpenCL Plotter
// Usage: mmx_opencl_plotter <plot_id_hex> <farmer_key_hex> [output_dir] [options]
#include "gpu_device.h"
#include "gpu_manager.h"
#include "plot_pipeline.h"
#include "multi_pipeline.h"
#include "plot_writer.h"
#include "pid_derive.h"
#include <mmx/addr_t.hpp>
#include <vnx/vnx.h>
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
        std::cerr << "  --num-gpus N    Number of GPUs to use (default: 1)" << std::endl;
        std::cerr << "  --ramdisk DIR   Use tmpfs at DIR for fast plotting" << std::endl;
        std::cerr << "  --no-meta       SSD mode (no metadata table, ~55% smaller)" << std::endl;
        std::cerr << "  --clevel N      Compression level 0-15 (default: 0)" << std::endl;
        std::cerr << "  --nft           NFT plot (requires --contract)" << std::endl;
        std::cerr << "  --contract HEX  Contract address for NFT plot" << std::endl;
        std::cerr << "  --derive        Derive plot_id from seed (arg1) + farmer_key (arg2)" << std::endl;
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
    int num_gpus = 1;
    bool has_meta = true;
    bool is_nft = false;
    bool do_derive = false;
    std::string contract_str;

    for(int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if(arg == "--k" && i+1 < argc) ksize = std::stoi(argv[++i]);
        else if(arg == "--limit" && i+1 < argc) limit = std::stoull(argv[++i]);
        else if(arg == "--device" && i+1 < argc) device_id = std::stoi(argv[++i]);
        else if(arg == "--num-gpus" && i+1 < argc) num_gpus = std::stoi(argv[++i]);
        else if(arg == "--ramdisk" && i+1 < argc) ramdisk_dir = argv[++i];
        else if(arg == "--no-meta") has_meta = false;
        else if(arg == "--clevel" && i+1 < argc) clevel = std::stoi(argv[++i]);
        else if(arg == "--nft") is_nft = true;
        else if(arg == "--contract" && i+1 < argc) contract_str = argv[++i];
        else if(arg == "--derive") do_derive = true;
        else output_dir = arg;
    }

    if(ksize < mmx::MIN_KSIZE || ksize > mmx::MAX_KSIZE) {
        std::cerr << "Error: ksize must be " << mmx::MIN_KSIZE << "-" << mmx::MAX_KSIZE << std::endl;
        return 1;
    }

    // Parse keys: first arg is plot_id_hex (or seed_hex if --derive)
    mmx::hash_t plot_id;
    mmx::pubkey_t farmer_key;
    mmx::addr_t contract_addr;
    bool have_contract = false;
    
    try {
        if(do_derive) {
            // First arg is seed, derive plot_id from seed+farm+ksize
            mmx::hash_t seed;
            seed.from_string(pid_str);
            farmer_key.from_string(fk_str);
            
            if(is_nft) {
                if(contract_str.empty()) {
                    std::cerr << "Error: NFT plot requires --contract <hex>" << std::endl;
                    return 1;
                }
                // Parse contract as hex bytes → hash_t → addr_t
                std::vector<uint8_t> raw = vnx::from_hex_string(contract_str);
                if(raw.size() != 32) {
                    std::cerr << "Error: contract must be 32 hex bytes (64 hex chars)" << std::endl;
                    return 1;
                }
                std::array<uint8_t, 32> arr;
                std::memcpy(arr.data(), raw.data(), 32);
                contract_addr = mmx::addr_t(mmx::hash_t(arr));
                have_contract = true;
            }
            
            plot_id = mmx::derive_plot_id(seed, ksize, farmer_key, is_nft, 
                have_contract ? std::optional<mmx::addr_t>(contract_addr) : std::nullopt);
        } else {
            plot_id.from_string(pid_str);
            farmer_key.from_string(fk_str);
            if(is_nft || !contract_str.empty()) {
                std::vector<uint8_t> raw = vnx::from_hex_string(contract_str);
                if(raw.size() != 32) {
                    std::cerr << "Error: contract must be 32 hex bytes (64 hex chars)" << std::endl;
                    return 1;
                }
                std::array<uint8_t, 32> arr;
                std::memcpy(arr.data(), raw.data(), 32);
                contract_addr = mmx::addr_t(mmx::hash_t(arr));
                have_contract = true;
            }
        }
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
    if(is_nft) std::cout << "  Type: NFT" << std::endl;
    if(have_contract) std::cout << "  Contract: " << contract_addr.to_string() << std::endl;

    // GPU init
    std::unique_ptr<mmx::GPUManager> gpu_mgr;
    std::unique_ptr<mmx::GPUDevice> single_gpu;
    
    if(num_gpus > 1) {
        gpu_mgr = std::make_unique<mmx::GPUManager>();
        gpu_mgr->init(device_id, num_gpus);
    } else {
        single_gpu = std::make_unique<mmx::GPUDevice>();
        single_gpu->init(device_id);
        single_gpu->print_info();
    }

    // Pipeline
    std::unique_ptr<mmx::MultiPipeline> multi_pipe;
    std::unique_ptr<mmx::PlotPipeline> single_pipe;
    
    if(num_gpus > 1) {
        multi_pipe = std::make_unique<mmx::MultiPipeline>(*gpu_mgr, ksize);
        multi_pipe->init();
    } else {
        single_pipe = std::make_unique<mmx::PlotPipeline>(*single_gpu, ksize);
        single_pipe->init();
    }

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
    try {
        if(num_gpus > 1) {
            multi_pipe->run_full_pipeline(X_values, plot_id_raw, result);
        } else {
            single_pipe->run_full_pipeline(X_values, plot_id_raw, result);
        }
    } catch(const std::exception& e) {
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
    uint32_t xbits_val = clevel;  // compression level C
    std::cout << "\n[Plot] Writing (C" << clevel << ", xbits=" << (ksize - clevel) << ")..." << std::endl;
    mmx::PlotWriter writer(ksize, xbits_val, has_meta);

    uint8_t fk_raw[33];
    std::memset(fk_raw, 0, 33);
    std::memcpy(fk_raw, farmer_key.data(), std::min((size_t)32, farmer_key.size()));

    try {
        writer.write_file(output_path,
            (const uint8_t*)plot_id.data(),
            fk_raw,
            result,
            have_contract ? (const uint8_t*)contract_addr.data() : nullptr);
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
