// mmx_opencl_plotter — MMX OpenCL Plotter
// 1:1 CLI compatible with reference mmx_cuda_plot
// Usage: mmx_opencl_plotter -f <farmer_key> [options]
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
#include <ctime>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <csignal>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

static volatile bool sig_interrupt = false;
static volatile bool sig_force = false;

static void interrupt_handler(int) {
    if(sig_interrupt) { sig_force = true; }
    else { sig_interrupt = true; }
}

static std::string hex_str(const uint8_t* data, size_t len) {
    std::string out(len * 2, ' ');
    for(size_t i = 0; i < len; i++) {
        out[i*2]   = "0123456789ABCDEF"[data[i] >> 4];
        out[i*2+1] = "0123456789ABCDEF"[data[i] & 0xF];
    }
    return out;
}

int main(int argc, char** argv) {
    // Silence unused function warning
    (void)hex_str;

    int ksize = KSIZE;
    std::string farmer_key_str;
    std::string contract_addr_str;
    std::vector<std::string> tmp_dirs;
    std::string tmp_dir2 = "@RAM";
    std::string tmp_dir3 = "@RAM";
    std::vector<std::string> final_dirs;
    int C = 0;
    int device = 0;
    int num_devices = 1;
    int num_plots = 1;
    bool ssd_mode = false;
    bool show_help = false;
    bool show_version = false;

    for(int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      ((arg == "-C" || arg == "--level") && i+1 < argc)       C = std::stoi(argv[++i]);
        else if(arg == "--ssd")                                          ssd_mode = true;
        else if((arg == "-n" || arg == "--count") && i+1 < argc)        num_plots = std::stoi(argv[++i]);
        else if((arg == "-g" || arg == "--device") && i+1 < argc)       device = std::stoi(argv[++i]);
        else if((arg == "-r" || arg == "--ndevices") && i+1 < argc)     num_devices = std::stoi(argv[++i]);
        else if((arg == "-t" || arg == "--tmpdir") && i+1 < argc)       tmp_dirs.push_back(argv[++i]);
        else if((arg == "-2" || arg == "--tmpdir2") && i+1 < argc)      tmp_dir2 = argv[++i];
        else if((arg == "-3" || arg == "--tmpdir3") && i+1 < argc)      tmp_dir3 = argv[++i];
        else if((arg == "-d" || arg == "--finaldir") && i+1 < argc)     final_dirs.push_back(argv[++i]);
        else if((arg == "-c" || arg == "--contract") && i+1 < argc)     contract_addr_str = argv[++i];
        else if((arg == "-f" || arg == "--farmerkey") && i+1 < argc)    farmer_key_str = argv[++i];
        else if(arg == "-h" || arg == "--help")                         show_help = true;
        else if(arg == "--version")                                      show_version = true;
        else if(arg == "-S" || arg == "--streams")     { if(i+1 < argc) i++; }
        else if(arg == "-B" || arg == "--chunksize")   { if(i+1 < argc) i++; }
        else if(arg == "-Q" || arg == "--maxtmp")      { if(i+1 < argc) i++; }
        else if(arg == "-A" || arg == "--copylimit")   { if(i+1 < argc) i++; }
        else if(arg == "-W" || arg == "--maxcopy")     { if(i+1 < argc) i++; }
        else if(arg == "-M" || arg == "--memory")      { if(i+1 < argc) i++; }
        else if(arg == "-z" || arg == "--dstport")     { if(i+1 < argc) i++; }
        else if(arg == "-w" || arg == "--waitforcopy") { }
        else { std::cerr << "Unknown: " << arg << std::endl; return -2; }
    }

    if(show_version) {
        std::cout << "MMX OpenCL Plotter k" << ksize << " - " TOSTRING(GIT_COMMIT_HASH) << std::endl;
        _exit(0);  // avoid static destructor issues
    }

    if(show_help || argc <= 1 || farmer_key_str.empty()) {
        std::cout << "MMX k" << ksize << " OpenCL plotter - " TOSTRING(GIT_COMMIT_HASH) "\n\n"
            "For <farmerkey> see output of `mmx wallet keys`.\n"
            "To plot for pooling, specify -c <contract> address.\n\n"
            "Usage:\n"
            "  mmx_opencl_plotter [OPTION...]\n\n"
            "  -C, --level arg      Compression level (0 to 15)\n"
            "      --ssd            Make SSD plots\n"
            "  -n, --count arg      Number of plots (default = 1, unlimited = -1)\n"
            "  -g, --device arg     CUDA device (default = 0)\n"
            "  -r, --ndevices arg   Number of CUDA devices (default = 1)\n"
            "  -t, --tmpdir arg     Temporary directories\n"
            "  -2, --tmpdir2 arg    Temporary dir 2 (default = @RAM)\n"
            "  -3, --tmpdir3 arg    Temporary dir 3 (default = @RAM)\n"
            "  -d, --finaldir arg   Final destinations\n"
            "  -c, --contract arg   Pool Contract Address\n"
            "  -f, --farmerkey arg  Farmer Public Key (33 bytes)\n"
            "      --version        Print version\n"
            "  -h, --help           Print help\n"
            << std::endl;
        return 0;
    }

    if(C < 0 || C > 15) {
        std::cerr << "Invalid compression level: " << C << std::endl;
        return -2;
    }

    if(tmp_dirs.empty()) tmp_dirs.push_back("./");
    if(tmp_dir3 != "@RAM" && tmp_dir2 == "@RAM") tmp_dir2 = tmp_dir3;

    // Parse farmer key (33 hex bytes)
    auto fk_raw = vnx::from_hex_string(farmer_key_str);
    if(fk_raw.size() != 33) {
        std::cerr << "Invalid farmer key: " << farmer_key_str << " (needs 33 bytes)" << std::endl;
        return -2;
    }
    mmx::pubkey_t farmer_key;
    std::memcpy((void*)farmer_key.data(), fk_raw.data(), 33);

    // Parse contract (bech32 address)
    mmx::addr_t contract;
    bool have_contract = false;
    if(!contract_addr_str.empty()) {
        try {
            contract.from_string(contract_addr_str);
            have_contract = true;
        } catch(...) {
            std::cerr << "Invalid contract address" << std::endl;
            return -2;
        }
    }

    bool is_nft = have_contract;

    // Generate random seed (matching reference)
    std::array<uint8_t, 32> seed_bytes;
    vnx::secure_random_bytes(seed_bytes.data(), 32);
    mmx::hash_t seed;
    std::memcpy(seed.data(), seed_bytes.data(), 32);

    // Derive plot_id from seed + farmer_key + optional contract
    mmx::hash_t plot_id = mmx::derive_plot_id(
        seed, ksize, farmer_key, is_nft,
        have_contract ? std::optional<mmx::addr_t>(contract) : std::nullopt);

    if(num_plots != 1) {
        std::signal(SIGINT, interrupt_handler);
        std::signal(SIGTERM, interrupt_handler);
    }

    const uint64_t num_x = uint64_t(1) << ksize;
    // Build plot name matching reference convention:
    // plot-mmx-<hdd|ssd>-k<ksize>-c<C>[-nft<prefix>]-<date>-<plot_id>
    time_t now = time(nullptr);
    char date_buf[32];
    strftime(date_buf, sizeof(date_buf), "%Y-%m-%d-%H-%M", localtime(&now));
    
    std::string nft_tag;
    if(is_nft) {
        std::string cstr = contract_addr_str;
        nft_tag = "-nft" + (cstr.size() > 10 ? cstr.substr(3, 7) : cstr);
    }
    
    std::string prefix = "plot-mmx-" + std::string(ssd_mode ? "ssd" : "hdd");
    std::string plot_name = prefix + "-k" + std::to_string(ksize) + "-c" + std::to_string(C)
        + nft_tag + "-" + date_buf + "-" + plot_id.to_string();
    
    std::string out_dir = final_dirs.empty() ? tmp_dirs[0] : final_dirs[0];
    std::string output_path = out_dir + "/" + plot_name + ".plot";

    std::cout << "Working Directory:   " << tmp_dirs[0] << std::endl;
    if(tmp_dir2 != "@RAM") std::cout << "Working Directory 2: " << tmp_dir2 << std::endl;
    if(tmp_dir3 != "@RAM") std::cout << "Working Directory 3: " << tmp_dir3 << std::endl;
    std::cout << "Compression Level: C" << C << " (" << (ssd_mode ? "SSD" : "HDD") << ")" << std::endl;
    std::cout << "Plot Name: " << plot_name << std::endl;

    // GPU init
    mmx::GPUManager gpu_mgr;
    mmx::GPUDevice single_gpu;
    if(num_devices > 1) { gpu_mgr.init(device, num_devices); }
    else { single_gpu.init(device); single_gpu.print_info(); }

    // Pipeline
    std::unique_ptr<mmx::MultiPipeline> multi_pipe;
    std::unique_ptr<mmx::PlotPipeline> single_pipe;
    if(num_devices > 1) {
        multi_pipe = std::make_unique<mmx::MultiPipeline>(gpu_mgr, ksize);
        multi_pipe->init();
    } else {
        single_pipe = std::make_unique<mmx::PlotPipeline>(single_gpu, ksize);
        single_pipe->init();
    }

    std::cout << "[F1] Generating " << num_x << " X values..." << std::endl;
    std::vector<uint32_t> X_values(num_x);
    #pragma omp parallel for
    for(uint64_t i = 0; i < num_x; i++) X_values[i] = (uint32_t)i;

    uint32_t pid32[8];
    std::memcpy(pid32, plot_id.data(), 32);

    auto t0 = std::chrono::high_resolution_clock::now();
    mmx::PlotData result;
    try {
        if(num_devices > 1) multi_pipe->run_full_pipeline(X_values, pid32, result);
        else single_pipe->run_full_pipeline(X_values, pid32, result);
    } catch(const std::exception& e) {
        std::cerr << "Pipeline failed: " << e.what() << std::endl;
        return 1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Pipeline time: " << std::chrono::duration<double>(t1 - t0).count() << "s" << std::endl;

    if(result.final_Y.empty()) { std::cout << "No matches!" << std::endl; return 1; }

    // Write plot
    std::cout << "[Plot] Writing..." << std::endl;
    mmx::PlotWriter writer(ksize, C, !ssd_mode);
    uint8_t fk33[33];
    std::memcpy(fk33, farmer_key.data(), 33);

    try {
        writer.write_file(output_path,
            (const uint8_t*)plot_id.data(), fk33, result,
            have_contract ? (const uint8_t*)contract.data() : nullptr);
    } catch(const std::exception& e) {
        std::cerr << "Write failed: " << e.what() << std::endl;
        return 1;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t2 - t0).count();
    std::ifstream fs(output_path, std::ios::binary | std::ios::ate);
    uint64_t fsize = fs.tellg();

    std::cout << "Total time: " << total_sec << "s" << std::endl;
    std::cout << "Plot size: " << (fsize / 1e9) << " GB" << std::endl;

    // Copy to additional destinations
    for(size_t i = 1; i < final_dirs.size(); i++) {
        std::string dest = final_dirs[i] + "/" + plot_name;
        std::filesystem::copy(output_path, dest, std::filesystem::copy_options::overwrite_existing);
    }

    return 0;
}
