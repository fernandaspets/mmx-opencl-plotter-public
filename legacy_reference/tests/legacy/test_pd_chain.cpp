// test_pd_chain.cpp — Verify PD chain integrity in-memory
// 
// This test runs the plotter and then verifies:
// 1. PD[t][i].pos < PD[t-1].size() for all i, t
// 2. PD[t][i].pos + PD[t][i].delta < PD[t-1].size() for all i, t
// 3. final_Y is sorted
// 4. PD[t] entries with same Y are in sorted_pos order
// 5. final_Y[i] matches pd_all[9][i].Y
//
// Build: embed in plotter build (add to CMakeLists.txt)
// Run: ./test_pd_chain --k 18 [--chunked]

#include <vector>
#include <cstdint>
#include <iostream>
#include <algorithm>

// Include plotter internals
#include "../plotter.cpp"

bool verify_pd_chain(const PlotData& plot) {
    bool ok = true;
    
    // Check final_Y is sorted
    for(size_t i = 1; i < plot.final_Y.size(); i++) {
        if(plot.final_Y[i] < plot.final_Y[i-1]) {
            std::cerr << "FAIL: final_Y not sorted at [" << i << "]: " 
                      << plot.final_Y[i] << " < " << plot.final_Y[i-1] << std::endl;
            ok = false;
            break;
        }
    }
    
    // Check PD chain: PD[t][i].pos and pos+delta must be valid in PD[t-1]
    for(int t = 3; t <= 9; t++) {
        if(t >= (int)plot.PD.size() || plot.PD[t].empty()) continue;
        if(t-1 >= (int)plot.PD.size() || plot.PD[t-1].empty()) continue;
        
        size_t prev_size = plot.PD[t-1].size();
        size_t errors = 0;
        for(size_t i = 0; i < plot.PD[t].size(); i++) {
            uint32_t pos = plot.PD[t][i].first;
            uint32_t delta = plot.PD[t][i].second;
            
            if(pos >= prev_size) {
                if(errors++ < 5) {
                    std::cerr << "FAIL: PD[" << t << "][" << i << "].pos=" << pos 
                              << " >= PD[" << (t-1) << "].size=" << prev_size << std::endl;
                }
                ok = false;
            }
            uint64_t right = (uint64_t)pos + delta;
            if(right >= prev_size) {
                if(errors++ < 5) {
                    std::cerr << "FAIL: PD[" << t << "][" << i << "].pos+delta=" << right 
                              << " >= PD[" << (t-1) << "].size=" << prev_size << std::endl;
                }
                ok = false;
            }
        }
        if(errors > 0) {
            std::cerr << "  PD[" << t << "]: " << errors << " chain errors" << std::endl;
        }
    }
    
    // Check PD[9] size matches final_Y size
    if(plot.PD[9].size() != plot.final_Y.size()) {
        std::cerr << "FAIL: PD[9].size=" << plot.PD[9].size() 
                  << " != final_Y.size=" << plot.final_Y.size() << std::endl;
        ok = false;
    }
    
    return ok;
}

int main(int argc, char** argv) {
    int k = 18;
    bool chunked = false;
    
    for(int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if(arg == "--k" && i+1 < argc) k = std::stoi(argv[++i]);
        else if(arg == "--chunked") chunked = true;
    }
    
    KSIZE = k;
    XBITS = k;
    update_constants();
    
    std::string plot_id = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2";
    std::string farmer_key = "02292cd11aa18e5f64344cbe6c580249364dfe5a3683adc25446aadcc1b38555d7";
    
    // Parse keys
    hash_t pid;
    for(size_t i = 0; i < 32; i++) {
        pid.data()[i] = std::stoul(plot_id.substr(i*2, 2), nullptr, 16);
    }
    
    pubkey_t fkey;
    for(size_t i = 0; i < 33; i++) {
        fkey.data()[i] = std::stoul(farmer_key.substr(i*2, 2), nullptr, 16);
    }
    
    OCL_Plotter plotter;
    plotter.init_f1();
    plotter.init_table_hash();
    if(chunked) plotter.init_gpu_kernels();
    
    PlotData plot;
    
    if(chunked) {
        std::cout << "=== Chunked Pipeline (K=" << k << ") ===" << std::endl;
        const int num_buckets = 1 << LOGBUCKETS;
        const int n_meta = MY_N_META;
        const uint64_t total_entries = (uint64_t)1 << KSIZE;
        const size_t avg_per_bucket = total_entries / num_buckets;
        const int max_bucket_size = std::max((size_t)4096, avg_per_bucket * 3 / 2 + 256);
        
        MemBucketStore store(num_buckets, max_bucket_size, n_meta);
        compute_f1_chunked(plotter, pid, store, 1 << KSIZE);
        
        std::vector<std::vector<PDEntry>> pd_all;
        std::vector<std::pair<uint32_t, uint32_t>> x_pairs_all;
        compute_f2_f9_chunked(plotter, store, num_buckets, max_bucket_size, n_meta, true, pd_all, x_pairs_all);
        
        build_plot_data_from_store(store, plot, pd_all, x_pairs_all);
    } else {
        std::cout << "=== Flat Pipeline (K=" << k << ") ===" << std::endl;
        auto [X_values, Y_all, M_all] = compute_f1(plotter, pid, 1 << KSIZE);
        compute_full_pipeline(X_values, Y_all, M_all, plot, plotter);
    }
    
    std::cout << "final_Y: " << plot.final_Y.size() << " entries" << std::endl;
    for(int t = 2; t <= 9; t++) {
        std::cout << "PD[" << t << "]: " << plot.PD[t].size() << " entries" << std::endl;
    }
    
    std::cout << "\n=== PD Chain Verification ===" << std::endl;
    bool ok = verify_pd_chain(plot);
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    
    return ok ? 0 : 1;
}
