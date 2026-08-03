// test_chunked.cpp — Test chunked pipeline with small k-size
#include "bucket_store.h"
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>
#include <chrono>

// Include the chunked pipeline
#include "chunked_pipeline.cpp"

// Globals (defined in plotter.cpp, but we need our own for testing)
int KSIZE = 20;
int XBITS = 20;
int LOGBUCKETS = 8;  // Use 8 for chunked (256 buckets)
int MY_N_META = 14;
int MY_N_META_OUT = 12;
int MY_N_TABLE = 9;
int DSIZE_ = 5;
int PSIZE_ = 21;
int PDSIZE = 40;
int XBITS_GLOBAL = 20;
uint32_t KMASK;
int MY_N_META_GLOBAL = 14;

void update_constants() {
    KMASK = ((uint32_t(1) << KSIZE) - 1);
    PSIZE_ = KSIZE + 1;
    PDSIZE = ((PSIZE_ + DSIZE_ + 7) / 8) * 8;
}

int main() {
    update_constants();
    
    // Init OpenCL
    cl_int err;
    cl_platform_id plat; clGetPlatformIDs(1, &plat, nullptr);
    cl_device_id dev; clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    
    char dn[256]; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dn), dn, nullptr);
    std::cout << "[OCL] Device: " << dn << std::endl;
    
    OCL_Plotter plotter;
    plotter.init(ctx, dev);
    plotter.init_table_hash();
    
    // Generate random plot ID
    hash_t plot_id;
    std::mt19937 rng(42);
    for(size_t i = 0; i < plot_id.size; i++) plot_id.data[i] = rng() & 0xFF;
    
    // Create bucket store
    const int num_buckets = 1 << LOGBUCKETS;  // 256
    const int max_bucket_size = ((1 << KSIZE) / num_buckets) * 2 + 256;  // 2x avg + slack
    const int n_meta = MY_N_META;
    
    std::cout << "k=" << KSIZE << ", buckets=" << num_buckets
              << ", max_bucket_size=" << max_bucket_size << std::endl;
    
    MemBucketStore store(num_buckets, max_bucket_size, n_meta);
    
    // F1 → bucket store
    compute_f1_chunked(plotter, plot_id, store, 1 << 18);
    
    // F2-F9 chunked
    compute_f2_f9_chunked(plotter, store, num_buckets, max_bucket_size, n_meta);
    
    // Print final stats
    uint64_t total = 0;
    for(int y = 0; y < num_buckets; y++) total += store.counts[y];
    std::cout << "\nFinal entries: " << total << std::endl;
    
    return 0;
}
