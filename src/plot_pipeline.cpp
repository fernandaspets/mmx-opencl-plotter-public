#include "plot_pipeline.h"
#include "bitmap_match.h"
#include <omp.h>
#include <fstream>
#include <cassert>

namespace mmx {

PlotPipeline::PlotPipeline(GPUDevice& gpu, uint32_t ksize)
    : gpu(gpu), ksize(ksize), kmask((1u << ksize) - 1) {
    if(ksize < MIN_KSIZE || ksize > MAX_KSIZE) {
        throw std::runtime_error("KSIZE must be " + std::to_string(MIN_KSIZE)
            + ".." + std::to_string(MAX_KSIZE));
    }
}

PlotPipeline::~PlotPipeline() {
    if(f1_X_buf) clReleaseMemObject(f1_X_buf);
    if(f1_ID_buf) clReleaseMemObject(f1_ID_buf);
    if(f1_Y_buf) clReleaseMemObject(f1_Y_buf);
    if(f1_M_buf) clReleaseMemObject(f1_M_buf);
    if(hash_L_buf) clReleaseMemObject(hash_L_buf);
    if(hash_R_buf) clReleaseMemObject(hash_R_buf);
    if(hash_Y_buf) clReleaseMemObject(hash_Y_buf);
    if(hash_M_buf) clReleaseMemObject(hash_M_buf);
}

void PlotPipeline::init() {
    load_kernels();
}

void PlotPipeline::load_kernels() {
    // Find kernel files
    auto find_kernel = [](const std::string& name) -> std::string {
        for(const auto& path : {"kernels/" + name, "../kernels/" + name, "../../kernels/" + name}) {
            std::ifstream f(path);
            if(f.good()) return path;
        }
        throw std::runtime_error("Cannot find kernels/" + name);
    };

    // F1 kernel
    gpu.load_program_from_file("f1", find_kernel("f1.cl"));
    k_f1 = gpu.get_kernel("compute_f1_kernel");
    // Try loading warp-parallel F1 kernels (3-kernel pipeline)
    try {
        k_gen_mem = gpu.get_kernel("gen_mem_array_v2_kernel");
        k_scatter_f1 = gpu.get_kernel("scatter_f1_v2_kernel");
        try {
            k_calc_memhash = gpu.get_kernel("calc_mem_hash_warp_kernel");
            std::cout << "[OCL] Using warp-parallel calc_mem_hash (32 threads/entry)" << std::endl;
        } catch(...) {
            k_calc_memhash = gpu.get_kernel("calc_mem_hash_v2_kernel");
        }
        f1_warp_loaded = true;
        std::cout << "[OCL] Split F1 kernels loaded (3-kernel pipeline)" << std::endl;
    } catch(...) {
        std::cout << "[OCL] Split F1 kernels not found, using monolithic F1" << std::endl;
    }

    // Table hash kernel
    gpu.load_program_from_file("table_hash", find_kernel("table_hash.cl"));
    k_table_hash = gpu.get_kernel("hash_table_entries");
}

void PlotPipeline::ensure_f1_buffers(size_t num_x) {
    if(num_x <= f1_buf_size) return;
    cl_int err;

    if(f1_X_buf) clReleaseMemObject(f1_X_buf);
    if(f1_ID_buf) clReleaseMemObject(f1_ID_buf);
    if(f1_Y_buf) clReleaseMemObject(f1_Y_buf);
    if(f1_M_buf) clReleaseMemObject(f1_M_buf);

    f1_X_buf = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY,
                              num_x * sizeof(uint32_t), nullptr, &err);
    GPUDevice::check(err, "f1_X_buf");
    f1_ID_buf = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY,
                               8 * sizeof(uint32_t), nullptr, &err);
    GPUDevice::check(err, "f1_ID_buf");
    f1_Y_buf = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY,
                              num_x * sizeof(uint32_t), nullptr, &err);
    GPUDevice::check(err, "f1_Y_buf");
    f1_M_buf = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY,
                              num_x * N_META * sizeof(uint32_t), nullptr, &err);
    GPUDevice::check(err, "f1_M_buf");

    f1_buf_size = num_x;
}

void PlotPipeline::ensure_hash_buffers(size_t num_matches) {
    if(num_matches <= hash_buf_size) return;
    cl_int err;

    if(hash_L_buf) clReleaseMemObject(hash_L_buf);
    if(hash_R_buf) clReleaseMemObject(hash_R_buf);
    if(hash_Y_buf) clReleaseMemObject(hash_Y_buf);
    if(hash_M_buf) clReleaseMemObject(hash_M_buf);

    hash_L_buf = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY,
                                num_matches * N_META * sizeof(uint32_t), nullptr, &err);
    GPUDevice::check(err, "hash_L_buf");
    hash_R_buf = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY,
                                num_matches * N_META * sizeof(uint32_t), nullptr, &err);
    GPUDevice::check(err, "hash_R_buf");
    hash_Y_buf = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY,
                                num_matches * sizeof(uint32_t), nullptr, &err);
    GPUDevice::check(err, "hash_Y_buf");
    hash_M_buf = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY,
                                num_matches * N_META * sizeof(uint32_t), nullptr, &err);
    GPUDevice::check(err, "hash_M_buf");

    hash_buf_size = num_matches;
}

bool PlotPipeline::init_hash_lr_kernel() {
    // Reload table_hash.cl to pick up hash_table_lr kernel
    for(const auto& path : {"kernels/table_hash.cl", "../kernels/table_hash.cl", "../../kernels/table_hash.cl"}) {
        std::ifstream f(path);
        if(f.good()) {
            std::cout << "[OCL] Loading LR kernel from " << path << std::endl;
            try {
                gpu.load_program_from_file("table_hash", path);
                // Re-create both kernels from reloaded program
                k_table_hash = gpu.get_kernel("hash_table_entries");
                k_table_hash_lr = gpu.get_kernel("hash_table_lr");
                if(k_table_hash_lr) {
                    std::cout << "[OCL] hash_table_lr loaded successfully" << std::endl;
                    break;
                }
            } catch(const std::exception& e) {
                std::cout << "[OCL] Failed to load LR kernel: " << e.what() << std::endl;
                k_table_hash_lr = nullptr;
            }
        }
    }
    if(!k_table_hash_lr) {
        std::cout << "[OCL] hash_table_lr not available, using legacy hash" << std::endl;
        return false;
    }
    use_gpu_resident = true;
    char vendor[256] = {};
    clGetDeviceInfo(gpu.device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
    bool is_nvidia = (strstr(vendor, "NVIDIA") != nullptr);
    g_hash_local = is_nvidia ? 64 : 256;
    // Also parse OpenCL version — NVIDIA OpenCL 1.2 may not support global atomics in our kernel
    std::cout << "[OCL] GPU-resident M_curr: hash_table_lr (local=" << g_hash_local << ")" << std::endl;
    
    // Test the kernel with a tiny invocation to verify it works
    ensure_gpu_resident_buffers(64);
    // Write test metadata to M_curr_gpu (avoids AMD driver crash on uninitialized buffer reads)
    std::vector<uint32_t> test_meta(64 * N_META, 0);
    for(int i = 0; i < 64 * N_META; i++) test_meta[i] = (uint32_t)(i * 2654435761u);
    clEnqueueWriteBuffer(gpu.queue, M_curr_gpu, CL_TRUE, 0, 64 * N_META * sizeof(uint32_t), test_meta.data(), 0, nullptr, nullptr);
    uint32_t test_lr_data[4] = {0, 1, 2, 3};
    cl_mem test_LR = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4*sizeof(uint32_t), test_lr_data, nullptr);
    cl_mem test_Y = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY, 2*sizeof(uint32_t), nullptr, nullptr);
    uint32_t test_mask = 0x3F;
    uint32_t test_nmatch = 2;
    uint32_t test_ntotal = 64;
    clSetKernelArg(k_table_hash_lr, 0, sizeof(cl_mem), &M_curr_gpu);
    clSetKernelArg(k_table_hash_lr, 1, sizeof(cl_mem), &test_LR);
    clSetKernelArg(k_table_hash_lr, 2, sizeof(cl_mem), &test_Y);
    clSetKernelArg(k_table_hash_lr, 3, sizeof(cl_mem), &M_out_gpu);
    clSetKernelArg(k_table_hash_lr, 4, sizeof(uint32_t), &test_mask);
    clSetKernelArg(k_table_hash_lr, 5, sizeof(uint32_t), &test_nmatch);
    clSetKernelArg(k_table_hash_lr, 6, sizeof(uint32_t), &test_ntotal);
    size_t local = (size_t)g_hash_local;
    size_t global = 2;
    cl_int err = clEnqueueNDRangeKernel(gpu.queue, k_table_hash_lr, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        std::cout << "[OCL] hash_table_lr failed test invocation (err=" << err << "), falling back to legacy" << std::endl;
        use_gpu_resident = false;
        k_table_hash_lr = nullptr;
        clReleaseMemObject(test_LR);
        clReleaseMemObject(test_Y);
        return false;
    }
    gpu.finish();
    clReleaseMemObject(test_LR);
    clReleaseMemObject(test_Y);
    std::cout << "[OCL] hash_table_lr verified OK" << std::endl;
    return true;
}

void PlotPipeline::ensure_gpu_resident_buffers(size_t num_entries) {
    if(num_entries <= gpu_resident_capacity) return;
    cl_int err;
    if(M_curr_gpu) clReleaseMemObject(M_curr_gpu);
    if(M_out_gpu)  clReleaseMemObject(M_out_gpu);
    // Allocate 1.1x for growth (entries grow ~2% per table)
    size_t cap = (size_t)(num_entries * 11 / 10 + 256);
    size_t sz = cap * N_META * sizeof(uint32_t);
    M_curr_gpu = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, sz, nullptr, &err);
    GPUDevice::check(err, "M_curr_gpu");
    M_out_gpu = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, sz, nullptr, &err);
    GPUDevice::check(err, "M_out_gpu");
    gpu_resident_capacity = cap;
}

void PlotPipeline::compute_f1(
    const std::vector<uint32_t>& X_values,
    const uint32_t* plot_id,
    std::vector<uint32_t>& Y_out,
    std::vector<uint32_t>& M_out)
{
    const uint32_t num_x = X_values.size();
    const uint32_t xbits = 0;  // C0: no compression
    Y_out.resize(num_x);
    M_out.resize(num_x * N_META);

    // Use warp-parallel F1 when available (3 kernels, 32 threads/entry)
    if(f1_warp_loaded && k_gen_mem && k_calc_memhash && k_scatter_f1) {
        auto tw_start = std::chrono::high_resolution_clock::now();
        const uint32_t SUB_BATCH = 131072;
        const uint32_t x_base = X_values[0];
        const uint32_t num_iter = 256;

        ensure_f1_buffers(SUB_BATCH);
        cl_int err;
        err = clEnqueueWriteBuffer(gpu.queue, f1_ID_buf, CL_FALSE, 0,
            8 * sizeof(uint32_t), plot_id, 0, nullptr, nullptr);
        GPUDevice::check(err, "write ID");

        cl_mem mem_buf = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE,
            (size_t)SUB_BATCH * 1024 * sizeof(uint32_t), nullptr, &err);
        GPUDevice::check(err, "mem_buf");
        cl_mem key_buf = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE,
            (size_t)SUB_BATCH * 16 * sizeof(uint32_t), nullptr, &err);
        GPUDevice::check(err, "key_buf");
        cl_mem hash_buf = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE,
            (size_t)SUB_BATCH * 32 * sizeof(uint32_t), nullptr, &err);
        GPUDevice::check(err, "hash_buf");

        uint32_t num_subs = (num_x + SUB_BATCH - 1) / SUB_BATCH;
        for(uint32_t sb = 0; sb < num_subs; sb++) {
            uint32_t offset = sb * SUB_BATCH;
            uint32_t count = std::min(SUB_BATCH, num_x - offset);
            uint32_t sub_x_base = x_base + offset;

            clSetKernelArg(k_gen_mem, 0, sizeof(cl_mem), &mem_buf);
            clSetKernelArg(k_gen_mem, 1, sizeof(cl_mem), &key_buf);
            clSetKernelArg(k_gen_mem, 2, sizeof(cl_mem), &f1_ID_buf);
            clSetKernelArg(k_gen_mem, 3, sizeof(uint32_t), &count);
            clSetKernelArg(k_gen_mem, 4, sizeof(uint32_t), &sub_x_base);
            size_t gs = ((count + 255) / 256) * 256;
            clEnqueueNDRangeKernel(gpu.queue, k_gen_mem, 1, nullptr, &gs, nullptr, 0, nullptr, nullptr);

            size_t local = 128;
            size_t global = ((count * 32 + local - 1) / local) * local;
            clSetKernelArg(k_calc_memhash, 0, sizeof(cl_mem), &mem_buf);
            clSetKernelArg(k_calc_memhash, 1, sizeof(cl_mem), &hash_buf);
            clSetKernelArg(k_calc_memhash, 2, sizeof(uint32_t), &count);
            clSetKernelArg(k_calc_memhash, 3, sizeof(uint32_t), &num_iter);
            clEnqueueNDRangeKernel(gpu.queue, k_calc_memhash, 1, nullptr, &global, &local, 0, nullptr, nullptr);

            uint32_t total = num_x;
            clSetKernelArg(k_scatter_f1, 0, sizeof(cl_mem), &key_buf);
            clSetKernelArg(k_scatter_f1, 1, sizeof(cl_mem), &hash_buf);
            clSetKernelArg(k_scatter_f1, 2, sizeof(cl_mem), &f1_Y_buf);
            clSetKernelArg(k_scatter_f1, 3, sizeof(cl_mem), &f1_M_buf);
            clSetKernelArg(k_scatter_f1, 4, sizeof(uint32_t), &kmask);
            clSetKernelArg(k_scatter_f1, 5, sizeof(uint32_t), &count);
            clSetKernelArg(k_scatter_f1, 6, sizeof(uint32_t), &sub_x_base);
            clSetKernelArg(k_scatter_f1, 7, sizeof(uint32_t), &total);
            clEnqueueNDRangeKernel(gpu.queue, k_scatter_f1, 1, nullptr, &gs, nullptr, 0, nullptr, nullptr);
            // No gpu.finish() here — in-order queue handles kernel deps.
            // CL_TRUE on readback blocks until all kernels complete.

            err = clEnqueueReadBuffer(gpu.queue, f1_Y_buf, CL_TRUE, 0,
                count * sizeof(uint32_t), Y_out.data() + offset, 0, nullptr, nullptr);
            GPUDevice::check(err, "read Y sub");
            err = clEnqueueReadBuffer(gpu.queue, f1_M_buf, CL_TRUE, 0,
                count * N_META * sizeof(uint32_t), M_out.data() + offset * N_META, 0, nullptr, nullptr);
            GPUDevice::check(err, "read M sub");
        }

        auto tw_end = std::chrono::high_resolution_clock::now();
        std::cout << "[WARP-F1] " << (tw_end - tw_start).count() / 1e6 << "ms" << std::endl;
        clReleaseMemObject(mem_buf);
        clReleaseMemObject(key_buf);
        clReleaseMemObject(hash_buf);
        return;
    }

    // Monolithic F1 fallback
    ensure_f1_buffers(num_x);

    cl_int err;
    err = clEnqueueWriteBuffer(gpu.queue, f1_X_buf, CL_FALSE, 0,
                               num_x * sizeof(uint32_t), X_values.data(), 0, nullptr, nullptr);
    GPUDevice::check(err, "write X");
    err = clEnqueueWriteBuffer(gpu.queue, f1_ID_buf, CL_FALSE, 0,
                               8 * sizeof(uint32_t), plot_id, 0, nullptr, nullptr);
    GPUDevice::check(err, "write ID");

    clSetKernelArg(k_f1, 0, sizeof(cl_mem), &f1_X_buf);
    clSetKernelArg(k_f1, 1, sizeof(cl_mem), &f1_ID_buf);
    clSetKernelArg(k_f1, 2, sizeof(cl_mem), &f1_Y_buf);
    clSetKernelArg(k_f1, 3, sizeof(cl_mem), &f1_M_buf);
    clSetKernelArg(k_f1, 4, sizeof(uint32_t), &kmask);
    clSetKernelArg(k_f1, 5, sizeof(uint32_t), &xbits);
    clSetKernelArg(k_f1, 6, sizeof(uint32_t), &num_x);

    size_t local = 64;
    size_t global = ((num_x + local - 1) / local) * local;
    err = clEnqueueNDRangeKernel(gpu.queue, k_f1, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    GPUDevice::check(err, "enqueue f1");
    gpu.finish();

    err = clEnqueueReadBuffer(gpu.queue, f1_Y_buf, CL_TRUE, 0,
                              num_x * sizeof(uint32_t), Y_out.data(), 0, nullptr, nullptr);
    GPUDevice::check(err, "read Y");
    gpu.finish();

    err = clEnqueueReadBuffer(gpu.queue, f1_M_buf, CL_TRUE, 0,
                              num_x * N_META * sizeof(uint32_t), M_out.data(), 0, nullptr, nullptr);
    GPUDevice::check(err, "read M");
}


// Parallel LSD radix sort for (Y, index) pairs by Y.
// 8-bit radix, 4 passes for 32-bit Y (but ksize bits are significant).
static void radix_sort_pairs(std::vector<std::pair<uint32_t, uint32_t>>& entries, uint32_t ksize) {
    const size_t n = entries.size();
    if(n <= 1) return;
    
    const int nthreads = omp_get_max_threads();
    const int BITS_PER_PASS = 8;
    const int num_passes = (ksize + BITS_PER_PASS - 1) / BITS_PER_PASS;
    
    std::vector<std::pair<uint32_t, uint32_t>> temp(n);
    
    for(int pass = 0; pass < num_passes; pass++) {
        const uint32_t shift = pass * BITS_PER_PASS;
        const uint32_t bits_this_pass = std::min((uint32_t)BITS_PER_PASS, ksize - shift);
        const uint32_t mask = (1u << bits_this_pass) - 1;
        const int num_bins = 1u << bits_this_pass;
        if(num_bins <= 0) break;
        
        // Per-thread histograms (vector of vectors avoids false sharing)
        std::vector<std::vector<uint32_t>> thread_hists(nthreads, std::vector<uint32_t>(num_bins, 0));
        
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < n; i++) {
            int ti = omp_get_thread_num();
            uint32_t digit = (entries[i].first >> shift) & mask;
            thread_hists[ti][digit]++;
        }
        
        // Global histogram (prefix sum across threads)
        std::vector<uint32_t> global_hist(num_bins, 0);
        for(int ti = 0; ti < nthreads; ti++)
            for(int b = 0; b < num_bins; b++)
                global_hist[b] += thread_hists[ti][b];
        
        // Exclusive prefix sum → global offsets
        std::vector<uint32_t> global_offset(num_bins, 0);
        uint32_t sum = 0;
        for(int b = 0; b < num_bins; b++) {
            global_offset[b] = sum;
            sum += global_hist[b];
        }
        
        // Per-thread starting offsets
        std::vector<std::vector<uint32_t>> thread_offsets(nthreads, std::vector<uint32_t>(num_bins));
        for(int b = 0; b < num_bins; b++) {
            uint32_t off = global_offset[b];
            for(int ti = 0; ti < nthreads; ti++) {
                thread_offsets[ti][b] = off;
                off += thread_hists[ti][b];
            }
        }
        
        // Scatter (stable within each thread)
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < n; i++) {
            int ti = omp_get_thread_num();
            uint32_t digit = (entries[i].first >> shift) & mask;
            uint32_t pos = thread_offsets[ti][digit]++;
            temp[pos] = entries[i];
        }
        
        std::swap(entries, temp);
    }
}

void PlotPipeline::sort_entries_by_y(std::vector<PlotEntry>& entries)
{
	const size_t n = entries.size();
    if(n < 2) return;

    // Check if already sorted (skip for t>2 since entries are sorted by Y from previous match)
    // Sample every 64th entry to check ordering
    bool needs_sort = false;
    for(size_t i = 64; i < n && !needs_sort; i += 64) {
        if(entries[i].Y < entries[i-64].Y) needs_sort = true;
    }
    if(!needs_sort && n > 256) return;

    // 8-bit radix sort (bucket sort by top LOGBUCKETS bits of Y)
    constexpr int LOGBUCKETS = 8;
    const int shift = ksize - LOGBUCKETS;
    const size_t num_buckets = 1u << LOGBUCKETS;

    // Count per bucket
    std::vector<uint32_t> bucket_counts(num_buckets, 0);
    for(const auto& e : entries) {
        uint32_t b = std::min(e.Y >> shift, (uint32_t)num_buckets - 1);
        bucket_counts[b]++;
    }

    // Prefix sum
    std::vector<uint32_t> bucket_offsets(num_buckets + 1, 0);
    for(size_t i = 0; i < num_buckets; i++) {
        bucket_offsets[i + 1] = bucket_offsets[i] + bucket_counts[i];
    }

    // Scatter into buckets (parallel with padded per-thread counters)
    std::vector<PlotEntry> sorted(n);
    const int nthr = omp_get_max_threads();
    const int pad = 64 / sizeof(uint32_t);  // 64-byte cache line
    std::vector<uint32_t> thr_pos(nthr * (num_buckets + pad), 0);
    
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < n; i++) {
        int ti = omp_get_thread_num();
        uint32_t b = std::min(entries[i].Y >> shift, (uint32_t)num_buckets - 1);
        thr_pos[ti * (num_buckets + pad) + b]++;
    }
    
    // Prefix sum across threads
    for(uint32_t b = 0; b < num_buckets; b++) {
        uint32_t sum = bucket_offsets[b];
        for(int t = 0; t < nthr; t++) {
            uint32_t cnt = thr_pos[t * (num_buckets + pad) + b];
            thr_pos[t * (num_buckets + pad) + b] = sum;
            sum += cnt;
        }
    }
    
    // Scatter using per-thread offsets
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < n; i++) {
        int ti = omp_get_thread_num();
        uint32_t b = std::min(entries[i].Y >> shift, (uint32_t)num_buckets - 1);
        uint32_t pos = thr_pos[ti * (num_buckets + pad) + b]++;
        sorted[pos] = entries[i];
    }

    // Sort within each bucket (parallel)
    #pragma omp parallel for schedule(dynamic, 8)
    for(size_t b = 0; b < num_buckets; b++) {
        uint32_t start = bucket_offsets[b];
        uint32_t cnt = bucket_counts[b];
        if(cnt > 1) {
            std::sort(sorted.begin() + start, sorted.begin() + start + cnt,
                [](const PlotEntry& a, const PlotEntry& b) {
                    if(a.Y != b.Y) return a.Y < b.Y;
                    return a.M < b.M;  // metadata as tiebreaker
                });
        }
    }

    entries = std::move(sorted);
}

std::vector<MatchPair> PlotPipeline::match_entries(
    const std::vector<PlotEntry>& entries)
{
    const size_t n = entries.size();
    if(n < 2) return {};

    // Compute bucket structure for parallelism
    constexpr int LOGBUCKETS = 8;
    const int shift = ksize - LOGBUCKETS;
    const size_t num_buckets = 1u << LOGBUCKETS;

    // Recompute bucket boundaries (entries are already sorted)
    std::vector<uint32_t> bucket_counts(num_buckets, 0);
    for(const auto& e : entries) {
        uint32_t b = std::min(e.Y >> shift, (uint32_t)num_buckets - 1);
        bucket_counts[b]++;
    }
    std::vector<uint32_t> bucket_offsets(num_buckets + 1, 0);
    for(size_t i = 0; i < num_buckets; i++) {
        bucket_offsets[i + 1] = bucket_offsets[i] + bucket_counts[i];
    }

    // Parallel matching
    const int nthreads = std::min(omp_get_max_threads(), (int)num_buckets);
    std::vector<std::vector<MatchPair>> thread_matches(nthreads);

    #pragma omp parallel for schedule(dynamic, 1)
    for(size_t b = 0; b < num_buckets; b++) {
        int ti = omp_get_thread_num();
        uint32_t start = bucket_offsets[b];
        uint32_t cnt = bucket_counts[b];
        if(cnt == 0) continue;

        // Match within bucket
        for(uint32_t x = start; x < start + cnt; x++) {
            const uint32_t YL = entries[x].Y;
            for(uint32_t y = x + 1; y < start + cnt && entries[y].Y <= YL + 1; y++) {
                if(entries[y].Y == YL + 1) {
                    thread_matches[ti].emplace_back(x, y);
                }
            }
        }

        // Cross-bucket matching
        if(b + 1 < num_buckets) {
            uint32_t next_start = bucket_offsets[b + 1];
            uint32_t next_cnt = bucket_counts[b + 1];
            if(next_cnt > 0) {
                const uint32_t last_Y = entries[start + cnt - 1].Y;
                // Use int to avoid unsigned wrap
                for(int xi = (int)(start + cnt - 1); xi >= (int)start; xi--) {
                    uint32_t x = (uint32_t)xi;
                    const uint32_t YL = entries[x].Y;
                    if(YL < last_Y) break;
                    for(uint32_t y = next_start; y < next_start + next_cnt && entries[y].Y <= YL + 1; y++) {
                        if(entries[y].Y == YL + 1) {
                            thread_matches[ti].emplace_back(x, y);
                        }
                    }
                }
            }
        }
    }

    // Flatten
    size_t total = 0;
    for(int ti = 0; ti < nthreads; ti++) total += thread_matches[ti].size();

    std::vector<MatchPair> result;
    result.reserve(total);
    for(int ti = 0; ti < nthreads; ti++) {
        result.insert(result.end(), thread_matches[ti].begin(), thread_matches[ti].end());
    }
    return result;
}

TableTiming PlotPipeline::process_table(
    std::vector<PlotEntry>& entries,
    const std::vector<uint32_t>* x_values_orig,
    std::vector<PDEntry>* pd_out,
    std::vector<std::pair<uint32_t, uint32_t>>* LR_pairs)
{
    TableTiming timing;
    timing.n_entries = entries.size();
    auto t0 = std::chrono::high_resolution_clock::now();

    // Sort by Y
    sort_entries_by_y(entries);
    auto t1 = std::chrono::high_resolution_clock::now();
    timing.sort_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Check Y distribution for debugging (verify no fixed-point dupes)
    if(entries.size() > 1000) {
        uint32_t max_dup = 0, max_dup_y = 0, cur_y = entries[0].Y, cur_cnt = 1;
        for(size_t di = 1; di < entries.size(); di++) {
            if(entries[di].Y == cur_y) { cur_cnt++; }
            else { if(cur_cnt > max_dup) { max_dup = cur_cnt; max_dup_y = cur_y; } cur_y = entries[di].Y; cur_cnt = 1; }
        }
        if(cur_cnt > max_dup) { max_dup = cur_cnt; max_dup_y = cur_y; }
        std::cerr << "[Ydist] n=" << entries.size() << " max_dup=" << max_dup << "@Y=" << max_dup_y << "\n";
    }
    // Match Y,Y+1
    auto matches = match_entries(entries);
    timing.n_matches = matches.size();
    auto t2 = std::chrono::high_resolution_clock::now();
    timing.match_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    if(matches.empty()) return timing;

    const size_t n_matches = matches.size();

    auto t3 = std::chrono::high_resolution_clock::now();

    // GPU hash — use hash_table_lr when available (GPU-resident M_curr)
    ensure_hash_buffers(n_matches);

    cl_int err;
    std::vector<uint32_t> Y_out(n_matches);
    std::vector<uint32_t> M_out;

    // GPU-resident hash: skip on AMD (driver returns wrong data for large-buffer random reads)
    bool use_gpu_lr = use_gpu_resident && M_curr_gpu && M_out_gpu && k_table_hash_lr;
    if(use_gpu_lr) {
        // On first call, detect vendor and disable on AMD
        static int gpu_ok = -1;
        if(gpu_ok == -1) {
            char buf[256]={};
            clGetDeviceInfo(gpu.device, CL_DEVICE_VENDOR, sizeof(buf), buf, nullptr);
            gpu_ok = (strstr(buf, "NVIDIA") != nullptr) ? 1 : 0;
            if(gpu_ok) std::cout << "[GPU-Res] hash enabled (NVIDIA)" << std::endl;
            else std::cout << "[GPU-Res] hash disabled (AMD driver large-buffer bug)" << std::endl;
        }
        // AMD: previously disabled due to uninit buffer read + missing resize causing garbage Y.
        // Both issues are now fixed — re-enable GPU-resident hash on all platforms.
        if(!gpu_ok) { std::cout << "[GPU-Res] hash enabled (AMD)" << std::endl; }
        else {
            // Ensure buffers have enough capacity for this table (entries grow)
            ensure_gpu_resident_buffers(std::max(entries.size(), n_matches));
        }
    }
    if(use_gpu_lr) {
        // Ensure buffers have enough capacity for this table (entries grow)<
        ensure_gpu_resident_buffers(std::max((size_t)entries.size(), n_matches));
        // Build LR_flat (DISABLED — AMD driver returns wrong data for large-buffer random reads) with ORIGINAL indices (entries[i].orig_idx references M_curr_gpu)
        std::vector<uint32_t> LR_flat(n_matches * 2);
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < n_matches; i++) {
            const auto& m = matches[i];
            LR_flat[i * 2]     = entries[m.first].orig_idx;
            LR_flat[i * 2 + 1] = entries[m.second].orig_idx;
        }
        auto t_ext = std::chrono::high_resolution_clock::now();
        timing.extract_ms = std::chrono::duration<double, std::milli>(t_ext - t3).count();

        // Check max orig_idx vs entries size
        {
            uint32_t max_orig = 0;
            for(size_t ci = 0; ci < n_matches; ci++) {
                if(LR_flat[ci*2] > max_orig) max_orig = LR_flat[ci*2];
                if(LR_flat[ci*2+1] > max_orig) max_orig = LR_flat[ci*2+1];
            }
            if(max_orig >= entries.size()) {
                std::cerr << "[LR] BOUNDS: max_orig=" << max_orig << " >= entries.size()=" << entries.size() << " (cap=" << gpu_resident_capacity << ")" << std::endl;
            }
        }

        // Upload LR pairs (2 uint32s per match vs 28 for L_meta + 28 for R_meta)
        cl_mem LR_buf = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            n_matches * 2 * sizeof(uint32_t), LR_flat.data(), &err);
        GPUDevice::check(err, "LR_buf");
        cl_mem Y_buf_lr = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE,
            n_matches * sizeof(uint32_t), nullptr, &err);
        GPUDevice::check(err, "Y_buf_lr");

        uint32_t n_total_local = (uint32_t)entries.size();
        uint32_t n_match = (uint32_t)n_matches;
        clSetKernelArg(k_table_hash_lr, 0, sizeof(cl_mem), &M_curr_gpu);
        clSetKernelArg(k_table_hash_lr, 1, sizeof(cl_mem), &LR_buf);
        clSetKernelArg(k_table_hash_lr, 2, sizeof(cl_mem), &Y_buf_lr);
        clSetKernelArg(k_table_hash_lr, 3, sizeof(cl_mem), &M_out_gpu);
        clSetKernelArg(k_table_hash_lr, 4, sizeof(uint32_t), &kmask);
        clSetKernelArg(k_table_hash_lr, 5, sizeof(uint32_t), &n_match);
        clSetKernelArg(k_table_hash_lr, 6, sizeof(uint32_t), &n_total_local);

        size_t local = (size_t)g_hash_local;
        size_t global = ((n_matches + local - 1) / local) * local;
        err = clEnqueueNDRangeKernel(gpu.queue, k_table_hash_lr, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        GPUDevice::check(err, "enqueue hash_lr");
        gpu.finish();

        // Download Y_out only — M_out stays on GPU
        M_out.clear();  // signal legacy path: avoid M_out download
        err = clEnqueueReadBuffer(gpu.queue, Y_buf_lr, CL_TRUE, 0,
            n_matches * sizeof(uint32_t), Y_out.data(), 0, nullptr, nullptr);
        GPUDevice::check(err, "read Y_lr");

        clReleaseMemObject(LR_buf);
        clReleaseMemObject(Y_buf_lr);

        // Swap M buffers (M_out → M_curr) — no data copy!
        std::swap(M_curr_gpu, M_out_gpu);
    } else {
        // Legacy path: extract L_meta/R_meta on CPU, upload to GPU
        std::vector<uint32_t> L_meta(n_matches * N_META);
        std::vector<uint32_t> R_meta(n_matches * N_META);
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < n_matches; i++) {
            const auto& m = matches[i];
            const auto& entry_L = entries[m.first];
            const auto& entry_R = entries[m.second];
            for(int j = 0; j < N_META; j++) {
                L_meta[i * N_META + j] = entry_L.M[j];
                R_meta[i * N_META + j] = entry_R.M[j];
            }
        }
        auto t_ext = std::chrono::high_resolution_clock::now();
        timing.extract_ms = std::chrono::duration<double, std::milli>(t_ext - t3).count();
        err = clEnqueueWriteBuffer(gpu.queue, hash_L_buf, CL_FALSE, 0,
                                   n_matches * N_META * sizeof(uint32_t), L_meta.data(), 0, nullptr, nullptr);
        GPUDevice::check(err, "write L");
        err = clEnqueueWriteBuffer(gpu.queue, hash_R_buf, CL_FALSE, 0,
                                   n_matches * N_META * sizeof(uint32_t), R_meta.data(), 0, nullptr, nullptr);
        GPUDevice::check(err, "write R");

        clSetKernelArg(k_table_hash, 0, sizeof(cl_mem), &hash_L_buf);
        clSetKernelArg(k_table_hash, 1, sizeof(cl_mem), &hash_R_buf);
        clSetKernelArg(k_table_hash, 2, sizeof(cl_mem), &hash_Y_buf);
        clSetKernelArg(k_table_hash, 3, sizeof(cl_mem), &hash_M_buf);
        clSetKernelArg(k_table_hash, 4, sizeof(uint32_t), &kmask);
        uint32_t n_u32 = (uint32_t)n_matches;
        clSetKernelArg(k_table_hash, 5, sizeof(uint32_t), &n_u32);

        size_t local = 64;
        size_t global = ((n_matches + local - 1) / local) * local;
        err = clEnqueueNDRangeKernel(gpu.queue, k_table_hash, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        GPUDevice::check(err, "enqueue hash");
        gpu.finish();

        M_out.resize(n_matches * N_META);
        err = clEnqueueReadBuffer(gpu.queue, hash_Y_buf, CL_TRUE, 0,
                                  n_matches * sizeof(uint32_t), Y_out.data(), 0, nullptr, nullptr);
        GPUDevice::check(err, "read Y");
        err = clEnqueueReadBuffer(gpu.queue, hash_M_buf, CL_TRUE, 0,
                                  n_matches * N_META * sizeof(uint32_t), M_out.data(), 0, nullptr, nullptr);
        GPUDevice::check(err, "read M");
    }

    auto t4 = std::chrono::high_resolution_clock::now();
    timing.hash_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    // Compute PD entries if requested
    if(pd_out) {
        pd_out->resize(n_matches);
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < n_matches; i++) {
            const auto& m = matches[i];
            uint16_t delta = (uint16_t)(m.second - m.first - 1);
            (*pd_out)[i] = {m.first, delta};
        }
    }

    // Build next table entries — Y + orig_idx only (M on GPU or downloaded here)
    entries.resize(n_matches);
    if(!M_out.empty()) {
        // Legacy path: M_out already downloaded, populate M
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < n_matches; i++) {
            entries[i].Y = Y_out[i];
            entries[i].orig_idx = (uint32_t)i;
            for(int j = 0; j < N_META; j++) {
                entries[i].M[j] = M_out[i * N_META + j];
            }
        }
    } else {
        // GPU-resident path: M stays on GPU, only Y + orig_idx
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < n_matches; i++) {
            entries[i].Y = Y_out[i];
            entries[i].orig_idx = (uint32_t)i;
        }
    }

    // Sort entries by Y so next table finds them pre-sorted (skip redundant sort)
    // Only needed for non-bitmap path (matches are in hash order, not Y order)
    sort_entries_by_y(entries);

    auto t5 = std::chrono::high_resolution_clock::now();
    timing.build_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

    return timing;
}

void PlotPipeline::run_full_pipeline(
    const std::vector<uint32_t>& X_values,
    const uint32_t* plot_id,
    PlotData& result)
{
    constexpr int N_TABLE = 9;
    const size_t num_x = X_values.size();
    const uint32_t kmask = this->kmask;

    result.final_Y.clear();
    result.table_entries.clear();
    result.pd_data.clear();
    result.x_pairs.clear();
    result.timings.clear();

    // F1
    std::vector<uint32_t> Y_flat, M_flat;
    compute_f1(X_values, plot_id, Y_flat, M_flat);

    // === ocl2.0-svm fast pipeline ===
    // Use lightweight pairs (Y, orig_idx) — 8 bytes each, NOT 64-byte PlotEntry.
    // M_curr stays as flat uint32_t array, uploaded to GPU once.
    std::vector<std::pair<uint32_t, uint32_t>> entries(num_x);
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < num_x; i++) {
        entries[i] = {Y_flat[i], (uint32_t)i};
    }

    // M_curr_flat: flat metadata array [num_entries * N_META]
    std::vector<uint32_t> M_curr_flat(num_x * N_META);
    std::memcpy(M_curr_flat.data(), M_flat.data(), num_x * N_META * sizeof(uint32_t));

    // Detect GPU vendor for sort tiebreaker (NVIDIA needs metadata sort)
    char dev_vendor[256] = {};
    clGetDeviceInfo(gpu.device, CL_DEVICE_VENDOR, sizeof(dev_vendor), dev_vendor, nullptr);
    bool is_nvidia = strstr(dev_vendor, "NVIDIA") != nullptr;
    bool use_meta_sort = is_nvidia;

    // Initialize GPU-resident hash
    if(!use_gpu_resident) {
        init_hash_lr_kernel();
    }
    cl_mem M_curr_gpu = nullptr, M_out_gpu = nullptr;
    bool gpu_resident = use_gpu_resident && k_table_hash_lr != nullptr;
    if(gpu_resident) {
        // Allocate 1.1x for growth (entries grow ~2% per table) — allocate ONCE, never resize
        // (resizing releases and recreates buffers, losing M_curr data)
        size_t max_entries = (size_t)(num_x * 11 / 10 + 256);
        ensure_gpu_resident_buffers(max_entries);
        M_curr_gpu = this->M_curr_gpu;
        M_out_gpu = this->M_out_gpu;
        clEnqueueWriteBuffer(gpu.queue, M_curr_gpu, CL_TRUE, 0,
            num_x * N_META * sizeof(uint32_t), M_curr_flat.data(), 0, nullptr, nullptr);
        std::cout << "[GPU-Res] Uploaded F1 metadata to GPU: " << num_x << " entries (cap=" << max_entries << ")" << std::endl;
    }

    // PD storage: PD[t] = vector of (sorted_L, delta) in sorted Y order
    std::vector<std::vector<PDEntry>> PD(N_TABLE + 1);
    // LR[t] = (sorted_L_pos, sorted_R_pos) per match — for PD build
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> LR(N_TABLE + 1);
    // X pairs for table 2
    std::vector<uint32_t> x_pairs_flat;

    auto t0 = std::chrono::high_resolution_clock::now();

    for(int t = 2; t <= N_TABLE; t++) {
        auto t_table = std::chrono::high_resolution_clock::now();
        const size_t n = entries.size();

        std::vector<std::pair<uint32_t, uint32_t>> lr_pairs;      // (orig_L, orig_R)
        std::vector<std::pair<uint32_t, uint32_t>> sorted_entries;   // (Y, orig_idx) sorted by Y
        std::vector<uint32_t> inv_map;                                // orig_pos → sorted_pos

        if(t == 2) {
            // T2: entries come unsorted from F1 — radix sort first
            radix_sort_pairs(entries, ksize);
        }
        // entries are now sorted by Y (T2: just sorted, T3+: from previous match sort)
        sorted_entries = entries;  // already sorted
        // Bucket match on sorted entries (LOGBUCKETS=6 like ocl2.0-svm)
        {
            const int log_buckets = std::min(6, (int)ksize - 1);
            const int shift = ksize - log_buckets;
            const size_t num_buckets = 1u << log_buckets;
            std::vector<uint32_t> bucket_counts(num_buckets, 0);
            for(size_t i = 0; i < n; i++)
                bucket_counts[std::min(entries[i].first >> shift, (uint32_t)num_buckets - 1)]++;
            std::vector<uint32_t> bucket_offsets(num_buckets + 1, 0);
            for(size_t i = 0; i < num_buckets; i++)
                bucket_offsets[i + 1] = bucket_offsets[i] + bucket_counts[i];
            const int nthreads = std::min(omp_get_max_threads(), (int)num_buckets);
            std::vector<std::vector<std::pair<uint32_t, uint32_t>>> thread_lr(nthreads);
            // Pre-allocate to avoid reallocation during emplace_back
            for(int ti = 0; ti < nthreads; ti++) thread_lr[ti].reserve(n / nthreads * 2);
            #pragma omp parallel for schedule(dynamic, 1)
            for(size_t b = 0; b < num_buckets; b++) {
                int ti = omp_get_thread_num();
                uint32_t start = bucket_offsets[b], cnt = bucket_counts[b];
                if(cnt == 0) continue;
                for(uint32_t x = start; x < start + cnt; x++) {
                    const auto YL = entries[x].first;
                    for(uint32_t y = x + 1; y < start + cnt && entries[y].first <= YL + 1; y++)
                        if(entries[y].first == YL + 1) thread_lr[ti].emplace_back(entries[x].second, entries[y].second);
                }
                if(b + 1 < num_buckets) {
                    uint32_t ns = bucket_offsets[b+1], nc = bucket_counts[b+1];
                    if(nc > 0) {
                        uint32_t last_Y = entries[start + cnt - 1].first;
                        for(int64_t x = (int64_t)start + cnt - 1; x >= (int64_t)start; x--) {
                            const auto YL = entries[x].first;
                            if(YL < last_Y) break;
                            for(uint32_t y = ns; y < ns + nc && entries[y].first <= YL + 1; y++)
                                if(entries[y].first == YL + 1) thread_lr[ti].emplace_back(entries[x].second, entries[y].second);
                        }
                    }
                }
            }
            size_t total = 0;
            for(int ti = 0; ti < nthreads; ti++) total += thread_lr[ti].size();
            lr_pairs.clear(); lr_pairs.reserve(total);
            for(int ti = 0; ti < nthreads; ti++)
                for(const auto& p : thread_lr[ti]) lr_pairs.push_back(p);
            inv_map.resize(n);
            for(size_t i = 0; i < n; i++) inv_map[entries[i].second] = (uint32_t)i;
        }

        size_t total_matches = lr_pairs.size();

        auto t_bm = std::chrono::high_resolution_clock::now();
        double bm_ms = std::chrono::duration<double, std::milli>(t_bm - t_table).count();

        if(total_matches == 0) throw std::runtime_error("zero matches at table " + std::to_string(t));

        // --- GPU hash ---
        std::vector<uint32_t> Y_results;
        std::vector<uint32_t> M_results;  // only used in non-resident path

        // Build LR_flat: (orig_L, orig_R) indices into M_curr
        std::vector<uint32_t> LR_flat(total_matches * 2);
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < total_matches; i++) {
            LR_flat[i * 2]     = lr_pairs[i].first;
            LR_flat[i * 2 + 1] = lr_pairs[i].second;
        }

        auto t_hash_start = std::chrono::high_resolution_clock::now();

        if(gpu_resident && total_matches > 0) {
            // GPU-resident: M_curr stays on GPU, only upload LR + download Y
            // Buffer already sized to 1.1x at init — no resize here (would lose data)
            cl_mem LR_buf = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                total_matches * 2 * sizeof(uint32_t), LR_flat.data(), nullptr);
            cl_mem Y_buf = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY,
                total_matches * sizeof(uint32_t), nullptr, nullptr);

            uint32_t n_match_u32 = (uint32_t)total_matches;
            uint32_t n_total_u32 = (uint32_t)n;
            clSetKernelArg(k_table_hash_lr, 0, sizeof(cl_mem), &M_curr_gpu);
            clSetKernelArg(k_table_hash_lr, 1, sizeof(cl_mem), &LR_buf);
            clSetKernelArg(k_table_hash_lr, 2, sizeof(cl_mem), &Y_buf);
            clSetKernelArg(k_table_hash_lr, 3, sizeof(cl_mem), &M_out_gpu);
            clSetKernelArg(k_table_hash_lr, 4, sizeof(uint32_t), &kmask);
            clSetKernelArg(k_table_hash_lr, 5, sizeof(uint32_t), &n_match_u32);
            clSetKernelArg(k_table_hash_lr, 6, sizeof(uint32_t), &n_total_u32);

            size_t local = (size_t)g_hash_local;
            size_t global = ((total_matches + local - 1) / local) * local;
            cl_int err = clEnqueueNDRangeKernel(gpu.queue, k_table_hash_lr, 1, nullptr, &global, &local, 0, nullptr, nullptr);
            if(err != CL_SUCCESS) {
                // Fallback to non-resident
                std::cerr << "[GPU-Res] hash failed (err=" << err << "), falling back" << std::endl;
                gpu_resident = false;
            } else {
                gpu.finish();
                Y_results.resize(total_matches);
                clEnqueueReadBuffer(gpu.queue, Y_buf, CL_TRUE, 0,
                    total_matches * sizeof(uint32_t), Y_results.data(), 0, nullptr, nullptr);
                std::swap(M_curr_gpu, M_out_gpu);
            }
            clReleaseMemObject(LR_buf);
            clReleaseMemObject(Y_buf);
        }

        if(!gpu_resident && total_matches > 0) {
            // Non-resident: upload M_curr + LR, download Y + M_out
            ensure_hash_buffers(total_matches);
            cl_int err;
            cl_mem Mb = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                M_curr_flat.size() * sizeof(uint32_t), M_curr_flat.data(), &err);
            cl_mem LRb = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                total_matches * 2 * sizeof(uint32_t), LR_flat.data(), &err);
            cl_mem Yb = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY,
                total_matches * sizeof(uint32_t), nullptr, &err);
            cl_mem Mb_out = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY,
                total_matches * N_META * sizeof(uint32_t), nullptr, &err);

            uint32_t n_match_u32 = (uint32_t)total_matches;
            uint32_t n_total_u32 = (uint32_t)(M_curr_flat.size() / N_META);
            clSetKernelArg(k_table_hash_lr, 0, sizeof(cl_mem), &Mb);
            clSetKernelArg(k_table_hash_lr, 1, sizeof(cl_mem), &LRb);
            clSetKernelArg(k_table_hash_lr, 2, sizeof(cl_mem), &Yb);
            clSetKernelArg(k_table_hash_lr, 3, sizeof(cl_mem), &Mb_out);
            clSetKernelArg(k_table_hash_lr, 4, sizeof(uint32_t), &kmask);
            clSetKernelArg(k_table_hash_lr, 5, sizeof(uint32_t), &n_match_u32);
            clSetKernelArg(k_table_hash_lr, 6, sizeof(uint32_t), &n_total_u32);

            size_t local = (size_t)g_hash_local;
            size_t global = ((total_matches + local - 1) / local) * local;
            clEnqueueNDRangeKernel(gpu.queue, k_table_hash_lr, 1, nullptr, &global, &local, 0, nullptr, nullptr);
            gpu.finish();

            Y_results.resize(total_matches);
            M_results.resize(total_matches * N_META);
            clEnqueueReadBuffer(gpu.queue, Yb, CL_TRUE, 0, total_matches * sizeof(uint32_t), Y_results.data(), 0, nullptr, nullptr);
            clEnqueueReadBuffer(gpu.queue, Mb_out, CL_TRUE, 0, total_matches * N_META * sizeof(uint32_t), M_results.data(), 0, nullptr, nullptr);

            clReleaseMemObject(Mb); clReleaseMemObject(LRb);
            clReleaseMemObject(Yb); clReleaseMemObject(Mb_out);

            M_curr_flat = std::move(M_results);
        }

        auto t_hash_end = std::chrono::high_resolution_clock::now();
        double hash_ms = std::chrono::duration<double, std::milli>(t_hash_end - t_hash_start).count();

        // --- Build matches = (Y, match_idx) and sort by Y ---
        std::vector<std::pair<uint32_t, uint32_t>> matches(total_matches);
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < total_matches; i++) {
            matches[i] = {Y_results[i], (uint32_t)i};
        }

        // Convert LR pairs from original positions to sorted positions (for PD)
        LR[t].resize(total_matches);
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < total_matches; i++) {
            LR[t][i] = {inv_map[lr_pairs[i].first], inv_map[lr_pairs[i].second]};
        }

        radix_sort_pairs(matches, ksize);
        auto t_sort_end = std::chrono::high_resolution_clock::now();

        // --- Build PD in sorted order ---
        PD[t].resize(total_matches);
        #pragma omp parallel for schedule(static)
        for(size_t k = 0; k < total_matches; k++) {
            uint32_t match_idx = matches[k].second;
            uint32_t sorted_L = LR[t][match_idx].first;
            uint32_t sorted_R = LR[t][match_idx].second;
            uint16_t delta = (uint16_t)(sorted_R - sorted_L - 1);  // CUDA reference: delta = R - L - 1
            PD[t][k] = {sorted_L, delta};
        }

        // --- X pairs for table 2 ---
        if(t == 2) {
            // sorted_entries already moved to `entries`, but we have inv_map + lr_pairs
            // sorted_to_x[sorted_pos] = X_values[orig_idx at that sorted_pos]
            // entries[sorted_pos] = (Y, orig_idx) after bitmap_match
            x_pairs_flat.resize(total_matches * 2);
            #pragma omp parallel for schedule(static)
            for(size_t k = 0; k < total_matches; k++) {
                uint32_t match_idx = matches[k].second;
                uint32_t sorted_L = LR[t][match_idx].first;
                uint32_t sorted_R = LR[t][match_idx].second;
                x_pairs_flat[k * 2]     = X_values[entries[sorted_L].second];
                x_pairs_flat[k * 2 + 1] = X_values[entries[sorted_R].second];
            }
        }

        // entries = matches (Y, idx into M_next) — pre-sorted by Y for next table
        entries = std::move(matches);
        auto t_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_table).count();
        double sort_ms = std::chrono::duration<double, std::milli>(t_sort_end - t_hash_end).count();

        std::cout << "[T" << t << "] " << total_matches << " matches (match=" << bm_ms
                  << "ms hash=" << hash_ms << "ms sort=" << sort_ms
                  << "ms total=" << total_ms << "ms)" << std::endl;

        if(entries.empty()) break;
    }

    // --- Download final M from GPU (resident mode) ---
    if(gpu_resident && M_curr_gpu && entries.size() > 0) {
        size_t n = entries.size();
        M_curr_flat.resize(n * N_META);
        clEnqueueReadBuffer(gpu.queue, M_curr_gpu, CL_TRUE, 0,
            n * N_META * sizeof(uint32_t), M_curr_flat.data(), 0, nullptr, nullptr);
        clReleaseMemObject(M_curr_gpu);
        clReleaseMemObject(M_out_gpu);
        this->M_curr_gpu = nullptr;
        this->M_out_gpu = nullptr;
    }

    // --- Convert to PlotData for plot_writer ---
    result.final_Y.resize(entries.size());
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < entries.size(); i++) {
        result.final_Y[i] = entries[i].first;
    }

    // Build final table entries (PlotEntry) for plot_writer metadata table
    {
        std::vector<PlotEntry> final_entries(entries.size());
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < entries.size(); i++) {
            final_entries[i].Y = entries[i].first;
            final_entries[i].orig_idx = entries[i].second;
            for(int j = 0; j < N_META; j++) {
                final_entries[i].M[j] = M_curr_flat[entries[i].second * N_META + j];
            }
        }
        result.table_entries.push_back(final_entries);
    }

    // PD data: pd_data[0] = PD[2], pd_data[1] = PD[3], ..., pd_data[7] = PD[9]
    for(int t = 2; t <= N_TABLE; t++) {
        result.pd_data.push_back(PD[t]);
    }

    // X pairs
    result.x_pairs = std::move(x_pairs_flat);

    auto t_total = std::chrono::high_resolution_clock::now();
    double pipeline_ms = std::chrono::duration<double, std::milli>(t_total - t0).count();
    std::cout << "Pipeline time: " << pipeline_ms / 1000.0 << "s" << std::endl;
}

} // namespace mmx
