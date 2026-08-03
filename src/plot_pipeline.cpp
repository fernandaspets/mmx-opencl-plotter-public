#include "plot_pipeline.h"
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
    // Reload table_hash.cl to pick up hash_table_entries_lr kernel
    for(const auto& path : {"kernels/table_hash.cl", "../kernels/table_hash.cl", "../../kernels/table_hash.cl"}) {
        std::ifstream f(path);
        if(f.good()) {
            std::cout << "[OCL] Loading LR kernel from " << path << std::endl;
            try {
                gpu.load_program_from_file("table_hash", path);
                // Re-create both kernels from reloaded program
                k_table_hash = gpu.get_kernel("hash_table_entries");
                k_table_hash_lr = gpu.get_kernel("hash_table_entries_lr");
                if(k_table_hash_lr) {
                    std::cout << "[OCL] hash_table_entries_lr loaded successfully" << std::endl;
                    break;
                }
            } catch(const std::exception& e) {
                std::cout << "[OCL] Failed to load LR kernel: " << e.what() << std::endl;
                k_table_hash_lr = nullptr;
            }
        }
    }
    if(!k_table_hash_lr) {
        std::cout << "[OCL] hash_table_entries_lr not available, using legacy hash" << std::endl;
        return false;
    }
    use_gpu_resident = true;
    char vendor[256] = {};
    clGetDeviceInfo(gpu.device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
    bool is_nvidia = (strstr(vendor, "NVIDIA") != nullptr);
    g_hash_local = is_nvidia ? 64 : 256;
    // Also parse OpenCL version — NVIDIA OpenCL 1.2 may not support global atomics in our kernel
    std::cout << "[OCL] GPU-resident M_curr: hash_table_entries_lr (local=" << g_hash_local << ")" << std::endl;
    
    // Test the kernel with a tiny invocation to verify it works
    ensure_gpu_resident_buffers(64);
    cl_mem test_LR = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY, 4*sizeof(uint32_t), nullptr, nullptr);
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
        std::cout << "[OCL] hash_table_entries_lr failed test invocation (err=" << err << "), falling back to legacy" << std::endl;
        use_gpu_resident = false;
        k_table_hash_lr = nullptr;
        clReleaseMemObject(test_LR);
        clReleaseMemObject(test_Y);
        return false;
    }
    gpu.finish();
    clReleaseMemObject(test_LR);
    clReleaseMemObject(test_Y);
    std::cout << "[OCL] hash_table_entries_lr verified OK" << std::endl;
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
            gpu.finish();

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
    if(n < 2) return;
    const int passes = (ksize + 7) / 8;  // number of 8-bit radix passes
    const int num_threads = omp_get_max_threads();
    std::vector<std::pair<uint32_t, uint32_t>> temp(n);
    for(int pass = 0; pass < passes; pass++) {
        const int shift = pass * 8;
        const uint32_t mask = 0xFF;
        // Count per digit
        std::vector<uint32_t> counts(256 * num_threads, 0);
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < n; i++) {
            int ti = omp_get_thread_num();
            uint32_t digit = (entries[i].first >> shift) & mask;
            counts[ti * 256 + digit]++;
        }
        // Prefix sum
        for(int ti = 0; ti < num_threads; ti++) {
            uint32_t sum = 0;
            for(int d = 0; d < 256; d++) {
                uint32_t c = counts[ti * 256 + d];
                counts[ti * 256 + d] = sum;
                sum += c;
            }
        }
        std::vector<uint32_t> base(256, 0);
        for(int d = 0; d < 256; d++) {
            for(int ti = 0; ti < num_threads; ti++) {
                base[d] += counts[ti * 256 + d];
            }
        }
        uint32_t total = 0;
        for(int d = 0; d < 256; d++) {
            uint32_t c = base[d];
            base[d] = total;
            total += c;
        }
        for(int ti = 0; ti < num_threads; ti++) {
            for(int d = 0; d < 256; d++) {
                counts[ti * 256 + d] += base[d];
            }
        }
        // Scatter
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < n; i++) {
            int ti = omp_get_thread_num();
            uint32_t digit = (entries[i].first >> shift) & mask;
            uint32_t pos = counts[ti * 256 + digit]++;
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

    // Scatter into buckets
    std::vector<PlotEntry> sorted(n);
    std::vector<uint32_t> write_pos = bucket_offsets;
    for(const auto& e : entries) {
        uint32_t b = std::min(e.Y >> shift, (uint32_t)num_buckets - 1);
        sorted[write_pos[b]++] = e;
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

    // Match Y,Y+1
    auto matches = match_entries(entries);
    timing.n_matches = matches.size();
    auto t2 = std::chrono::high_resolution_clock::now();
    timing.match_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    if(matches.empty()) return timing;

    const size_t n_matches = matches.size();

    auto t3 = std::chrono::high_resolution_clock::now();

    // GPU hash — use hash_table_entries_lr when available (GPU-resident M_curr)
    ensure_hash_buffers(n_matches);

    cl_int err;
    std::vector<uint32_t> Y_out(n_matches);
    std::vector<uint32_t> M_out;

    if(use_gpu_resident && M_curr_gpu && M_out_gpu && k_table_hash_lr) {
        // Build LR_flat with ORIGINAL indices (entries[i].orig_idx references M_curr_gpu)
        std::vector<uint32_t> LR_flat(n_matches * 2);
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < n_matches; i++) {
            const auto& m = matches[i];
            LR_flat[i * 2]     = entries[m.first].orig_idx;
            LR_flat[i * 2 + 1] = entries[m.second].orig_idx;
        }
        auto t_ext = std::chrono::high_resolution_clock::now();
        timing.extract_ms = std::chrono::duration<double, std::milli>(t_ext - t3).count();

        // Upload LR pairs (2 uint32s per match vs 28 for L_meta + 28 for R_meta)
        cl_mem LR_buf = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            n_matches * 2 * sizeof(uint32_t), LR_flat.data(), &err);
        GPUDevice::check(err, "LR_buf");
        cl_mem Y_buf_lr = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE,
            n_matches * sizeof(uint32_t), nullptr, &err);
        GPUDevice::check(err, "Y_buf_lr");

        uint32_t n_total = (uint32_t)entries.size();
        uint32_t n_match = (uint32_t)n_matches;
        clSetKernelArg(k_table_hash_lr, 0, sizeof(cl_mem), &M_curr_gpu);
        clSetKernelArg(k_table_hash_lr, 1, sizeof(cl_mem), &LR_buf);
        clSetKernelArg(k_table_hash_lr, 2, sizeof(cl_mem), &Y_buf_lr);
        clSetKernelArg(k_table_hash_lr, 3, sizeof(cl_mem), &M_out_gpu);
        clSetKernelArg(k_table_hash_lr, 4, sizeof(uint32_t), &kmask);
        clSetKernelArg(k_table_hash_lr, 5, sizeof(uint32_t), &n_match);
        clSetKernelArg(k_table_hash_lr, 6, sizeof(uint32_t), &n_total);

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

    result.final_Y.clear();
    result.table_entries.clear();
    result.pd_data.clear();
    result.x_pairs.clear();
    result.timings.clear();

    // F1
    std::vector<uint32_t> Y_flat, M_flat;
    compute_f1(X_values, plot_id, Y_flat, M_flat);

    // Build table 1 entries with original indices
    std::vector<PlotEntry> entries(num_x);
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < num_x; i++) {
        entries[i].Y = Y_flat[i];
        entries[i].orig_idx = (uint32_t)i;
        for(int j = 0; j < N_META; j++) {
            entries[i].M[j] = M_flat[i * N_META + j];
        }
    }

    result.table_entries.push_back(entries);

    // Initialize GPU-resident hash if available
    if(!use_gpu_resident) {
        init_hash_lr_kernel();
        if(use_gpu_resident) {
            ensure_gpu_resident_buffers(num_x);
            clEnqueueWriteBuffer(gpu.queue, M_curr_gpu, CL_TRUE, 0,
                num_x * N_META * sizeof(uint32_t), M_flat.data(), 0, nullptr, nullptr);
        }
    }

    // F2-F9
    for(int t = 2; t <= N_TABLE; t++) {
        std::vector<PDEntry> table_pd;
        
        std::cout << "[T" << t << "] " << entries.size() << " entries..." << std::flush;
        
        // Save sorted entries BEFORE processing for X-pair lookup
        std::vector<PlotEntry> prev_sorted_entries;
        if(t == 2) {
            prev_sorted_entries = entries;
            sort_entries_by_y(prev_sorted_entries);
        }
        
        auto tt = process_table(entries,
            (t == 2) ? &X_values : nullptr,
            &table_pd);
        
        result.timings.push_back(tt);
        result.table_entries.push_back(entries);
        result.pd_data.push_back(table_pd);
        
        // For table 2: compute X pairs from original indices
        if(t == 2 && tt.n_matches > 0) {
            std::vector<uint32_t> sorted_to_x(prev_sorted_entries.size());
            for(size_t j = 0; j < prev_sorted_entries.size(); j++) {
                sorted_to_x[j] = X_values[prev_sorted_entries[j].orig_idx];
            }
            
            result.x_pairs.resize(tt.n_matches * 2);
            for(uint32_t j = 0; j < tt.n_matches; j++) {
                auto& pd = result.pd_data.back()[j];
                uint32_t sorted_L = pd.first;
                uint32_t delta = pd.second;
                uint32_t sorted_R = sorted_L + delta + 1;
                
                result.x_pairs[j * 2]     = sorted_to_x[sorted_L];
                result.x_pairs[j * 2 + 1] = sorted_to_x[sorted_R];
            }
            std::cout << " (" << result.x_pairs.size()/2 << " X pairs)" << std::flush;
        }
        
        std::cout << " " << tt.n_matches << " matches (sort=" << tt.sort_ms
                  << "ms match=" << tt.match_ms << "ms hash=" << tt.hash_ms
                  << "ms extract=" << tt.extract_ms
                  << "ms build=" << tt.build_ms
                  << "ms)" << std::endl;
        
        if(entries.empty()) break;
    }
    // Save final Y values
    result.final_Y.resize(entries.size());
    for(size_t i = 0; i < entries.size(); i++) {
        result.final_Y[i] = entries[i].Y;
    }

    // Download final M from GPU for plot writer (GPU-resident path)
    if(use_gpu_resident && M_curr_gpu && entries.size() > 0) {
        size_t n = entries.size();
        std::vector<uint32_t> M_final(n * N_META);
        clEnqueueReadBuffer(gpu.queue, M_curr_gpu, CL_TRUE, 0,
            n * N_META * sizeof(uint32_t), M_final.data(), 0, nullptr, nullptr);
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < n; i++) {
            for(int j = 0; j < N_META; j++) {
                entries[i].M[j] = M_final[i * N_META + j];
            }
        }
    }
}

} // namespace mmx
