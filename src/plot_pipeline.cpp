#include "plot_pipeline.h"
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
    try {
        k_table_hash_lr = gpu.get_kernel("hash_table_entries_lr");
    } catch(...) {
        std::cout << "[OCL] hash_table_entries_lr not available, using legacy hash path" << std::endl;
        return false;
    }
    use_gpu_resident = true;
    std::cout << "[OCL] GPU-resident M_curr: using hash_table_entries_lr" << std::endl;
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

    ensure_f1_buffers(num_x);

    // Upload X values and plot ID
    cl_int err;
    err = clEnqueueWriteBuffer(gpu.queue, f1_X_buf, CL_FALSE, 0,
                               num_x * sizeof(uint32_t), X_values.data(), 0, nullptr, nullptr);
    GPUDevice::check(err, "write X");
    err = clEnqueueWriteBuffer(gpu.queue, f1_ID_buf, CL_FALSE, 0,
                               8 * sizeof(uint32_t), plot_id, 0, nullptr, nullptr);
    GPUDevice::check(err, "write ID");

    // Set kernel args
    clSetKernelArg(k_f1, 0, sizeof(cl_mem), &f1_X_buf);
    clSetKernelArg(k_f1, 1, sizeof(cl_mem), &f1_ID_buf);
    clSetKernelArg(k_f1, 2, sizeof(cl_mem), &f1_Y_buf);
    clSetKernelArg(k_f1, 3, sizeof(cl_mem), &f1_M_buf);
    clSetKernelArg(k_f1, 4, sizeof(uint32_t), &kmask);
    clSetKernelArg(k_f1, 5, sizeof(uint32_t), &xbits);
    clSetKernelArg(k_f1, 6, sizeof(uint32_t), &num_x);

    // Launch
    size_t local = 64;
    size_t global = ((num_x + local - 1) / local) * local;
    err = clEnqueueNDRangeKernel(gpu.queue, k_f1, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    GPUDevice::check(err, "enqueue f1");
    gpu.finish();

    // Read back
    Y_out.resize(num_x);
    M_out.resize(num_x * N_META);
    err = clEnqueueReadBuffer(gpu.queue, f1_Y_buf, CL_TRUE, 0,
                              num_x * sizeof(uint32_t), Y_out.data(), 0, nullptr, nullptr);
    GPUDevice::check(err, "read Y");
    err = clEnqueueReadBuffer(gpu.queue, f1_M_buf, CL_TRUE, 0,
                              num_x * N_META * sizeof(uint32_t), M_out.data(), 0, nullptr, nullptr);
    GPUDevice::check(err, "read M");
}

void PlotPipeline::sort_entries_by_y(std::vector<PlotEntry>& entries)
{
	const size_t n = entries.size();
    if(n < 2) return;

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

    // Extract L_meta, R_meta for GPU hash
    const size_t n_matches = matches.size();

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

    auto t3 = std::chrono::high_resolution_clock::now();
    timing.extract_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // GPU hash — kmask is member variable
    ensure_hash_buffers(n_matches);

    cl_int err;
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

    std::vector<uint32_t> Y_out(n_matches);
    std::vector<uint32_t> M_out(n_matches * N_META);
    err = clEnqueueReadBuffer(gpu.queue, hash_Y_buf, CL_TRUE, 0,
                              n_matches * sizeof(uint32_t), Y_out.data(), 0, nullptr, nullptr);
    GPUDevice::check(err, "read Y");
    err = clEnqueueReadBuffer(gpu.queue, hash_M_buf, CL_TRUE, 0,
                              n_matches * N_META * sizeof(uint32_t), M_out.data(), 0, nullptr, nullptr);
    GPUDevice::check(err, "read M");

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

    // Build next table entries with original indices
    entries.resize(n_matches);
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < n_matches; i++) {
        entries[i].Y = Y_out[i];
        entries[i].orig_idx = (uint32_t)i;  // tracks position in this table's match list
        for(int j = 0; j < N_META; j++) {
            entries[i].M[j] = M_out[i * N_META + j];
        }
    }

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
                  << "ms)" << std::endl;
        
        if(entries.empty()) break;
    }
    // Save final Y values
    result.final_Y.resize(entries.size());
    for(size_t i = 0; i < entries.size(); i++) {
        result.final_Y[i] = entries[i].Y;
    }
}

} // namespace mmx
