#include "multi_pipeline.h"
#include <algorithm>
#include <omp.h>

namespace mmx {

void MultiPipeline::ensure_gpu_bufs(int gpu_idx, size_t n_matches) {
    auto& bufs = gpu_bufs[gpu_idx];
    if(n_matches <= bufs.capacity) return;
    
    auto& gpu = gpu_mgr[gpu_idx];
    cl_int err;
    
    auto free_buf = [](cl_mem& buf) { if(buf) { clReleaseMemObject(buf); buf = nullptr; } };
    
    free_buf(bufs.L_buf); free_buf(bufs.R_buf);
    free_buf(bufs.Y_buf); free_buf(bufs.M_buf);
    
    bufs.L_buf = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY,
                                n_matches * N_META * sizeof(uint32_t), nullptr, &err);
    bufs.R_buf = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY,
                                n_matches * N_META * sizeof(uint32_t), nullptr, &err);
    bufs.Y_buf = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY,
                                n_matches * sizeof(uint32_t), nullptr, &err);
    bufs.M_buf = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY,
                                n_matches * N_META * sizeof(uint32_t), nullptr, &err);
    bufs.capacity = n_matches;
}

void MultiPipeline::compute_f1(
    const std::vector<uint32_t>& X_values,
    const uint32_t* plot_id,
    std::vector<uint32_t>& Y_out,
    std::vector<uint32_t>& M_out)
{
    const size_t N = gpu_mgr.size();
    const size_t total = X_values.size();
    const size_t chunk = (total + N - 1) / N;
    
    std::vector<std::vector<uint32_t>> Y_chunks(N), M_chunks(N);
    
    #pragma omp parallel for num_threads(N)
    for(size_t g = 0; g < N; g++) {
        size_t start = g * chunk;
        size_t count = std::min(chunk, total - start);
        if(count == 0) continue;
        
        std::vector<uint32_t> X_chunk(X_values.begin() + start, X_values.begin() + start + count);
        pipelines[g]->compute_f1(X_chunk, plot_id, Y_chunks[g], M_chunks[g]);
    }
    
    // Merge results
    size_t total_y = 0, total_m = 0;
    for(size_t g = 0; g < N; g++) { total_y += Y_chunks[g].size(); total_m += M_chunks[g].size(); }
    
    Y_out.resize(total_y);
    M_out.resize(total_m);
    
    size_t y_off = 0, m_off = 0;
    for(size_t g = 0; g < N; g++) {
        std::memcpy(Y_out.data() + y_off, Y_chunks[g].data(), Y_chunks[g].size() * sizeof(uint32_t));
        std::memcpy(M_out.data() + m_off, M_chunks[g].data(), M_chunks[g].size() * sizeof(uint32_t));
        y_off += Y_chunks[g].size();
        m_off += M_chunks[g].size();
    }
}

TableTiming MultiPipeline::process_table(
    std::vector<PlotEntry>& entries,
    const std::vector<uint32_t>* x_values_orig,
    std::vector<PDEntry>* pd_out)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    const size_t N = gpu_mgr.size();
    
    // Sort on CPU (single-threaded by design)
    pipelines[0]->sort_entries_by_y(entries);
    auto t1 = std::chrono::high_resolution_clock::now();
    double sort_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    // Match on CPU
    auto matches = pipelines[0]->match_entries(entries);
    auto t2 = std::chrono::high_resolution_clock::now();
    double match_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    
    TableTiming timing;
    timing.n_entries = entries.size();
    timing.n_matches = matches.size();
    timing.sort_ms = sort_ms;
    timing.match_ms = match_ms;
    
    if(matches.empty()) return timing;
    
    const size_t n_matches = matches.size();
    const size_t chunk = (n_matches + N - 1) / N;
    
    // Build per-GPU metadata chunks
    std::vector<std::vector<uint32_t>> L_chunks(N), R_chunks(N);
    
    #pragma omp parallel for num_threads(N)
    for(size_t g = 0; g < N; g++) {
        size_t start = g * chunk;
        size_t count = std::min(chunk, n_matches - start);
        if(count == 0) continue;
        
        L_chunks[g].resize(count * N_META);
        R_chunks[g].resize(count * N_META);
        
        for(size_t i = 0; i < count; i++) {
            const auto& m = matches[start + i];
            for(int j = 0; j < N_META; j++) {
                L_chunks[g][i * N_META + j] = entries[m.first].M[j];
                R_chunks[g][i * N_META + j] = entries[m.second].M[j];
            }
        }
        
        // Upload and hash on GPU g
        auto& gpu = gpu_mgr[g];
        gpu_bufs.resize(std::max(gpu_bufs.size(), N));
        ensure_gpu_bufs(g, count);
        
        cl_int err;
        err = clEnqueueWriteBuffer(gpu.queue, gpu_bufs[g].L_buf, CL_FALSE, 0,
                                   count * N_META * sizeof(uint32_t), L_chunks[g].data(), 0, nullptr, nullptr);
        err = clEnqueueWriteBuffer(gpu.queue, gpu_bufs[g].R_buf, CL_FALSE, 0,
                                   count * N_META * sizeof(uint32_t), R_chunks[g].data(), 0, nullptr, nullptr);
        
        cl_kernel k_hash = pipelines[g]->get_table_hash_kernel();
        clSetKernelArg(k_hash, 0, sizeof(cl_mem), &gpu_bufs[g].L_buf);
        clSetKernelArg(k_hash, 1, sizeof(cl_mem), &gpu_bufs[g].R_buf);
        clSetKernelArg(k_hash, 2, sizeof(cl_mem), &gpu_bufs[g].Y_buf);
        clSetKernelArg(k_hash, 3, sizeof(cl_mem), &gpu_bufs[g].M_buf);
        clSetKernelArg(k_hash, 4, sizeof(uint32_t), &kmask);
        uint32_t n_u32 = (uint32_t)count;
        clSetKernelArg(k_hash, 5, sizeof(uint32_t), &n_u32);
        
        size_t local = 64;
        size_t global = ((count + local - 1) / local) * local;
        clEnqueueNDRangeKernel(gpu.queue, k_hash, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    }
    
    // Wait for all GPUs
    for(size_t g = 0; g < N; g++) gpu_mgr[g].finish();
    auto t3 = std::chrono::high_resolution_clock::now();
    timing.hash_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    
    // Read back and merge results
    std::vector<uint32_t> Y_out(n_matches), M_out(n_matches * N_META);
    
    #pragma omp parallel for num_threads(N)
    for(size_t g = 0; g < N; g++) {
        size_t start = g * chunk;
        size_t count = std::min(chunk, n_matches - start);
        if(count == 0) continue;
        
        clEnqueueReadBuffer(gpu_mgr[g].queue, gpu_bufs[g].Y_buf, CL_TRUE, 0,
                            count * sizeof(uint32_t), Y_out.data() + start, 0, nullptr, nullptr);
        clEnqueueReadBuffer(gpu_mgr[g].queue, gpu_bufs[g].M_buf, CL_TRUE, 0,
                            count * N_META * sizeof(uint32_t), M_out.data() + start * N_META, 0, nullptr, nullptr);
    }
    
    auto t4 = std::chrono::high_resolution_clock::now();
    timing.extract_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    
    // Build next entries
    entries.resize(n_matches);
    #pragma omp parallel for
    for(size_t i = 0; i < n_matches; i++) {
        entries[i].Y = Y_out[i];
        entries[i].orig_idx = (uint32_t)i;
        for(int j = 0; j < N_META; j++) {
            entries[i].M[j] = M_out[i * N_META + j];
        }
    }
    
    // PD entries
    if(pd_out) {
        pd_out->resize(n_matches);
        for(size_t i = 0; i < n_matches; i++) {
            (*pd_out)[i] = {matches[i].first, (uint16_t)(matches[i].second - matches[i].first - 1)};
        }
    }
    
    auto t5 = std::chrono::high_resolution_clock::now();
    timing.build_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
    
    return timing;
}

void MultiPipeline::run_full_pipeline(
    const std::vector<uint32_t>& X_values,
    const uint32_t* plot_id,
    PlotData& result)
{
    // Same as PlotPipeline::run_full_pipeline but using multi-GPU
    constexpr int N_TABLE = 9;
    const size_t num_x = X_values.size();
    
    result.final_Y.clear();
    result.table_entries.clear();
    result.pd_data.clear();
    result.x_pairs.clear();
    result.timings.clear();
    
    // F1 on all GPUs
    std::vector<uint32_t> Y_flat, M_flat;
    compute_f1(X_values, plot_id, Y_flat, M_flat);
    
    // Build entries
    std::vector<PlotEntry> entries(num_x);
    #pragma omp parallel for
    for(size_t i = 0; i < num_x; i++) {
        entries[i].Y = Y_flat[i];
        entries[i].orig_idx = (uint32_t)i;
        for(int j = 0; j < N_META; j++) entries[i].M[j] = M_flat[i * N_META + j];
    }
    result.table_entries.push_back(entries);
    
    // Tables F2-F9
    for(int t = 2; t <= N_TABLE; t++) {
        std::cout << "[T" << t << "] " << entries.size() << " entries..." << std::flush;
        
        std::vector<PDEntry> table_pd;
        std::vector<PlotEntry> prev_sorted;
        if(t == 2) { prev_sorted = entries; pipelines[0]->sort_entries_by_y(prev_sorted); }
        
        auto tt = process_table(entries, nullptr, &table_pd);
        result.timings.push_back(tt);
        result.table_entries.push_back(entries);
        result.pd_data.push_back(table_pd);
        
        // X pairs for table 2
        if(t == 2 && tt.n_matches > 0 && !prev_sorted.empty()) {
            std::vector<uint32_t> sorted_to_x(prev_sorted.size());
            for(size_t j = 0; j < prev_sorted.size(); j++)
                sorted_to_x[j] = X_values[prev_sorted[j].orig_idx];
            
            result.x_pairs.resize(tt.n_matches * 2);
            for(uint32_t j = 0; j < tt.n_matches; j++) {
                auto& pd = result.pd_data.back()[j];
                uint32_t sorted_L = pd.first;
                uint32_t sorted_R = sorted_L + pd.second + 1;
                result.x_pairs[j * 2] = sorted_to_x[sorted_L];
                result.x_pairs[j * 2 + 1] = sorted_to_x[sorted_R];
            }
            std::cout << " (" << result.x_pairs.size()/2 << " X pairs)" << std::flush;
        }
        
        std::cout << " " << tt.n_matches << " matches (sort=" << tt.sort_ms
                  << "ms match=" << tt.match_ms << "ms hash=" << tt.hash_ms
                  << "ms)" << std::endl;
        
        if(entries.empty()) break;
    }
    
    result.final_Y.resize(entries.size());
    for(size_t i = 0; i < entries.size(); i++) result.final_Y[i] = entries[i].Y;
}

} // namespace mmx
