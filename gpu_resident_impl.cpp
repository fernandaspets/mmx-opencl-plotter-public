// This file is included inline in plotter.cpp before compute_full_pipeline.
// It provides compute_gpu_resident() — a fully GPU-resident F2-F9 pipeline.

static void compute_gpu_resident(
    const std::vector<uint32_t>& Y_all,
    const std::vector<uint32_t>& M_all,
    PlotData& plot,
    OCL_Plotter& gpu_plotter,
    const hash_t& plot_id)
{
    const int n_meta = MY_N_META;
    const int n_meta_out = MY_N_META_OUT;
    const int n_table = MY_N_TABLE;
    const int log_b = LOGBUCKETS;
    const int log_b2 = KSIZE - LOGBUCKETS - 9;
    const int num_l1 = 1 << log_b;
    const int num_sub = 1 << log_b2;
    const uint32_t total = (uint32_t)Y_all.size();
    const int shift_l1 = KSIZE - log_b;
    
    const uint32_t entries_per_l1 = total / num_l1;
    const uint32_t max_bs = entries_per_l1 * 3 / 2 + 256;
    const uint32_t avg_sub = entries_per_l1 / num_sub;
    const uint32_t max_bs2 = std::max(1024u, avg_sub * 3 / 2 + 256);
    
    // VRAM estimate
    size_t c_sz = (size_t)num_l1 * max_bs * n_meta * 4;
    size_t py_sz = (size_t)num_sub * max_bs2 * 8;
    size_t lr_sz = (size_t)total * 2 * 4 * 4;  // max 4x matches
    size_t pd_sz = (size_t)total * 4;
    size_t total_vram = 2 * c_sz + py_sz + lr_sz + pd_sz + num_l1 * 4 * 3 + num_sub * 4 * 2;
    
    std::cout << "[GPU-Res] L1=" << num_l1 << " sub=" << num_sub
              << " max_bs=" << max_bs << " max_bs2=" << max_bs2
              << " VRAM~" << total_vram / 1e9 << "GB" << std::endl;
    
    size_t vram_avail = 0;
    clGetDeviceInfo(gpu_plotter.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(vram_avail), &vram_avail, nullptr);
    if(total_vram > vram_avail * 8 / 10) {
        throw std::runtime_error("[GPU-Res] Not enough VRAM: need " + std::to_string(total_vram/1e9) + "GB");
    }
    
    cl_int err;
    int zero = 0;
    auto t0 = my_time_ms();
    
    // Allocate GPU buffers
    cl_mem C_in = clCreateBuffer(gpu_plotter.context, CL_MEM_READ_WRITE, c_sz, nullptr, &err);
    cl_mem C_out = clCreateBuffer(gpu_plotter.context, CL_MEM_READ_WRITE, c_sz, nullptr, &err);
    cl_mem tmp_c = clCreateBuffer(gpu_plotter.context, CL_MEM_READ_ONLY, max_bs * n_meta * 4, nullptr, &err);
    cl_mem PY = clCreateBuffer(gpu_plotter.context, CL_MEM_READ_WRITE, py_sz, nullptr, &err);
    cl_mem sub_cnt = clCreateBuffer(gpu_plotter.context, CL_MEM_READ_WRITE, num_sub * 4, nullptr, &err);
    cl_mem sub_off = clCreateBuffer(gpu_plotter.context, CL_MEM_READ_WRITE, (num_sub + 1) * 4, nullptr, &err);
    cl_mem LR = clCreateBuffer(gpu_plotter.context, CL_MEM_WRITE_ONLY, lr_sz, nullptr, &err);
    cl_mem PD_match = clCreateBuffer(gpu_plotter.context, CL_MEM_WRITE_ONLY, pd_sz, nullptr, &err);
    cl_mem num_matches = clCreateBuffer(gpu_plotter.context, CL_MEM_READ_WRITE, 4, nullptr, &err);
    cl_mem new_bs = clCreateBuffer(gpu_plotter.context, CL_MEM_READ_WRITE, num_l1 * 4, nullptr, &err);
    cl_mem bucket_count = clCreateBuffer(gpu_plotter.context, CL_MEM_READ_WRITE, num_l1 * 4, nullptr, &err);
    cl_mem null_mem = nullptr;
    
    // Per-table LR storage (for final PD/X_pairs)
    std::vector<cl_mem> lr_per_table(n_table + 1);
    std::vector<uint32_t> lr_count_per_table(n_table + 1, 0);
    
    // PD output from eval_p1_tx (bit-packed, per L1 bucket)
    size_t pd_out_sz = (size_t)num_l1 * max_bs * PDSIZE / 8;
    std::vector<cl_mem> pd_out_per_table(n_table + 1);
    
    // ========================================================================
    // Step 1: Upload F1 metadata to C_in, set bucket_count
    // ========================================================================
    std::vector<uint32_t> f1_counts(num_l1, 0);
    {
        std::vector<std::vector<uint32_t>> l1_meta(num_l1);
        for(size_t i = 0; i < total; i++) {
            uint32_t Y = Y_all[i];
            uint32_t b = Y >> shift_l1;
            if(b >= (uint32_t)num_l1) b = num_l1 - 1;
            for(int j = 0; j < n_meta; j++)
                l1_meta[b].push_back(M_all[i * n_meta + j]);
            f1_counts[b]++;
        }
        for(int y = 0; y < num_l1; y++) {
            uint32_t cnt = f1_counts[y];
            if(cnt == 0) continue;
            size_t off = (size_t)y * max_bs * n_meta * 4;
            clEnqueueWriteBuffer(gpu_plotter.queue, C_in, CL_FALSE, off,
                cnt * n_meta * 4, l1_meta[y].data(), 0, nullptr, nullptr);
        }
        clEnqueueWriteBuffer(gpu_plotter.queue, bucket_count, CL_TRUE, 0,
            num_l1 * 4, f1_counts.data(), 0, nullptr, nullptr);
        clFinish(gpu_plotter.queue);
    }
    std::cout << "[GPU-Res] F1 uploaded in " << (my_time_ms() - t0) / 1000.0 << "s" << std::endl;
    
    // ========================================================================
    // Step 2: Process tables 2-9 on GPU
    // ========================================================================
    cl_mem active_count = bucket_count;
    cl_mem active_c_in = C_in;
    cl_mem active_c_out = C_out;
    
    for(int t = 2; t <= n_table; t++) {
        auto tt = my_time_ms();
        
        // Clear new bucket sizes for this table's output
        clEnqueueFillBuffer(gpu_plotter.queue, new_bs, &zero, 4, 0, num_l1 * 4, 0, nullptr, nullptr);
        clEnqueueFillBuffer(gpu_plotter.queue, num_matches, &zero, 4, 0, 4, 0, nullptr, nullptr);
        
        // Read bucket counts for this table (small: 64 uint32s)
        std::vector<uint32_t> counts(num_l1);
        clEnqueueReadBuffer(gpu_plotter.queue, active_count, CL_TRUE, 0,
            num_l1 * 4, counts.data(), 0, nullptr, nullptr);
        
        uint32_t total_matches_t = 0;
        
        for(int y = 0; y < num_l1; y++) {
            uint32_t count_y = counts[y];
            if(count_y == 0) continue;
            
            // Copy L1 bucket metadata from active_c_in to tmp_c (GPU-to-GPU copy)
            size_t src_off = (size_t)y * max_bs * n_meta * 4;
            clEnqueueCopyBuffer(gpu_plotter.queue, active_c_in, tmp_c,
                src_off, 0, count_y * n_meta * 4, 0, nullptr, nullptr);
            clFinish(gpu_plotter.queue);
            
            // scatter_2: tmp_c → PY + sub_cnt
            clEnqueueFillBuffer(gpu_plotter.queue, sub_cnt, &zero, 4, 0, num_sub * 4, 0, nullptr, nullptr);
            
            uint32_t count_u32 = count_y;
            uint32_t max_bs2_u32 = max_bs2;
            clSetKernelArg(gpu_plotter.k_scatter2, 0, sizeof(cl_mem), &PY);
            clSetKernelArg(gpu_plotter.k_scatter2, 1, sizeof(cl_mem), &sub_cnt);
            clSetKernelArg(gpu_plotter.k_scatter2, 2, sizeof(cl_mem), &null_mem);
            clSetKernelArg(gpu_plotter.k_scatter2, 3, sizeof(cl_mem), &tmp_c);
            clSetKernelArg(gpu_plotter.k_scatter2, 4, sizeof(uint32_t), &count_u32);
            clSetKernelArg(gpu_plotter.k_scatter2, 5, sizeof(uint32_t), &max_bs2_u32);
            size_t gs = ((count_y + 63) / 64) * 64;
            clEnqueueNDRangeKernel(gpu_plotter.queue, gpu_plotter.k_scatter2, 1, nullptr, &gs, nullptr, 0, nullptr, nullptr);
            
            // prefix_sum: sub_cnt → sub_off
            uint32_t num_sub_u32 = num_sub;
            clSetKernelArg(gpu_plotter.prefix_sum_kernel, 0, sizeof(cl_mem), &sub_cnt);
            clSetKernelArg(gpu_plotter.prefix_sum_kernel, 1, sizeof(cl_mem), &sub_off);
            clSetKernelArg(gpu_plotter.prefix_sum_kernel, 2, sizeof(uint32_t) * (num_sub + 1), nullptr);
            clSetKernelArg(gpu_plotter.prefix_sum_kernel, 3, sizeof(uint32_t), &num_sub_u32);
            size_t ps_g = num_sub, ps_l = num_sub;
            clEnqueueNDRangeKernel(gpu_plotter.queue, gpu_plotter.prefix_sum_kernel, 1, nullptr, &ps_g, &ps_l, 0, nullptr, nullptr);
            
            // hybrid_sort_y: sort PY within sub-buckets
            uint32_t max_bs_sort = max_bs2;
            clSetKernelArg(gpu_plotter.k_simple_sort, 0, sizeof(cl_mem), &PY);
            clSetKernelArg(gpu_plotter.k_simple_sort, 1, sizeof(cl_mem), &sub_cnt);
            clSetKernelArg(gpu_plotter.k_simple_sort, 2, sizeof(uint32_t), &max_bs_sort);
            clSetKernelArg(gpu_plotter.k_simple_sort, 3, sizeof(uint32_t), &num_sub_u32);
            size_t sort_g[2] = {256, (size_t)num_sub}, sort_l[2] = {256, 1};
            clEnqueueNDRangeKernel(gpu_plotter.queue, gpu_plotter.k_simple_sort, 2, nullptr, sort_g, sort_l, 0, nullptr, nullptr);
            
            // match_p1: find Y,Y+1 pairs
            uint32_t max_total = count_y * 4;
            uint32_t write_pd = 0;
            clSetKernelArg(gpu_plotter.k_match_p1, 0, sizeof(cl_mem), &LR);
            clSetKernelArg(gpu_plotter.k_match_p1, 1, sizeof(cl_mem), &PD_match);
            clSetKernelArg(gpu_plotter.k_match_p1, 2, sizeof(cl_mem), &num_matches);
            clSetKernelArg(gpu_plotter.k_match_p1, 3, sizeof(cl_mem), &PY);
            clSetKernelArg(gpu_plotter.k_match_p1, 4, sizeof(cl_mem), &sub_cnt);
            clSetKernelArg(gpu_plotter.k_match_p1, 5, sizeof(cl_mem), &sub_off);
            clSetKernelArg(gpu_plotter.k_match_p1, 6, sizeof(uint32_t), &num_sub_u32);
            clSetKernelArg(gpu_plotter.k_match_p1, 7, sizeof(uint32_t), &max_bs_sort);
            clSetKernelArg(gpu_plotter.k_match_p1, 8, sizeof(uint32_t), &max_total);
            clSetKernelArg(gpu_plotter.k_match_p1, 9, sizeof(uint32_t), &write_pd);
            int groups_per_sub = (max_bs2 + 127) / 128;
            size_t match_g[2] = {(size_t)(128 * groups_per_sub), (size_t)num_sub}, match_l[2] = {128, 1};
            clEnqueueNDRangeKernel(gpu_plotter.queue, gpu_plotter.k_match_p1, 2, nullptr, match_g, match_l, 0, nullptr, nullptr);
            
            // eval_p1_tx: hash LR pairs → C_out (scattered to new L1 buckets)
            // Read match count
            uint32_t match_count = 0;
            clEnqueueReadBuffer(gpu_plotter.queue, num_matches, CL_TRUE, 0, 4, &match_count, 0, nullptr, nullptr);
            total_matches_t += match_count;
            
            if(match_count > 0) {
                uint32_t max_bs_u32 = max_bs;
                uint32_t table_u32 = t;
                uint32_t write_y = 0;  // don't write Y_out (scatter_2 computes Y from C)
                uint32_t write_c = 1;
                uint32_t has_pd = (t >= 3) ? 1 : 0;
                uint32_t has_x = (t == 2) ? 1 : 0;
                ulong pd_0 = 0;
                uint32_t x2size = 2 * XBITS - 1;
                uint32_t xbits = XBITS;
                
                // For t=2, we need X_in (F1 indices). Upload per L1 bucket.
                cl_mem x_buf = null_mem;
                if(t == 2) {
                    // Upload X values for this L1 bucket
                    // X values are the F1 indices for entries in this L1 bucket
                    // We need to track which F1 indices are in this L1 bucket
                    // TODO: upload X values alongside metadata
                }
                
                clSetKernelArg(gpu_plotter.eval_p1_tx_kernel, 0, sizeof(cl_mem), &null_mem);  // Y_out = null
                clSetKernelArg(gpu_plotter.eval_p1_tx_kernel, 1, sizeof(cl_mem), &active_c_out);  // C_out
                clSetKernelArg(gpu_plotter.eval_p1_tx_kernel, 2, sizeof(cl_mem), &null_mem);  // PD_out = null (TODO)
                clSetKernelArg(gpu_plotter.eval_p1_tx_kernel, 3, sizeof(cl_mem), &new_bs);  // bucket_size
                clSetKernelArg(gpu_plotter.eval_p1_tx_kernel, 4, sizeof(cl_mem), &tmp_c);  // C_in
                clSetKernelArg(gpu_plotter.eval_p1_tx_kernel, 5, sizeof(cl_mem), &null_mem);  // PD_in
                clSetKernelArg(gpu_plotter.eval_p1_tx_kernel, 6, sizeof(cl_mem), &x_buf);  // X_in
                clSetKernelArg(gpu_plotter.eval_p1_tx_kernel, 7, sizeof(cl_mem), &LR);  // LR pairs
                clSetKernelArg(gpu_plotter.eval_p1_tx_kernel, 8, sizeof(cl_mem), &num_matches);  // num_found
                clSetKernelArg(gpu_plotter.eval_p1_tx_kernel, 9, sizeof(ulong), &pd_0);
                clSetKernelArg(gpu_plotter.eval_p1_tx_kernel, 10, sizeof(uint32_t), &max_bs_u32);
                clSetKernelArg(gpu_plotter.eval_p1_tx_kernel, 11, sizeof(uint32_t), &x2size);
                clSetKernelArg(gpu_plotter.eval_p1_tx_kernel, 12, sizeof(uint32_t), &xbits);
                clSetKernelArg(gpu_plotter.eval_p1_tx_kernel, 13, sizeof(uint32_t), &table_u32);
                clSetKernelArg(gpu_plotter.eval_p1_tx_kernel, 14, sizeof(uint32_t), &write_y);
                clSetKernelArg(gpu_plotter.eval_p1_tx_kernel, 15, sizeof(uint32_t), &write_c);
                clSetKernelArg(gpu_plotter.eval_p1_tx_kernel, 16, sizeof(uint32_t), &has_pd);
                clSetKernelArg(gpu_plotter.eval_p1_tx_kernel, 17, sizeof(uint32_t), &has_x);
                
                size_t hash_g = ((match_count + 63) / 64) * 64;
                clEnqueueNDRangeKernel(gpu_plotter.queue, gpu_plotter.eval_p1_tx_kernel, 1, nullptr, &hash_g, nullptr, 0, nullptr, nullptr);
            }
            
            clFinish(gpu_plotter.queue);
        }
        
        // Save LR pairs for this table (copy to per-table buffer)
        if(total_matches_t > 0) {
            lr_per_table[t] = clCreateBuffer(gpu_plotter.context, CL_MEM_READ_WRITE, total_matches_t * 2 * 4, nullptr, &err);
            clEnqueueCopyBuffer(gpu_plotter.queue, LR, lr_per_table[t], 0, 0, total_matches_t * 2 * 4, 0, nullptr, nullptr);
            lr_count_per_table[t] = total_matches_t;
        }
        
        // Swap: C_in ↔ C_out, bucket_count ↔ new_bs
        std::swap(active_c_in, active_c_out);
        std::swap(active_count, new_bs);
        
        std::cout << "[T" << t << "] " << total_matches_t << " matches ("
                  << (my_time_ms() - tt) / 1000.0 << "s)" << std::endl;
    }
    
    // ========================================================================
    // Step 3: Download and build PlotData
    // ========================================================================
    auto t_dl = my_time_ms();
    
    // Download final C_in (metadata, scattered per L1 bucket)
    std::vector<uint32_t> final_counts(num_l1);
    clEnqueueReadBuffer(gpu_plotter.queue, active_count, CL_TRUE, 0, num_l1 * 4, final_counts.data(), 0, nullptr, nullptr);
    
    uint32_t total_final = 0;
    for(int y = 0; y < num_l1; y++) total_final += final_counts[y];
    std::cout << "[GPU-Res] Total final entries: " << total_final << std::endl;
    
    // Download metadata per L1 bucket and build final_Y + final_meta
    plot.final_Y.resize(total_final);
    plot.final_meta.resize(total_final);
    uint32_t pos = 0;
    for(int y = 0; y < num_l1; y++) {
        uint32_t cnt = final_counts[y];
        if(cnt == 0) continue;
        std::vector<uint32_t> meta(cnt * n_meta);
        size_t off = (size_t)y * max_bs * n_meta * 4;
        clEnqueueReadBuffer(gpu_plotter.queue, active_c_in, CL_TRUE, off,
            cnt * n_meta * 4, meta.data(), 0, nullptr, nullptr);
        // Compute Y from metadata XOR
        for(uint32_t i = 0; i < cnt; i++) {
            uint32_t Y = 0;
            for(int j = 0; j < n_meta; j++) Y ^= meta[i * n_meta + j];
            Y &= KMASK;
            plot.final_Y[pos + i] = Y;
            std::memcpy(plot.final_meta[pos + i].data(), &meta[i * n_meta], n_meta * sizeof(uint32_t));
        }
        pos += cnt;
    }
    
    // Download LR pairs per table and build PD + X_pairs
    // TODO: implement PD and X_pairs build from LR pairs
    
    std::cout << "[GPU-Res] Download in " << (my_time_ms() - t_dl) / 1000.0 << "s" << std::endl;
    std::cout << "[GPU-Res] Total: " << (my_time_ms() - t0) / 1000.0 << "s" << std::endl;
    
    // Cleanup
    clReleaseMemObject(C_in); clReleaseMemObject(C_out); clReleaseMemObject(tmp_c);
    clReleaseMemObject(PY); clReleaseMemObject(sub_cnt); clReleaseMemObject(sub_off);
    clReleaseMemObject(LR); clReleaseMemObject(PD_match); clReleaseMemObject(num_matches);
    clReleaseMemObject(new_bs); clReleaseMemObject(bucket_count);
    for(auto& lr : lr_per_table) if(lr) clReleaseMemObject(lr);
}
