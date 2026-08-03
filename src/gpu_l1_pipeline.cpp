// GPU L1-bucket pipeline for F2-F9
// Keeps ALL metadata on GPU throughout the table loop.
// Uses f2_f9.cl kernels: scatter_2, calc_offset_sum, simple_sort_y, match_p1, eval_p1_tx
// Only downloads LR pairs per table for PD + X pair construction.

#include "plot_pipeline.h"
#include "plot_config.h"
#include <fstream>
#include <cstring>
#include <omp.h>

namespace mmx {

static bool load_f2f9_kernels(GPUDevice& gpu, int ksize,
    cl_kernel& k_scatter2, cl_kernel& k_offset_sum, cl_kernel& k_simple_sort,
    cl_kernel& k_match_p1, cl_kernel& k_eval)
{
    std::string kdir;
    for(const auto& d : {"kernels", "../kernels", "../../kernels"}) {
        if(std::ifstream(std::string(d) + "/f2_f9.cl").good()) { kdir = d; break; }
    }
    if(kdir.empty()) return false;

    int lb2 = std::max(0, ksize - 8 - 9);
    std::string opts = "-cl-std=CL1.2"
        " -DKSIZE=" + std::to_string(ksize) +
        " -DLOGBUCKETS=8" +
        " -DLOGBUCKETS2=" + std::to_string(lb2) +
        " -DN_META=" + std::to_string(N_META) +
        " -DN_META_OUT=12" +
        " -DN_TABLE=9" +
        " -DDSIZE_=5 -DPSIZE_=" + std::to_string(ksize + 1) +
        " -DPDSIZE=" + std::to_string(5 + ksize) +
        " -DX2SIZE=" + std::to_string(2 * ksize - 1) +
        " -DXBITS=" + std::to_string(ksize) +
        " -DHYBRID_SORT_LOG_THREADS=6 -DNUM_THREADS=64" +
        " -DKMASK=" + std::to_string((1u << ksize) - 1) +
        " -DDMASK=31 -DMETA_BYTES=" + std::to_string(N_META * 4) +
        " -DMAX_LOCAL_SIZE=40";
    std::string src;
    for(const auto& fn : {"/f2_f9.cl", "/simple_sort.cl", "/prefix_sum.cl"}) {
        std::ifstream f(kdir + fn);
        if(!f.good()) return false;
        src += std::string(std::istreambuf_iterator<char>(f), {}) + "\n";
    }
    try {
        gpu.load_program("f2f9", src, opts);
        k_scatter2   = gpu.get_kernel("scatter_2");
        k_offset_sum = gpu.get_kernel("calc_offset_sum");
        k_simple_sort= gpu.get_kernel("simple_sort_y");
        k_match_p1   = gpu.get_kernel("match_p1");
        k_eval       = gpu.get_kernel("eval_p1_tx");
        return (k_scatter2 && k_offset_sum && k_simple_sort && k_match_p1 && k_eval);
    } catch(...) { return false; }
}

bool run_gpu_l1_pipeline(
    GPUDevice& gpu,
    int ksize,
    uint32_t total_entries,
    const std::vector<uint32_t>& M_flat,
    const std::vector<uint32_t>& X_flat,
    std::vector<PlotEntry>& entries,
    std::vector<uint32_t>& final_Y,
    std::vector<std::vector<PDEntry>>& pd_all,
    std::vector<uint32_t>& x_pairs)
{
    cl_kernel k_scatter2=nullptr, k_offset_sum=nullptr, k_simple_sort=nullptr;
    cl_kernel k_match_p1=nullptr, k_eval=nullptr;
    if(!load_f2f9_kernels(gpu, ksize, k_scatter2, k_offset_sum, k_simple_sort,
                          k_match_p1, k_eval)) {
        return false;
    }
    const uint32_t KMASK = (1u << ksize) - 1;
    const int n_l1 = 256, n_sub = std::max(1, 1 << (ksize - 17));
    const uint32_t avg_l1 = (total_entries + n_l1 - 1) / n_l1;
    const uint32_t max_bs = avg_l1 * 3 + 256;
    const uint32_t avg_sub = total_entries / (n_l1 * n_sub) + 1;
    const uint32_t max_bs2 = std::max(1024u, avg_sub * 3 + 256);
    size_t c_sz = (size_t)n_l1 * max_bs * N_META * 4;
    size_t py_sz = (size_t)n_sub * max_bs2 * 8;
    size_t vram_needed = 2*c_sz + py_sz + (size_t)total_entries * 12;
    size_t vram_avail = 0;
    clGetDeviceInfo(gpu.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(vram_avail), &vram_avail, nullptr);

    std::cout << "[GPU-L1] L1=" << n_l1 << " sub=" << n_sub
              << " max_bs=" << max_bs << " max_bs2=" << max_bs2
              << " VRAM~" << (vram_needed/1e9) << "GB / " << (vram_avail/1e9) << "GB" << std::endl;
    if(vram_needed > vram_avail * 9 / 10) return false;

    cl_int err; int zero = 0;
    std::vector<cl_mem> bufs;
    auto ab = [&](size_t sz) { cl_mem m = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, sz, nullptr, &err);
        GPUDevice::check(err); bufs.push_back(m); return m; };

    cl_mem C_in = ab(c_sz), C_out = ab(c_sz), tmp_c = ab(max_bs * N_META * 4);
    cl_mem PY = ab(py_sz), scnt = ab((size_t)n_sub * 4), soff = ab((size_t)(n_sub+1) * 4);
    cl_mem LRbuf = ab((size_t)total_entries * 8), PDbuf = ab((size_t)total_entries * 4);
    cl_mem nm = ab(4), nbs = ab((size_t)n_l1 * 4), bcnt = ab((size_t)n_l1 * 4);
    cl_mem Xbuf = ab((size_t)max_bs * 4), Ybuf = ab((size_t)total_entries * 4);
    auto cleanup = [&](){ for(auto m : bufs) clReleaseMemObject(m); };

    auto t0 = std::chrono::high_resolution_clock::now();

    // Build Y from M, scatter into L1 buckets, upload
    const int shift = ksize - 8;
    std::vector<uint32_t> f1c(n_l1, 0), Y_all(total_entries);
    std::vector<std::vector<uint32_t>> l1m(n_l1), l1x(n_l1);
    for(size_t i = 0; i < total_entries; i++) {
        uint32_t Y = 0;
        for(int j = 0; j < N_META; j++) Y ^= M_flat[i * N_META + j];
        Y &= KMASK; Y_all[i] = Y;
        uint32_t b = std::min(Y >> shift, (uint32_t)(n_l1 - 1));
        f1c[b]++;
        for(int j = 0; j < N_META; j++) l1m[b].push_back(M_flat[i * N_META + j]);
        l1x[b].push_back(X_flat.empty() ? (uint32_t)i : X_flat[i]);
    }
    for(int y = 0; y < n_l1; y++) {
        if(f1c[y] == 0) continue;
        clEnqueueWriteBuffer(gpu.queue, C_in, CL_FALSE, (size_t)y * max_bs * N_META * 4,
            f1c[y] * N_META * 4, l1m[y].data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(gpu.queue, Xbuf, CL_FALSE, (size_t)y * max_bs * 4,
            f1c[y] * 4, l1x[y].data(), 0, nullptr, nullptr);
    }
    clEnqueueWriteBuffer(gpu.queue, Ybuf, CL_FALSE, 0, total_entries * 4, Y_all.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(gpu.queue, bcnt, CL_TRUE, 0, n_l1 * 4, f1c.data(), 0, nullptr, nullptr);
    gpu.finish();
    std::cout << "[GPU-L1] Upload: "
              << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() << "s" << std::endl;

    // Process tables 2-9
    cl_mem ac_in = C_in, ac_out = C_out, acnt = bcnt;
    std::vector<std::vector<uint32_t>> lr_t(10), pd_t(10);

    for(int t = 2; t <= 9; t++) {
        auto tt0 = std::chrono::high_resolution_clock::now();
        clEnqueueFillBuffer(gpu.queue, nbs, &zero, 4, 0, n_l1 * 4, 0, nullptr, nullptr);
        clEnqueueFillBuffer(gpu.queue, nm, &zero, 4, 0, 4, 0, nullptr, nullptr);
        gpu.finish();

        std::vector<uint32_t> cnts(n_l1);
        clEnqueueReadBuffer(gpu.queue, acnt, CL_TRUE, 0, n_l1 * 4, cnts.data(), 0, nullptr, nullptr);
        uint32_t tmatches = 0;

        for(int y = 0; y < n_l1; y++) {
            if(cnts[y] == 0) continue;
            uint32_t cy = cnts[y];

            size_t off = (size_t)y * max_bs * N_META * 4;
            clEnqueueCopyBuffer(gpu.queue, ac_in, tmp_c, off, 0, cy * N_META * 4, 0, nullptr, nullptr);
            gpu.finish();

            clEnqueueFillBuffer(gpu.queue, scnt, &zero, 4, 0, n_sub * 4, 0, nullptr, nullptr);
            uint32_t cyu = cy, mbs2 = max_bs2;
            cl_mem yb = nullptr;  // compute Y from C_in metadata on GPU
            clSetKernelArg(k_scatter2, 0, 8, &PY);
            clSetKernelArg(k_scatter2, 1, 8, &scnt);
            clSetKernelArg(k_scatter2, 2, 8, &yb);
            clSetKernelArg(k_scatter2, 3, 8, &tmp_c);
            clSetKernelArg(k_scatter2, 4, 4, &cyu);
            clSetKernelArg(k_scatter2, 5, 4, &mbs2);
            size_t gs = ((cy + 63) / 64) * 64;
            clEnqueueNDRangeKernel(gpu.queue, k_scatter2, 1, nullptr, &gs, nullptr, 0, nullptr, nullptr);

            uint32_t nsu = (uint32_t)n_sub, wt = 1;
            clSetKernelArg(k_offset_sum, 0, 8, &soff);
            clSetKernelArg(k_offset_sum, 1, 8, &scnt);
            clSetKernelArg(k_offset_sum, 2, 4, &nsu);
            clSetKernelArg(k_offset_sum, 3, 4, &wt);
            size_t pl = std::min((size_t)n_sub, (size_t)1024);
            clEnqueueNDRangeKernel(gpu.queue, k_offset_sum, 1, nullptr, &pl, &pl, 0, nullptr, nullptr);

            clSetKernelArg(k_simple_sort, 0, 8, &PY);
            clSetKernelArg(k_simple_sort, 1, 8, &scnt);
            clSetKernelArg(k_simple_sort, 2, 4, &mbs2);
            clSetKernelArg(k_simple_sort, 3, 4, &nsu);
            size_t sg[2] = {256, (size_t)n_sub}, sl[2] = {256, 1};
            clEnqueueNDRangeKernel(gpu.queue, k_simple_sort, 2, nullptr, sg, sl, 0, nullptr, nullptr);

            uint32_t mt = (uint32_t)((size_t)cy * 4), wp = (t >= 3) ? 1 : 0;
            clSetKernelArg(k_match_p1, 0, 8, &LRbuf);
            clSetKernelArg(k_match_p1, 1, 8, &PDbuf);
            clSetKernelArg(k_match_p1, 2, 8, &nm);
            clSetKernelArg(k_match_p1, 3, 8, &PY);
            clSetKernelArg(k_match_p1, 4, 8, &scnt);
            clSetKernelArg(k_match_p1, 5, 8, &soff);
            clSetKernelArg(k_match_p1, 6, 4, &nsu);
            clSetKernelArg(k_match_p1, 7, 4, &mbs2);
            clSetKernelArg(k_match_p1, 8, 4, &mt);
            clSetKernelArg(k_match_p1, 9, 4, &wp);
            int gps = (max_bs2 + 127) / 128;
            size_t mg[2] = {(size_t)(128 * gps), (size_t)n_sub}, ml[2] = {128, 1};
            clEnqueueNDRangeKernel(gpu.queue, k_match_p1, 2, nullptr, mg, ml, 0, nullptr, nullptr);
            gpu.finish();

            // Read match count for this bucket
            uint32_t nm_before = tmatches;  // from previous buckets
            uint32_t nm_after = 0;
            clEnqueueReadBuffer(gpu.queue, nm, CL_TRUE, 0, 4, &nm_after, 0, nullptr, nullptr);
            uint32_t mc = nm_after - nm_before;
            if(mc > 0) {
                uint32_t mx = max_bs, tu = (uint32_t)t;
                uint32_t wy = 0, wc = 1, hp = (t >= 3), hx = (t == 2);
                uint64_t pdo = 0; uint32_t x2s = 2 * ksize - 1;
                cl_mem nm2 = nullptr;
                // Overwrite nm with mc (this bucket's count only) for eval
                clEnqueueWriteBuffer(gpu.queue, nm, CL_FALSE, 0, 4, &mc, 0, nullptr, nullptr);
                clSetKernelArg(k_eval, 0, 8, &nm2);
                clSetKernelArg(k_eval, 1, 8, &ac_out);
                clSetKernelArg(k_eval, 2, 8, &nm2);
                clSetKernelArg(k_eval, 3, 8, &nbs);
                clSetKernelArg(k_eval, 4, 8, &tmp_c);
                clSetKernelArg(k_eval, 5, 8, &nm2);
                clSetKernelArg(k_eval, 6, 8, hx ? &Xbuf : &nm2);
                clSetKernelArg(k_eval, 7, 8, &LRbuf);
                clSetKernelArg(k_eval, 8, 8, &nm);
                clSetKernelArg(k_eval, 9, 8, &pdo);
                clSetKernelArg(k_eval, 10, 4, &mx);
                clSetKernelArg(k_eval, 11, 4, &x2s);
                clSetKernelArg(k_eval, 12, 4, &ksize);
                clSetKernelArg(k_eval, 13, 4, &tu);
                clSetKernelArg(k_eval, 14, 4, &wy);
                clSetKernelArg(k_eval, 15, 4, &wc);
                clSetKernelArg(k_eval, 16, 4, &hp);
                clSetKernelArg(k_eval, 17, 4, &hx);
                size_t hg = ((mc + 63) / 64) * 64;
                clEnqueueNDRangeKernel(gpu.queue, k_eval, 1, nullptr, &hg, nullptr, 0, nullptr, nullptr);
                gpu.finish();
                tmatches = nm_after;  // accumulate
                // Restore cumulative count for next bucket
                clEnqueueWriteBuffer(gpu.queue, nm, CL_FALSE, 0, 4, &nm_after, 0, nullptr, nullptr);
            } else {
                // No matches this bucket, restore nm for safety
                clEnqueueWriteBuffer(gpu.queue, nm, CL_FALSE, 0, 4, &nm_after, 0, nullptr, nullptr);
            }
        }

        lr_t[t].resize(tmatches * 2);
        clEnqueueReadBuffer(gpu.queue, LRbuf, CL_TRUE, 0, tmatches * 8, lr_t[t].data(), 0, nullptr, nullptr);
        if(t >= 3) {
            pd_t[t].resize(tmatches);
            clEnqueueReadBuffer(gpu.queue, PDbuf, CL_TRUE, 0, tmatches * 4, pd_t[t].data(), 0, nullptr, nullptr);
        }
        std::swap(ac_in, ac_out);
        std::swap(acnt, nbs);
        double te = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tt0).count();
        std::cout << "[T" << t << "] " << tmatches << " matches (" << te << "s)" << std::endl;
    }

    // Download final results
    std::vector<uint32_t> fc(n_l1);
    clEnqueueReadBuffer(gpu.queue, acnt, CL_TRUE, 0, n_l1 * 4, fc.data(), 0, nullptr, nullptr);
    uint32_t tf = 0;
    for(auto c : fc) tf += c;

    entries.resize(tf); final_Y.resize(tf);
    size_t pos = 0;
    for(int y = 0; y < n_l1; y++) {
        if(fc[y] == 0) continue;
        uint32_t cnt = fc[y];
        std::vector<uint32_t> meta(cnt * N_META);
        size_t off = (size_t) y * max_bs * N_META * 4;
        clEnqueueReadBuffer(gpu.queue, ac_in, CL_TRUE, off, cnt * N_META * 4, meta.data(), 0, nullptr, nullptr);
        for(uint32_t i = 0; i < cnt; i++, pos++) {
            uint32_t Y = 0;
            for(int j = 0; j < N_META; j++) {
                entries[pos].M[j] = meta[i * N_META + j];
                Y ^= meta[i * N_META + j];
            }
            entries[pos].Y = Y & KMASK;
            entries[pos].orig_idx = (uint32_t)pos;
            final_Y[pos] = entries[pos].Y;
        }
    }

    pd_all.assign(10, {});
    for(int t = 3; t <= 9; t++) {
        pd_all[t].resize(pd_t[t].size());
        for(size_t i = 0; i < pd_t[t].size(); i++) {
            pd_all[t][i] = {pd_t[t][i] >> 5, (uint16_t)(pd_t[t][i] & 0x1F)};
        }
    }
    x_pairs.clear();
    for(size_t i = 0; i < lr_t[2].size(); i += 2) {
        x_pairs.push_back(lr_t[2][i]);
        x_pairs.push_back(lr_t[2][i + 1]);
    }

    cleanup();
    double total = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << "[GPU-L1] Total: " << total << "s" << std::endl;
    return true;
}

} // namespace mmx
