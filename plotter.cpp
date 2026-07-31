/*
 * OpenCL Plotter for MMX — AMD GPU (gfx1100) compatible
 * 
 * Computes F1 on GPU (OpenCL), F2-F9 on CPU, writes plot file.
 * This avoids the gfx1100 hardware bug that breaks HIP/CUDA/ZLUDA plotting.
 *
 * Usage: mmx_opencl_plotter <plot_id_hex> <farmer_key_hex> [output_dir] [--k KSIZE] [--test] [--limit N]
 */

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <vnx/vnx.h>
#include <mmx/PlotHeader.hxx>
#include <mmx/hash_t.hpp>
#include <mmx/hash_512_t.hpp>
#include <mmx/pos/encoding.h>
#include <mmx/pos/config.h>
#include <mmx/pos/util.h>
#include <mmx/utils.h>

#include <cstdio>
#include <cstdint>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <mutex>
#include <set>
#include <unordered_map>
#include <ctime>
#include <cctype>
#include <parallel/algorithm>
#include <omp.h>

using namespace mmx;
using namespace mmx::pos;

// Runtime-configurable constants (set from command line)
static int KSIZE = 26;
static int XBITS = 26;
static int LOGBUCKETS = 6;

// Derived constants (computed at runtime)
static int MY_N_META = 14;
static int MY_N_META_OUT = 12;
static int MY_N_TABLE = 9;
static int PSIZE_;
static int DSIZE_ = 5;
static int PDBYTES;
static int PDSIZE;
static uint32_t DMASK;
static uint32_t KMASK;
static int LPX2SIZE;

static int PARK_SIZE_X = 2048;
static int PARK_SIZE_Y = 8192;
static int PARK_SIZE_PD = 2048;
static int PARK_SIZE_META = 256;

static double MAX_AVG_OFFSET_BITS = 2.65;
static double MAX_AVG_YDELTA_BITS = 2.25;

// ============================================================================
// Utility
// ============================================================================

static int64_t my_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static uint64_t cdiv(uint64_t a, uint64_t b) {
    return (a + b - 1) / b;
}

static void update_constants() {
    KMASK = ((uint32_t(1) << KSIZE) - 1);
    DMASK = ((1u << DSIZE_) - 1);
    PSIZE_ = KSIZE + 1;
    PDBYTES = (PSIZE_ + DSIZE_ + 7) / 8;
    PDSIZE = PDBYTES * 8;
    LPX2SIZE = 2 * XBITS - 1;
}

// encode_symbol: same as mmx::pos::encode_symbol but we declare it here
// since it's not exported in the header
namespace mmx { namespace pos {
extern std::pair<uint32_t, uint32_t> encode_symbol(const uint8_t sym);
} }

// Bit writer for park encoding
// Fast bit writer: writes value into buf at bit_offset, num_bits wide.
// buf must be pre-allocated to sufficient size. Little-endian bit order (bit 0 = LSB of first byte).
static void write_bits(std::vector<uint8_t>& buf, uint64_t value, uint64_t bit_offset, uint64_t num_bits) {
    if(num_bits == 0) return;
    uint64_t byte_pos = bit_offset / 8;
    uint32_t bit_pos = bit_offset % 8;
    
    // Handle first byte if not aligned
    if(bit_pos != 0) {
        uint32_t bits_in_first = std::min((uint64_t)(8 - bit_pos), num_bits);
        uint8_t mask = (uint8_t(1) << bits_in_first) - 1;
        buf[byte_pos] |= (uint8_t)(value & mask) << bit_pos;
        value >>= bits_in_first;
        num_bits -= bits_in_first;
        byte_pos++;
    }
    
    // Write full bytes
    while(num_bits >= 8) {
        buf[byte_pos] = (uint8_t)(value & 0xFF);
        value >>= 8;
        num_bits -= 8;
        byte_pos++;
    }
    
    // Handle last byte if remaining bits
    if(num_bits > 0) {
        buf[byte_pos] |= (uint8_t)(value & ((uint8_t(1) << num_bits) - 1));
    }
}

// Line point functions
static uint64_t get_x_enc(uint32_t x) {
    uint32_t a = x, b = x - 1;
    if(a % 2 == 0) a /= 2;
    else b /= 2;
    return uint64_t(a) * b;
}

static uint64_t calc_line_point(uint32_t x, uint32_t y) {
    return get_x_enc(std::max(x, y)) + std::min(x, y);
}

static uint64_t calc_line_point2(uint32_t x, uint32_t y) {
    return calc_line_point(x + 1, y + 1);
}

// ============================================================================
// OpenCL F1 Computation
// ============================================================================

class OCL_Plotter {
public:
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    cl_program program;
    cl_kernel f1_kernel;
    
    void init(cl_context ctx, cl_device_id dev) {
        context = ctx;
        device = dev;
        cl_int err;
        queue = clCreateCommandQueue(context, device, 0, &err);
        if(err != CL_SUCCESS) throw std::runtime_error("Failed to create command queue");
        
        std::string kernel_path = "pos_recompute.cl";
        FILE* f = fopen(kernel_path.c_str(), "r");
        if(!f) throw std::runtime_error("Cannot open " + kernel_path);
        fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
        char* src = new char[sz+1]; size_t rd = fread(src, 1, sz, f); src[sz] = 0; fclose(f);
        
        program = clCreateProgramWithSource(context, 1, (const char**)&src, &sz, &err);
        delete[] src;
        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if(err != CL_SUCCESS) {
            char log[4096]; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
            throw std::runtime_error(std::string("OpenCL build failed: ") + log);
        }
        
        f1_kernel = clCreateKernel(program, "compute_f1_kernel", &err);
        if(err != CL_SUCCESS) throw std::runtime_error("compute_f1_kernel not found");
        
        std::cout << "[OCL] F1 kernel loaded" << std::endl;
    }
    
    // Table hash is initialized separately via init_table_hash()
    
    void compute_f1_batch(
        const std::vector<uint32_t>& X_values,
        const hash_t& plot_id,
        std::vector<uint32_t>& Y_out,
        std::vector<uint32_t>& M_out)
    {
        const size_t num_x = X_values.size();
        if(num_x == 0) return;
        
        std::vector<uint32_t> id_u32(8);
        std::memcpy(id_u32.data(), plot_id.data(), 32);
        
        cl_int err;
        cl_mem X_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      num_x * sizeof(uint32_t), (void*)X_values.data(), &err);
        cl_mem ID_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       8 * sizeof(uint32_t), (void*)id_u32.data(), &err);
        cl_mem Y_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                      num_x * sizeof(uint32_t), nullptr, &err);
        cl_mem M_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                      num_x * MY_N_META * sizeof(uint32_t), nullptr, &err);
        
        uint32_t kmask = KMASK;
        uint32_t xbits = 0;
        uint32_t num_x_u32 = (uint32_t)num_x;
        
        clSetKernelArg(f1_kernel, 0, sizeof(cl_mem), &X_buf);
        clSetKernelArg(f1_kernel, 1, sizeof(cl_mem), &ID_buf);
        clSetKernelArg(f1_kernel, 2, sizeof(cl_mem), &Y_buf);
        clSetKernelArg(f1_kernel, 3, sizeof(cl_mem), &M_buf);
        clSetKernelArg(f1_kernel, 4, sizeof(uint32_t), &kmask);
        clSetKernelArg(f1_kernel, 5, sizeof(uint32_t), &xbits);
        clSetKernelArg(f1_kernel, 6, sizeof(uint32_t), &num_x_u32);
        
        size_t global = num_x;
        size_t local = 64;
        if(global % local != 0) global = ((global / local) + 1) * local;
        
        err = clEnqueueNDRangeKernel(queue, f1_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        if(err != CL_SUCCESS) throw std::runtime_error("F1 kernel launch failed: " + std::to_string(err));
        
        Y_out.resize(num_x);
        M_out.resize(num_x * MY_N_META);
        clEnqueueReadBuffer(queue, Y_buf, CL_TRUE, 0, num_x * sizeof(uint32_t), Y_out.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, M_buf, CL_TRUE, 0, num_x * MY_N_META * sizeof(uint32_t), M_out.data(), 0, nullptr, nullptr);
        
        clReleaseMemObject(X_buf);
        clReleaseMemObject(ID_buf);
        clReleaseMemObject(Y_buf);
        clReleaseMemObject(M_buf);
    }
    
    void compute_all_f1(
        const hash_t& plot_id,
        std::vector<uint32_t>& Y_all,
        std::vector<uint32_t>& M_all,
        const uint32_t batch_size = 1 << 20,
        bool test_mode = false, uint64_t test_limit = 0)
    {
        uint64_t total = uint64_t(1) << KSIZE;
        if(test_mode && test_limit > 0) {
            total = std::min(total, test_limit);
            std::cout << "[F1] TEST MODE: limiting to " << total << " entries" << std::endl;
        }
        const uint64_t num_batches = (total + batch_size - 1) / batch_size;
        
        Y_all.resize(total);
        M_all.resize(total * MY_N_META);
        
        std::cout << "[F1] Computing " << total << " entries in " << num_batches << " batches..." << std::endl;
        auto t0 = my_time_ms();
        
        std::vector<uint32_t> X_batch(batch_size);
        std::vector<uint32_t> Y_batch, M_batch;
        
        for(uint64_t b = 0; b < num_batches; b++) {
            uint64_t start = b * batch_size;
            uint32_t count = (uint32_t)std::min((uint64_t)batch_size, total - start);
            
            for(uint32_t i = 0; i < count; i++) {
                X_batch[i] = (uint32_t)(start + i);
            }
            X_batch.resize(count);
            
            compute_f1_batch(X_batch, plot_id, Y_batch, M_batch);
            
            std::memcpy(Y_all.data() + start, Y_batch.data(), count * sizeof(uint32_t));
            std::memcpy(M_all.data() + start * MY_N_META, M_batch.data(), count * MY_N_META * sizeof(uint32_t));
            
            if((b + 1) % 16 == 0 || b == num_batches - 1) {
                auto elapsed = my_time_ms() - t0;
                auto pct = (b + 1) * 100.0 / num_batches;
                std::cout << "[F1] Batch " << (b+1) << "/" << num_batches << " (" << pct << "%) "
                          << elapsed / 1000.0 << "s" << std::endl;
            }
        }
        
        auto elapsed = my_time_ms() - t0;
        std::cout << "[F1] Done in " << elapsed / 1000.0 << " sec" << std::endl;
    }
    
    cl_kernel table_hash_kernel = nullptr;
    cl_kernel hash_lr_kernel = nullptr;
    cl_kernel eval_p1_tx_kernel = nullptr;
    cl_program f2f9_program = nullptr;
    
    void init_table_hash() {
        std::string kernel_path = "table_hash.cl";
        FILE* f = fopen(kernel_path.c_str(), "r");
        if(!f) throw std::runtime_error("Cannot open " + kernel_path);
        fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
        char* src = new char[sz+1]; size_t rd = fread(src, 1, sz, f); src[sz] = 0; fclose(f);
        
        cl_int err;
        cl_program prog = clCreateProgramWithSource(context, 1, (const char**)&src, &sz, &err);
        delete[] src;
        err = clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr);
        if(err != CL_SUCCESS) {
            char log[4096]; clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
            throw std::runtime_error(std::string("table_hash build failed: ") + log);
        }
        
        table_hash_kernel = clCreateKernel(prog, "hash_table_entries", &err);
        hash_lr_kernel = clCreateKernel(prog, "hash_table_lr", &err);
        if(err != CL_SUCCESS) throw std::runtime_error("hash_table_lr not found");
        if(err != CL_SUCCESS) throw std::runtime_error("hash_table_entries not found");
        std::cout << "[OCL] Table hash kernel loaded" << std::endl;
    }
    
    // Hash LR pairs on GPU. Returns Y and M arrays.
    void gpu_hash_table(
        const std::vector<uint32_t>& L_meta,  // [num * 14]
        const std::vector<uint32_t>& R_meta,  // [num * 14]
        std::vector<uint32_t>& Y_out,
        std::vector<uint32_t>& M_out,
        uint32_t kmask)
    {
        const size_t num = L_meta.size() / 14;
        if(num == 0) { Y_out.clear(); M_out.clear(); return; }
        
        cl_int err;
        cl_mem Lb = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, L_meta.size()*4, (void*)L_meta.data(), &err);
        cl_mem Rb = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, R_meta.size()*4, (void*)R_meta.data(), &err);
        cl_mem Yb = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num*4, nullptr, &err);
        cl_mem Mb = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num*14*4, nullptr, &err);
        
        uint32_t num_u32 = (uint32_t)num;
        clSetKernelArg(table_hash_kernel,0,sizeof(cl_mem),&Lb);
        clSetKernelArg(table_hash_kernel,1,sizeof(cl_mem),&Rb);
        clSetKernelArg(table_hash_kernel,2,sizeof(cl_mem),&Yb);
        clSetKernelArg(table_hash_kernel,3,sizeof(cl_mem),&Mb);
        clSetKernelArg(table_hash_kernel,4,sizeof(uint32_t),&kmask);
        clSetKernelArg(table_hash_kernel,5,sizeof(uint32_t),&num_u32);
        
        size_t global = num, local = 64;
        if(global % local) global = ((global/local)+1)*local;
        clEnqueueNDRangeKernel(queue, table_hash_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        
        Y_out.resize(num);
        M_out.resize(num * 14);
        clEnqueueReadBuffer(queue, Yb, CL_TRUE, 0, num*4, Y_out.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, Mb, CL_TRUE, 0, num*14*4, M_out.data(), 0, nullptr, nullptr);
        
        clReleaseMemObject(Lb); clReleaseMemObject(Rb);
        clReleaseMemObject(Yb); clReleaseMemObject(Mb);
    }
    
    // Optimized: hash using M_curr + LR pairs directly (no CPU meta extraction)
    void gpu_hash_table_lr(
        const std::vector<uint32_t>& M_curr_flat,  // [num_total * 14] all metadata
        const std::vector<uint32_t>& LR_flat,      // [num_matches * 2] P1,P2 indices
        std::vector<uint32_t>& Y_out,
        std::vector<uint32_t>& M_out,
        uint32_t kmask)
    {
        const size_t num = LR_flat.size() / 2;
        if(num == 0) { Y_out.clear(); M_out.clear(); return; }
        
        cl_int err2;
        cl_mem Mb = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            M_curr_flat.size() * 4, (void*)M_curr_flat.data(), &err2);
        cl_mem LRb = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            LR_flat.size() * 4, (void*)LR_flat.data(), &err2);
        cl_mem Yb = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num * 4, nullptr, &err2);
        cl_mem Mb_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num * MY_N_META * 4, nullptr, &err2);
        
        uint32_t num_u32 = (uint32_t)num;
        clSetKernelArg(hash_lr_kernel, 0, sizeof(cl_mem), &Mb);
        clSetKernelArg(hash_lr_kernel, 1, sizeof(cl_mem), &LRb);
        clSetKernelArg(hash_lr_kernel, 2, sizeof(cl_mem), &Yb);
        clSetKernelArg(hash_lr_kernel, 3, sizeof(cl_mem), &Mb_out);
        clSetKernelArg(hash_lr_kernel, 4, sizeof(uint32_t), &kmask);
        clSetKernelArg(hash_lr_kernel, 5, sizeof(uint32_t), &num_u32);
        
        size_t global = num, local = 64;
        if(global % local) global = ((global/local)+1)*local;
        clEnqueueNDRangeKernel(queue, hash_lr_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        
        Y_out.resize(num);
        M_out.resize(num * MY_N_META);
        clEnqueueReadBuffer(queue, Yb, CL_TRUE, 0, num*4, Y_out.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, Mb_out, CL_TRUE, 0, num*MY_N_META*4, M_out.data(), 0, nullptr, nullptr);
        
        clReleaseMemObject(Mb); clReleaseMemObject(LRb);
        clReleaseMemObject(Yb); clReleaseMemObject(Mb_out);
    }
    
    void init_eval_kernel() {
        std::string kernel_path = "f2_f9.cl";
        FILE* f = fopen(kernel_path.c_str(), "r");
        if(!f) throw std::runtime_error("Cannot open " + kernel_path);
        fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
        char* src = new char[sz+1]; size_t rd = fread(src, 1, sz, f); src[sz] = 0; fclose(f);
        
        cl_int err2;
        f2f9_program = clCreateProgramWithSource(context, 1, (const char**)&src, &sz, &err2);
        delete[] src;
        
        // Build with current constants
        std::string opts = "-DKSIZE=" + std::to_string(KSIZE)
                         + " -DLOGBUCKETS=" + std::to_string(LOGBUCKETS)
                         + " -DLOGBUCKETS2=" + std::to_string(KSIZE - LOGBUCKETS - 9)
                         + " -DN_META=" + std::to_string(MY_N_META)
                         + " -DN_META_OUT=" + std::to_string(MY_N_META_OUT)
                         + " -DN_TABLE=" + std::to_string(MY_N_TABLE)
                         + " -DDSIZE_=5 -DPSIZE_=" + std::to_string(KSIZE + 1)
                         + " -DPDSIZE=" + std::to_string(PDSIZE)
                         + " -DX2SIZE=" + std::to_string(2 * XBITS - 1)
                         + " -DXBITS=" + std::to_string(XBITS)
                         + " -DHYBRID_SORT_LOG_THREADS=6 -DNUM_THREADS=64"
                         + " -DKMASK=" + std::to_string(KMASK)
                         + " -DDMASK=31 -DMETA_BYTES=" + std::to_string(MY_N_META * 4)
                         + " -DMAX_LOCAL_SIZE=40";
        
        err2 = clBuildProgram(f2f9_program, 1, &device, opts.c_str(), nullptr, nullptr);
        if(err2 != CL_SUCCESS) {
            char log[4096]; clGetProgramBuildInfo(f2f9_program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
            throw std::runtime_error(std::string("f2_f9 build failed: ") + log);
        }
        
        eval_p1_tx_kernel = clCreateKernel(f2f9_program, "eval_p1_tx", &err2);
        if(err2 != CL_SUCCESS) throw std::runtime_error("eval_p1_tx not found");
        std::cout << "[OCL] eval_p1_tx kernel loaded" << std::endl;
    }
    
    // New method: hash table using eval_p1_tx kernel (indexes C_in directly via LR pairs)
    void gpu_eval_table(
        const std::vector<uint32_t>& C_in,      // [num_entries * N_META] metadata for all entries
        const std::vector<uint32_t>& LR_flat,   // [num_matches * 2] P_1, P_2 pairs
        std::vector<uint32_t>& Y_out,
        std::vector<uint32_t>& M_out,
        int table)
    {
        const size_t num = LR_flat.size() / 2;
        if(num == 0) { Y_out.clear(); M_out.clear(); return; }
        
        const int num_buckets = 1 << LOGBUCKETS;
        // Max entries per output bucket. For k entries distributed into num_buckets,
        // avg per bucket = k / num_buckets. Use 4x avg + 4096 as safety margin.
        const size_t avg_per_bucket = num / num_buckets;
        const int max_bucket_size = std::max((size_t)4096, avg_per_bucket * 2 + 4096);
        
        cl_int err2;
        cl_mem C_in_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            C_in.size() * 4, (void*)C_in.data(), &err2);
        cl_mem LR_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            LR_flat.size() * 4, (void*)LR_flat.data(), &err2);
        uint32_t num32 = (uint32_t)num;
        cl_mem num_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            4, (void*)&num32, &err2);
        
        ulong PD_0 = 0;
        uint max_bs = max_bucket_size;
        uint x2size = 2 * XBITS - 1, xbits = XBITS;
        uint write_y = (table >= MY_N_TABLE) ? 1 : 0;
        uint write_c = 1, has_pd_in = 0, has_x_in = 0;
        
        // Y_out: only needed for table >= N_TABLE, otherwise minimal
        size_t y_buf_size = write_y ? (size_t)num_buckets * max_bucket_size * 4 : 4;
        // C_out: always needed (metadata for next table)
        size_t c_buf_size = (size_t)num_buckets * max_bucket_size * MY_N_META * 4;
        // PD_out: not needed for our CPU-driven pipeline (PD computed separately on CPU)
        // But kernel always writes to it, so allocate minimal size
        size_t pd_buf_size = 4;
        
        cl_mem Y_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, y_buf_size, nullptr, &err2);
        cl_mem C_out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, c_buf_size, nullptr, &err2);
        cl_mem cnt_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, num_buckets * 4, nullptr, &err2);
        cl_mem PD_out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, pd_buf_size, nullptr, &err2);
        cl_mem null_mem = nullptr;
        
        int zero = 0;
        clEnqueueFillBuffer(queue, cnt_buf, &zero, 4, 0, num_buckets * 4, 0, nullptr, nullptr);
        clEnqueueFillBuffer(queue, PD_out_buf, &zero, 4, 0, pd_buf_size, 0, nullptr, nullptr);
        
        clSetKernelArg(eval_p1_tx_kernel, 0, sizeof(cl_mem), &Y_buf);
        clSetKernelArg(eval_p1_tx_kernel, 1, sizeof(cl_mem), &C_out_buf);
        clSetKernelArg(eval_p1_tx_kernel, 2, sizeof(cl_mem), &PD_out_buf);
        clSetKernelArg(eval_p1_tx_kernel, 3, sizeof(cl_mem), &cnt_buf);
        clSetKernelArg(eval_p1_tx_kernel, 4, sizeof(cl_mem), &C_in_buf);
        clSetKernelArg(eval_p1_tx_kernel, 5, sizeof(cl_mem), &null_mem);
        clSetKernelArg(eval_p1_tx_kernel, 6, sizeof(cl_mem), &null_mem);
        clSetKernelArg(eval_p1_tx_kernel, 7, sizeof(cl_mem), &LR_buf);
        clSetKernelArg(eval_p1_tx_kernel, 8, sizeof(cl_mem), &num_buf);
        clSetKernelArg(eval_p1_tx_kernel, 9, sizeof(ulong), &PD_0);
        clSetKernelArg(eval_p1_tx_kernel, 10, sizeof(uint), &max_bs);
        clSetKernelArg(eval_p1_tx_kernel, 11, sizeof(uint), &x2size);
        clSetKernelArg(eval_p1_tx_kernel, 12, sizeof(uint), &xbits);
        uint table_u32 = (uint)table;
        clSetKernelArg(eval_p1_tx_kernel, 13, sizeof(uint), &table_u32);
        clSetKernelArg(eval_p1_tx_kernel, 14, sizeof(uint), &write_y);
        clSetKernelArg(eval_p1_tx_kernel, 15, sizeof(uint), &write_c);
        clSetKernelArg(eval_p1_tx_kernel, 16, sizeof(uint), &has_pd_in);
        clSetKernelArg(eval_p1_tx_kernel, 17, sizeof(uint), &has_x_in);
        
        size_t global = num;
        if(global % 64) global = ((global / 64) + 1) * 64;
        clEnqueueNDRangeKernel(queue, eval_p1_tx_kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        
        // Read back bucket counts
        std::vector<uint32_t> cnt(num_buckets);
        clEnqueueReadBuffer(queue, cnt_buf, CL_TRUE, 0, num_buckets * 4, cnt.data(), 0, nullptr, nullptr);
        
        // Read back Y (only if write_y) and C
        std::vector<uint32_t> Y_gpu(write_y ? num_buckets * max_bucket_size : 0, 0);
        std::vector<uint32_t> C_gpu(num_buckets * max_bucket_size * MY_N_META, 0);
        if(write_y) {
            clEnqueueReadBuffer(queue, Y_buf, CL_TRUE, 0, num_buckets * max_bucket_size * 4, Y_gpu.data(), 0, nullptr, nullptr);
        }
        clEnqueueReadBuffer(queue, C_out_buf, CL_TRUE, 0, num_buckets * max_bucket_size * MY_N_META * 4, C_gpu.data(), 0, nullptr, nullptr);
        
        // Flatten output
        Y_out.clear(); M_out.clear();
        for(int b = 0; b < num_buckets; b++) {
            for(int j = 0; j < (int)cnt[b]; j++) {
                if(write_y && b * max_bucket_size + j < (int)Y_gpu.size()) {
                    Y_out.push_back(Y_gpu[b * max_bucket_size + j]);
                } else {
                    // Compute Y from metadata
                    uint32_t Y = 0;
                    for(int m = 0; m < MY_N_META; m++) Y ^= C_gpu[(b * max_bucket_size + j) * MY_N_META + m];
                    Y_out.push_back(Y & KMASK);
                }
                for(int m = 0; m < MY_N_META; m++) {
                    M_out.push_back(C_gpu[(b * max_bucket_size + j) * MY_N_META + m]);
                }
            }
        }
        
        clReleaseMemObject(C_in_buf); clReleaseMemObject(LR_buf); clReleaseMemObject(num_buf);
        clReleaseMemObject(Y_buf); clReleaseMemObject(C_out_buf); clReleaseMemObject(cnt_buf);
        clReleaseMemObject(PD_out_buf);
    }
};

// ============================================================================
// CPU F2-F9 Computation (with LR pointers for proof reconstruction)
// ============================================================================

struct PlotData {
    std::vector<uint32_t> final_Y;
    std::vector<std::array<uint32_t, 14>> final_meta;
    
    // PD tree: PD[table][entry] = (sorted_pos_in_prev_table, delta)
    // PD[t] has num_entries[t] entries (one per table t entry)
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> PD;
    
    // X pairs for table 2 (indexed by table 2 sorted position)
    std::vector<std::pair<uint32_t, uint32_t>> X_pairs;
    
    std::vector<uint64_t> num_entries;
};

void compute_full_pipeline(
    const std::vector<uint32_t>& X_values,
    const std::vector<uint32_t>& Y_in,
    const std::vector<uint32_t>& M_in,
    PlotData& plot,
    OCL_Plotter& gpu_plotter)
{
    const uint32_t kmask = KMASK;
    
    std::vector<std::array<uint32_t, 14>> M_curr(Y_in.size());
    for(size_t i = 0; i < Y_in.size(); i++) {
        for(int j = 0; j < MY_N_META; j++) {
            M_curr[i][j] = M_in[i * MY_N_META + j];
        }
    }
    
    std::vector<std::pair<uint32_t, uint32_t>> entries;
    entries.reserve(Y_in.size());
    std::vector<uint32_t> final_indices;
    for(size_t i = 0; i < Y_in.size(); i++) {
        entries.emplace_back(Y_in[i], (uint32_t)i);
    }
    
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> LR(MY_N_TABLE + 1);
    std::vector<std::vector<uint32_t>> entries_map(MY_N_TABLE + 1);  // entries_map[t] = sorted->original mapping for table t
    
    auto sort_func = [&M_curr](const auto& L, const auto& R) {
        if(L.first == R.first) return M_curr[L.second] < M_curr[R.second];
        return L < R;
    };
    
    plot.num_entries.resize(MY_N_TABLE + 1);
    plot.num_entries[1] = entries.size();
    
    for(int t = 2; t <= MY_N_TABLE; t++) {
        auto t_table = my_time_ms();
        const size_t n = entries.size();
        std::cerr << "[T" << t << "] Sorting " << n << " entries...               \r" << std::flush;
        
        // Radix sort by Y (bucket sort — O(n) instead of O(n log n))
        // Use LOGBUCKETS bits for bucketing, then sort within buckets
        const int log_buckets = std::min(LOGBUCKETS, KSIZE - 1);
        const uint32_t bucket_mask = (1u << log_buckets) - 1;
        const int shift = KSIZE - log_buckets;
        const size_t num_buckets = 1u << log_buckets;
        
        // Count entries per bucket
        std::vector<uint32_t> bucket_counts(num_buckets, 0);
        for(size_t i = 0; i < n; i++) {
            uint32_t bucket = entries[i].first >> shift;
            bucket_counts[std::min(bucket, (uint32_t)num_buckets - 1)]++;
        }
        
        // Compute bucket offsets (prefix sum)
        std::vector<uint32_t> bucket_offsets(num_buckets + 1, 0);
        for(size_t i = 0; i < num_buckets; i++) {
            bucket_offsets[i + 1] = bucket_offsets[i] + bucket_counts[i];
        }
        
        // Scatter entries into buckets (stable within bucket)
        std::vector<std::pair<uint32_t, uint32_t>> sorted_entries(n);
        std::vector<uint32_t> write_pos(num_buckets);
        for(size_t i = 0; i < num_buckets; i++) write_pos[i] = bucket_offsets[i];
        for(size_t i = 0; i < n; i++) {
            uint32_t bucket = std::min(entries[i].first >> shift, (uint32_t)num_buckets - 1);
            sorted_entries[write_pos[bucket]++] = entries[i];
        }
        
        // Sort within each bucket in parallel (small buckets — fast)
        std::cerr << "[T" << t << "] Sorting buckets...                        \r" << std::flush;
        #pragma omp parallel for schedule(dynamic, 4)
        for(size_t b = 0; b < num_buckets; b++) {
            uint32_t start = bucket_offsets[b];
            uint32_t count = bucket_counts[b];
            if(count > 1) {
                std::sort(sorted_entries.begin() + start, 
                          sorted_entries.begin() + start + count, sort_func);
            }
        }
        entries = std::move(sorted_entries);
        
        auto t_sorted = my_time_ms();
        std::cerr << "[T" << t << "] Sorted in " << (t_sorted - t_table) / 1000.0 << "s. Matching...     \r" << std::flush;
        
        // Parallel matching: each thread handles a range of buckets
        // Y and Y+1 are always in the same or adjacent bucket
        const int nthreads = std::min(omp_get_max_threads(), (int)num_buckets);
        
        // Each thread collects LR pairs
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> thread_lr(nthreads);
        
        #pragma omp parallel for schedule(dynamic, 1)
        for(size_t b = 0; b < num_buckets; b++) {
            int ti = omp_get_thread_num();
            uint32_t start = bucket_offsets[b];
            uint32_t count = bucket_counts[b];
            if(count == 0) continue;
            
            // Match within this bucket — store SORTED indices
            for(uint32_t x = start; x < start + count; x++) {
                const auto YL = entries[x].first;
                for(uint32_t y = x + 1; y < start + count && entries[y].first <= YL + 1; y++) {
                    if(entries[y].first == YL + 1) {
                        // Store sorted positions, not original indices
                        thread_lr[ti].emplace_back(x, y);
                    }
                }
            }
            
            // Cross-bucket matching
            if(b + 1 < num_buckets) {
                uint32_t next_start = bucket_offsets[b + 1];
                uint32_t next_count = bucket_counts[b + 1];
                if(next_count > 0) {
                    uint32_t last_Y = entries[start + count - 1].first;
                    for(uint32_t x = start + count - 1; x >= start; x--) {
                        const auto YL = entries[x].first;
                        if(YL < last_Y) break;
                        for(uint32_t y = next_start; y < next_start + next_count && entries[y].first <= YL + 1; y++) {
                            if(entries[y].first == YL + 1) {
                                thread_lr[ti].emplace_back(x, y);
                            }
                        }
                    }
                }
            }
        }
        
        // Count total matches
        size_t total_matches = 0;
        for(int ti = 0; ti < nthreads; ti++) total_matches += thread_lr[ti].size();
        
        std::cerr << "[T" << t << "] " << total_matches << " matches. Hashing...            \r" << std::flush;
        
        auto t_meta_start = my_time_ms();
        
        // Flatten LR pairs
        std::vector<std::pair<uint32_t, uint32_t>> all_lr;
        all_lr.reserve(total_matches);
        for(int ti = 0; ti < nthreads; ti++) {
            for(const auto& p : thread_lr[ti]) all_lr.push_back(p);
        }
        
        // GPU hash: upload M_curr flat + LR pairs, kernel indexes directly
        // No CPU meta extraction needed!
        std::vector<uint32_t> M_curr_flat(M_curr.size() * MY_N_META);
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < M_curr.size(); i++) {
            for(int j = 0; j < MY_N_META; j++) {
                M_curr_flat[i * MY_N_META + j] = M_curr[i][j];
            }
        }
        std::vector<uint32_t> LR_flat(total_matches * 2);
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < total_matches; i++) {
            LR_flat[i * 2] = entries[all_lr[i].first].second;
            LR_flat[i * 2 + 1] = entries[all_lr[i].second].second;
        }
        
        auto t_hash_start = my_time_ms();
        
        std::vector<uint32_t> Y_results, M_results;
        gpu_plotter.gpu_hash_table_lr(M_curr_flat, LR_flat, Y_results, M_results, KMASK);
        
        auto t_hash_end = my_time_ms();
        double meta_time = (t_hash_start - t_meta_start) / 1000.0;
        double hash_time = (t_hash_end - t_hash_start) / 1000.0;
        std::cerr << "[T" << t << "] meta=" << meta_time << "s hash=" << hash_time << "s             \r" << std::flush;
        
        // Convert to M_next format
        std::vector<std::array<uint32_t, 14>> M_next(total_matches);
        for(size_t i = 0; i < total_matches; i++) {
            for(int j = 0; j < MY_N_META; j++) {
                M_next[i][j] = M_results[i * MY_N_META + j];
            }
        }
        
        // Build matches vector (Y, index into M_next)
        std::vector<std::pair<uint32_t, uint32_t>> matches;
        matches.reserve(total_matches);
        for(size_t i = 0; i < total_matches; i++) {
            matches.emplace_back(Y_results[i], (uint32_t)i);
        }
        
        std::vector<std::pair<uint32_t, uint32_t>> lr_pairs = std::move(all_lr);
        
        if(matches.empty()) {
            throw std::runtime_error("zero matches at table " + std::to_string(t));
        }
        
        LR[t] = std::move(lr_pairs);
        // Save the entries mapping: sorted_idx -> original_idx
        entries_map[t-1] = std::vector<uint32_t>(entries.size());
        for(size_t k = 0; k < entries.size(); k++) {
            entries_map[t-1][k] = entries[k].second;
        }
        // Save PD for EVERY entry in this table (in MATCH ORDER)
        // Store SORTED positions (index into sorted entries of previous table)
        // delta = sorted_R - sorted_L - 1 (always >= 0 since sorted_R > sorted_L)
        plot.PD.resize(MY_N_TABLE + 1);
        plot.PD[t].resize(matches.size());
        for(size_t k = 0; k < matches.size(); k++) {
            uint32_t sorted_L = LR[t][k].first;
            uint32_t sorted_R = LR[t][k].second;
            plot.PD[t][k] = {sorted_L, sorted_R - sorted_L};
        }
        std::cerr << "[Debug] PD[" << t << "] saved with " << plot.PD[t].size() << " entries" << std::flush;
        M_curr = std::move(M_next);
        entries = std::move(matches);
        plot.num_entries[t] = entries.size();
        
        auto t_end = my_time_ms();
        std::cout << "[T" << t << "] " << plot.num_entries[t] << " entries ("
                  << (t_end - t_table) / 1000.0 << "s)                    " << std::endl;
    }
    
    // Save entries_map[9] (table 9 mapping: sorted_idx -> original_idx)
    entries_map[MY_N_TABLE] = std::vector<uint32_t>(entries.size());
    for(size_t k = 0; k < entries.size(); k++) {
        entries_map[MY_N_TABLE][k] = entries[k].second;
    }
    
// Reorder PD[2..8] by sorted position so PD[t][sorted_pos] gives the right entry.
    for(int t = 2; t <= 8; t++) {
        auto old_pd = std::move(plot.PD[t]);
        plot.PD[t].resize(old_pd.size());
        for(size_t sorted_pos = 0; sorted_pos < entries_map[t].size() && sorted_pos < old_pd.size(); sorted_pos++) {
            uint32_t match_idx = entries_map[t][sorted_pos];
            if(match_idx < old_pd.size()) {
                plot.PD[t][sorted_pos] = old_pd[match_idx];
            }
        }
    }
    
    // Final sort using radix sort (same as table sort)
    std::cerr << "[Final] Sorting " << entries.size() << " entries...           \r" << std::flush;
    {
        const int log_buckets = std::min(LOGBUCKETS, KSIZE - 1);
        const size_t num_buckets = 1u << log_buckets;
        const int shift = KSIZE - log_buckets;
        
        std::vector<uint32_t> bucket_counts(num_buckets, 0);
        for(size_t i = 0; i < entries.size(); i++) {
            bucket_counts[std::min(entries[i].first >> shift, (uint32_t)num_buckets - 1)]++;
        }
        std::vector<uint32_t> bucket_offsets(num_buckets + 1, 0);
        for(size_t i = 0; i < num_buckets; i++)
            bucket_offsets[i + 1] = bucket_offsets[i] + bucket_counts[i];
        
        std::vector<std::pair<uint32_t, uint32_t>> sorted_entries(entries.size());
        std::vector<uint32_t> write_pos(num_buckets);
        for(size_t i = 0; i < num_buckets; i++) write_pos[i] = bucket_offsets[i];
        for(size_t i = 0; i < entries.size(); i++) {
            uint32_t b = std::min(entries[i].first >> shift, (uint32_t)num_buckets - 1);
            sorted_entries[write_pos[b]++] = entries[i];
        }
        
        #pragma omp parallel for schedule(dynamic, 4)
        for(size_t b = 0; b < num_buckets; b++) {
            uint32_t start = bucket_offsets[b];
            uint32_t count = bucket_counts[b];
            if(count > 1)
                std::sort(sorted_entries.begin() + start, sorted_entries.begin() + start + count, sort_func);
        }
        entries = std::move(sorted_entries);
    }
    
    // Deduplicate using hash set (much faster than std::set BST)
    // Use a flat hash map: meta array -> bool
    // Since meta is 14 uint32s = 56 bytes, use a simple approach:
    // Sort by meta, then dedup linearly
    std::cerr << "[Final] Deduplicating " << entries.size() << " entries...     \r" << std::flush;
    {
        // Build index array sorted by meta
        std::vector<uint32_t> meta_idx(entries.size());
        for(size_t i = 0; i < entries.size(); i++) meta_idx[i] = (uint32_t)i;
        
        auto meta_cmp = [&entries, &M_curr](uint32_t a, uint32_t b) {
            return M_curr[entries[a].second] < M_curr[entries[b].second];
        };
        
        __gnu_parallel::sort(meta_idx.begin(), meta_idx.end(), meta_cmp,
            __gnu_parallel::parallel_tag(omp_get_max_threads()));
        
        // Linear dedup
        std::array<uint32_t, 14> prev_meta = {};
        bool first = true;
        for(size_t i = 0; i < meta_idx.size(); i++) {
            const auto& entry = entries[meta_idx[i]];
            const auto& meta = M_curr[entry.second];
            if(first || meta != prev_meta) {
                plot.final_Y.push_back(entry.first);
                plot.final_meta.push_back(meta);
                final_indices.push_back(entry.second);
                prev_meta = meta;
                first = false;
            }
        }
    }
    
    std::cout << "[Final] " << plot.final_Y.size() << " unique entries          " << std::endl;
    
    // Re-sort final entries by Y (dedup destroyed Y order)
    std::cerr << "[Final] Re-sorting by Y...                       \r" << std::flush;
    {
        size_t nf = plot.final_Y.size();
        std::vector<uint32_t> idx(nf);
        for(size_t i = 0; i < nf; i++) idx[i] = (uint32_t)i;
        
        // Sort index by Y value
        __gnu_parallel::sort(idx.begin(), idx.end(),
            [&](uint32_t a, uint32_t b) { return plot.final_Y[a] < plot.final_Y[b]; },
            __gnu_parallel::parallel_tag(omp_get_max_threads()));
        
        // Reorder all arrays
        std::vector<uint32_t> new_Y(nf);
        std::vector<std::array<uint32_t, 14>> new_meta(nf);
        std::vector<uint32_t> new_indices(nf);
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < nf; i++) {
            new_Y[i] = plot.final_Y[idx[i]];
            new_meta[i] = plot.final_meta[idx[i]];
            new_indices[i] = final_indices[idx[i]];
        }
        plot.final_Y = std::move(new_Y);
        plot.final_meta = std::move(new_meta);
        final_indices = std::move(new_indices);
    }
    
// PD[9] dedup: keep only entries for deduped final entries
    {
        auto old_pd9 = std::move(plot.PD[9]);
        plot.PD[9].resize(final_indices.size());
        for(size_t i = 0; i < final_indices.size(); i++) {
            plot.PD[9][i] = old_pd9[final_indices[i]];
        }
    }
    
    // X pairs: compute in match order, then reorder by sorted position
    {
        std::vector<std::pair<uint32_t, uint32_t>> x_match_order(LR[2].size());
        #pragma omp parallel for schedule(static)
        for(size_t k = 0; k < LR[2].size(); k++) {
            uint32_t sorted_L = LR[2][k].first;
            uint32_t sorted_R = LR[2][k].second;
            x_match_order[k] = {entries_map[1][sorted_L], entries_map[1][sorted_R]};
        }
        plot.X_pairs.resize(x_match_order.size());
        for(size_t sorted_pos = 0; sorted_pos < entries_map[2].size() && sorted_pos < x_match_order.size(); sorted_pos++) {
            uint32_t match_idx = entries_map[2][sorted_pos];
            if(match_idx < x_match_order.size()) {
                plot.X_pairs[sorted_pos] = x_match_order[match_idx];
            }
        }
    }
}

// ============================================================================
// Plot File Writer
// ============================================================================

void write_plot(
    const std::string& file_path,
    const hash_t& plot_id,
    const pubkey_t& farmer_key,
    const PlotData& plot,
    bool is_hdd_plot = true)
{
    auto header = PlotHeader::create();
    uint64_t header_size = 0;
    header->version = 0;
    header->ksize = KSIZE;
    header->xbits = XBITS;
    header->has_meta = is_hdd_plot;
    header->seed = hash_t();
    header->plot_id = plot_id;
    header->farmer_key = farmer_key;
    header->park_size_x = PARK_SIZE_X;
    header->park_size_y = PARK_SIZE_Y;
    header->park_size_pd = PARK_SIZE_PD;
    header->park_size_meta = PARK_SIZE_META;
    
    const uint32_t max_park_bytes_x = cdiv(PARK_SIZE_X * LPX2SIZE, 8);
    const uint32_t max_park_bytes_meta = cdiv(PARK_SIZE_META * KSIZE * MY_N_META_OUT, 8);
    const uint32_t max_park_bytes_y = 4 + cdiv(uint32_t((PARK_SIZE_Y - 1) * MAX_AVG_YDELTA_BITS), 8);
    const uint32_t max_park_bytes_pd = cdiv(PARK_SIZE_PD * KSIZE, 8) + cdiv(uint32_t(PARK_SIZE_PD * MAX_AVG_OFFSET_BITS), 8);
    
    header->park_bytes_x = max_park_bytes_x;
    header->park_bytes_y = max_park_bytes_y;
    header->park_bytes_pd = max_park_bytes_pd;
    header->park_bytes_meta = max_park_bytes_meta;
    header->entry_bits_x = LPX2SIZE;
    header->num_entries_y = plot.final_Y.size();
    std::cout << "[Debug] num_entries_y = " << plot.final_Y.size() << ", PD[9] = " << plot.PD[9].size() << std::endl;
    
    // Set ALL header fields before writing, so the header size is correct
    const uint64_t num_entries = plot.final_Y.size();
    
    // Pre-allocate table_offset_pd so header size is consistent
    header->table_offset_pd.resize(7);
    
    // Use placeholder offsets (0) for first write to measure header size
    // The offsets will be filled in after measuring
    header->table_offset_y = 0;
    header->table_offset_meta = 0;
    header->table_offset_x = 0;
    header->plot_size = 0;
    for(int t = 0; t < 7; t++) header->table_offset_pd[t] = 0;
    
    // Write header once to measure its size
    {
        vnx::write_to_file(file_path, header);
        std::ifstream tmp(file_path, std::ios::binary | std::ios::ate);
        header_size = tmp.tellg();
    }
    
    // Align header to 4096 bytes (like the CUDA plotter does)
    header_size = (header_size + 4095) & ~4095;
    
    // Now compute real offsets
    uint64_t offset = header_size;
    header->table_offset_y = offset;
    const uint64_t num_parks_y = cdiv(num_entries, (uint64_t)PARK_SIZE_Y);
    offset += num_parks_y * max_park_bytes_y;
    
    if(is_hdd_plot) {
        header->table_offset_meta = offset;
        const uint64_t num_parks_meta = cdiv(num_entries, (uint64_t)PARK_SIZE_META);
        offset += num_parks_meta * max_park_bytes_meta;
    }
    
    // PD tables in reverse order: table_offset_pd[0] = PD[8] (for table 9→8),
    // table_offset_pd[1] = PD[7] (for table 8→7), ..., table_offset_pd[6] = PD[2] (for table 3→2)
    // PD tables: table_offset_pd[0] = PD for table 9→8 (num_entries[9] entries)
    // table_offset_pd[1] = PD for table 8→7 (num_entries[8] entries)
    // ... table_offset_pd[6] = PD for table 3→2 (num_entries[3] entries)
    for(int t = 0; t < 7; t++) {
        header->table_offset_pd[t] = offset;
        int pd_table = MY_N_TABLE - t;  // 9, 8, 7, 6, 5, 4, 3
        const uint64_t num_entries_pd = plot.PD[pd_table].size();
        const uint64_t num_parks_pd = cdiv(num_entries_pd, (uint64_t)PARK_SIZE_PD);
        offset += num_parks_pd * max_park_bytes_pd;
    }
    
    header->table_offset_x = offset;
    // X table has one entry per table 2 match
    const uint64_t num_x_entries = plot.num_entries[2];
    const uint64_t num_parks_x = cdiv(num_x_entries, (uint64_t)PARK_SIZE_X);
    offset += num_parks_x * max_park_bytes_x;
    
    header->plot_size = offset;
    
    // Write header with correct offsets (same size since all fields were already set)
    vnx::write_to_file(file_path, header);
    
    std::cout << "[Plot] Header size: " << header_size << std::endl;
    std::cout << "[Plot] Y table offset: " << header->table_offset_y << std::endl;
    std::cout << "[Plot] Meta table offset: " << header->table_offset_meta << std::endl;
    std::cout << "[Plot] X table offset: " << header->table_offset_x << std::endl;
    std::cout << "[Plot] Total plot size: " << offset << " bytes (" << offset / 1e9 << " GB)" << std::endl;
    
    // Open file for appending parks
    // First, pad the header to 4096-byte alignment
    {
        std::ifstream tmp(file_path, std::ios::binary | std::ios::ate);
        uint64_t current_size = tmp.tellg();
        tmp.close();
        if(current_size < header_size) {
            std::ofstream pad(file_path, std::ios::binary | std::ios::app);
            std::vector<char> zeros(header_size - current_size, 0);
            pad.write(zeros.data(), zeros.size());
            pad.close();
        }
    }
    
    std::ofstream out(file_path, std::ios::binary | std::ios::app);
    if(!out.good()) throw std::runtime_error("Cannot open " + file_path);
    
    // Write Y table (parallel chunk generation)
    {
        std::cout << "[Plot] Writing Y table (" << num_parks_y << " parks)..." << std::endl;
        
        const uint64_t CHUNK = 1024;
        for(uint64_t chunk_start = 0; chunk_start < num_parks_y; chunk_start += CHUNK) {
            uint64_t chunk_end = std::min(chunk_start + CHUNK, num_parks_y);
            uint64_t chunk_size = chunk_end - chunk_start;
            std::vector<std::vector<uint8_t>> parks_buf(chunk_size);
            
            #pragma omp parallel for schedule(dynamic, 16)
            for(uint64_t pi = 0; pi < chunk_size; pi++) {
                uint64_t p = chunk_start + pi;
                std::vector<uint8_t> park(max_park_bytes_y * 2, 0);
                uint64_t start = p * PARK_SIZE_Y;
                uint64_t count = std::min((uint64_t)PARK_SIZE_Y, num_entries - start);
                if(count == 0) { parks_buf[pi] = std::move(park); continue; }
                
                uint32_t Y_first = plot.final_Y[start];
                write_bits(park, Y_first, 0, KSIZE);
                
                // Encode Y deltas using encode_symbol (matching CUDA plotter)
                uint64_t delta_bit_offset = 32; // after 4-byte header (KSIZE bits for first Y)
                // Actually, first Y is KSIZE bits, but the header is 4 bytes = 32 bits.
                // The bit stream starts at bit 32 (byte 4).
                for(uint64_t i = 1; i < count; i++) {
                    uint32_t delta = plot.final_Y[start + i] - plot.final_Y[start + i - 1];
                    if(delta > 255) throw std::runtime_error("Y delta too large: " + std::to_string(delta));
                    auto sym = mmx::pos::encode_symbol((uint8_t)delta);
                    write_bits(park, sym.first, delta_bit_offset, sym.second);
                    delta_bit_offset += sym.second;
                }
                park.resize(max_park_bytes_y, 0);
                parks_buf[pi] = std::move(park);
            }
            
            for(uint64_t pi = 0; pi < chunk_size; pi++) {
                out.write((char*)parks_buf[pi].data(), max_park_bytes_y);
            }
        }
    }
    
    // Write Meta table (parallel chunk generation)
    if(is_hdd_plot) {
        const uint64_t num_parks_meta = cdiv(num_entries, (uint64_t)PARK_SIZE_META);
        std::cout << "[Plot] Writing Meta table (" << num_parks_meta << " parks)..." << std::endl;
        
        const uint64_t CHUNK = 4096;
        for(uint64_t chunk_start = 0; chunk_start < num_parks_meta; chunk_start += CHUNK) {
            uint64_t chunk_end = std::min(chunk_start + CHUNK, num_parks_meta);
            uint64_t chunk_size = chunk_end - chunk_start;
            std::vector<std::vector<uint8_t>> parks_buf(chunk_size);
            
            #pragma omp parallel for schedule(dynamic, 16)
            for(uint64_t pi = 0; pi < chunk_size; pi++) {
                uint64_t p = chunk_start + pi;
                std::vector<uint8_t> park(max_park_bytes_meta, 0);
                uint64_t start = p * PARK_SIZE_META;
                uint64_t count = std::min((uint64_t)PARK_SIZE_META, num_entries - start);
                
                for(uint64_t i = 0; i < count; i++) {
                    for(int j = 0; j < MY_N_META_OUT; j++) {
                        uint64_t bit_offset = (i * MY_N_META_OUT + j) * KSIZE;
                        write_bits(park, plot.final_meta[start + i][j], bit_offset, KSIZE);
                    }
                }
                parks_buf[pi] = std::move(park);
            }
            
            for(uint64_t pi = 0; pi < chunk_size; pi++) {
                out.write((char*)parks_buf[pi].data(), max_park_bytes_meta);
            }
        }
    }
    
    // Debug: check PD delta ranges
    for(int t = 2; t <= MY_N_TABLE; t++) std::cout << "[Debug] PD[" << t << " size = " << plot.PD[t].size() << std::endl;
    for(int t = 0; t < 7; t++) {
        int pd_table = MY_N_TABLE - t;
        uint32_t max_delta = 0;
        for(size_t i = 0; i < plot.PD[pd_table].size(); i++) {
            max_delta = std::max(max_delta, plot.PD[pd_table][i].second);
        }
        std::cout << "[Debug] PD table " << pd_table << ": max delta = " << max_delta
                  << " (must be <= 255), entries = " << plot.PD[pd_table].size() << std::endl;
    }
    
    // Write PD tables in reverse order: PD[8] first, PD[2] last
    // (matches how Prover reads them: table_offset_pd[0] is for table 9→8)
    for(int t = 0; t < 7; t++) {
        int pd_table = MY_N_TABLE - t;  // 9, 8, 7, 6, 5, 4, 3
        const uint64_t num_entries_t = plot.PD[pd_table].size();
        const uint64_t num_parks_pd = cdiv(num_entries_t, (uint64_t)PARK_SIZE_PD);
        std::cout << "[Plot] Writing PD table " << pd_table << " (" << num_parks_pd << " parks)..." << std::endl;
        
        const uint64_t CHUNK = 4096;
        for(uint64_t chunk_start = 0; chunk_start < num_parks_pd; chunk_start += CHUNK) {
            uint64_t chunk_end = std::min(chunk_start + CHUNK, num_parks_pd);
            uint64_t chunk_size = chunk_end - chunk_start;
            std::vector<std::vector<uint8_t>> parks_buf(chunk_size);
            
            #pragma omp parallel for schedule(dynamic, 16)
            for(uint64_t pi = 0; pi < chunk_size; pi++) {
                uint64_t p = chunk_start + pi;
                // Use 2x buffer to handle delta overflow
                uint32_t park_buf_size = max_park_bytes_pd * 2;
                std::vector<uint8_t> park(park_buf_size, 0);
                uint64_t start = p * PARK_SIZE_PD;
                uint64_t count = std::min((uint64_t)PARK_SIZE_PD, num_entries_t - start);
                if(count == 0) { parks_buf[pi] = std::move(park); continue; }
                
                uint64_t pos_bit_offset = 0;
                for(uint64_t i = 0; i < count; i++) {
                    uint32_t pos = plot.PD[pd_table][start + i].first;
                    write_bits(park, pos, pos_bit_offset, KSIZE);
                    pos_bit_offset += KSIZE;
                }
                
                // Encode PD deltas using encode_symbol (matching CUDA plotter)
                uint64_t delta_bit_offset = (uint64_t)PARK_SIZE_PD * KSIZE;
                for(uint64_t i = 0; i < count; i++) {
                    uint32_t delta = plot.PD[pd_table][start + i].second;
                    if(delta > 255) throw std::runtime_error("PD delta too large: " + std::to_string(delta));
                    auto sym = mmx::pos::encode_symbol((uint8_t)delta);
                    write_bits(park, sym.first, delta_bit_offset, sym.second);
                    delta_bit_offset += sym.second;
                }
                park.resize(max_park_bytes_pd, 0);
                parks_buf[pi] = std::move(park);
            }
            
            for(uint64_t pi = 0; pi < chunk_size; pi++) {
                out.write((char*)parks_buf[pi].data(), max_park_bytes_pd);
            }
        }
    }
    
    // Write X table (parallel chunk generation)
    {
        const uint64_t num_x_entries = plot.X_pairs.size();
        const uint64_t num_parks_x = cdiv(num_x_entries, (uint64_t)PARK_SIZE_X);
        std::cout << "[Plot] Writing X table (" << num_parks_x << " parks, " << num_x_entries << " entries)..." << std::endl;
        
        const uint64_t CHUNK = 4096;
        for(uint64_t chunk_start = 0; chunk_start < num_parks_x; chunk_start += CHUNK) {
            uint64_t chunk_end = std::min(chunk_start + CHUNK, num_parks_x);
            uint64_t chunk_size = chunk_end - chunk_start;
            std::vector<std::vector<uint8_t>> parks_buf(chunk_size);
            
            #pragma omp parallel for schedule(dynamic, 16)
            for(uint64_t pi = 0; pi < chunk_size; pi++) {
                uint64_t p = chunk_start + pi;
                std::vector<uint8_t> park(max_park_bytes_x, 0);
                uint64_t start = p * PARK_SIZE_X;
                uint64_t count = std::min((uint64_t)PARK_SIZE_X, num_x_entries - start);
                
                for(uint64_t i = 0; i < count; i++) {
                    // Use calc_line_point2 for compressed (xbits < ksize), calc_line_point for uncompressed
                    uint64_t LP = (XBITS < KSIZE) ?
                        calc_line_point2(plot.X_pairs[start + i].first, plot.X_pairs[start + i].second) :
                        calc_line_point(plot.X_pairs[start + i].first, plot.X_pairs[start + i].second);
                    write_bits(park, LP, i * LPX2SIZE, LPX2SIZE);
                }
                parks_buf[pi] = std::move(park);
            }
            
            for(uint64_t pi = 0; pi < chunk_size; pi++) {
                out.write((char*)parks_buf[pi].data(), max_park_bytes_x);
            }
        }
    }
    
    out.close();
    
    std::cout << "[Plot] Plot file written: " << file_path << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv)
{
    std::string output_dir = "./";  // default: current directory
    std::string plot_name;
    bool test_mode = false;
    uint64_t test_limit = 0;
    bool use_ramdisk = false;
    std::string final_dir;  // if set, copy plot here after writing to ramdisk
    
    if(argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <plot_id_hex> <farmer_key_hex> [output_dir] [options]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --k N           Set plot k-size (default: 26)" << std::endl;
        std::cerr << "  --ramdisk DIR   Use tmpfs at DIR for plotting, then copy to output_dir" << std::endl;
        std::cerr << "  --test          Run in test mode" << std::endl;
        std::cerr << "  --limit N       Limit entries (test mode)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "To use a RAM disk (recommended for speed):" << std::endl;
        std::cerr << "  sudo mount -t tmpfs -o size=8G tmpfs /mnt/ramdisk" << std::endl;
        std::cerr << "  ./mmx_opencl_plotter <pid> <fkey> /mnt/hdd/ --ramdisk /mnt/ramdisk --k 26" << std::endl;
        return 1;
    }
    
    std::string pid_str = argv[1];
    std::string fk_str = argv[2];
    
    for(int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if(arg == "--test") test_mode = true;
        else if(arg == "--limit" && i+1 < argc) test_limit = std::stoull(argv[++i]);
        else if(arg == "--k" && i+1 < argc) { KSIZE = std::stoi(argv[++i]); XBITS = KSIZE; }
        else if(arg == "--ramdisk" && i+1 < argc) {
            use_ramdisk = true;
            final_dir = output_dir;
            output_dir = argv[++i];
        }
        else output_dir = arg;
    }
    
    update_constants();
    
    if(pid_str.size() != 64) { std::cerr << "plot_id must be 64 hex chars" << std::endl; return 1; }
    if(fk_str.size() != 66) { std::cerr << "farmer_key must be 66 hex chars" << std::endl; return 1; }
    
    hash_t plot_id;
    plot_id.from_string(pid_str);
    pubkey_t farmer_key;
    farmer_key.from_string(fk_str);
    
    // Generate plot name: plot-mmx-hdd-k<K>-c<LEVEL>-YYYY-MM-DD-HH-MM-<PLOT_ID_HEX>.plot
    std::string plot_type = "hdd";  // we write HDD plots (has_meta=true)
    int clevel = 0;  // compression level 0 (uncompressed)
    {
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        char date_str[32];
        std::strftime(date_str, sizeof(date_str), "%Y-%m-%d-%H-%M", &tm);
        // plot_id as uppercase hex (pid_str is already lowercase hex, uppercase it)
        std::string pid_upper = pid_str;
        std::transform(pid_upper.begin(), pid_upper.end(), pid_upper.begin(), ::toupper);
        plot_name = "plot-mmx-" + plot_type + "-k" + std::to_string(KSIZE)
                  + "-c" + std::to_string(clevel)
                  + "-" + date_str
                  + "-" + pid_upper + ".plot";
    }
    std::string plot_path = output_dir;
    if(!plot_path.empty() && plot_path.back() != '/') plot_path += "/";
    plot_path += plot_name;
    
    std::cout << "=== MMX OpenCL Plotter ===" << std::endl;
    std::cout << "Plot ID: " << plot_id.to_string() << std::endl;
    std::cout << "Farmer Key: " << farmer_key.to_string() << std::endl;
    std::cout << "KSIZE: " << KSIZE << ", XBITS: " << XBITS << std::endl;
    std::cout << "Output: " << plot_path << std::endl;
    if(use_ramdisk) {
        std::cout << "RAM disk: " << output_dir << " (plot written here first)" << std::endl;
        std::cout << "Final dir: " << final_dir << " (plot copied here after)" << std::endl;
    }
    std::cout << std::endl;
    
    // Initialize OpenCL
    cl_int err;
    cl_uint np; clGetPlatformIDs(0, nullptr, &np);
    cl_platform_id plat; clGetPlatformIDs(1, &plat, nullptr);
    cl_uint nd; clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd);
    cl_device_id dev; clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
    char dn[256]; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dn), dn, nullptr);
    std::cout << "[OCL] Device: " << dn << std::endl;
    
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if(err != CL_SUCCESS) { std::cerr << "Failed to create OpenCL context" << std::endl; return 1; }
    
    // Compute F1 on GPU
    OCL_Plotter plotter;
    plotter.init(ctx, dev);
    plotter.init_table_hash();
    
    std::vector<uint32_t> Y_all, M_all;
    plotter.compute_all_f1(plot_id, Y_all, M_all, 1 << 20, test_mode, test_limit);
    
    // Compute F2-F9 on CPU
    std::cout << "\n[CPU] Computing F2-F9..." << std::endl;
    auto t0 = my_time_ms();
    
    std::vector<uint32_t> X_values(Y_all.size());
    for(size_t i = 0; i < X_values.size(); i++) X_values[i] = (uint32_t)i;
    
    PlotData plot;
    compute_full_pipeline(X_values, Y_all, M_all, plot, plotter);
    
    auto t1 = my_time_ms();
    std::cout << "[CPU] F2-F9 done in " << (t1 - t0) / 1000.0 << " sec" << std::endl;
    
    // Write plot file
    std::cout << "\n[Plot] Writing plot file..." << std::endl;
    write_plot(plot_path, plot_id, farmer_key, plot, true);
    
    // If using ramdisk, copy plot to final destination
    if(use_ramdisk && !final_dir.empty()) {
        std::string final_path = final_dir;
        if(!final_path.empty() && final_path.back() != '/') final_path += "/";
        final_path += plot_name;
        std::cout << "[Plot] Copying from RAM disk to " << final_path << "..." << std::endl;
        std::ifstream src(plot_path, std::ios::binary);
        std::ofstream dst(final_path, std::ios::binary);
        if(!dst.good()) {
            std::cerr << "ERROR: Cannot write to " << final_path << std::endl;
            std::cerr << "Plot remains on RAM disk at: " << plot_path << std::endl;
            clReleaseContext(ctx);
            return 1;
        }
        dst << src.rdbuf();
        dst.close();
        src.close();
        // Remove from RAM disk to free memory
        std::remove(plot_path.c_str());
        plot_path = final_path;
    }
    
    auto t2 = my_time_ms();
    std::cout << "\n[Done] Total time: " << (t2 - t0) / 1000.0 << " sec" << std::endl;
    std::cout << "[Done] Plot file: " << plot_path << std::endl;
    
    clReleaseContext(ctx);
    
    return 0;
}
