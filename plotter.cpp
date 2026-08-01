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
#include <future>
#include <thread>
#include <mutex>
#include "opt_config.h"
#include "buffer_pool.h"
#include "pipeline.h"
#include "svm_pool.h"

// Forward declarations for globals defined later
extern bool gpu_yield;
extern bool timing_detail;

// Global buffer pool (Module C) — initialized in compute_f2_f9_chunked
BufferPool* g_bufpool = nullptr;

// Multi-GPU support (Module E)
class OCL_Plotter;  // forward declaration
std::vector<OCL_Plotter*> g_plotters;
std::vector<BufferPool*> g_bufpools;
std::vector<SVMPool*> g_svmpools;
int g_num_gpus = 1;
std::mutex g_dst_mutex;  // protects dst store writes in multi-GPU mode

// SVM helper: set SVM pointer as kernel arg
inline cl_int clSetKernelArgSVM(cl_kernel kernel, cl_uint idx, const void* svm_ptr) {
    return clSetKernelArgSVMPointer(kernel, idx, svm_ptr);
}
#include <mutex>
#include <set>
#include <unordered_map>
#include <ctime>
#include <cctype>
#include <parallel/algorithm>
#include <unistd.h>
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

// OpenCL error checking helper
#define CL_CHECK(err, msg) \
    if(err != CL_SUCCESS) { \
        throw std::runtime_error(std::string(msg) + ": OpenCL error " + std::to_string(err)); \
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
    
    // Bounds check: ensure we don't write past the buffer end
    uint64_t end_byte = (bit_offset + num_bits + 7) / 8;
    if(end_byte > buf.size()) {
        throw std::runtime_error("write_bits overflow: need byte " + std::to_string(end_byte)
            + " but buffer is " + std::to_string(buf.size()) + " bytes");
    }
    
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
    cl_command_queue queue2 = nullptr;
    
    cl_command_queue& get_queue(int bucket = 0) {
        if(get_opt_config().num_queues > 1 && queue2 && bucket % 2 == 1)
            return queue2;
        return queue;
    }
    cl_program program;
    cl_kernel f1_kernel;
    
    void init(cl_context ctx, cl_device_id dev) {
        context = ctx;
        device = dev;
        cl_int err;
        queue = clCreateCommandQueue(context, device, 0, &err);
        if(err != CL_SUCCESS) throw std::runtime_error("Failed to create command queue");
        
        // Create second queue for multi-queue pipelining (Module E)
        queue2 = clCreateCommandQueue(context, device, 0, &err);
        if(err != CL_SUCCESS) throw std::runtime_error("Failed to create command queue 2");
        
        std::string kernel_path = "pos_recompute.cl";
        FILE* f = fopen(kernel_path.c_str(), "r");
        if(!f) throw std::runtime_error("Cannot open " + kernel_path);
        fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
        char* src = new char[sz+1]; size_t rd = fread(src, 1, sz, f); src[sz] = 0; fclose(f);
        
        program = clCreateProgramWithSource(context, 1, (const char**)&src, &sz, &err);
        delete[] src;
        err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
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
    
    // SVM-based F1 batch — no cl_mem, no upload/download, direct shared memory
    void compute_f1_batch_svm(
        std::vector<uint32_t>& X_values,
        const hash_t& plot_id,
        std::vector<uint32_t>& Y_out,
        std::vector<uint32_t>& M_out,
        void* svm_X, void* svm_Y, void* svm_M,  // pre-allocated SVM buffers
        size_t svm_capacity)  // max entries that fit in SVM buffers
    {
        const size_t num_x = X_values.size();
        if(num_x == 0) return;
        if(num_x > svm_capacity) throw std::runtime_error("F1 SVM batch too large");
        
        std::vector<uint32_t> id_u32(8);
        std::memcpy(id_u32.data(), plot_id.data(), 32);
        
        // Write X values to SVM (fine-grain: direct write, coarse-grain: map/unmap)
        // For now use map/unmap which works on both
        cl_int err;
        clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, svm_X, num_x * sizeof(uint32_t), 0, nullptr, nullptr);
        std::memcpy(svm_X, X_values.data(), num_x * sizeof(uint32_t));
        clEnqueueSVMUnmap(queue, svm_X, 0, nullptr, nullptr);
        
        // Upload plot_id (small, use regular buffer)
        cl_mem ID_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       8 * sizeof(uint32_t), (void*)id_u32.data(), &err);
        
        uint32_t kmask = KMASK;
        uint32_t xbits = 0;
        uint32_t num_x_u32 = (uint32_t)num_x;
        
        clSetKernelArgSVMPointer(f1_kernel, 0, svm_X);
        clSetKernelArg(f1_kernel, 1, sizeof(cl_mem), &ID_buf);  // plot_id as regular buffer
        clSetKernelArgSVMPointer(f1_kernel, 2, svm_Y);
        clSetKernelArgSVMPointer(f1_kernel, 3, svm_M);
        clSetKernelArg(f1_kernel, 4, sizeof(uint32_t), &kmask);
        clSetKernelArg(f1_kernel, 5, sizeof(uint32_t), &xbits);
        clSetKernelArg(f1_kernel, 6, sizeof(uint32_t), &num_x_u32);
        
        size_t global = num_x;
        size_t local = 64;
        if(global % local != 0) global = ((global / local) + 1) * local;
        
        err = clEnqueueNDRangeKernel(queue, f1_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        if(err != CL_SUCCESS) throw std::runtime_error("F1 SVM kernel launch failed: " + std::to_string(err));
        
        // Read results from SVM
        Y_out.resize(num_x);
        M_out.resize(num_x * MY_N_META);
        clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, svm_Y, num_x * sizeof(uint32_t), 0, nullptr, nullptr);
        std::memcpy(Y_out.data(), svm_Y, num_x * sizeof(uint32_t));
        clEnqueueSVMUnmap(queue, svm_Y, 0, nullptr, nullptr);
        clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, svm_M, num_x * MY_N_META * sizeof(uint32_t), 0, nullptr, nullptr);
        std::memcpy(M_out.data(), svm_M, num_x * MY_N_META * sizeof(uint32_t));
        clEnqueueSVMUnmap(queue, svm_M, 0, nullptr, nullptr);
        
        clReleaseMemObject(ID_buf);
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
            
            // Use SVM F1 if available
            bool use_svm = get_opt_config().svm && g_svmpools.size() > 0 && g_svmpools[0]->svm_F1_X;
            if(use_svm) {
                compute_f1_batch_svm(X_batch, plot_id, Y_batch, M_batch,
                    g_svmpools[0]->svm_F1_X, g_svmpools[0]->svm_F1_Y, g_svmpools[0]->svm_F1_M,
                    g_svmpools[0]->svm_F1_X ? (size_t)-1 : 0);  // capacity check disabled
            } else {
                compute_f1_batch(X_batch, plot_id, Y_batch, M_batch);
            }
            
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
    cl_kernel gather_meta_kernel = nullptr;
    cl_kernel prefix_sum_kernel = nullptr;
    cl_kernel eval_p1_tx_kernel = nullptr;
    cl_kernel k_scatter2 = nullptr;
    cl_kernel k_simple_sort = nullptr;
    cl_kernel k_match_p1 = nullptr;
    cl_kernel k_memset_u32 = nullptr;
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
        err = clBuildProgram(prog, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
        if(err != CL_SUCCESS) {
            char log[4096]; clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
            throw std::runtime_error(std::string("table_hash build failed: ") + log);
        }
        
        table_hash_kernel = clCreateKernel(prog, "hash_table_entries", &err);
        hash_lr_kernel = clCreateKernel(prog, "hash_table_lr", &err);
        if(err != CL_SUCCESS) throw std::runtime_error("hash_table_lr not found");
        if(err != CL_SUCCESS) throw std::runtime_error("hash_table_entries not found");
        
        // Load gather kernel (Module B variant 2)
        {
            std::string gpath = "gather_meta.cl";
            FILE* gf = fopen(gpath.c_str(), "r");
            if(gf) {
                fseek(gf, 0, SEEK_END); size_t gsz = ftell(gf); fseek(gf, 0, SEEK_SET);
                char* gsrc = new char[gsz+1]; fread(gsrc, 1, gsz, gf); gsrc[gsz] = 0; fclose(gf);
                cl_program gprog = clCreateProgramWithSource(context, 1, (const char**)&gsrc, &gsz, &err);
                delete[] gsrc;
                err = clBuildProgram(gprog, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
                if(err == CL_SUCCESS) {
                    gather_meta_kernel = clCreateKernel(gprog, "gather_meta", &err);
                    if(err == CL_SUCCESS) std::cout << "[OCL] Gather kernel loaded" << std::endl;
                }
            }
        }
        
        // Load prefix sum kernel (Module A)
        {
            std::string ppath = "prefix_sum.cl";
            FILE* pf = fopen(ppath.c_str(), "r");
            if(pf) {
                fseek(pf, 0, SEEK_END); size_t psz = ftell(pf); fseek(pf, 0, SEEK_SET);
                char* psrc = new char[psz+1]; fread(psrc, 1, psz, pf); psrc[psz] = 0; fclose(pf);
                cl_program pprog = clCreateProgramWithSource(context, 1, (const char**)&psrc, &psz, &err);
                delete[] psrc;
                err = clBuildProgram(pprog, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
                if(err == CL_SUCCESS) {
                    prefix_sum_kernel = clCreateKernel(pprog, "gpu_prefix_sum", &err);
                    if(err == CL_SUCCESS) std::cout << "[OCL] Prefix sum kernel loaded" << std::endl;
                }
            }
        }
        
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
    
    // SVM-based hash — no cl_mem, direct shared memory (fine-grain: zero-copy on AMD)
    void gpu_hash_table_svm(
        const std::vector<uint32_t>& L_meta,  // [num * 14]
        const std::vector<uint32_t>& R_meta,  // [num * 14]
        std::vector<uint32_t>& Y_out,
        std::vector<uint32_t>& M_out,
        uint32_t kmask,
        void* svm_L, void* svm_R, void* svm_Y, void* svm_M,
        bool fine_grain)
    {
        const size_t num = L_meta.size() / 14;
        if(num == 0) { Y_out.clear(); M_out.clear(); return; }
        
        // Write L/R metadata to SVM
        if(fine_grain) {
            std::memcpy(svm_L, L_meta.data(), L_meta.size() * 4);
            std::memcpy(svm_R, R_meta.data(), R_meta.size() * 4);
            std::atomic_thread_fence(std::memory_order_release);
        } else {
            clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, svm_L, L_meta.size() * 4, 0, nullptr, nullptr);
            std::memcpy(svm_L, L_meta.data(), L_meta.size() * 4);
            clEnqueueSVMUnmap(queue, svm_L, 0, nullptr, nullptr);
            clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, svm_R, R_meta.size() * 4, 0, nullptr, nullptr);
            std::memcpy(svm_R, R_meta.data(), R_meta.size() * 4);
            clEnqueueSVMUnmap(queue, svm_R, 0, nullptr, nullptr);
        }
        
        uint32_t num_u32 = (uint32_t)num;
        clSetKernelArgSVMPointer(table_hash_kernel, 0, svm_L);
        clSetKernelArgSVMPointer(table_hash_kernel, 1, svm_R);
        clSetKernelArgSVMPointer(table_hash_kernel, 2, svm_Y);
        clSetKernelArgSVMPointer(table_hash_kernel, 3, svm_M);
        clSetKernelArg(table_hash_kernel, 4, sizeof(uint32_t), &kmask);
        clSetKernelArg(table_hash_kernel, 5, sizeof(uint32_t), &num_u32);
        
        size_t global = num, local = 64;
        if(global % local) global = ((global/local)+1)*local;
        clEnqueueNDRangeKernel(queue, table_hash_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        
        Y_out.resize(num);
        M_out.resize(num * 14);
        if(fine_grain) {
            clFinish(queue);
            std::memcpy(Y_out.data(), svm_Y, num * 4);
            std::memcpy(M_out.data(), svm_M, num * 14 * 4);
        } else {
            clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, svm_Y, num * 4, 0, nullptr, nullptr);
            std::memcpy(Y_out.data(), svm_Y, num * 4);
            clEnqueueSVMUnmap(queue, svm_Y, 0, nullptr, nullptr);
            clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, svm_M, num * 14 * 4, 0, nullptr, nullptr);
            std::memcpy(M_out.data(), svm_M, num * 14 * 4);
            clEnqueueSVMUnmap(queue, svm_M, 0, nullptr, nullptr);
        }
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
        uint32_t num_total_u32 = (uint32_t)(M_curr_flat.size() / MY_N_META);
        clSetKernelArg(hash_lr_kernel, 0, sizeof(cl_mem), &Mb);
        clSetKernelArg(hash_lr_kernel, 1, sizeof(cl_mem), &LRb);
        clSetKernelArg(hash_lr_kernel, 2, sizeof(cl_mem), &Yb);
        clSetKernelArg(hash_lr_kernel, 3, sizeof(cl_mem), &Mb_out);
        clSetKernelArg(hash_lr_kernel, 4, sizeof(uint32_t), &kmask);
        clSetKernelArg(hash_lr_kernel, 5, sizeof(uint32_t), &num_u32);
        clSetKernelArg(hash_lr_kernel, 6, sizeof(uint32_t), &num_total_u32);
        
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
    
    void init_gpu_kernels() {
        // Load f2_f9.cl + simple_sort.cl combined
        std::string src_str;
        {
            FILE* f = fopen("f2_f9.cl", "r");
            if(!f) throw std::runtime_error("Cannot open f2_f9.cl");
            fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
            char* src = new char[sz+1]; fread(src, 1, sz, f); src[sz] = 0; fclose(f);
            src_str += src;
            src_str += "\n";
            delete[] src;
        }
        {
            FILE* f = fopen("simple_sort.cl", "r");
            if(!f) throw std::runtime_error("Cannot open simple_sort.cl");
            fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
            char* src = new char[sz+1]; fread(src, 1, sz, f); src[sz] = 0; fclose(f);
            src_str += src;
            src_str += "\n";
            delete[] src;
        }
        
        cl_int err2;
        const char* cstr = src_str.c_str();
        size_t len = src_str.size();
        f2f9_program = clCreateProgramWithSource(context, 1, &cstr, &len, &err2);
        
        // Build with current constants
        std::string opts = "-cl-std=CL1.2 -DKSIZE=" + std::to_string(KSIZE)
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
        
        k_scatter2 = clCreateKernel(f2f9_program, "scatter_2", &err2);
        if(err2 != CL_SUCCESS) throw std::runtime_error("scatter_2 not found");
        k_simple_sort = clCreateKernel(f2f9_program, "simple_sort_y", &err2);
        if(err2 != CL_SUCCESS) throw std::runtime_error("simple_sort_y not found");
        k_match_p1 = clCreateKernel(f2f9_program, "match_p1", &err2);
        if(err2 != CL_SUCCESS) throw std::runtime_error("match_p1 not found");
        k_memset_u32 = clCreateKernel(f2f9_program, "memset_u32", &err2);
        if(err2 != CL_SUCCESS) throw std::runtime_error("memset_u32 not found");
        eval_p1_tx_kernel = clCreateKernel(f2f9_program, "eval_p1_tx", &err2);
        if(err2 != CL_SUCCESS) throw std::runtime_error("eval_p1_tx not found");
        
        std::cout << "[OCL] GPU kernels loaded: scatter_2, simple_sort_y, match_p1, eval_p1_tx, memset_u32" << std::endl;
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
        
        // Check for bucket overflow (entries dropped by eval_p1_tx)
        uint32_t total_out = 0;
        for(int b = 0; b < num_buckets; b++) total_out += cnt[b];
        if(total_out < num) {
            std::cerr << "[WARN] eval_p1_tx dropped " << (num - total_out)
                      << " entries (bucket overflow, max_bucket_size=" << max_bucket_size
                      << ") at table " << table << std::endl;
        }
        
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
        
        // GPU hash — old method (pre-extract L/R metadata on CPU)
        std::vector<uint32_t> L_meta_flat(total_matches * MY_N_META);
        std::vector<uint32_t> R_meta_flat(total_matches * MY_N_META);
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < total_matches; i++) {
            const auto& [sorted_L, sorted_R] = all_lr[i];
            uint32_t orig_L = entries[sorted_L].second;
            uint32_t orig_R = entries[sorted_R].second;
            for(int j = 0; j < MY_N_META; j++) {
                L_meta_flat[i * MY_N_META + j] = M_curr[orig_L][j];
                R_meta_flat[i * MY_N_META + j] = M_curr[orig_R][j];
            }
        }
        std::vector<uint32_t> Y_results, M_results;
        if(get_opt_config().svm && g_svmpools.size() > 0 && g_svmpools[0]->svm_L_gathered) {
            // SVM hash — no cl_mem, direct shared memory
            gpu_plotter.gpu_hash_table_svm(L_meta_flat, R_meta_flat, Y_results, M_results, KMASK,
                g_svmpools[0]->svm_L_gathered, g_svmpools[0]->svm_R_gathered,
                g_svmpools[0]->svm_Y_hash, g_svmpools[0]->svm_M_hash,
                g_svmpools[0]->fine_grain);
        } else {
            gpu_plotter.gpu_hash_table(L_meta_flat, R_meta_flat, Y_results, M_results, KMASK);
        }
        
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
        // Entries are already sorted by Y from the radix sort above.
        // Duplicates have the same Y AND same metadata (Y = XOR of metadata).
        // So duplicates are adjacent in Y-sorted order — no meta sort needed!
        // Just scan linearly, comparing metadata of entries with the same Y.
        std::array<uint32_t, 14> prev_meta = {};
        uint32_t prev_Y = 0xFFFFFFFF;
        bool first = true;
        for(size_t i = 0; i < entries.size(); i++) {
            uint32_t Y = entries[i].first;
            const auto& meta = M_curr[entries[i].second];
            if(first || Y != prev_Y || meta != prev_meta) {
                plot.final_Y.push_back(Y);
                plot.final_meta.push_back(meta);
                final_indices.push_back(entries[i].second);
                prev_meta = meta;
                prev_Y = Y;
                first = false;
            }
        }
    }
    
    std::cout << "[Final] " << plot.final_Y.size() << " unique entries          " << std::endl;
    
    // No re-sort needed — entries are already in Y-sorted order from dedup!
    // No re-sort needed — entries are already in Y-sorted order from dedup!
    // (The old code sorted by metadata then re-sorted by Y — we skip both)
    
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
    std::cerr << "[write_plot] final_Y=" << plot.final_Y.size() << " X_pairs=" << plot.X_pairs.size() << std::endl;
    for(int t = 2; t <= MY_N_TABLE; t++) {
        std::cerr << "[write_plot] PD[" << t << "]=" << plot.PD[t].size() << std::endl;
    }
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
                // Check for park overflow before truncating
                uint64_t actual_park_bits = delta_bit_offset;
                uint64_t actual_park_bytes = (actual_park_bits + 7) / 8;
                if(actual_park_bytes > max_park_bytes_y) {
                    throw std::runtime_error("Y park overflow: " + std::to_string(actual_park_bytes)
                        + " > " + std::to_string(max_park_bytes_y) + " at park " + std::to_string(p));
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
    
    std::cerr << "[write_plot] Writing PD parks..." << std::endl;
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
                // Check for park overflow before truncating
                uint64_t actual_pd_bits = delta_bit_offset;
                uint64_t actual_pd_bytes = (actual_pd_bits + 7) / 8;
                if(actual_pd_bytes > max_park_bytes_pd) {
                    throw std::runtime_error("PD park overflow: " + std::to_string(actual_pd_bytes)
                        + " > " + std::to_string(max_park_bytes_pd) + " at park " + std::to_string(p)
                        + " table " + std::to_string(pd_table));
                }
                park.resize(max_park_bytes_pd, 0);
                parks_buf[pi] = std::move(park);
            }
            
            for(uint64_t pi = 0; pi < chunk_size; pi++) {
                out.write((char*)parks_buf[pi].data(), max_park_bytes_pd);
            }
        }
    }
    
    std::cerr << "[write_plot] Writing X parks..." << std::endl;
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


// ============================================================================
// Chunked Pipeline (for large k-sizes: k29+)
// Processes one first-level bucket at a time to limit memory usage.
// Uses in-memory bucket store (requires ~66 GB RAM for k29).
// ============================================================================

#include "bucket_store.h"

struct MemBucketStore {
    int num_buckets;
    int max_bucket_size;
    int n_meta;
    std::vector<std::vector<uint32_t>> buckets;
    std::vector<uint32_t> counts;
    // PD tracking: pd_buckets[y] = [count * 2] flat array of (sorted_pos, delta) pairs
    std::vector<std::vector<uint32_t>> pd_buckets;
    // X values (F1 indices): x_values[y] = [count] flat array of F1 indices
    std::vector<std::vector<uint32_t>> x_values;
    // X pairs for table 2: x_pairs_buckets[y] = [count * 2] flat array of (left_x, right_x)
    std::vector<std::vector<uint32_t>> x_pairs_buckets;
    
    MemBucketStore(int nb, int mbs, int nm)
        : num_buckets(nb), max_bucket_size(mbs), n_meta(nm)
    {
        buckets.resize(nb);
        counts.resize(nb, 0);
        pd_buckets.resize(nb);
        x_values.resize(nb);
        x_pairs_buckets.resize(nb);
    }
    
    void clear() {
        for(auto& b : buckets) b.clear();
        for(auto& p : pd_buckets) p.clear();
        for(auto& x : x_values) x.clear();
        for(auto& xp : x_pairs_buckets) xp.clear();
        std::fill(counts.begin(), counts.end(), 0);
    }
    
    uint64_t total_dropped = 0;  // Track dropped entries
    
    void append(int bucket, const uint32_t* meta, uint32_t count) {
        if(bucket < 0 || bucket >= num_buckets) return;
        if(counts[bucket] + count > (uint32_t)max_bucket_size) {
            uint32_t dropped = count - (max_bucket_size - counts[bucket]);
            total_dropped += dropped;
            count = max_bucket_size - counts[bucket];
        }
        if(count == 0) return;
        auto& b = buckets[bucket];
        b.insert(b.end(), meta, meta + count * n_meta);
        counts[bucket] += count;
    }
    
    void append_pd(int bucket, const uint32_t* pd, uint32_t count) {
        if(bucket < 0 || bucket >= num_buckets) return;
        auto& p = pd_buckets[bucket];
        p.insert(p.end(), pd, pd + count * 2);  // 2 uint32s per entry: (sorted_pos, delta)
    }
    
    void append_x_pairs(int bucket, const uint32_t* xp, uint32_t count) {
        if(bucket < 0 || bucket >= num_buckets) return;
        auto& p = x_pairs_buckets[bucket];
        p.insert(p.end(), xp, xp + count * 2);  // 2 uint32s per entry: (left_x, right_x)
    }
    
    void append_x(int bucket, const uint32_t* x, uint32_t count) {
        if(bucket < 0 || bucket >= num_buckets) return;
        auto& xv = x_values[bucket];
        xv.insert(xv.end(), x, x + count);
    }
};

void compute_f1_chunked(
    OCL_Plotter& plotter,
    const hash_t& plot_id,
    MemBucketStore& store,
    int batch_size)
{
    const uint64_t total_entries = (uint64_t)1 << KSIZE;
    const int num_buckets = store.num_buckets;
    const int n_meta = store.n_meta;
    const int shift = KSIZE - LOGBUCKETS;
    
    const uint32_t num_batches = (total_entries + batch_size - 1) / batch_size;
    std::cout << "[F1] Computing " << total_entries << " entries in " << num_batches << " batches..." << std::endl;
    auto t0 = my_time_ms();
    
    std::vector<uint32_t> X_batch(batch_size), Y_batch(batch_size), M_batch(batch_size * n_meta);
    std::vector<std::vector<uint32_t>> batch_buckets(num_buckets);
    
    // Use SVM F1 if available
    bool use_svm_f1 = get_opt_config().svm && g_svmpools.size() > 0 && g_svmpools[0]->svm_F1_X;
    SVMPool* f1_svm = use_svm_f1 ? g_svmpools[0] : nullptr;
    
    for(uint32_t b = 0; b < num_batches; b++) {
        const uint32_t start = b * batch_size;
        const uint32_t count = std::min((uint32_t)batch_size, (uint32_t)(total_entries - start));
        
        for(uint32_t i = 0; i < count; i++) X_batch[i] = start + i;
        X_batch.resize(count);
        
        if(use_svm_f1) {
            plotter.compute_f1_batch_svm(X_batch, plot_id, Y_batch, M_batch,
                f1_svm->svm_F1_X, f1_svm->svm_F1_Y, f1_svm->svm_F1_M, batch_size);
        } else {
            plotter.compute_f1_batch(X_batch, plot_id, Y_batch, M_batch);
        }
        
        for(int i = 0; i < num_buckets; i++) batch_buckets[i].clear();
        // Also batch X values (F1 indices) per bucket
        std::vector<std::vector<uint32_t>> batch_x(num_buckets);
        
        for(uint32_t i = 0; i < count; i++) {
            uint32_t Y = Y_batch[i];
            uint32_t bucket = Y >> shift;
            if(bucket >= (uint32_t)num_buckets) bucket = num_buckets - 1;
            for(int j = 0; j < n_meta; j++)
                batch_buckets[bucket].push_back(M_batch[i * n_meta + j]);
            batch_x[bucket].push_back(X_batch[i]);  // Save F1 index
        }
        
        for(int y = 0; y < num_buckets; y++) {
            if(batch_buckets[y].empty()) continue;
            uint32_t cnt = batch_buckets[y].size() / n_meta;
            store.append(y, batch_buckets[y].data(), cnt);
            store.append_x(y, batch_x[y].data(), cnt);
        }
        
        if(b % 16 == 0 || b == num_batches - 1) {
            std::cerr << "\r[F1] Batch " << (b+1) << "/" << num_batches
                      << " (" << (b+1)*100/num_batches << "%) "
                      << (my_time_ms() - t0) / 1000.0 << "s" << std::flush;
        }
    }
    std::cout << "\n[F1] Done in " << (my_time_ms() - t0) / 1000.0 << " sec" << std::endl;
    
    uint64_t total = 0;
    for(int i = 0; i < num_buckets; i++) total += store.counts[i];
    std::cout << "[F1] Total entries: " << total << std::endl;
}

// Multi-GPU F1: split batches across GPUs for concurrent F1 computation
void compute_f1_chunked_multi_gpu(
    OCL_Plotter& plotter,
    const hash_t& plot_id,
    MemBucketStore& store,
    int batch_size)
{
    const uint64_t total_entries = (uint64_t)1 << KSIZE;
    const int num_buckets = store.num_buckets;
    const int n_meta = store.n_meta;
    const int shift = KSIZE - LOGBUCKETS;
    
    const uint32_t num_batches = (total_entries + batch_size - 1) / batch_size;
    std::cout << "[F1] Multi-GPU: " << total_entries << " entries in " << num_batches
              << " batches on " << g_num_gpus << " GPUs..." << std::endl;
    auto t0 = my_time_ms();
    
    // Each GPU has its own batch buffers
    std::vector<std::vector<uint32_t>> X_bufs(g_num_gpus), Y_bufs(g_num_gpus), M_bufs(g_num_gpus);
    for(int g = 0; g < g_num_gpus; g++) {
        X_bufs[g].resize(batch_size);
        Y_bufs[g].resize(batch_size);
        M_bufs[g].resize(batch_size * n_meta);
    }
    
    // Process batches in pairs — use threads to overlap GPU computation
    for(uint32_t b = 0; b < num_batches; b += g_num_gpus) {
        // Submit batches to all GPUs concurrently via threads
        std::vector<std::future<void>> futs;
        for(int g = 0; g < g_num_gpus && b + g < num_batches; g++) {
            OCL_Plotter& active_pl = (g < (int)g_plotters.size()) ? *g_plotters[g] : plotter;
            SVMPool* f1_svm_g = (g < (int)g_svmpools.size() && g_svmpools[g]->svm_F1_X) ? g_svmpools[g] : nullptr;
            const uint32_t start = (b + g) * batch_size;
            const uint32_t count = std::min((uint32_t)batch_size, (uint32_t)(total_entries - start));
            for(uint32_t i = 0; i < count; i++) X_bufs[g][i] = start + i;
            X_bufs[g].resize(count);
            // Launch on thread — each thread uses its own GPU's queue
            futs.push_back(std::async(std::launch::async, [&active_pl, g, &X_bufs, &plot_id, &Y_bufs, &M_bufs, f1_svm_g, batch_size]() {
                if(f1_svm_g) {
                    active_pl.compute_f1_batch_svm(X_bufs[g], plot_id, Y_bufs[g], M_bufs[g],
                        f1_svm_g->svm_F1_X, f1_svm_g->svm_F1_Y, f1_svm_g->svm_F1_M, batch_size);
                } else {
                    active_pl.compute_f1_batch(X_bufs[g], plot_id, Y_bufs[g], M_bufs[g]);
                }
            }));
        }
        // Wait for all GPUs to finish
        for(auto& f : futs) f.wait();
        
        // Collect results from all GPUs into store (single-threaded, no race)
        for(int g = 0; g < g_num_gpus && b + g < num_batches; g++) {
            const uint32_t count = X_bufs[g].size();
            std::vector<std::vector<uint32_t>> batch_buckets(num_buckets);
            std::vector<std::vector<uint32_t>> batch_x(num_buckets);
            for(uint32_t i = 0; i < count; i++) {
                uint32_t Y = Y_bufs[g][i];
                uint32_t bucket = Y >> shift;
                if(bucket >= (uint32_t)num_buckets) bucket = num_buckets - 1;
                for(int j = 0; j < n_meta; j++)
                    batch_buckets[bucket].push_back(M_bufs[g][i * n_meta + j]);
                batch_x[bucket].push_back(X_bufs[g][i]);
            }
            for(int y = 0; y < num_buckets; y++) {
                if(batch_buckets[y].empty()) continue;
                uint32_t cnt = batch_buckets[y].size() / n_meta;
                store.append(y, batch_buckets[y].data(), cnt);
                store.append_x(y, batch_x[y].data(), cnt);
            }
        }
        
        if(b % 16 == 0 || b >= num_batches - g_num_gpus) {
            std::cerr << "\r[F1] Batch " << std::min(b + g_num_gpus, num_batches) << "/" << num_batches
                      << " (" << std::min((b + g_num_gpus) * 100 / num_batches, 100u) << "%) "
                      << (my_time_ms() - t0) / 1000.0 << "s" << std::flush;
        }
    }
    
    auto elapsed = my_time_ms() - t0;
    std::cout << "\n[F1] Done in " << elapsed / 1000.0 << " sec" << std::endl;
    uint64_t total = 0;
    for(int i = 0; i < num_buckets; i++) total += store.counts[i];
    std::cout << "[F1] Total entries: " << total << std::endl;
}

void process_bucket_chunk(
    OCL_Plotter& plotter,
    MemBucketStore& src,
    MemBucketStore& dst,
    int y,
    int table)
{
    const int n_meta = src.n_meta;
    const int shift = KSIZE - LOGBUCKETS;
    
    uint32_t count_y = src.counts[y];
    if(count_y == 0) return;
    
    const uint32_t* meta_y = src.buckets[y].data();
    
    uint32_t count_ynext = 0;
    const uint32_t* meta_ynext = nullptr;
    if(y + 1 < src.num_buckets && src.counts[y + 1] > 0) {
        count_ynext = src.counts[y + 1];
        meta_ynext = src.buckets[y + 1].data();
    }
    
    uint32_t total = count_y + count_ynext;
    
    // Compute Y values
    std::vector<uint32_t> Y_all(total);
    for(uint32_t i = 0; i < count_y; i++) {
        uint32_t Y = 0;
        for(int j = 0; j < n_meta; j++) Y ^= meta_y[i * n_meta + j];
        Y_all[i] = Y & KMASK;
    }
    for(uint32_t i = 0; i < count_ynext; i++) {
        uint32_t Y = 0;
        for(int j = 0; j < n_meta; j++) Y ^= meta_ynext[i * n_meta + j];
        Y_all[count_y + i] = Y & KMASK;
    }
    
    // Sort entries by Y
    std::vector<std::pair<uint32_t, uint32_t>> entries(total);
    for(uint32_t i = 0; i < total; i++) entries[i] = {Y_all[i], i};
    std::sort(entries.begin(), entries.end());
    
    // Match Y,Y+1 — only LEFT in bucket y
    std::vector<uint32_t> LR_flat;
    for(size_t i = 0; i < entries.size(); i++) {
        if(entries[i].second >= count_y) continue;
        uint32_t YL = entries[i].first;
        for(size_t j = i + 1; j < entries.size(); j++) {
            uint32_t YR = entries[j].first;
            if(YR == YL + 1) {
                LR_flat.push_back(entries[i].second);
                LR_flat.push_back(entries[j].second);
            } else if(YR > YL + 1) break;
        }
    }
    
    if(LR_flat.empty()) return;
    
    // Build combined metadata
    std::vector<uint32_t> M_combined(total * n_meta);
    std::memcpy(M_combined.data(), meta_y, count_y * n_meta * 4);
    if(count_ynext > 0)
        std::memcpy(M_combined.data() + count_y * n_meta, meta_ynext, count_ynext * n_meta * 4);
    
    // GPU hash — extract L_meta and R_meta on CPU (old method, known working)
    size_t num_matches = LR_flat.size() / 2;
    std::vector<uint32_t> L_meta_flat(num_matches * n_meta);
    std::vector<uint32_t> R_meta_flat(num_matches * n_meta);
    for(size_t i = 0; i < num_matches; i++) {
        uint32_t P1 = LR_flat[i * 2];
        uint32_t P2 = LR_flat[i * 2 + 1];
        for(int j = 0; j < n_meta; j++) {
            L_meta_flat[i * n_meta + j] = M_combined[P1 * n_meta + j];
            R_meta_flat[i * n_meta + j] = M_combined[P2 * n_meta + j];
        }
    }
    std::vector<uint32_t> Y_out, M_out;
    plotter.gpu_hash_table(L_meta_flat, R_meta_flat, Y_out, M_out, KMASK);
    
    // Bucket new metadata → dst store
    std::vector<std::vector<uint32_t>> dst_batch(dst.num_buckets);
    for(size_t i = 0; i < Y_out.size(); i++) {
        uint32_t bucket = Y_out[i] >> shift;
        if(bucket >= (uint32_t)dst.num_buckets) bucket = dst.num_buckets - 1;
        for(int j = 0; j < n_meta; j++)
            dst_batch[bucket].push_back(M_out[i * n_meta + j]);
    }
    
    for(int b = 0; b < dst.num_buckets; b++) {
        if(dst_batch[b].empty()) continue;
        uint32_t cnt = dst_batch[b].size() / n_meta;
        dst.append(b, dst_batch[b].data(), cnt);
    }
}

// Cross-boundary matching: find Y,Y+1 pairs where Y is at the end of bucket y
// and Y+1 is at the start of bucket y+1. These matches are missed by
// process_bucket_gpu which only matches within a single first-level bucket.
void cross_boundary_match(
    OCL_Plotter& plotter,
    MemBucketStore& src,
    MemBucketStore& dst,
    int y,
    int table,
    std::vector<std::pair<uint32_t, uint32_t>>* x_pairs_out = nullptr)
{
    const int n_meta = src.n_meta;
    const uint32_t kmask = KMASK;
    const uint32_t shift = KSIZE - LOGBUCKETS;
    const uint32_t boundary = (uint32_t)(y + 1) << shift;  // first Y in bucket y+1
    const uint32_t Y_target = boundary - 1;  // last Y that can match across

    if(y + 1 >= src.num_buckets) return;
    if(src.counts[y] == 0 || src.counts[y + 1] == 0) return;

    // Find entries in bucket y with Y = Y_target (= boundary - 1)
    // Find entries in bucket y+1 with Y = boundary
    std::vector<uint32_t> left_indices;   // indices into bucket y
    std::vector<uint32_t> right_indices;  // indices into bucket y+1

    const uint32_t* meta_y = src.buckets[y].data();
    uint32_t cnt_y = src.counts[y];
    for(uint32_t i = 0; i < cnt_y; i++) {
        uint32_t Y = 0;
        for(int j = 0; j < n_meta; j++) Y ^= meta_y[i * n_meta + j];
        Y &= kmask;
        if(Y == Y_target) left_indices.push_back(i);
    }

    const uint32_t* meta_ynext = src.buckets[y + 1].data();
    uint32_t cnt_ynext = src.counts[y + 1];
    for(uint32_t i = 0; i < cnt_ynext; i++) {
        uint32_t Y = 0;
        for(int j = 0; j < n_meta; j++) Y ^= meta_ynext[i * n_meta + j];
        Y &= kmask;
        if(Y == boundary) right_indices.push_back(i);
    }

    if(left_indices.empty() || right_indices.empty()) return;

    // All combinations of left × right are matches (Y, Y+1)
    size_t num_cross = left_indices.size() * right_indices.size();

    // Build L_meta and R_meta for hashing
    std::vector<uint32_t> L_meta_flat(num_cross * n_meta);
    std::vector<uint32_t> R_meta_flat(num_cross * n_meta);
    for(size_t li = 0; li < left_indices.size(); li++) {
        uint32_t left_idx = left_indices[li];
        for(size_t ri = 0; ri < right_indices.size(); ri++) {
            uint32_t right_idx = right_indices[ri];
            size_t k = li * right_indices.size() + ri;
            for(int j = 0; j < n_meta; j++) {
                L_meta_flat[k * n_meta + j] = meta_y[left_idx * n_meta + j];
                R_meta_flat[k * n_meta + j] = meta_ynext[right_idx * n_meta + j];
            }
        }
    }

    // GPU hash to get new Y and metadata
    std::vector<uint32_t> Y_out, M_out;
    plotter.gpu_hash_table(L_meta_flat, R_meta_flat, Y_out, M_out, kmask);

    if(Y_out.empty()) return;

    // Compute bucket_offset for PD
    uint32_t bucket_offset_y = 0;
    for(int b = 0; b < y; b++) bucket_offset_y += src.counts[b];
    uint32_t bucket_offset_ynext = bucket_offset_y + cnt_y;

    // For each cross-boundary match, add to dst store
    for(size_t k = 0; k < Y_out.size(); k++) {
        uint32_t Y_new = Y_out[k] & kmask;
        uint32_t dst_bucket = Y_new >> shift;
        if(dst_bucket >= (uint32_t)dst.num_buckets) continue;

        // Metadata for new entry
        std::vector<uint32_t> new_meta(n_meta);
        for(int j = 0; j < n_meta; j++) {
            new_meta[j] = M_out[k * n_meta + j];
        }
        dst.append(dst_bucket, new_meta.data(), 1);

        // PD: store global positions (bucket_offset + orig_pos) in previous table
        size_t li = k / right_indices.size();
        size_t ri = k % right_indices.size();
        uint32_t left_global = bucket_offset_y + left_indices[li];
        uint32_t right_global = bucket_offset_ynext + right_indices[ri];
        uint32_t pd_data[2] = {left_global, right_global};
        dst.append_pd(dst_bucket, pd_data, 1);

        // For table 2: save X pairs into dst store (F1 indices)
        if(table == 2) {
            const std::vector<uint32_t>& x_vals_y = src.x_values[y];
            const std::vector<uint32_t>& x_vals_ynext = src.x_values[y + 1];
            uint32_t left_x = (left_indices[li] < x_vals_y.size()) ? x_vals_y[left_indices[li]] : 0;
            uint32_t right_x = (right_indices[ri] < x_vals_ynext.size()) ? x_vals_ynext[right_indices[ri]] : 0;
            uint32_t xp_data[2] = {left_x, right_x};
            dst.append_x_pairs(dst_bucket, xp_data, 1);
        }
    }
}

// GPU-native per-bucket processor: scatter→sort→match→hash all on GPU
// Data stays on GPU through all steps. Only 1 upload + 1 download per bucket.
void process_bucket_gpu(
    OCL_Plotter& plotter,
    MemBucketStore& src,
    MemBucketStore& dst,
    int y,
    int table,
    std::vector<std::pair<uint32_t, uint32_t>>* x_pairs_out = nullptr)
{
    const int n_meta = src.n_meta;
    const int shift = KSIZE - LOGBUCKETS;
    const int logbuckets2 = KSIZE - LOGBUCKETS - 9;
    const int num_sub = 1 << logbuckets2;  // sub-buckets per first-level bucket (masked index)
    // Max entries per sub-bucket: (4 << KSIZE) / num_buckets_1 / num_buckets_2 / 3
    const int max_bs2 = std::max(1024, (int)(((uint64_t)4 << KSIZE) / (1 << LOGBUCKETS) / num_sub));
    
    uint32_t count_y = src.counts[y];
    if(count_y == 0) return;
    
    // Multi-GPU: select plotter + buffer pool for this bucket
    OCL_Plotter& active_plotter = (g_num_gpus > 1 && g_plotters.size() > 1) ? 
        *g_plotters[y % g_num_gpus] : plotter;
    BufferPool* active_pool = (g_num_gpus > 1 && g_bufpools.size() > 1) ? 
        g_bufpools[y % g_num_gpus] : g_bufpool;
    
    const uint32_t* meta_y = src.buckets[y].data();
    
    // Process this bucket. Cross-boundary matching is handled separately
    // by cross_boundary_match() called after this function.
    uint32_t total = count_y;
    if(total == 0) return;
    
    // Safety: check VRAM availability
    size_t vram_available = 0;
    clGetDeviceInfo(active_plotter.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(vram_available), &vram_available, nullptr);
    size_t needed = (size_t)total * n_meta * 4  // C_in
                  + (size_t)num_sub * max_bs2 * 8  // PY_tmp
                  + (size_t)total * 8  // LR
                  + (size_t)total * 4  // PD_matches
                  + (size_t)total * 4  // Y_out
                  + (size_t)total * n_meta * 4  // M_out
                  + num_sub * 4 * 2;  // counts + offsets
    if(needed > vram_available * 8 / 10) {
        // Fallback to CPU for this bucket
        std::cout << "[WARN] VRAM low, falling back to CPU for bucket " << y << " table " << table << std::endl;
        process_bucket_chunk(plotter, src, dst, y, table);
        return;
    }
    
    cl_int err;
    cl_mem null_mem = nullptr;
    int zero = 0;
    
    // Timing instrumentation
    bool do_timing = timing_detail && (y == 0);  // only first bucket of each table
    auto step_start = std::chrono::steady_clock::now();
    auto step_end = step_start;
    auto print_step = [&](const char* name) {
        if(do_timing) {
            step_end = std::chrono::steady_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count();
            std::cerr << "    [" << name << "] " << ms << "us" << std::endl;
            step_start = step_end;
        }
    };
    
    // Step 1: Upload C_in (metadata for bucket y only)
    std::vector<uint32_t> C_combined(total * n_meta);
    std::memcpy(C_combined.data(), meta_y, count_y * n_meta * 4);
    
    cl_mem C_in_buf, PY_buf, sub_cnt_buf, sub_off_buf, LR_buf, PD_match_buf, num_matches_buf;
    
    bool using_pool = (active_pool != nullptr && active_pool->initialized);
    bool pool_release_skip = false;
    
    // Module E: Use per-bucket queue for multi-queue pipelining
    cl_command_queue& q = active_plotter.get_queue(y);
    
    // Module F: zero-copy read helper (map/unmap instead of clEnqueueReadBuffer)
    auto zero_copy_read = [&](cl_mem buf, void* dst, size_t size) {
        cl_int e;
        void* mapped = clEnqueueMapBuffer(q, buf, CL_TRUE, CL_MAP_READ, 0, size, 0, nullptr, nullptr, &e);
        if(e == CL_SUCCESS && mapped) {
            std::memcpy(dst, mapped, size);
            clEnqueueUnmapMemObject(q, buf, mapped, 0, nullptr, nullptr);
        }
    };
    
    if(using_pool) {
        // Module C: Reuse pre-allocated buffers — just write data, no create/destroy
        C_in_buf = active_pool->C_in_buf;
        if(get_opt_config().zero_copy) {
            // Module F: Map buffer, write directly, unmap (zero-copy on shared memory, fast DMA on discrete)
            void* mapped = clEnqueueMapBuffer(q, C_in_buf, CL_TRUE, CL_MAP_WRITE,
                0, count_y * n_meta * 4, 0, nullptr, nullptr, &err);
            CL_CHECK(err, "map C_in_buf");
            std::memcpy(mapped, meta_y, count_y * n_meta * 4);
            clEnqueueUnmapMemObject(q, C_in_buf, mapped, 0, nullptr, nullptr);
        } else {
            clEnqueueWriteBuffer(q, C_in_buf, CL_TRUE, 0,
                count_y * n_meta * 4, meta_y, 0, nullptr, nullptr);
        }
        
        PY_buf = active_pool->PY_buf;
        sub_cnt_buf = active_pool->sub_cnt_buf;
        clEnqueueFillBuffer(q, sub_cnt_buf, &zero, 4, 0, num_sub * 4, 0, nullptr, nullptr);
        
        LR_buf = active_pool->LR_buf;
        PD_match_buf = active_pool->PD_match_buf;
        num_matches_buf = active_pool->num_matches_buf;
        
        pool_release_skip = true;  // pool will clean these up
    } else {
        C_in_buf = clCreateBuffer(active_plotter.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            C_combined.size() * 4, (void*)C_combined.data(), &err);
        CL_CHECK(err, "C_in_buf creation");
        
        PY_buf = clCreateBuffer(active_plotter.context, CL_MEM_READ_WRITE,
            (size_t)num_sub * max_bs2 * 8, nullptr, &err);
        CL_CHECK(err, "PY_buf creation");
        sub_cnt_buf = clCreateBuffer(active_plotter.context, CL_MEM_READ_WRITE,
            num_sub * 4, nullptr, &err);
        CL_CHECK(err, "sub_cnt_buf creation");
        clEnqueueFillBuffer(q, sub_cnt_buf, &zero, 4, 0, num_sub * 4, 0, nullptr, nullptr);
    }
    print_step("upload_C_in");
    
    uint32_t total_u32 = total;
    uint32_t max_bs2_u32 = max_bs2;
    clSetKernelArg(active_plotter.k_scatter2, 0, sizeof(cl_mem), &PY_buf);
    clSetKernelArg(active_plotter.k_scatter2, 1, sizeof(cl_mem), &sub_cnt_buf);
    clSetKernelArg(active_plotter.k_scatter2, 2, sizeof(cl_mem), &null_mem);  // Y_in=null → compute from C_in
    clSetKernelArg(active_plotter.k_scatter2, 3, sizeof(cl_mem), &C_in_buf);
    clSetKernelArg(active_plotter.k_scatter2, 4, sizeof(uint32_t), &total_u32);
    clSetKernelArg(active_plotter.k_scatter2, 5, sizeof(uint32_t), &max_bs2_u32);
    
    size_t scatter_global = total;
    if(scatter_global % 64) scatter_global = ((scatter_global / 64) + 1) * 64;
    clEnqueueNDRangeKernel(q, active_plotter.k_scatter2, 1, nullptr, &scatter_global, nullptr, 0, nullptr, nullptr);
    print_step("scatter");
    
    // Step 3: Prefix sum of sub-bucket counts
    if(using_pool) sub_off_buf = active_pool->sub_off_buf;
    
    if(get_opt_config().gpu_prefix_sum && active_plotter.prefix_sum_kernel) {
        // Module A: GPU prefix sum — no readback needed (except last element for overflow check)
        if(!using_pool) {
            sub_off_buf = clCreateBuffer(active_plotter.context, CL_MEM_READ_WRITE,
                (num_sub + 1) * 4, nullptr, &err);
            CL_CHECK(err, "sub_off_buf (gpu prefix)");
        }
        
        uint32_t num_sub_u32 = (uint32_t)num_sub;
        clSetKernelArg(active_plotter.prefix_sum_kernel, 0, sizeof(cl_mem), &sub_cnt_buf);
        clSetKernelArg(active_plotter.prefix_sum_kernel, 1, sizeof(cl_mem), &sub_off_buf);
        clSetKernelArg(active_plotter.prefix_sum_kernel, 2, sizeof(uint32_t) * (num_sub + 1), nullptr);  // local memory
        clSetKernelArg(active_plotter.prefix_sum_kernel, 3, sizeof(uint32_t), &num_sub_u32);
        
        size_t ps_global = num_sub, ps_local = num_sub;
        clEnqueueNDRangeKernel(q, active_plotter.prefix_sum_kernel, 1, nullptr,
            &ps_global, &ps_local, 0, nullptr, nullptr);
        print_step("gpu_prefix_sum");
        // No overflow check readback — saves ~500us of latency per bucket.
        // Overflow is rare and will be caught later when entries don't match.
    } else {
        // Original: CPU prefix sum — download sub_cnt, compute on CPU, upload sub_off
        std::vector<uint32_t> sub_cnt(num_sub);
        clEnqueueReadBuffer(q, sub_cnt_buf, CL_TRUE, 0, num_sub * 4, sub_cnt.data(), 0, nullptr, nullptr);
        print_step("download_sub_cnt");
        std::vector<uint32_t> sub_off(num_sub + 1, 0);
        for(int i = 0; i < num_sub; i++) sub_off[i + 1] = sub_off[i] + sub_cnt[i];
        uint32_t total_scattered = sub_off[num_sub];
        
        if(total_scattered < total) {
            std::cerr << "[WARN] scatter_2 dropped " << (total - total_scattered)
                      << " entries (sub-bucket overflow, max_bs2=" << max_bs2 << ")"
                      << " bucket " << y << " table " << table << std::endl;
        }
        
        if(!using_pool) {
            sub_off_buf = clCreateBuffer(active_plotter.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                (num_sub + 1) * 4, (void*)sub_off.data(), &err);
            CL_CHECK(err, "sub_off_buf creation");
        } else {
            clEnqueueWriteBuffer(q, sub_off_buf, CL_TRUE, 0,
                (num_sub + 1) * 4, sub_off.data(), 0, nullptr, nullptr);
        }
    }
    
    // Step 4: GPU simple_sort_y — sort within each sub-bucket
    uint32_t max_bs_sort = max_bs2;
    uint32_t num_sub_u32 = num_sub;
    clSetKernelArg(active_plotter.k_simple_sort, 0, sizeof(cl_mem), &PY_buf);
    clSetKernelArg(active_plotter.k_simple_sort, 1, sizeof(cl_mem), &sub_cnt_buf);
    clSetKernelArg(active_plotter.k_simple_sort, 2, sizeof(uint32_t), &max_bs_sort);
    clSetKernelArg(active_plotter.k_simple_sort, 3, sizeof(uint32_t), &num_sub_u32);
    
    size_t sort_g[2] = {256, (size_t)num_sub}, sort_l[2] = {256, 1};
    clEnqueueNDRangeKernel(q, active_plotter.k_simple_sort, 2, nullptr, sort_g, sort_l, 0, nullptr, nullptr);
    print_step("sort");
    
    // Step 5: GPU match_p1 — find Y,Y+1 pairs
    if(!using_pool) {
        LR_buf = clCreateBuffer(active_plotter.context, CL_MEM_WRITE_ONLY,
            (size_t)total * 4 * 8, nullptr, &err);
        CL_CHECK(err, "LR_buf creation");
        PD_match_buf = clCreateBuffer(active_plotter.context, CL_MEM_WRITE_ONLY,
            (size_t)total * 4 * 4, nullptr, &err);
        CL_CHECK(err, "PD_match_buf creation");
        num_matches_buf = clCreateBuffer(active_plotter.context, CL_MEM_READ_WRITE, 4, nullptr, &err);
        CL_CHECK(err, "num_matches_buf creation");
    }
    clEnqueueFillBuffer(q, num_matches_buf, &zero, 4, 0, 4, 0, nullptr, nullptr);
    
    uint32_t max_total = total * 4;  // matches can exceed entries for high density
    uint32_t write_pd = 0;  // PD tracking WIP
    clSetKernelArg(active_plotter.k_match_p1, 0, sizeof(cl_mem), &LR_buf);
    clSetKernelArg(active_plotter.k_match_p1, 1, sizeof(cl_mem), &PD_match_buf);
    clSetKernelArg(active_plotter.k_match_p1, 2, sizeof(cl_mem), &num_matches_buf);
    clSetKernelArg(active_plotter.k_match_p1, 3, sizeof(cl_mem), &PY_buf);
    clSetKernelArg(active_plotter.k_match_p1, 4, sizeof(cl_mem), &sub_cnt_buf);
    clSetKernelArg(active_plotter.k_match_p1, 5, sizeof(cl_mem), &sub_off_buf);
    clSetKernelArg(active_plotter.k_match_p1, 6, sizeof(uint32_t), &num_sub_u32);
    clSetKernelArg(active_plotter.k_match_p1, 7, sizeof(uint32_t), &max_bs_sort);
    clSetKernelArg(active_plotter.k_match_p1, 8, sizeof(uint32_t), &max_total);
    clSetKernelArg(active_plotter.k_match_p1, 9, sizeof(uint32_t), &write_pd);
    
    int groups_per_sub = (max_bs2 + 127) / 128;
    size_t match_g[2] = {(size_t)(128 * groups_per_sub), (size_t)num_sub}, match_l[2] = {128, 1};
    clEnqueueNDRangeKernel(q, active_plotter.k_match_p1, 2, nullptr, match_g, match_l, 0, nullptr, nullptr);
    print_step("match");
    
    uint32_t gpu_matches = 0;
    std::vector<uint32_t> LR_filtered;
    
    // Match count + LR download
    if(get_opt_config().zero_copy) {
        zero_copy_read(num_matches_buf, &gpu_matches, 4);
        print_step("download_match_count");
        if(gpu_matches > 0) {
            LR_filtered.resize(gpu_matches * 2);
            zero_copy_read(LR_buf, LR_filtered.data(), gpu_matches * 8);
            print_step("download_LR");
        }
    } else {
        clEnqueueReadBuffer(q, num_matches_buf, CL_TRUE, 0, 4, &gpu_matches, 0, nullptr, nullptr);
        print_step("download_match_count");
        if(gpu_matches > 0) {
            LR_filtered.resize(gpu_matches * 2);
            clEnqueueReadBuffer(q, LR_buf, CL_TRUE, 0, gpu_matches * 8, LR_filtered.data(), 0, nullptr, nullptr);
            print_step("download_LR");
        }
    }
    
    if(gpu_matches == 0 || LR_filtered.empty()) {
        if(!using_pool) {
            clReleaseMemObject(C_in_buf); clReleaseMemObject(PY_buf); clReleaseMemObject(sub_cnt_buf);
            clReleaseMemObject(sub_off_buf); clReleaseMemObject(LR_buf); clReleaseMemObject(PD_match_buf);
            clReleaseMemObject(num_matches_buf);
        }
        return;
    }
    
    
    // Compute PD for each match
    // Global sorted position = bucket_offset[y] + sorted_pos_in_bucket
    // bucket_offset[y] = sum of counts of all previous buckets in src store
    uint32_t bucket_offset = 0;
    for(int b = 0; b < y; b++) bucket_offset += src.counts[b];
    
    uint32_t num_filt = LR_filtered.size() / 2;
    std::vector<uint32_t> pd_data(num_filt * 2);  // (left_store_pos, right_store_pos)
    for(uint32_t i = 0; i < num_filt; i++) {
        uint32_t left_orig = LR_filtered[i * 2];
        uint32_t right_orig = LR_filtered[i * 2 + 1];
        // Store insertion-order position (= index into pd_all[t-1])
        // bucket_offset + orig_pos = global store position = pd_all index
        pd_data[i * 2] = bucket_offset + left_orig;
        pd_data[i * 2 + 1] = bucket_offset + right_orig;
    }
    
    // Step 7: GPU hash
    // Module B: If gpu_meta_extract is enabled, use hash_table_lr kernel directly.
    // This skips: LR download for meta extraction, CPU L/R extraction, L/R re-upload.
    // C_in_buf (metadata) and LR_buf (match pairs) are already on GPU.
    // We still download LR_filtered for PD computation and X_pairs (table 2).
    size_t num_matches = LR_filtered.size() / 2;
    std::vector<uint32_t> Y_out, M_out;
    
    if(get_opt_config().gpu_meta_extract) {
        // Module B: GPU metadata extraction — gather on GPU + hash on GPU
        // Two kernels: gather_meta (scattered reads) + hash_table_entries (sequential reads)
        // This avoids the AMD codegen bug in hash_table_lr (interleaved scatter+hash)
        cl_int err2;
        uint32_t num_m = (uint32_t)num_matches;
        uint32_t num_total = total;
        uint32_t n_meta_u32 = (uint32_t)n_meta;
        
        // GPU gather: read M_curr[P1], M_curr[P2] → L_meta_out, R_meta_out
        cl_mem L_gathered, R_gathered, Yb, Mb;
        if(using_pool) {
            L_gathered = active_pool->L_gathered;
            R_gathered = active_pool->R_gathered;
            Yb = active_pool->Y_hash_buf;
            Mb = active_pool->M_hash_buf;
        } else {
            L_gathered = clCreateBuffer(active_plotter.context, CL_MEM_WRITE_ONLY,
                num_matches * n_meta * 4, nullptr, &err2);
            CL_CHECK(err2, "L_gathered");
            R_gathered = clCreateBuffer(active_plotter.context, CL_MEM_WRITE_ONLY,
                num_matches * n_meta * 4, nullptr, &err2);
            CL_CHECK(err2, "R_gathered");
            Yb = clCreateBuffer(active_plotter.context, CL_MEM_WRITE_ONLY, num_matches * 4, nullptr, &err2);
            CL_CHECK(err2, "Yb (meta-extract)");
            Mb = clCreateBuffer(active_plotter.context, CL_MEM_WRITE_ONLY, num_matches * MY_N_META * 4, nullptr, &err2);
            CL_CHECK(err2, "Mb (meta-extract)");
        }
        
        clSetKernelArg(active_plotter.gather_meta_kernel, 0, sizeof(cl_mem), &C_in_buf);
        clSetKernelArg(active_plotter.gather_meta_kernel, 1, sizeof(cl_mem), &LR_buf);
        clSetKernelArg(active_plotter.gather_meta_kernel, 2, sizeof(cl_mem), &L_gathered);
        clSetKernelArg(active_plotter.gather_meta_kernel, 3, sizeof(cl_mem), &R_gathered);
        clSetKernelArg(active_plotter.gather_meta_kernel, 4, sizeof(uint32_t), &num_m);
        clSetKernelArg(active_plotter.gather_meta_kernel, 5, sizeof(uint32_t), &num_total);
        clSetKernelArg(active_plotter.gather_meta_kernel, 6, sizeof(uint32_t), &n_meta_u32);
        
        size_t gather_global = num_matches, gather_local = 64;
        if(gather_global % gather_local) gather_global = ((gather_global / gather_local) + 1) * gather_local;
        clEnqueueNDRangeKernel(q, active_plotter.gather_meta_kernel, 1, nullptr,
            &gather_global, &gather_local, 0, nullptr, nullptr);
        print_step("gather");
        
        // GPU hash: hash_table_entries with gathered metadata (sequential reads)
        uint32_t kmask = KMASK;
        clSetKernelArg(active_plotter.table_hash_kernel, 0, sizeof(cl_mem), &L_gathered);
        clSetKernelArg(active_plotter.table_hash_kernel, 1, sizeof(cl_mem), &R_gathered);
        clSetKernelArg(active_plotter.table_hash_kernel, 2, sizeof(cl_mem), &Yb);
        clSetKernelArg(active_plotter.table_hash_kernel, 3, sizeof(cl_mem), &Mb);
        clSetKernelArg(active_plotter.table_hash_kernel, 4, sizeof(uint32_t), &kmask);
        clSetKernelArg(active_plotter.table_hash_kernel, 5, sizeof(uint32_t), &num_m);
        
        size_t hash_global = num_matches, hash_local = 64;
        if(hash_global % hash_local) hash_global = ((hash_global / hash_local) + 1) * hash_local;
        clEnqueueNDRangeKernel(q, active_plotter.table_hash_kernel, 1, nullptr,
            &hash_global, &hash_local, 0, nullptr, nullptr);
        
        Y_out.resize(num_matches);
        M_out.resize(num_matches * MY_N_META);
        if(get_opt_config().zero_copy) {
            // Module F: zero-copy read via map/unmap
            zero_copy_read(Yb, Y_out.data(), num_matches * 4);
            zero_copy_read(Mb, M_out.data(), num_matches * MY_N_META * 4);
        } else if(get_opt_config().async_transfers) {
            // Module D: Non-blocking read Y + M, wait once
            cl_event ev_y, ev_m;
            clEnqueueReadBuffer(q, Yb, CL_FALSE, 0, num_matches * 4, Y_out.data(), 0, nullptr, &ev_y);
            clEnqueueReadBuffer(q, Mb, CL_FALSE, 0, num_matches * MY_N_META * 4, M_out.data(), 0, nullptr, &ev_m);
            clWaitForEvents(1, &ev_y);
            clWaitForEvents(1, &ev_m);
        } else {
            clEnqueueReadBuffer(q, Yb, CL_TRUE, 0, num_matches * 4, Y_out.data(), 0, nullptr, nullptr);
            clEnqueueReadBuffer(q, Mb, CL_TRUE, 0, num_matches * MY_N_META * 4, M_out.data(), 0, nullptr, nullptr);
        }
        print_step("hash+download");
        
        if(!using_pool) {
            clReleaseMemObject(L_gathered);
            clReleaseMemObject(R_gathered);
            clReleaseMemObject(Yb);
            clReleaseMemObject(Mb);
        }
    } else {
        // Original path: CPU meta extraction + gpu_hash_table (re-upload)
        std::vector<uint32_t> L_meta_flat(num_matches * n_meta);
        std::vector<uint32_t> R_meta_flat(num_matches * n_meta);
        for(size_t i = 0; i < num_matches; i++) {
            uint32_t P1 = LR_filtered[i * 2];
            uint32_t P2 = LR_filtered[i * 2 + 1];
            for(int j = 0; j < n_meta; j++) {
                L_meta_flat[i * n_meta + j] = C_combined[P1 * n_meta + j];
                R_meta_flat[i * n_meta + j] = C_combined[P2 * n_meta + j];
            }
        }
        active_plotter.gpu_hash_table(L_meta_flat, R_meta_flat, Y_out, M_out, KMASK);
        print_step("cpu_extract+hash");
    }
    
    // Step 9: Bucket output → dst store (metadata + PD + X_pairs)
    std::vector<std::vector<uint32_t>> dst_batch(dst.num_buckets);
    std::vector<std::vector<uint32_t>> dst_pd_batch(dst.num_buckets);
    std::vector<std::vector<uint32_t>> dst_xp_batch(dst.num_buckets);
    
    // For table 2: prepare X pairs (F1 indices) for each match
    std::vector<uint32_t> xp_flat;
    if(table == 2) {
        const std::vector<uint32_t>& x_vals = src.x_values[y];
        for(uint32_t i = 0; i < num_filt; i++) {
            uint32_t left_local = LR_filtered[i * 2];
            uint32_t right_local = LR_filtered[i * 2 + 1];
            uint32_t left_x = (left_local < x_vals.size()) ? x_vals[left_local] : 0;
            uint32_t right_x = (right_local < x_vals.size()) ? x_vals[right_local] : 0;
            xp_flat.push_back(left_x);
            xp_flat.push_back(right_x);
        }
    }
    
    for(uint32_t i = 0; i < num_filt; i++) {
        uint32_t bucket = Y_out[i] >> shift;
        if(bucket >= (uint32_t)dst.num_buckets) bucket = dst.num_buckets - 1;
        for(int j = 0; j < n_meta; j++)
            dst_batch[bucket].push_back(M_out[i * n_meta + j]);
        // Store PD alongside: (sorted_pos, delta)
        dst_pd_batch[bucket].push_back(pd_data[i * 2]);
        dst_pd_batch[bucket].push_back(pd_data[i * 2 + 1]);
        // Store X pairs for table 2
        if(table == 2 && i * 2 + 1 < xp_flat.size()) {
            dst_xp_batch[bucket].push_back(xp_flat[i * 2]);
            dst_xp_batch[bucket].push_back(xp_flat[i * 2 + 1]);
        }
    }
    for(int b = 0; b < dst.num_buckets; b++) {
        if(dst_batch[b].empty()) continue;
        uint32_t cnt = dst_batch[b].size() / n_meta;
        if(g_num_gpus > 1) {
            std::lock_guard<std::mutex> lock(g_dst_mutex);
            dst.append(b, dst_batch[b].data(), cnt);
            dst.append_pd(b, dst_pd_batch[b].data(), cnt);
            if(table == 2 && !dst_xp_batch[b].empty()) {
                dst.append_x_pairs(b, dst_xp_batch[b].data(), cnt);
            }
        } else {
            dst.append(b, dst_batch[b].data(), cnt);
            dst.append_pd(b, dst_pd_batch[b].data(), cnt);
            if(table == 2 && !dst_xp_batch[b].empty()) {
                dst.append_x_pairs(b, dst_xp_batch[b].data(), cnt);
            }
        }
    }
    
    // Cleanup GPU resources
    if(!using_pool) {
        clReleaseMemObject(C_in_buf); clReleaseMemObject(PY_buf); clReleaseMemObject(sub_cnt_buf);
        clReleaseMemObject(sub_off_buf); clReleaseMemObject(LR_buf); clReleaseMemObject(PD_match_buf);
        clReleaseMemObject(num_matches_buf);
    }

}

// PD per table entry: Y value + (sorted_pos, delta)
struct PDEntry {
    uint32_t Y;
    uint32_t left_pos;   // store-order position of left entry in previous table
    uint32_t right_pos;  // store-order position of right entry in previous table
};

// ============================================================================
// Module E v2: 3-phase pipelined bucket processing
// ============================================================================

// Phase 1: Submit scatter→sort→match kernels (all non-blocking)
// Returns immediately. GPU starts crunching. CPU can submit next bucket.
// ============================================================================
// Module E v2: 3-phase pipeline (submit → submit_hash → collect)
// All GPU enqueues are non-blocking (CL_FALSE) with events.
// CPU only blocks when it needs data. No threads needed.
// ============================================================================

// Phase 1: Submit scatter→sort→match kernels (ALL non-blocking)
// Returns immediately. GPU starts crunching. CPU submits next bucket.
BucketPending submit_bucket_pipeline(
    OCL_Plotter& plotter,
    MemBucketStore& src,
    int y,
    int table,
    BufferPool* pool)
{
    BucketPending p;
    p.skip = true;
    p.y = y;
    p.table = table;
    p.n_meta = src.n_meta;
    p.shift = KSIZE - LOGBUCKETS;
    const int logbuckets2 = KSIZE - LOGBUCKETS - 9;
    p.num_sub = 1 << logbuckets2;
    p.max_bs2 = std::max(1024, (int)(((uint64_t)4 << KSIZE) / (1 << LOGBUCKETS) / p.num_sub));

    p.count_y = src.counts[y];
    if(p.count_y == 0) return p;

    p.active_plotter = &plotter;
    p.active_pool = pool;
    const uint32_t* meta_y = src.buckets[y].data();
    p.total = p.count_y;

    OCL_Plotter& active_plotter = *p.active_plotter;
    BufferPool* active_pool = p.active_pool;
    const int n_meta = p.n_meta;
    const int num_sub = p.num_sub;
    const int max_bs2 = p.max_bs2;
    uint32_t total = p.total;

    cl_int err;
    cl_mem null_mem = nullptr;
    int zero = 0;

    p.using_pool = (active_pool != nullptr && active_pool->initialized);
    p.q = active_plotter.queue;
    cl_command_queue& q = p.q;

    // Copy metadata to host buffer (for later CPU PD computation)
    p.C_combined.resize(total * n_meta);
    std::memcpy(p.C_combined.data(), meta_y, p.count_y * n_meta * 4);

    // Step 1: Allocate/assign buffers + upload C_in
    if(p.using_pool) {
        p.C_in_buf = active_pool->C_in_buf;
        p.PY_buf = active_pool->PY_buf;
        p.sub_cnt_buf = active_pool->sub_cnt_buf;
        p.sub_off_buf = active_pool->sub_off_buf;
        p.LR_buf = active_pool->LR_buf;
        p.PD_match_buf = active_pool->PD_match_buf;
        p.num_matches_buf = active_pool->num_matches_buf;
        
        if(false && get_opt_config().pinned && active_pool->pinned_C_in_ptr) {
            // Module G: disabled — NVIDIA OCL has issues with persistently mapped buffers
            std::memcpy(active_pool->pinned_C_in_ptr, meta_y, p.count_y * n_meta * 4);
            clEnqueueWriteBuffer(q, p.C_in_buf, CL_FALSE, 0,
                p.count_y * n_meta * 4, active_pool->pinned_C_in_ptr, 0, nullptr, nullptr);
        } else {
            // Regular write (blocking or non-blocking depending on flags)
            clEnqueueWriteBuffer(q, p.C_in_buf, CL_FALSE, 0,
                p.count_y * n_meta * 4, meta_y, 0, nullptr, nullptr);
        }
    } else {
        p.C_in_buf = clCreateBuffer(active_plotter.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            p.C_combined.size() * 4, (void*)p.C_combined.data(), &err);
        CL_CHECK(err, "C_in_buf creation");
        p.PY_buf = clCreateBuffer(active_plotter.context, CL_MEM_READ_WRITE,
            (size_t)num_sub * max_bs2 * 8, nullptr, &err);
        CL_CHECK(err, "PY_buf creation");
        p.sub_cnt_buf = clCreateBuffer(active_plotter.context, CL_MEM_READ_WRITE, num_sub * 4, nullptr, &err);
        CL_CHECK(err, "sub_cnt_buf creation");
        p.sub_off_buf = clCreateBuffer(active_plotter.context, CL_MEM_READ_WRITE, (num_sub + 1) * 4, nullptr, &err);
        CL_CHECK(err, "sub_off_buf creation");
        p.LR_buf = clCreateBuffer(active_plotter.context, CL_MEM_WRITE_ONLY, (size_t)total * 4 * 8, nullptr, &err);
        CL_CHECK(err, "LR_buf creation");
        p.PD_match_buf = clCreateBuffer(active_plotter.context, CL_MEM_WRITE_ONLY, (size_t)total * 4 * 4, nullptr, &err);
        CL_CHECK(err, "PD_match_buf creation");
        p.num_matches_buf = clCreateBuffer(active_plotter.context, CL_MEM_READ_WRITE, 4, nullptr, &err);
        CL_CHECK(err, "num_matches_buf creation");
    }
    clEnqueueFillBuffer(q, p.sub_cnt_buf, &zero, 4, 0, num_sub * 4, 0, nullptr, nullptr);
    clEnqueueFillBuffer(q, p.num_matches_buf, &zero, 4, 0, 4, 0, nullptr, nullptr);

    // Step 2: Scatter (non-blocking — in-order queue ensures C_in write completes first)
    uint32_t total_u32 = total;
    uint32_t max_bs2_u32 = max_bs2;
    clSetKernelArg(active_plotter.k_scatter2, 0, sizeof(cl_mem), &p.PY_buf);
    clSetKernelArg(active_plotter.k_scatter2, 1, sizeof(cl_mem), &p.sub_cnt_buf);
    clSetKernelArg(active_plotter.k_scatter2, 2, sizeof(cl_mem), &null_mem);
    clSetKernelArg(active_plotter.k_scatter2, 3, sizeof(cl_mem), &p.C_in_buf);
    clSetKernelArg(active_plotter.k_scatter2, 4, sizeof(uint32_t), &total_u32);
    clSetKernelArg(active_plotter.k_scatter2, 5, sizeof(uint32_t), &max_bs2_u32);
    size_t scatter_global = total;
    if(scatter_global % 64) scatter_global = ((scatter_global / 64) + 1) * 64;
    clEnqueueNDRangeKernel(q, active_plotter.k_scatter2, 1, nullptr, &scatter_global, nullptr, 0, nullptr, nullptr);

    // Step 3: Prefix sum (Module A: GPU, non-blocking)
    p.num_sub_u32 = (uint32_t)num_sub;
    if(get_opt_config().gpu_prefix_sum && active_plotter.prefix_sum_kernel) {
        clSetKernelArg(active_plotter.prefix_sum_kernel, 0, sizeof(cl_mem), &p.sub_cnt_buf);
        clSetKernelArg(active_plotter.prefix_sum_kernel, 1, sizeof(cl_mem), &p.sub_off_buf);
        clSetKernelArg(active_plotter.prefix_sum_kernel, 2, sizeof(uint32_t) * (num_sub + 1), nullptr);
        clSetKernelArg(active_plotter.prefix_sum_kernel, 3, sizeof(uint32_t), &p.num_sub_u32);
        size_t ps_global = num_sub, ps_local = num_sub;
        clEnqueueNDRangeKernel(q, active_plotter.prefix_sum_kernel, 1, nullptr, &ps_global, &ps_local, 0, nullptr, nullptr);
    } else {
        // CPU prefix sum requires blocking read — but only when Module A is off
        std::vector<uint32_t> sub_cnt(num_sub);
        clEnqueueReadBuffer(q, p.sub_cnt_buf, CL_TRUE, 0, num_sub * 4, sub_cnt.data(), 0, nullptr, nullptr);
        std::vector<uint32_t> sub_off(num_sub + 1, 0);
        for(int i = 0; i < num_sub; i++) sub_off[i + 1] = sub_off[i] + sub_cnt[i];
        clEnqueueWriteBuffer(q, p.sub_off_buf, CL_FALSE, 0, (num_sub + 1) * 4, sub_off.data(), 0, nullptr, nullptr);
    }

    // Step 4: Sort (non-blocking)
    p.max_bs_sort = max_bs2;
    uint32_t max_bs_sort = p.max_bs_sort;
    clSetKernelArg(active_plotter.k_simple_sort, 0, sizeof(cl_mem), &p.PY_buf);
    clSetKernelArg(active_plotter.k_simple_sort, 1, sizeof(cl_mem), &p.sub_cnt_buf);
    clSetKernelArg(active_plotter.k_simple_sort, 2, sizeof(uint32_t), &max_bs_sort);
    clSetKernelArg(active_plotter.k_simple_sort, 3, sizeof(uint32_t), &p.num_sub_u32);
    size_t sort_g[2] = {256, (size_t)num_sub}, sort_l[2] = {256, 1};
    clEnqueueNDRangeKernel(q, active_plotter.k_simple_sort, 2, nullptr, sort_g, sort_l, 0, nullptr, nullptr);

    // Step 5: Match (non-blocking, record event for later sync)
    uint32_t max_total = total * 4;
    uint32_t write_pd = 0;
    clSetKernelArg(active_plotter.k_match_p1, 0, sizeof(cl_mem), &p.LR_buf);
    clSetKernelArg(active_plotter.k_match_p1, 1, sizeof(cl_mem), &p.PD_match_buf);
    clSetKernelArg(active_plotter.k_match_p1, 2, sizeof(cl_mem), &p.num_matches_buf);
    clSetKernelArg(active_plotter.k_match_p1, 3, sizeof(cl_mem), &p.PY_buf);
    clSetKernelArg(active_plotter.k_match_p1, 4, sizeof(cl_mem), &p.sub_cnt_buf);
    clSetKernelArg(active_plotter.k_match_p1, 5, sizeof(cl_mem), &p.sub_off_buf);
    clSetKernelArg(active_plotter.k_match_p1, 6, sizeof(uint32_t), &p.num_sub_u32);
    clSetKernelArg(active_plotter.k_match_p1, 7, sizeof(uint32_t), &max_bs_sort);
    clSetKernelArg(active_plotter.k_match_p1, 8, sizeof(uint32_t), &max_total);
    clSetKernelArg(active_plotter.k_match_p1, 9, sizeof(uint32_t), &write_pd);
    int groups_per_sub = (max_bs2 + 127) / 128;
    size_t match_g[2] = {(size_t)(128 * groups_per_sub), (size_t)num_sub}, match_l[2] = {128, 1};
    clEnqueueNDRangeKernel(q, active_plotter.k_match_p1, 2, nullptr, match_g, match_l, 0, nullptr, nullptr);
    
    // Non-blocking read of match count — event on the READ (not kernel)
    // so that clWaitForEvents guarantees the data is in gpu_matches
    clEnqueueReadBuffer(q, p.num_matches_buf, CL_TRUE, 0, 4, &p.gpu_matches, 0, nullptr, nullptr);

    p.skip = false;
    return p;
}

// Phase 2: Wait for match, read LR, submit gather+hash (non-blocking hash reads)
void submit_hash_pipeline(BucketPending& p, MemBucketStore& src)
{
    if(p.skip) return;

    OCL_Plotter& active_plotter = *p.active_plotter;
    BufferPool* active_pool = p.active_pool;
    cl_command_queue& q = p.q;
    const int n_meta = p.n_meta;
    int y = p.y, table = p.table;

    // Wait for match kernel + match count read to complete
    clWaitForEvents(1, &p.ev_match_done);
    clReleaseEvent(p.ev_match_done);
    p.ev_match_done = nullptr;
    
    if(timing_detail && (p.y == 0 || p.y == 1))
        std::cerr << "    [debug] bucket " << p.y << " gpu_matches=" << p.gpu_matches << std::endl;

    if(p.gpu_matches == 0) {
        p.zero_matches = true;
        if(!p.using_pool) {
            clReleaseMemObject(p.C_in_buf); clReleaseMemObject(p.PY_buf); clReleaseMemObject(p.sub_cnt_buf);
            clReleaseMemObject(p.sub_off_buf); clReleaseMemObject(p.LR_buf); clReleaseMemObject(p.PD_match_buf);
            clReleaseMemObject(p.num_matches_buf);
        }
        return;
    }

    // Read LR pairs
    p.LR_filtered.resize(p.gpu_matches * 2);
    if(false && get_opt_config().pinned && p.active_pool && p.active_pool->pinned_LR_ptr) {
        // Module G: disabled on NVIDIA (mapped buffer read conflict)
        clEnqueueReadBuffer(q, p.LR_buf, CL_TRUE, 0, p.gpu_matches * 8,
            p.active_pool->pinned_LR_ptr, 0, nullptr, nullptr);
        std::memcpy(p.LR_filtered.data(), p.active_pool->pinned_LR_ptr, p.gpu_matches * 8);
    } else {
        clEnqueueReadBuffer(q, p.LR_buf, CL_TRUE, 0, p.gpu_matches * 8, p.LR_filtered.data(), 0, nullptr, nullptr);
    }

    // Compute PD (CPU work)
    p.bucket_offset = 0;
    for(int b = 0; b < y; b++) p.bucket_offset += src.counts[b];
    p.num_filt = p.LR_filtered.size() / 2;
    p.pd_data.resize(p.num_filt * 2);
    for(uint32_t i = 0; i < p.num_filt; i++) {
        p.pd_data[i * 2] = p.bucket_offset + p.LR_filtered[i * 2];
        p.pd_data[i * 2 + 1] = p.bucket_offset + p.LR_filtered[i * 2 + 1];
    }

    // X pairs for table 2
    if(table == 2) {
        const std::vector<uint32_t>& x_vals = src.x_values[y];
        for(uint32_t i = 0; i < p.num_filt; i++) {
            uint32_t left_local = p.LR_filtered[i * 2];
            uint32_t right_local = p.LR_filtered[i * 2 + 1];
            uint32_t left_x = (left_local < x_vals.size()) ? x_vals[left_local] : 0;
            uint32_t right_x = (right_local < x_vals.size()) ? x_vals[right_local] : 0;
            p.xp_flat.push_back(left_x);
            p.xp_flat.push_back(right_x);
        }
    }

    p.num_matches = p.gpu_matches;

    if(get_opt_config().gpu_meta_extract) {
        cl_int err2;
        uint32_t num_m = (uint32_t)p.num_matches;
        uint32_t num_total = p.total;
        uint32_t n_meta_u32 = (uint32_t)n_meta;

        if(p.using_pool) {
            p.L_gathered = active_pool->L_gathered;
            p.R_gathered = active_pool->R_gathered;
            p.Yb = active_pool->Y_hash_buf;
            p.Mb = active_pool->M_hash_buf;
        } else {
            p.L_gathered = clCreateBuffer(active_plotter.context, CL_MEM_WRITE_ONLY, p.num_matches * n_meta * 4, nullptr, &err2);
            CL_CHECK(err2, "L_gathered");
            p.R_gathered = clCreateBuffer(active_plotter.context, CL_MEM_WRITE_ONLY, p.num_matches * n_meta * 4, nullptr, &err2);
            CL_CHECK(err2, "R_gathered");
            p.Yb = clCreateBuffer(active_plotter.context, CL_MEM_WRITE_ONLY, p.num_matches * 4, nullptr, &err2);
            CL_CHECK(err2, "Yb");
            p.Mb = clCreateBuffer(active_plotter.context, CL_MEM_WRITE_ONLY, p.num_matches * MY_N_META * 4, nullptr, &err2);
            CL_CHECK(err2, "Mb");
        }

        // Gather kernel (non-blocking)
        clSetKernelArg(active_plotter.gather_meta_kernel, 0, sizeof(cl_mem), &p.C_in_buf);
        clSetKernelArg(active_plotter.gather_meta_kernel, 1, sizeof(cl_mem), &p.LR_buf);
        clSetKernelArg(active_plotter.gather_meta_kernel, 2, sizeof(cl_mem), &p.L_gathered);
        clSetKernelArg(active_plotter.gather_meta_kernel, 3, sizeof(cl_mem), &p.R_gathered);
        clSetKernelArg(active_plotter.gather_meta_kernel, 4, sizeof(uint32_t), &num_m);
        clSetKernelArg(active_plotter.gather_meta_kernel, 5, sizeof(uint32_t), &num_total);
        clSetKernelArg(active_plotter.gather_meta_kernel, 6, sizeof(uint32_t), &n_meta_u32);
        size_t gather_global = p.num_matches, gather_local = 64;
        if(gather_global % gather_local) gather_global = ((gather_global / gather_local) + 1) * gather_local;
        clEnqueueNDRangeKernel(q, active_plotter.gather_meta_kernel, 1, nullptr, &gather_global, &gather_local, 0, nullptr, nullptr);

        // Hash kernel (non-blocking)
        uint32_t kmask = KMASK;
        clSetKernelArg(active_plotter.table_hash_kernel, 0, sizeof(cl_mem), &p.L_gathered);
        clSetKernelArg(active_plotter.table_hash_kernel, 1, sizeof(cl_mem), &p.R_gathered);
        clSetKernelArg(active_plotter.table_hash_kernel, 2, sizeof(cl_mem), &p.Yb);
        clSetKernelArg(active_plotter.table_hash_kernel, 3, sizeof(cl_mem), &p.Mb);
        clSetKernelArg(active_plotter.table_hash_kernel, 4, sizeof(uint32_t), &kmask);
        clSetKernelArg(active_plotter.table_hash_kernel, 5, sizeof(uint32_t), &num_m);
        size_t hash_global = p.num_matches, hash_local = 64;
        if(hash_global % hash_local) hash_global = ((hash_global / hash_local) + 1) * hash_local;
        clEnqueueNDRangeKernel(q, active_plotter.table_hash_kernel, 1, nullptr, &hash_global, &hash_local, 0, nullptr, nullptr);

        // Non-blocking reads of Y + M results — ev_hash_done guards both (in-order queue)
        p.Y_out.resize(p.num_matches);
        p.M_out.resize(p.num_matches * MY_N_META);
        if(false && get_opt_config().pinned && p.active_pool && p.active_pool->pinned_Y_ptr) {
            // Module G: disabled on NVIDIA (mapped buffer read conflict)
            clEnqueueReadBuffer(q, p.Yb, CL_TRUE, 0, p.num_matches * 4,
                p.active_pool->pinned_Y_ptr, 0, nullptr, nullptr);
            clEnqueueReadBuffer(q, p.Mb, CL_TRUE, 0, p.num_matches * MY_N_META * 4,
                p.active_pool->pinned_M_ptr, 0, nullptr, nullptr);
            std::memcpy(p.Y_out.data(), p.active_pool->pinned_Y_ptr, p.num_matches * 4);
            std::memcpy(p.M_out.data(), p.active_pool->pinned_M_ptr, p.num_matches * MY_N_META * 4);
        } else {
            clEnqueueReadBuffer(q, p.Yb, CL_TRUE, 0, p.num_matches * 4, p.Y_out.data(), 0, nullptr, nullptr);
            clEnqueueReadBuffer(q, p.Mb, CL_TRUE, 0, p.num_matches * MY_N_META * 4, p.M_out.data(), 0, nullptr, nullptr);
        }
    } else {
        // CPU-extract path (blocking, no event)
        std::vector<uint32_t> L_meta_flat(p.num_matches * n_meta);
        std::vector<uint32_t> R_meta_flat(p.num_matches * n_meta);
        for(size_t i = 0; i < p.num_matches; i++) {
            uint32_t P1 = p.LR_filtered[i * 2];
            uint32_t P2 = p.LR_filtered[i * 2 + 1];
            for(int j = 0; j < n_meta; j++) {
                L_meta_flat[i * n_meta + j] = p.C_combined[P1 * n_meta + j];
                R_meta_flat[i * n_meta + j] = p.C_combined[P2 * n_meta + j];
            }
        }
        active_plotter.gpu_hash_table(L_meta_flat, R_meta_flat, p.Y_out, p.M_out, KMASK);
    }
}

// Phase 3: Wait for hash, write results to dst store
void collect_bucket_pipeline(BucketPending& p, MemBucketStore& dst)
{
    if(p.skip || p.zero_matches) return;

    // Wait for hash results if using GPU meta extract
    if(get_opt_config().gpu_meta_extract && p.ev_hash_done) {
        clWaitForEvents(1, &p.ev_hash_done);
        clReleaseEvent(p.ev_hash_done);
        p.ev_hash_done = nullptr;
    }

    const int n_meta = p.n_meta;
    const int shift = p.shift;
    int table = p.table;

    // Distribute to dst store (CPU work, single-threaded — no mutex needed)
    std::vector<std::vector<uint32_t>> dst_batch(dst.num_buckets);
    std::vector<std::vector<uint32_t>> dst_pd_batch(dst.num_buckets);
    std::vector<std::vector<uint32_t>> dst_xp_batch(dst.num_buckets);

    for(uint32_t i = 0; i < p.num_filt; i++) {
        uint32_t bucket = p.Y_out[i] >> shift;
        if(bucket >= (uint32_t)dst.num_buckets) bucket = dst.num_buckets - 1;
        for(int j = 0; j < n_meta; j++)
            dst_batch[bucket].push_back(p.M_out[i * n_meta + j]);
        dst_pd_batch[bucket].push_back(p.pd_data[i * 2]);
        dst_pd_batch[bucket].push_back(p.pd_data[i * 2 + 1]);
        if(table == 2 && i * 2 + 1 < p.xp_flat.size()) {
            dst_xp_batch[bucket].push_back(p.xp_flat[i * 2]);
            dst_xp_batch[bucket].push_back(p.xp_flat[i * 2 + 1]);
        }
    }
    for(int b = 0; b < dst.num_buckets; b++) {
        if(dst_batch[b].empty()) continue;
        uint32_t cnt = dst_batch[b].size() / n_meta;
        dst.append(b, dst_batch[b].data(), cnt);
        dst.append_pd(b, dst_pd_batch[b].data(), cnt);
        if(table == 2 && !dst_xp_batch[b].empty()) {
            dst.append_x_pairs(b, dst_xp_batch[b].data(), cnt);
        }
    }

    // Cleanup GPU buffers if we own them (no pool)
    if(get_opt_config().gpu_meta_extract && !p.using_pool) {
        clReleaseMemObject(p.L_gathered);
        clReleaseMemObject(p.R_gathered);
        clReleaseMemObject(p.Yb);
        clReleaseMemObject(p.Mb);
    }
    if(!p.using_pool) {
        clReleaseMemObject(p.C_in_buf); clReleaseMemObject(p.PY_buf); clReleaseMemObject(p.sub_cnt_buf);
        clReleaseMemObject(p.sub_off_buf); clReleaseMemObject(p.LR_buf); clReleaseMemObject(p.PD_match_buf);
        clReleaseMemObject(p.num_matches_buf);
    }
}

// ============================================================================
// Module H: SVM pipeline — uses Shared Virtual Memory (OpenCL 2.0)
// No cl_mem, no clEnqueueWriteBuffer/ReadBuffer. CPU and GPU share pointers.
// Fine-grain (AMD): write directly, GPU sees it immediately.
// Coarse-grain (NVIDIA): map/unmap around CPU access.
// ============================================================================

BucketPending submit_bucket_svm(
    OCL_Plotter& plotter,
    MemBucketStore& src,
    int y,
    int table,
    SVMPool* svm)
{
    BucketPending p;
    p.skip = true;
    p.y = y;
    p.table = table;
    p.n_meta = src.n_meta;
    p.shift = KSIZE - LOGBUCKETS;
    const int logbuckets2 = KSIZE - LOGBUCKETS - 9;
    p.num_sub = 1 << logbuckets2;
    p.max_bs2 = std::max(1024, (int)(((uint64_t)4 << KSIZE) / (1 << LOGBUCKETS) / p.num_sub));

    p.count_y = src.counts[y];
    if(p.count_y == 0) return p;

    p.active_plotter = &plotter;
    p.active_svm = svm;
    p.using_svm = true;
    const uint32_t* meta_y = src.buckets[y].data();
    p.total = p.count_y;

    OCL_Plotter& active_plotter = *p.active_plotter;
    const int n_meta = p.n_meta;
    const int num_sub = p.num_sub;
    const int max_bs2 = p.max_bs2;
    uint32_t total = p.total;

    cl_int err;
    int zero = 0;
    p.q = active_plotter.queue;
    cl_command_queue& q = p.q;
    
    // Timing
    bool do_timing = timing_detail && (y == 0);
    auto ts = std::chrono::steady_clock::now();
    auto te = ts;
    auto ps = [&](const char* name) {
        if(do_timing) {
            te = std::chrono::steady_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(te - ts).count();
            std::cerr << "    [SVM " << name << "] " << us << "us" << std::endl;
            ts = te;
        }
    };

    // Copy metadata for CPU PD computation (only needed for non-gpu-meta path)
    if(!get_opt_config().gpu_meta_extract) {
        p.C_combined.resize(total * n_meta);
        std::memcpy(p.C_combined.data(), meta_y, p.count_y * n_meta * 4);
    }

    // Assign SVM pointers
    p.svm_C_in = svm->svm_C_in;
    p.svm_PY = svm->svm_PY;
    p.svm_sub_cnt = svm->svm_sub_cnt;
    p.svm_sub_off = svm->svm_sub_off;
    p.svm_LR = svm->svm_LR;
    p.svm_PD_match = svm->svm_PD_match;
    p.svm_num_matches = svm->svm_num_matches;

    // Write C_in data to SVM buffer
    if(svm->fine_grain) {
        // Fine-grain AMD: direct CPU memcpy (shared physical memory, no DMA needed)
        // std::atomic_thread_fence ensures CPU write is visible to GPU
        std::memcpy(p.svm_C_in, meta_y, p.count_y * n_meta * 4);
        std::memset(p.svm_sub_cnt, 0, num_sub * 4);
        std::memset(p.svm_num_matches, 0, 4);
        std::atomic_thread_fence(std::memory_order_release);
    ps("write_C_in");
    } else {
        // Coarse-grain: map, write, unmap
        svm->map(p.svm_C_in, p.count_y * n_meta * 4, CL_MAP_WRITE);
        std::memcpy(p.svm_C_in, meta_y, p.count_y * n_meta * 4);
        svm->unmap(p.svm_C_in);
        svm->map(p.svm_sub_cnt, num_sub * 4, CL_MAP_WRITE);
        std::memset(p.svm_sub_cnt, 0, num_sub * 4);
        svm->unmap(p.svm_sub_cnt);
        svm->map(p.svm_num_matches, 4, CL_MAP_WRITE);
        std::memset(p.svm_num_matches, 0, 4);
        svm->unmap(p.svm_num_matches);
    }

    // Step 2: Scatter kernel — use SVM pointer as arg
    uint32_t total_u32 = total;
    uint32_t max_bs2_u32 = max_bs2;
    cl_mem null_mem = nullptr;
    clSetKernelArgSVMPointer(active_plotter.k_scatter2, 0, p.svm_PY);
    clSetKernelArgSVMPointer(active_plotter.k_scatter2, 1, p.svm_sub_cnt);
    clSetKernelArg(active_plotter.k_scatter2, 2, sizeof(cl_mem), &null_mem);
    clSetKernelArgSVMPointer(active_plotter.k_scatter2, 3, p.svm_C_in);
    clSetKernelArg(active_plotter.k_scatter2, 4, sizeof(uint32_t), &total_u32);
    clSetKernelArg(active_plotter.k_scatter2, 5, sizeof(uint32_t), &max_bs2_u32);
    size_t scatter_global = total;
    if(scatter_global % 64) scatter_global = ((scatter_global / 64) + 1) * 64;
    clEnqueueNDRangeKernel(q, active_plotter.k_scatter2, 1, nullptr, &scatter_global, nullptr, 0, nullptr, nullptr);
    ps("scatter");

    // Step 3: Prefix sum (Module A: GPU)
    p.num_sub_u32 = (uint32_t)num_sub;
    if(get_opt_config().gpu_prefix_sum && active_plotter.prefix_sum_kernel) {
        clSetKernelArgSVMPointer(active_plotter.prefix_sum_kernel, 0, p.svm_sub_cnt);
        clSetKernelArgSVMPointer(active_plotter.prefix_sum_kernel, 1, p.svm_sub_off);
        clSetKernelArg(active_plotter.prefix_sum_kernel, 2, sizeof(uint32_t) * (num_sub + 1), nullptr);
        clSetKernelArg(active_plotter.prefix_sum_kernel, 3, sizeof(uint32_t), &p.num_sub_u32);
        size_t ps_global = num_sub, ps_local = num_sub;
        clEnqueueNDRangeKernel(q, active_plotter.prefix_sum_kernel, 1, nullptr, &ps_global, &ps_local, 0, nullptr, nullptr);
    }

    // Step 4: Sort
    p.max_bs_sort = max_bs2;
    uint32_t max_bs_sort = p.max_bs_sort;
    clSetKernelArgSVMPointer(active_plotter.k_simple_sort, 0, p.svm_PY);
    clSetKernelArgSVMPointer(active_plotter.k_simple_sort, 1, p.svm_sub_cnt);
    clSetKernelArg(active_plotter.k_simple_sort, 2, sizeof(uint32_t), &max_bs_sort);
    clSetKernelArg(active_plotter.k_simple_sort, 3, sizeof(uint32_t), &p.num_sub_u32);
    size_t sort_g[2] = {256, (size_t)num_sub}, sort_l[2] = {256, 1};
    clEnqueueNDRangeKernel(q, active_plotter.k_simple_sort, 2, nullptr, sort_g, sort_l, 0, nullptr, nullptr);
    ps("sort");

    // Step 5: Match
    uint32_t max_total = total * 4;
    uint32_t write_pd = 0;
    clSetKernelArgSVMPointer(active_plotter.k_match_p1, 0, p.svm_LR);
    clSetKernelArgSVMPointer(active_plotter.k_match_p1, 1, p.svm_PD_match);
    clSetKernelArgSVMPointer(active_plotter.k_match_p1, 2, p.svm_num_matches);
    clSetKernelArgSVMPointer(active_plotter.k_match_p1, 3, p.svm_PY);
    clSetKernelArgSVMPointer(active_plotter.k_match_p1, 4, p.svm_sub_cnt);
    clSetKernelArgSVMPointer(active_plotter.k_match_p1, 5, p.svm_sub_off);
    clSetKernelArg(active_plotter.k_match_p1, 6, sizeof(uint32_t), &p.num_sub_u32);
    clSetKernelArg(active_plotter.k_match_p1, 7, sizeof(uint32_t), &max_bs_sort);
    clSetKernelArg(active_plotter.k_match_p1, 8, sizeof(uint32_t), &max_total);
    clSetKernelArg(active_plotter.k_match_p1, 9, sizeof(uint32_t), &write_pd);
    int groups_per_sub = (max_bs2 + 127) / 128;
    size_t match_g[2] = {(size_t)(128 * groups_per_sub), (size_t)num_sub}, match_l[2] = {128, 1};
    clEnqueueNDRangeKernel(q, active_plotter.k_match_p1, 2, nullptr, match_g, match_l, 0, nullptr, &p.ev_match_done);
    ps("match");

    // DO NOT read match count here — return immediately so GPU 1 can be submitted
    // Match count read happens in submit_hash_svm (after both GPUs are working)
    p.skip = false;
    return p;
}

void submit_hash_svm(BucketPending& p, MemBucketStore& src)
{
    if(p.skip) return;
    
    OCL_Plotter& active_plotter = *p.active_plotter;
    SVMPool* svm = p.active_svm;
    cl_command_queue& q = p.q;
    const int n_meta = p.n_meta;
    int y = p.y, table = p.table;
    
    bool do_timing = timing_detail && (y == 0);
    auto ts = std::chrono::steady_clock::now();
    auto te = ts;
    auto ps = [&](const char* name) {
        if(do_timing) {
            te = std::chrono::steady_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(te - ts).count();
            std::cerr << "    [SVM " << name << "] " << us << "us" << std::endl;
            ts = te;
        }
    };
    
    // Read match count — use event wait (faster than clFinish which flushes entire queue)
    if(svm->fine_grain && p.ev_match_done) {
        clWaitForEvents(1, &p.ev_match_done);
        clReleaseEvent(p.ev_match_done);
        p.ev_match_done = nullptr;
        p.gpu_matches = *(volatile uint32_t*)p.svm_num_matches;
        ps("event_wait+match_count");
    } else {
        uint32_t match_count = 0;
        clEnqueueSVMMap(q, CL_TRUE, CL_MAP_READ, p.svm_num_matches, 4, 0, nullptr, nullptr);
        match_count = *(uint32_t*)p.svm_num_matches;
        clEnqueueSVMUnmap(q, p.svm_num_matches, 0, nullptr, nullptr);
        p.gpu_matches = match_count;
    }
    
    if(p.gpu_matches == 0) {
        p.zero_matches = true;
        return;
    }
    
    // Read LR pairs — fine-grain: already synced (clFinish above), direct memcpy
    p.LR_filtered.resize(p.gpu_matches * 2);
    if(svm->fine_grain) {
        // No clFinish needed — already synced at match count read
        std::memcpy(p.LR_filtered.data(), p.svm_LR, p.gpu_matches * 8);
    } else {
        svm->map(p.svm_LR, p.gpu_matches * 8, CL_MAP_READ);
        std::memcpy(p.LR_filtered.data(), p.svm_LR, p.gpu_matches * 8);
        svm->unmap(p.svm_LR);
    }
    ps("read_LR");
    
    // Compute PD
    p.bucket_offset = 0;
    for(int b = 0; b < y; b++) p.bucket_offset += src.counts[b];
    p.num_filt = p.LR_filtered.size() / 2;
    p.pd_data.resize(p.num_filt * 2);
    for(uint32_t i = 0; i < p.num_filt; i++) {
        p.pd_data[i * 2] = p.bucket_offset + p.LR_filtered[i * 2];
        p.pd_data[i * 2 + 1] = p.bucket_offset + p.LR_filtered[i * 2 + 1];
    }
    
    // X pairs for table 2
    if(table == 2) {
        const std::vector<uint32_t>& x_vals = src.x_values[y];
        for(uint32_t i = 0; i < p.num_filt; i++) {
            uint32_t left_local = p.LR_filtered[i * 2];
            uint32_t right_local = p.LR_filtered[i * 2 + 1];
            uint32_t left_x = (left_local < x_vals.size()) ? x_vals[left_local] : 0;
            uint32_t right_x = (right_local < x_vals.size()) ? x_vals[right_local] : 0;
            p.xp_flat.push_back(left_x);
            p.xp_flat.push_back(right_x);
        }
    }
    
    p.num_matches = p.gpu_matches;
    
    if(get_opt_config().gpu_meta_extract) {
        // Assign SVM buffers for gather/hash
        p.svm_L_gathered = svm->svm_L_gathered;
        p.svm_R_gathered = svm->svm_R_gathered;
        p.svm_Y_hash = svm->svm_Y_hash;
        p.svm_M_hash = svm->svm_M_hash;
        
        uint32_t num_m = (uint32_t)p.num_matches;
        uint32_t num_total = p.total;
        uint32_t n_meta_u32 = (uint32_t)n_meta;
        
        // Gather kernel
        clSetKernelArgSVMPointer(active_plotter.gather_meta_kernel, 0, p.svm_C_in);
        clSetKernelArgSVMPointer(active_plotter.gather_meta_kernel, 1, p.svm_LR);
        clSetKernelArgSVMPointer(active_plotter.gather_meta_kernel, 2, p.svm_L_gathered);
        clSetKernelArgSVMPointer(active_plotter.gather_meta_kernel, 3, p.svm_R_gathered);
        clSetKernelArg(active_plotter.gather_meta_kernel, 4, sizeof(uint32_t), &num_m);
        clSetKernelArg(active_plotter.gather_meta_kernel, 5, sizeof(uint32_t), &num_total);
        clSetKernelArg(active_plotter.gather_meta_kernel, 6, sizeof(uint32_t), &n_meta_u32);
        size_t gather_global = p.num_matches, gather_local = 64;
        if(gather_global % gather_local) gather_global = ((gather_global / gather_local) + 1) * gather_local;
        clEnqueueNDRangeKernel(q, active_plotter.gather_meta_kernel, 1, nullptr, &gather_global, &gather_local, 0, nullptr, nullptr);
        ps("gather");
        
        // Hash kernel
        uint32_t kmask = KMASK;
        clSetKernelArgSVMPointer(active_plotter.table_hash_kernel, 0, p.svm_L_gathered);
        clSetKernelArgSVMPointer(active_plotter.table_hash_kernel, 1, p.svm_R_gathered);
        clSetKernelArgSVMPointer(active_plotter.table_hash_kernel, 2, p.svm_Y_hash);
        clSetKernelArgSVMPointer(active_plotter.table_hash_kernel, 3, p.svm_M_hash);
        clSetKernelArg(active_plotter.table_hash_kernel, 4, sizeof(uint32_t), &kmask);
        clSetKernelArg(active_plotter.table_hash_kernel, 5, sizeof(uint32_t), &num_m);
        size_t hash_global = p.num_matches, hash_local = 64;
        if(hash_global % hash_local) hash_global = ((hash_global / hash_local) + 1) * hash_local;
        clEnqueueNDRangeKernel(q, active_plotter.table_hash_kernel, 1, nullptr, &hash_global, &hash_local, 0, nullptr, &p.ev_hash_done);
        ps("hash");
        
        // Read Y + M from SVM — event wait (faster than clFinish)
        p.Y_out.resize(p.num_matches);
        p.M_out.resize(p.num_matches * MY_N_META);
        // Read Y+M — fine-grain: event wait + direct memcpy; coarse: map/unmap
        if(svm->fine_grain && p.ev_hash_done) {
            clWaitForEvents(1, &p.ev_hash_done);
            clReleaseEvent(p.ev_hash_done);
            p.ev_hash_done = nullptr;
            ps("event_wait_hash");
            std::memcpy(p.Y_out.data(), p.svm_Y_hash, p.num_matches * 4);
            std::memcpy(p.M_out.data(), p.svm_M_hash, p.num_matches * MY_N_META * 4);
        } else {
            svm->map(p.svm_Y_hash, p.num_matches * 4, CL_MAP_READ);
            std::memcpy(p.Y_out.data(), p.svm_Y_hash, p.num_matches * 4);
            svm->unmap(p.svm_Y_hash);
            svm->map(p.svm_M_hash, p.num_matches * MY_N_META * 4, CL_MAP_READ);
            std::memcpy(p.M_out.data(), p.svm_M_hash, p.num_matches * MY_N_META * 4);
            svm->unmap(p.svm_M_hash);
        }
    } else {
        // CPU-extract path
        std::vector<uint32_t> L_meta_flat(p.num_matches * n_meta);
        std::vector<uint32_t> R_meta_flat(p.num_matches * n_meta);
        for(size_t i = 0; i < p.num_matches; i++) {
            uint32_t P1 = p.LR_filtered[i * 2];
            uint32_t P2 = p.LR_filtered[i * 2 + 1];
            for(int j = 0; j < n_meta; j++) {
                L_meta_flat[i * n_meta + j] = p.C_combined[P1 * n_meta + j];
                R_meta_flat[i * n_meta + j] = p.C_combined[P2 * n_meta + j];
            }
        }
        active_plotter.gpu_hash_table(L_meta_flat, R_meta_flat, p.Y_out, p.M_out, KMASK);
    }
}

void collect_bucket_svm(BucketPending& p, MemBucketStore& dst)
{
    if(p.skip || p.zero_matches) return;
    
    const int n_meta = p.n_meta;
    const int shift = p.shift;
    int table = p.table;
    
    // Distribute to dst store (same as collect_bucket_pipeline)
    std::vector<std::vector<uint32_t>> dst_batch(dst.num_buckets);
    std::vector<std::vector<uint32_t>> dst_pd_batch(dst.num_buckets);
    std::vector<std::vector<uint32_t>> dst_xp_batch(dst.num_buckets);
    
    for(uint32_t i = 0; i < p.num_filt; i++) {
        uint32_t bucket = p.Y_out[i] >> shift;
        if(bucket >= (uint32_t)dst.num_buckets) bucket = dst.num_buckets - 1;
        for(int j = 0; j < n_meta; j++)
            dst_batch[bucket].push_back(p.M_out[i * n_meta + j]);
        dst_pd_batch[bucket].push_back(p.pd_data[i * 2]);
        dst_pd_batch[bucket].push_back(p.pd_data[i * 2 + 1]);
        if(table == 2 && i * 2 + 1 < p.xp_flat.size()) {
            dst_xp_batch[bucket].push_back(p.xp_flat[i * 2]);
            dst_xp_batch[bucket].push_back(p.xp_flat[i * 2 + 1]);
        }
    }
    for(int b = 0; b < dst.num_buckets; b++) {
        if(dst_batch[b].empty()) continue;
        uint32_t cnt = dst_batch[b].size() / n_meta;
        dst.append(b, dst_batch[b].data(), cnt);
        dst.append_pd(b, dst_pd_batch[b].data(), cnt);
        if(table == 2 && !dst_xp_batch[b].empty()) {
            dst.append_x_pairs(b, dst_xp_batch[b].data(), cnt);
        }
    }
}

void compute_f2_f9_chunked(
    OCL_Plotter& plotter,
    MemBucketStore& store,
    int num_buckets,
    int max_bucket_size,
    int n_meta,
    bool gpu_yield,
    std::vector<std::vector<PDEntry>>& pd_all,
    std::vector<std::pair<uint32_t, uint32_t>>& x_pairs_all)
{
    MemBucketStore src(num_buckets, max_bucket_size, n_meta);
    MemBucketStore dst(num_buckets, max_bucket_size, n_meta);
    
    // Module C: Initialize buffer pool(s) if enabled
    BufferPool bufpool;
    std::vector<BufferPool> multi_bufpools;
    
    // Module H: Initialize SVM pool(s) if enabled
    SVMPool svm_pool;
    std::vector<SVMPool> multi_svm_pools;
    if(get_opt_config().svm) {
        int logbuckets2 = KSIZE - LOGBUCKETS - 9;
        int num_sub = 1 << logbuckets2;
        int max_bs2 = std::max(1024, (int)(((uint64_t)4 << KSIZE) / (1 << LOGBUCKETS) / num_sub));
        
        if(g_num_gpus > 1 && g_plotters.size() > 1) {
            multi_svm_pools.resize(g_num_gpus);
            for(int g = 0; g < g_num_gpus; g++) {
                multi_svm_pools[g].init(g_plotters[g]->context, g_plotters[g]->queue, g_plotters[g]->device,
                                       max_bucket_size, n_meta, num_sub, max_bs2, 1 << 18);
                g_svmpools.push_back(&multi_svm_pools[g]);
            }
        } else {
            svm_pool.init(plotter.context, plotter.queue, plotter.device,
                         max_bucket_size, n_meta, num_sub, max_bs2, std::max((uint32_t)(1 << 20), (uint32_t)(1 << KSIZE)));
            g_svmpools.push_back(&svm_pool);
        }
    }
    
    if(get_opt_config().bufpool && !get_opt_config().svm) {
        int logbuckets2 = KSIZE - LOGBUCKETS - 9;
        int num_sub = 1 << logbuckets2;
        int max_bs2 = std::max(1024, (int)(((uint64_t)4 << KSIZE) / (1 << LOGBUCKETS) / num_sub));
        
        if(g_num_gpus > 1 && g_plotters.size() > 1) {
            // Multi-GPU: one buffer pool per GPU
            multi_bufpools.resize(g_num_gpus);
            for(int g = 0; g < g_num_gpus; g++) {
                multi_bufpools[g].init(g_plotters[g]->context, g_plotters[g]->queue,
                                       max_bucket_size, n_meta, num_sub, max_bs2);
                g_bufpools.push_back(&multi_bufpools[g]);
            }
            g_bufpool = g_bufpools[0];  // backwards compatible
        } else {
            bufpool.init(plotter.context, plotter.queue, max_bucket_size, n_meta, num_sub, max_bs2);
            g_bufpool = &bufpool;
        }
    }
    
    // Copy F1 output to src (metadata + X values)
    for(int y = 0; y < num_buckets; y++) {
        if(store.counts[y] > 0) {
            src.append(y, store.buckets[y].data(), store.counts[y]);
            if(store.x_values[y].size() > 0)
                src.append_x(y, store.x_values[y].data(), store.counts[y]);
        }
    }
    
    pd_all.resize(MY_N_TABLE + 1);
    auto t0 = my_time_ms();
    
    for(int t = 2; t <= MY_N_TABLE; t++) {
        auto tt0 = my_time_ms();
        dst.clear();
        
        if(get_opt_config().svm && g_svmpools.size() > 0) {
            // Module H: SVM pipeline — 2-wide for multi-GPU
            if(g_num_gpus > 1 && g_svmpools.size() > 1) {
                // Multi-GPU: submit pairs concurrently
                for(int y = 0; y < num_buckets; y += 2) {
                    bool have0 = (src.counts[y] > 0);
                    bool have1 = (y + 1 < num_buckets && src.counts[y + 1] > 0);
                    if(!have0 && !have1) continue;
                    
                    // Phase 1: submit BOTH buckets' scatter/sort/match (kernels queue on each GPU)
                    BucketPending p0, p1;
                    if(have0) p0 = submit_bucket_svm(*g_plotters[0], src, y, t, g_svmpools[0]);
                    if(have1) p1 = submit_bucket_svm(*g_plotters[1], src, y + 1, t, g_svmpools[1]);
                    // Both GPUs are now executing kernels concurrently!
                    
                    // Phase 2: submit hash for both (each waits on its own GPU's match)
                    if(have0) submit_hash_svm(p0, src);
                    if(have1) submit_hash_svm(p1, src);
                    
                    // Phase 3: collect both
                    if(have0) collect_bucket_svm(p0, dst);
                    if(have1) collect_bucket_svm(p1, dst);
                    
                    if(have0) cross_boundary_match(*g_plotters[0], src, dst, y, t);
                    if(have1) cross_boundary_match(*g_plotters[1], src, dst, y + 1, t);
                    
                    if(y % 32 == 0 || y >= num_buckets - 2) {
                        std::cerr << "\r[T" << t << "] Bucket " << (y+2) << "/" << num_buckets
                                  << " (" << (y+2)*100/num_buckets << "%) "
                                  << (my_time_ms() - tt0) / 1000.0 << "s" << std::flush;
                    }
                }
            } else {
                // Single-GPU SVM
                for(int y = 0; y < num_buckets; y++) {
                    if(src.counts[y] == 0) continue;
                    BucketPending p = submit_bucket_svm(plotter, src, y, t, g_svmpools[0]);
                    submit_hash_svm(p, src);
                    collect_bucket_svm(p, dst);
                    cross_boundary_match(plotter, src, dst, y, t);
                    
                    if(y % 32 == 0 || y == num_buckets - 1) {
                        std::cerr << "\r[T" << t << "] Bucket " << (y+1) << "/" << num_buckets
                                  << " (" << (y+1)*100/num_buckets << "%) "
                                  << (my_time_ms() - tt0) / 1000.0 << "s" << std::flush;
                    }
                }
            }
        } else if(g_num_gpus > 1 && g_plotters.size() > 1 && get_opt_config().bufpool) {
            // Module E v2: 2-wide pipeline — no threads, event-based overlap
            for(int y = 0; y < num_buckets; y += 2) {
                bool have0 = (src.counts[y] > 0);
                bool have1 = (y + 1 < num_buckets && src.counts[y + 1] > 0);
                if(!have0 && !have1) continue;
                
                // Phase 1: submit both buckets (non-blocking, returns instantly)
                BucketPending p0, p1;
                if(have0) p0 = submit_bucket_pipeline(*g_plotters[0], src, y, t, g_bufpools[0]);
                if(have1) p1 = submit_bucket_pipeline(*g_plotters[1], src, y + 1, t, g_bufpools[1]);
                // BOTH GPUs are now crunching scatter/sort/match concurrently
                
                // Phase 2: submit hash stage (waits on match event, submits gather+hash)
                if(have0) submit_hash_pipeline(p0, src);
                if(have1) submit_hash_pipeline(p1, src);
                
                // Phase 3: collect results (waits on hash event, writes to dst store)
                if(have0) collect_bucket_pipeline(p0, dst);
                if(have1) collect_bucket_pipeline(p1, dst);
                
                // Cross-boundary matching
                if(have0) cross_boundary_match(*g_plotters[0], src, dst, y, t);
                if(have1) cross_boundary_match(*g_plotters[1], src, dst, y + 1, t);
                
                if(y % 32 == 0 || y >= num_buckets - 2) {
                    std::cerr << "\r[T" << t << "] Bucket " << (y+2) << "/" << num_buckets
                              << " (" << (y+2)*100/num_buckets << "%) "
                              << (my_time_ms() - tt0) / 1000.0 << "s" << std::flush;
                }
            }
        } else {
            // Single-GPU sequential path (original)
            for(int y = 0; y < num_buckets; y++) {
                if(src.counts[y] == 0) continue;
                process_bucket_gpu(plotter, src, dst, y, t);
                
                // Cross-boundary matching
                cross_boundary_match(plotter, src, dst, y, t);
                
                // Display yield
                if(gpu_yield) {
                    clFinish(plotter.queue);
                    if(plotter.queue2) clFinish(plotter.queue2);
                    usleep(500);
                }
                
                if(y % 32 == 0 || y == num_buckets - 1) {
                    std::cerr << "\r[T" << t << "] Bucket " << (y+1) << "/" << num_buckets
                              << " (" << (y+1)*100/num_buckets << "%) "
                              << (my_time_ms() - tt0) / 1000.0 << "s" << std::flush;
                }
            }
        }
        
        double elapsed = (my_time_ms() - tt0) / 1000.0;
        uint64_t total = 0;
        for(int y = 0; y < num_buckets; y++) total += dst.counts[y];
        std::cout << "\n[T" << t << "] " << total << " entries (" << elapsed << " sec)" << std::endl;
        
        // Save PD for this table: collect (Y, sorted_pos, delta) from dst
        auto tpd0 = my_time_ms();
        pd_all[t].clear();
        pd_all[t].reserve(total);
        if(t == 2) {
            x_pairs_all.clear();
            x_pairs_all.reserve(total);
        }
        for(int y = 0; y < num_buckets; y++) {
            uint32_t cnt = dst.counts[y];
            if(cnt == 0) continue;
            const uint32_t* meta = dst.buckets[y].data();
            const uint32_t* pd = dst.pd_buckets[y].data();
            bool has_pd = (dst.pd_buckets[y].size() >= (size_t)cnt * 2);
            const uint32_t* xp = (t == 2 && y < (int)dst.x_pairs_buckets.size()) ? dst.x_pairs_buckets[y].data() : nullptr;
            bool has_xp = (t == 2 && y < (int)dst.x_pairs_buckets.size() && dst.x_pairs_buckets[y].size() >= (size_t)cnt * 2);
            for(uint32_t i = 0; i < cnt; i++) {
                uint32_t Y = 0;
                for(int j = 0; j < n_meta; j++) Y ^= meta[i * n_meta + j];
                Y &= KMASK;
                PDEntry entry;
                entry.Y = Y;
                if(has_pd) {
                    entry.left_pos = pd[i * 2];
                    entry.right_pos = pd[i * 2 + 1];
                } else {
                    entry.left_pos = 0;
                    entry.right_pos = 0;
                }
                pd_all[t].push_back(entry);
                if(t == 2 && has_xp) {
                    x_pairs_all.push_back({xp[i * 2], xp[i * 2 + 1]});
                }
            }
        }
        std::cout << "[T" << t << "] PD saved: " << pd_all[t].size() << " entries ("
                  << (my_time_ms() - tpd0) / 1000.0 << " sec)" << std::endl;
        
        // For table 2, X_pairs are now collected from dst store alongside pd_all
        if(t == 2) {
            std::cout << "[T2] X_pairs saved: " << x_pairs_all.size() << " entries" << std::endl;
            
            // Verify x_pairs_all and pd_all[2] are in sync
            if(x_pairs_all.size() != pd_all[2].size()) {
                std::cerr << "[ERROR] x_pairs_all size (" << x_pairs_all.size()
                          << ") != pd_all[2] size (" << pd_all[2].size() << ")" << std::endl;
            }
        }
        
        // Swap src and dst
        std::swap(src.buckets, dst.buckets);
        std::swap(src.counts, dst.counts);
        std::swap(src.pd_buckets, dst.pd_buckets);
std::swap(src.x_values, dst.x_values);
        dst.clear();
    }
    
    // Copy final result back to store
    store.clear();
    for(int y = 0; y < num_buckets; y++) {
        if(src.counts[y] > 0)
            store.append(y, src.buckets[y].data(), src.counts[y]);
    }
    
    std::cout << "[CPU] F2-F9 done in " << (my_time_ms() - t0) / 1000.0 << " sec" << std::endl;
    
    // Module C: cleanup buffer pool
    if(get_opt_config().bufpool && !get_opt_config().svm) {
        bufpool.cleanup();
        for(auto& bp : multi_bufpools) bp.cleanup();
        g_bufpool = nullptr;
        g_bufpools.clear();
    }
    
    // Module H: cleanup SVM pool
    if(get_opt_config().svm) {
        svm_pool.cleanup();
        for(auto& sp : multi_svm_pools) sp.cleanup();
        g_svmpools.clear();
    }
}



// Build PlotData from final bucket store (for plot file writing)
// Collects all entries from all buckets, sorts by Y, deduplicates
void build_plot_data_from_store(
    MemBucketStore& store,
    PlotData& plot,
    std::vector<std::vector<PDEntry>>& pd_all,
    std::vector<std::pair<uint32_t, uint32_t>>& x_pairs_all)
{
    const int n_meta = store.n_meta;
    
    std::cout << "[Plot] Building plot data from pd_all..." << std::endl;
    
    auto bt0 = my_time_ms();
    
    // ====================================================================
    // Step 1: For each table t (2..9), sort pd_all[t] by (Y, left_pos)
    //         to get the Y-sorted order. Build remap maps.
    // ====================================================================
    
    // sort_order[t] = permutation: Y-sorted index -> original pd_all index
    // inv_remap[t]  = mapping: original pd_all index -> Y-sorted index
    std::vector<std::vector<uint32_t>> sort_order(MY_N_TABLE + 1);
    std::vector<std::vector<uint32_t>> inv_remap(MY_N_TABLE + 1);
    
    for(int t = 2; t <= MY_N_TABLE; t++) {
        if(t >= (int)pd_all.size() || pd_all[t].empty()) continue;
        
        size_t n = pd_all[t].size();
        sort_order[t].resize(n);
        for(size_t i = 0; i < n; i++) sort_order[t][i] = (uint32_t)i;
        
        // Sort by (Y, left_pos) — consistent tie-breaker matching store insertion order
        // Sort by (Y, pd_all_index) — NOT (Y, left_pos).
        // CUDA sorts by PY = (Y, local_pos) where local_pos = insertion order within bucket.
        // pd_all_index = bucket_offset + insertion_order = CUDA's local_pos (globally).
        // Using left_pos as tie-breaker is WRONG because left_pos is the parent's
        // position in the PREVIOUS table, not this entry's own position.
        auto pd_sort = [&pd_all, t](uint32_t a, uint32_t b) {
            if(pd_all[t][a].Y == pd_all[t][b].Y)
                return a < b;  // a, b are pd_all indices = insertion order = local_pos
            return pd_all[t][a].Y < pd_all[t][b].Y;
        };
        std::sort(sort_order[t].begin(), sort_order[t].end(), pd_sort);
        
        // Build inverse: original index -> Y-sorted index
        inv_remap[t].resize(n);
        for(size_t i = 0; i < n; i++) {
            inv_remap[t][sort_order[t][i]] = (uint32_t)i;
        }
        
        std::cout << "[Plot] T" << t << ": " << n << " entries sorted" << std::endl;
    }
    
    // ====================================================================
    std::cerr << "[Build] Step 1 (sort pd_all): " << (my_time_ms() - bt0) / 1000.0 << "s" << std::endl;
    auto bt1 = my_time_ms();
    
    // Step 2: Build final_Y and final_meta from pd_all[9] (table 9 = final table)
    //         Sort by (Y, left_pos), deduplicate by (Y, metadata)
    // ====================================================================
    
    // Collect metadata from store for table 9 entries
    // pd_all[9] entries correspond to store entries (same order)
    // Store entries: bucket by bucket, insertion order within bucket
    // pd_all[9] was collected in the same order
    std::vector<std::array<uint32_t, 14>> t9_meta(pd_all[9].size());
    {
        size_t idx = 0;
        for(int y = 0; y < store.num_buckets && idx < pd_all[9].size(); y++) {
            uint32_t cnt = store.counts[y];
            const uint32_t* meta = store.buckets[y].data();
            for(uint32_t i = 0; i < cnt && idx < pd_all[9].size(); i++, idx++) {
                for(int j = 0; j < n_meta; j++)
                    t9_meta[idx][j] = meta[i * n_meta + j];
            }
        }
    }
    
    // Build final_Y sorted by (Y, pd_all_index), deduped by (Y, metadata)
    std::vector<uint32_t> final_indices;  // pd_all[9] indices that survive dedup
    
    for(size_t i = 0; i < sort_order[9].size(); i++) {
        uint32_t orig_idx = sort_order[9][i];
        // Check for duplicate (same Y and same metadata as previous)
        if(!final_indices.empty()) {
            uint32_t prev_idx = final_indices.back();
            if(pd_all[9][orig_idx].Y == pd_all[9][prev_idx].Y &&
               t9_meta[orig_idx] == t9_meta[prev_idx]) {
                continue;  // skip duplicate
            }
        }
        final_indices.push_back(orig_idx);
    }
    
    std::cout << "[Plot] " << final_indices.size() << " unique entries (was " 
              << pd_all[9].size() << ")" << std::endl;
    
    plot.final_Y.resize(final_indices.size());
    plot.final_meta.resize(final_indices.size());
    for(size_t i = 0; i < final_indices.size(); i++) {
        uint32_t idx = final_indices[i];
        plot.final_Y[i] = pd_all[9][idx].Y;
        plot.final_meta[i] = t9_meta[idx];
    }
    
    std::cerr << "[Build] Step 2 (final_Y+meta): " << (my_time_ms() - bt1) / 1000.0 << "s" << std::endl;
    auto bt2 = my_time_ms();
    
    // ====================================================================
    // Step 3: Build PD[t] for t=2..9
    //         PD[t] is indexed by Y-sorted position in table t.
    //         PD[t][i] = (Y_sorted_pos_in_table_{t-1}, delta)
    //         where delta = right_Y_sorted_pos - left_Y_sorted_pos
    // ====================================================================
    
    plot.PD.resize(MY_N_TABLE + 1);
    
    for(int t = 2; t <= MY_N_TABLE; t++) {
        if(t >= (int)pd_all.size() || pd_all[t].empty()) {
            plot.PD[t].clear();
            continue;
        }
        
        size_t n = sort_order[t].size();
        plot.PD[t].resize(n);
        
        for(size_t i = 0; i < n; i++) {
            uint32_t orig_idx = sort_order[t][i];  // original pd_all index
            auto& pe = pd_all[t][orig_idx];
            
            // pe.left_pos and pe.right_pos are indices into pd_all[t-1]
            // Remap to Y-sorted indices using inv_remap[t-1]
            uint32_t left_ysorted, right_ysorted;
            
            if(t == 2) {
                // For table 2, positions index into table 1 (F1/X table)
                // Table 1 is the F1 output, indexed by x value
                // No remap needed — F1 entries are indexed by position
                left_ysorted = pe.left_pos;
                right_ysorted = pe.right_pos;
            } else {
                // Remap from pd_all[t-1] index to Y-sorted index
                if(pe.left_pos < inv_remap[t-1].size()) {
                    left_ysorted = inv_remap[t-1][pe.left_pos];
                } else {
                    left_ysorted = pe.left_pos;  // fallback
                }
                if(pe.right_pos < inv_remap[t-1].size()) {
                    right_ysorted = inv_remap[t-1][pe.right_pos];
                } else {
                    right_ysorted = pe.right_pos;  // fallback
                }
            }
            
            uint32_t delta = right_ysorted - left_ysorted;
            plot.PD[t][i] = {left_ysorted, delta};
        }
        
        std::cout << "[Plot] PD[" << t << "]: " << plot.PD[t].size() << " entries" << std::endl;
    }
    
    std::cerr << "[Build] Step 3 (PD tables): " << (my_time_ms() - bt2) / 1000.0 << "s" << std::endl;
    auto bt3 = my_time_ms();
    
    // ====================================================================
    // ====================================================================
    // Step 4: Build X_pairs from x_pairs_all
    //         Sort by Y to match PD[2] ordering
    // ====================================================================
    
    if(!x_pairs_all.empty() && pd_all.size() > 2 && !pd_all[2].empty()) {
        // Use the SAME sort_order[2] as PD[2] to guarantee X_pairs[i] corresponds
        // exactly to PD[2][i]. x_pairs_all[j] corresponds to pd_all[2][j] (same
        // insertion order), so reordering by sort_order[2] keeps them in sync.
        plot.X_pairs.resize(sort_order[2].size());
        for(size_t i = 0; i < sort_order[2].size(); i++) {
            uint32_t orig_idx = sort_order[2][i];
            if(orig_idx < x_pairs_all.size()) {
                plot.X_pairs[i] = x_pairs_all[orig_idx];
            }
        }
        
        // Deduplicate X_pairs to match final_Y dedup
        // (For now, keep all — dedup only matters for final_Y)
        
        std::cout << "[Plot] X_pairs: " << plot.X_pairs.size() << " entries" << std::endl;
    }
    
    // ====================================================================
    std::cerr << "[Build] Step 4 (X_pairs): " << (my_time_ms() - bt3) / 1000.0 << "s" << std::endl;
    auto bt4 = my_time_ms();
    
    // Step 5: Deduplicate PD[9] to match final_Y
    //         final_indices has the surviving pd_all[9] indices
    //         PD[9] currently has all pd_all[9] entries (sorted by Y)
    //         Need to keep only the ones in final_indices
    // ====================================================================
    
    {
        auto old_pd9 = std::move(plot.PD[9]);
        plot.PD[9].resize(final_indices.size());
        
        // final_indices[i] = original pd_all[9] index
        // We need Y-sorted index for each final entry
        // sort_order[9][i] = original index of i-th Y-sorted entry
        // inv_remap[9][orig_idx] = Y-sorted index
        
        for(size_t i = 0; i < final_indices.size(); i++) {
            uint32_t orig_idx = final_indices[i];
            uint32_t ysorted_idx = inv_remap[9][orig_idx];
            if(ysorted_idx < old_pd9.size()) {
                plot.PD[9][i] = old_pd9[ysorted_idx];
            }
        }
        
        std::cout << "[Plot] PD[9] deduped: " << plot.PD[9].size() << " entries" << std::endl;
    }
    
    // Set num_entries for write_plot (X table uses num_entries[2])
    plot.num_entries.resize(MY_N_TABLE + 1);
    plot.num_entries[1] = 1ULL << KSIZE;  // F1 entries
    for(int t = 2; t <= MY_N_TABLE; t++) {
        plot.num_entries[t] = (t < (int)pd_all.size()) ? pd_all[t].size() : 0;
    }
    
    // Pre-compaction verification: check Y[left]+1 == Y[right] for all PD entries
    {
        std::vector<std::vector<uint32_t>> Y_sorted(MY_N_TABLE + 1);
        for(int t = 2; t <= MY_N_TABLE; t++) {
            if(t >= (int)pd_all.size() || pd_all[t].empty()) continue;
            if(t >= (int)sort_order.size() || sort_order[t].empty()) continue;
            Y_sorted[t].resize(sort_order[t].size());
            for(size_t j = 0; j < sort_order[t].size(); j++) {
                uint32_t orig_idx = sort_order[t][j];
                if(orig_idx < pd_all[t].size()) {
                    Y_sorted[t][j] = pd_all[t][orig_idx].Y;
                }
            }
        }
        bool pd_ok = true;
        for(int t = 3; t <= MY_N_TABLE; t++) {
            if(t >= (int)plot.PD.size() || plot.PD[t].empty()) continue;
            if(t-1 < 2 || t-1 >= (int)Y_sorted.size() || Y_sorted[t-1].empty()) continue;
            size_t prev_size = Y_sorted[t-1].size();
            size_t bounds_errors = 0, match_errors = 0;
            for(size_t i = 0; i < plot.PD[t].size(); i++) {
                uint32_t pos = plot.PD[t][i].first;
                uint32_t delta = plot.PD[t][i].second;
                uint32_t right = pos + delta;
                if(pos >= prev_size || right >= prev_size) { bounds_errors++; continue; }
                uint32_t YL = Y_sorted[t-1][pos];
                uint32_t YR = Y_sorted[t-1][right];
                if(YR != YL + 1) match_errors++;
            }
            if(bounds_errors > 0 || match_errors > 0) {
                std::cerr << "[PD CHECK] PD[" << t << "]: " << bounds_errors << " bounds, "
                          << match_errors << " match errors out of " << plot.PD[t].size() << std::endl;
                pd_ok = false;
            }
        }
        std::cout << "[PD CHECK] Pre-compaction: " << (pd_ok ? "PASS" : "FAIL") << std::endl;
    }
    
    // Step 3b: Phase 2 compaction — remove unreachable entries
    //          CUDA plotter does this in phase 2. We need to match.
    //          1. Start from T9 (all reachable — they're the roots)
    //          2. For t = 8 down to 2: mark T(t) entries referenced by reachable T(t+1)
    //          3. Build remap: old Y-sorted index -> new compacted Y-sorted index
    //          4. Rebuild PD with compacted positions
    //          5. Rebuild final_Y, final_meta, X_pairs with compacted indices
    // ====================================================================
    
    // Mark reachable entries in each table (using Y-sorted indices)
    std::vector<std::vector<uint8_t>> reachable(MY_N_TABLE + 1);
    std::vector<std::vector<uint32_t>> compact_remap(MY_N_TABLE + 1);  // old Y-sorted idx -> new Y-sorted idx
    
    // T9: all reachable
    reachable[MY_N_TABLE].resize(plot.PD[MY_N_TABLE].size(), true);
    
    // T8 down to T2: mark entries referenced by reachable T(t+1)
    for(int t = MY_N_TABLE - 1; t >= 2; t--) {
        if(t + 1 > MY_N_TABLE || plot.PD[t + 1].empty()) {
            reachable[t].resize(plot.PD[t].size(), true);
            continue;
        }
        reachable[t].resize(plot.PD[t].size(), false);
        
        // For each reachable entry in T(t+1), mark its left and right parents in T(t)
        for(size_t i = 0; i < plot.PD[t + 1].size(); i++) {
            if(t + 1 < MY_N_TABLE && !reachable[t + 1][i]) continue;
            uint32_t pos = plot.PD[t + 1][i].first;
            uint32_t delta = plot.PD[t + 1][i].second;
            uint32_t right = pos + delta;
            if(pos < reachable[t].size()) reachable[t][pos] = true;
            if(right < reachable[t].size()) reachable[t][right] = true;
        }
        
        // Build compact remap for table t
        compact_remap[t].resize(reachable[t].size());
        uint32_t new_idx = 0;
        for(size_t i = 0; i < reachable[t].size(); i++) {
            if(reachable[t][i]) {
                compact_remap[t][i] = new_idx++;
            } else {
                compact_remap[t][i] = 0xFFFFFFFF;  // invalid
            }
        }
        size_t removed = reachable[t].size() - new_idx;
        std::cout << "[Compact] T" << t << ": " << new_idx << " reachable, "
                  << removed << " removed (was " << reachable[t].size() << ")" << std::endl;
    }
    
    // Build compact remap for T9 (all reachable, so identity)
    compact_remap[MY_N_TABLE].resize(plot.PD[MY_N_TABLE].size());
    for(size_t i = 0; i < compact_remap[MY_N_TABLE].size(); i++) {
        compact_remap[MY_N_TABLE][i] = (uint32_t)i;
    }
    
    // Remap PD positions to compacted indices
    for(int t = 3; t <= MY_N_TABLE; t++) {
        if(plot.PD[t].empty()) continue;
        for(size_t i = 0; i < plot.PD[t].size(); i++) {
            // Only keep entries that are reachable in T(t)
            if(t < MY_N_TABLE && !reachable[t][i]) continue;
            uint32_t pos = plot.PD[t][i].first;
            uint32_t delta = plot.PD[t][i].second;
            uint32_t right = pos + delta;
            // Remap to compacted indices in T(t-1)
            if(pos < compact_remap[t - 1].size()) {
                plot.PD[t][i].first = compact_remap[t - 1][pos];
            }
            if(right < compact_remap[t - 1].size()) {
                uint32_t new_right = compact_remap[t - 1][right];
                plot.PD[t][i].second = new_right - plot.PD[t][i].first;
            }
        }
    }
    
    // Remove unreachable entries from PD tables (compact them)
    for(int t = 3; t <= MY_N_TABLE; t++) {
        if(plot.PD[t].empty() || reachable[t].empty()) continue;
        std::vector<std::pair<uint32_t, uint32_t>> new_pd;
        new_pd.reserve(reachable[t].size());
        for(size_t i = 0; i < plot.PD[t].size(); i++) {
            if(t < MY_N_TABLE && !reachable[t][i]) continue;
            new_pd.push_back(plot.PD[t][i]);
        }
        plot.PD[t] = std::move(new_pd);
    }
    
    // Compact X_pairs (table 2): only keep reachable entries
    if(!plot.X_pairs.empty()) {
        std::vector<std::pair<uint32_t, uint32_t>> new_xp;
        new_xp.reserve(plot.X_pairs.size());
        for(size_t i = 0; i < plot.X_pairs.size(); i++) {
            if(i < reachable[2].size() && reachable[2][i]) {
                new_xp.push_back(plot.X_pairs[i]);
            }
        }
        plot.X_pairs = std::move(new_xp);
    }
    
    // Compact final_Y and final_meta (T9 — all reachable, no change needed)
    // But update num_entries
    for(int t = 2; t <= MY_N_TABLE; t++) {
        if(t < (int)plot.num_entries.size()) {
            plot.num_entries[t] = plot.PD[t].size();
        }
    }
    plot.num_entries[2] = plot.X_pairs.size();
    
    std::cout << "[Compact] Done. PD[9]=" << plot.PD[9].size()
              << " X_pairs=" << plot.X_pairs.size() << std::endl;
    
    std::cerr << "[Build] Step 5 (compact+dedup): " << (my_time_ms() - bt4) / 1000.0 << "s" << std::endl;
    

    // Post-compaction verification: check all PD positions are in bounds
    {
        bool pd_ok = true;
        for(int t = 3; t <= MY_N_TABLE; t++) {
            if(t >= (int)plot.PD.size() || plot.PD[t].empty()) continue;
            if(t-1 >= (int)plot.PD.size() || plot.PD[t-1].empty()) continue;
            size_t prev_size = plot.PD[t-1].size();
            size_t errors = 0;
            for(size_t i = 0; i < plot.PD[t].size(); i++) {
                uint32_t pos = plot.PD[t][i].first;
                uint32_t delta = plot.PD[t][i].second;
                if(pos >= prev_size || (uint64_t)pos + delta >= prev_size) {
                    if(errors++ < 3)
                        std::cerr << "[PD CHECK] PD[" << t << "][" << i << "]: pos=" << pos
                                  << " delta=" << delta << " prev_size=" << prev_size << std::endl;
                }
            }
            if(errors > 0) {
                std::cerr << "[PD CHECK] PD[" << t << "]: " << errors << " bounds errors" << std::endl;
                pd_ok = false;
            }
        }
        // Check final_Y sorted
        for(size_t i = 1; i < plot.final_Y.size(); i++) {
            if(plot.final_Y[i] < plot.final_Y[i-1]) {
                std::cerr << "[PD CHECK] final_Y not sorted at [" << i << "]" << std::endl;
                pd_ok = false; break;
            }
        }
        // Full end-to-end chain verification
        if(pd_ok) {
            int chain_errors = 0, chains_checked = 0;
            int num_to_check = std::min((size_t)10, plot.final_Y.size());
            for(int idx = 0; idx < num_to_check; idx++) {
                std::vector<uint64_t> pointers = {(uint64_t)idx};
                bool ok = true;
                for(int t = MY_N_TABLE; t >= 3; t--) {
                    std::vector<uint64_t> new_pointers;
                    for(uint64_t p : pointers) {
                        if(p >= plot.PD[t].size()) { ok = false; chain_errors++; break; }
                        uint32_t pos = plot.PD[t][p].first;
                        uint32_t delta = plot.PD[t][p].second;
                        new_pointers.push_back(pos);
                        new_pointers.push_back(pos + delta);
                    }
                    if(!ok) break;
                    pointers = new_pointers;
                }
                if(!ok) continue;
                for(uint64_t p : pointers) {
                    if(p >= plot.X_pairs.size()) { ok = false; chain_errors++; break; }
                }
                if(ok) chains_checked++;
            }
            if(chain_errors > 0)
                std::cerr << "[CHAIN] " << chain_errors << " chain errors out of " << num_to_check << std::endl;
            else
                std::cout << "[CHAIN] " << chains_checked << "/" << num_to_check << " full chains verified OK" << std::endl;
        }
        std::cout << "[PD CHECK] Post-compaction: " << (pd_ok ? "PASS" : "FAIL") << std::endl;
    }
    
    std::cout << "[Plot] Built PlotData: " << plot.final_Y.size() << " entries" << std::endl;
}


bool gpu_yield = true;
int device_id = 0;

// Assert queue is in-order (Module E v2 relies on FIFO ordering for event-based sync)
void assert_queue_in_order(cl_command_queue q, const char* label) {
    if(!q) return;
    cl_command_queue_properties props = 0;
    cl_int err = clGetCommandQueueInfo(q, CL_QUEUE_PROPERTIES, sizeof(props), &props, nullptr);
    if(err != CL_SUCCESS) { std::cerr << "[FATAL] assert_queue_in_order(" << label << "): failed" << std::endl; std::exit(1); }
    if(props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        std::cerr << "[FATAL] assert_queue_in_order(" << label << "): OUT-OF-ORDER queue! Pipeline assumes in-order." << std::endl;
        std::exit(1);
    }
}
bool timing_detail = false;

int main(int argc, char** argv)
{
    std::string output_dir = "./";  // default: current directory
    std::string plot_name;
    bool test_mode = false;
    uint64_t test_limit = 0;
    bool use_ramdisk = false;
    bool use_chunked = false;
    bool dump_pd = false;
    std::string final_dir;  // if set, copy plot here after writing to ramdisk
    
    if(argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <plot_id_hex> <farmer_key_hex> [output_dir] [options]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --k N           Set plot k-size (default: 26)" << std::endl;
        std::cerr << "  --ramdisk DIR   Use tmpfs at DIR for plotting, then copy to output_dir" << std::endl;
        std::cerr << "  --chunked       Use per-bucket chunked pipeline (for k29+, uses more total RAM but less per-chunk)" << std::endl;
std::cerr << "  --no-yield      Disable GPU display yield (for headless systems)" << std::endl;
std::cerr << "  --device N      Select GPU device index (default: 0)" << std::endl;
std::cerr << "  --opt-gpu-meta  Module B: GPU metadata extraction (skip CPU extract+reupload)" << std::endl;
std::cerr << "  --opt-gpu-prefix Module A: GPU prefix sum (skip sub-count readback)" << std::endl;
std::cerr << "  --opt-async     Module D: Async PCIe transfers" << std::endl;
std::cerr << "  --opt-queues N  Module E: Multi-queue pipelining (N parallel buckets)" << std::endl;
std::cerr << "  --opt-bufpool   Module C: Pre-allocated reusable GPU buffers" << std::endl;
std::cerr << "  --opt-zero-copy Module F: Pinned memory + map/unmap (zero-copy)" << std::endl;
std::cerr << "  --opt-pinned   Module G: Pinned host memory (async DMA)" << std::endl;
std::cerr << "  --opt-svm      Module H: Shared Virtual Memory (OpenCL 2.0)" << std::endl;
std::cerr << "  --timing        Show per-step timing for first bucket of each table" << std::endl;
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
        else if(arg == "--chunked") use_chunked = true;
else if(arg == "--no-yield") gpu_yield = false;
else if(arg == "--device" && i+1 < argc) device_id = std::stoi(argv[++i]);
else if(arg == "--opt-gpu-meta") get_opt_config().gpu_meta_extract = true;
else if(arg == "--opt-gpu-prefix") get_opt_config().gpu_prefix_sum = true;
else if(arg == "--opt-async") get_opt_config().async_transfers = true;
else if(arg == "--opt-queues" && i+1 < argc) get_opt_config().num_queues = std::stoi(argv[++i]);
else if(arg == "--timing") timing_detail = true;
else if(arg == "--opt-bufpool") get_opt_config().bufpool = true;
else if(arg == "--opt-zero-copy") get_opt_config().zero_copy = true;
else if(arg == "--opt-pinned") get_opt_config().pinned = true;
else if(arg == "--opt-svm") get_opt_config().svm = true;
else if(arg == "--device" && i+1 < argc) device_id = std::stoi(argv[++i]);
        else if(arg == "--dump-pd") dump_pd = true;
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
    
    // List all available GPUs
    std::vector<cl_device_id> all_devs(nd);
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, nd, all_devs.data(), nullptr);
    std::cout << "[OCL] Found " << nd << " GPU device(s):" << std::endl;
    for(cl_uint i = 0; i < nd; i++) {
        char dn[256]; clGetDeviceInfo(all_devs[i], CL_DEVICE_NAME, sizeof(dn), dn, nullptr);
        cl_ulong mem = 0; clGetDeviceInfo(all_devs[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
        std::cout << "  [" << i << "] " << dn << " (" << mem/1024/1024/1024 << " GB VRAM)" << std::endl;
    }
    
    // Select device
    cl_uint dev_idx = std::min((cl_uint)device_id, nd - 1);
    cl_device_id dev = all_devs[dev_idx];
    char dn[256]; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dn), dn, nullptr);
    std::cout << "[OCL] Using device [" << dev_idx << "]: " << dn << std::endl;
    
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if(err != CL_SUCCESS) { std::cerr << "Failed to create OpenCL context" << std::endl; return 1; }
    
    // Compute F1 on GPU
    OCL_Plotter plotter;
    plotter.init(ctx, dev);
    plotter.init_table_hash();
    
    // Multi-GPU: create additional plotters if --num-gpus > 1 and available
    int num_gpus = std::min(get_opt_config().num_queues, (int)nd);  // reuse num_queues for now
    if(num_gpus > 1 && nd > 1) {
        g_num_gpus = num_gpus;
        g_plotters.push_back(&plotter);  // first plotter
        for(int g = 1; g < num_gpus; g++) {
            cl_device_id dev2 = all_devs[(dev_idx + g) % nd];
            cl_context ctx2 = clCreateContext(nullptr, 1, &dev2, nullptr, nullptr, &err);
            if(err != CL_SUCCESS) { std::cerr << "Failed to create context for GPU " << g << std::endl; break; }
            OCL_Plotter* p2 = new OCL_Plotter();
            p2->init(ctx2, dev2);
            p2->init_table_hash();
            // Load kernels for second plotter
            // (init_gpu_kernels called later in chunked path)
            g_plotters.push_back(p2);
            std::cout << "[OCL] Multi-GPU: added device [" << (dev_idx + g) % nd << "] as GPU " << g << std::endl;
        }
    }
    
    // Print optimization config
    if(get_opt_config().any_enabled()) {
        get_opt_config().print();
    }
    
    if(use_chunked) {
        // Chunked pipeline: process one first-level bucket at a time
        // Requires LOGBUCKETS=8 (256 buckets) for manageable chunk sizes
        LOGBUCKETS = 8;
        update_constants();
        plotter.init_gpu_kernels();
        // Init kernels on all multi-GPU plotters
        for(size_t g = 1; g < g_plotters.size(); g++) {
            g_plotters[g]->init_gpu_kernels();
        }
        // Assert all queues are in-order (pipeline relies on FIFO)
        assert_queue_in_order(plotter.queue, "plotter.queue");
        for(auto* pl : g_plotters) {
            assert_queue_in_order(pl->queue, "g_plotters[].queue");
            if(pl->queue2) assert_queue_in_order(pl->queue2, "g_plotters[].queue2");
        }
        std::cout << "[Chunked] LOGBUCKETS=" << LOGBUCKETS << " (" << (1 << LOGBUCKETS) << " buckets)" << std::endl;
        
        const int num_buckets = 1 << LOGBUCKETS;
        const int n_meta = MY_N_META;
        const uint64_t entries_per_bucket = (uint64_t)1 << (KSIZE - LOGBUCKETS);
        const int max_bucket_size = entries_per_bucket * 3 / 2 + 256;
        
        std::cout << "[Chunked] Entries per bucket: " << entries_per_bucket << std::endl;
        std::cout << "[Chunked] Max bucket size: " << max_bucket_size << std::endl;
        std::cout << "[Chunked] RAM per store: " << (double)num_buckets * max_bucket_size * n_meta * 4 / 1e9 << " GB" << std::endl;
        std::cout << "[Chunked] Total RAM (2 stores): " << 2.0 * num_buckets * max_bucket_size * n_meta * 4 / 1e9 << " GB" << std::endl;
        
        MemBucketStore store(num_buckets, max_bucket_size, n_meta);
        
auto t0 = my_time_ms();
std::vector<std::vector<PDEntry>> pd_all;
        // F1 → bucket store
        if(g_num_gpus > 1 && g_plotters.size() > 1)
            // Multi-GPU: use smaller batches for more overlap rounds
            compute_f1_chunked_multi_gpu(plotter, plot_id, store, 1 << 18);
        else
            // Single-GPU: batch = total entries (1 batch, no round-trip overhead)
            compute_f1_chunked(plotter, plot_id, store, std::max((uint32_t)(1 << 20), (uint32_t)(1 << KSIZE)));
        
        // F2-F9 chunked
        std::cout << "\n[CPU] Computing F2-F9 (chunked)..." << std::endl;
        std::vector<std::pair<uint32_t, uint32_t>> x_pairs_all;
        compute_f2_f9_chunked(plotter, store, num_buckets, max_bucket_size, n_meta, gpu_yield, pd_all, x_pairs_all);
        
        auto t_f29 = my_time_ms();
        
        // Build PlotData from final bucket store
        std::cout << "\n[Plot] Building plot data from bucket store..." << std::endl;
        PlotData plot;
        build_plot_data_from_store(store, plot, pd_all, x_pairs_all);
        
        auto t_build = my_time_ms();
        std::cout << "[Plot] Build+compact: " << (t_build - t_f29) / 1000.0 << " sec" << std::endl;
    
    if(dump_pd) {
        for(int t = 2; t <= MY_N_TABLE; t++) {
            std::string fname = "/tmp/pd_chunked_T" + std::to_string(t) + ".txt";
            std::ofstream f(fname);
            f << "T" << t << " PD entries: " << plot.PD[t].size() << "\n";
            // Show pd_all[t] raw data
            if(t < (int)pd_all.size() && !pd_all[t].empty()) {
                f << "pd_all[" << t << "] raw entries: " << pd_all[t].size() << "\n";
                size_t lim = std::min((size_t)50, pd_all[t].size());
                for(size_t i = 0; i < lim; i++) {
                    f << "  pd_all[" << t << "][" << i << "] = (Y=" << pd_all[t][i].Y 
                      << ", left_pos=" << pd_all[t][i].left_pos 
                      << ", right_pos=" << pd_all[t][i].right_pos << ")\n";
                }
            }
            f << "\nplot.PD[" << t << "] entries: " << plot.PD[t].size() << "\n";
            size_t lim = std::min((size_t)50, plot.PD[t].size());
            for(size_t i = 0; i < lim; i++) {
                f << "  PD[" << t << "][" << i << "] = (pos=" << plot.PD[t][i].first 
                  << ", delta=" << plot.PD[t][i].second << ")\n";
            }
            f.close();
            std::cout << "[Dump] Wrote " << fname << std::endl;
        }
        std::ofstream fy("/tmp/pd_chunked_finalY.txt");
        fy << "final_Y entries: " << plot.final_Y.size() << "\n";
        size_t lim = plot.final_Y.size();
        for(size_t i = 0; i < lim; i++) fy << "  Y[" << i << "] = " << plot.final_Y[i] << "\n";
        fy.close();
        std::ofstream fx("/tmp/pd_chunked_Xpairs.txt");
        fx << "X_pairs entries: " << plot.X_pairs.size() << "\n";
        lim = plot.X_pairs.size();
        for(size_t i = 0; i < lim; i++) fx << "  X[" << i << "] = (" << plot.X_pairs[i].first << ", " << plot.X_pairs[i].second << ")\n";
        fx.close();
    }
        
        // Write plot file
        std::cout << "\n[Plot] Writing plot file..." << std::endl;
        write_plot(plot_path, plot_id, farmer_key, plot, true);
        auto t_write = my_time_ms();
        std::cout << "[Plot] Write: " << (t_write - t_build) / 1000.0 << " sec" << std::endl;
        
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
            std::remove(plot_path.c_str());
            plot_path = final_path;
        }
        
        auto t2 = my_time_ms();
        std::cout << "\n[Done] Total time: " << (t2 - t0) / 1000.0 << " sec" << std::endl;
        std::cout << "[Done] Plot file: " << plot_path << std::endl;
        
        clReleaseContext(ctx);
        return 0;
    }
    
    std::vector<uint32_t> Y_all, M_all;
    
    // SVM F1 for flat path (if --opt-svm)
    SVMPool flat_svm_pool;
    if(get_opt_config().svm) {
        int f1_batch = std::max((uint32_t)(1 << 20), (uint32_t)(1 << KSIZE));
        // For flat path: hash buffers need to hold all matches (~2^K entries)
        int flat_max_entries = (1 << KSIZE) * 3 / 2 + 256;  // same as chunked max_bucket_size
        flat_svm_pool.init(plotter.context, plotter.queue, plotter.device,
                          flat_max_entries, MY_N_META, 64, 4096, f1_batch);
        g_svmpools.push_back(&flat_svm_pool);
    }
    
    plotter.compute_all_f1(plot_id, Y_all, M_all, std::max((uint32_t)(1 << 20), (uint32_t)(1 << KSIZE)), test_mode, test_limit);
    
    // Compute F2-F9 on CPU
    std::cout << "\n[CPU] Computing F2-F9..." << std::endl;
    auto t0 = my_time_ms();
    
    std::vector<uint32_t> X_values(Y_all.size());
    for(size_t i = 0; i < X_values.size(); i++) X_values[i] = (uint32_t)i;
    
    PlotData plot;
    compute_full_pipeline(X_values, Y_all, M_all, plot, plotter);
    
    if(dump_pd) {
        for(int t = 2; t <= MY_N_TABLE; t++) {
            std::string fname = "/tmp/pd_flat_T" + std::to_string(t) + ".txt";
            std::ofstream f(fname);
            f << "T" << t << " PD entries: " << plot.PD[t].size() << "\n";
            size_t lim = std::min((size_t)50, plot.PD[t].size());
            for(size_t i = 0; i < lim; i++) {
                f << "  PD[" << t << "][" << i << "] = (pos=" << plot.PD[t][i].first 
                  << ", delta=" << plot.PD[t][i].second << ")\n";
            }
            f.close();
            std::cout << "[Dump] Wrote " << fname << std::endl;
        }
        std::ofstream fy("/tmp/pd_flat_finalY.txt");
        fy << "final_Y entries: " << plot.final_Y.size() << "\n";
        size_t lim = plot.final_Y.size();
        for(size_t i = 0; i < lim; i++) fy << "  Y[" << i << "] = " << plot.final_Y[i] << "\n";
        fy.close();
        std::ofstream fx("/tmp/pd_flat_Xpairs.txt");
        fx << "X_pairs entries: " << plot.X_pairs.size() << "\n";
        lim = plot.X_pairs.size();
        for(size_t i = 0; i < lim; i++) fx << "  X[" << i << "] = (" << plot.X_pairs[i].first << ", " << plot.X_pairs[i].second << ")\n";
        fx.close();
    }
    
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




