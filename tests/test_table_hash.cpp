// Test: table hash kernel — hash matched LR pairs to produce Y_out + M_out
#include "../src/gpu_device.h"
#include "../src/plot_config.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <fstream>

int main() {
    std::cout << "=== test_table_hash ===" << std::endl;

    // Initialize GPU
    mmx::GPUDevice gpu;
    gpu.init(0);
    gpu.print_info();

    // Find kernel file
    std::string kernel_path;
    for(const auto& path : {"kernels/table_hash.cl", "../kernels/table_hash.cl", "../../kernels/table_hash.cl"}) {
        std::ifstream f(path);
        if(f.good()) { kernel_path = path; break; }
    }
    if(kernel_path.empty()) {
        throw std::runtime_error("Cannot find kernels/table_hash.cl");
    }
    gpu.load_program_from_file("table_hash", kernel_path);
    cl_kernel kernel = gpu.get_kernel("hash_table_lr");

    // Test: hash 4 LR pairs
    const uint32_t num_matches = 4;
    const uint32_t ksize = 22;
    const uint32_t kmask = (1u << ksize) - 1;

    // Simulate M_curr (previous table's metadata): 14 uint32 per entry
    const uint32_t num_entries_m = 16;  // small test size
    std::vector<uint32_t> M_curr(num_entries_m * mmx::N_META, 0);
    for(uint32_t i = 0; i < num_entries_m * mmx::N_META; ++i) M_curr[i] = i;

    // LR pairs: (orig_L, orig_R) = indices into M_curr
    std::vector<uint32_t> LR(num_matches * 2);
    LR[0] = 0; LR[1] = 1;   // pair 0: entries[0] + entries[1]
    LR[2] = 2; LR[3] = 3;   // pair 1: entries[2] + entries[3]
    LR[4] = 5; LR[5] = 7;   // pair 2: entries[5] + entries[7]
    LR[6] = 10; LR[7] = 11;  // pair 3: entries[10] + entries[11]

    // Allocate GPU buffers
    cl_int err;
    cl_mem M_buf = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  num_entries_m * mmx::N_META * sizeof(uint32_t), M_curr.data(), &err);
    mmx::GPUDevice::check(err, "M_buf");

    cl_mem LR_buf = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   num_matches * 2 * sizeof(uint32_t), LR.data(), &err);
    mmx::GPUDevice::check(err, "LR_buf");

    cl_mem Y_buf = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY,
                                  num_matches * sizeof(uint32_t), nullptr, &err);
    mmx::GPUDevice::check(err, "Y_buf");

    cl_mem M_out_buf = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY,
                                      num_matches * mmx::N_META * sizeof(uint32_t), nullptr, &err);
    mmx::GPUDevice::check(err, "M_out_buf");

    // Set kernel args
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &M_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &LR_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &Y_buf);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &M_out_buf);
    clSetKernelArg(kernel, 4, sizeof(uint32_t), &kmask);
    clSetKernelArg(kernel, 5, sizeof(uint32_t), &num_matches);
    const uint32_t num_total = num_entries_m;
    clSetKernelArg(kernel, 6, sizeof(uint32_t), &num_total);

    // Launch kernel
    size_t local = 64;
    size_t global = ((num_matches + local - 1) / local) * local;
    err = clEnqueueNDRangeKernel(gpu.queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    mmx::GPUDevice::check(err, "enqueue hash_table_lr");
    gpu.finish();

    // Read results
    std::vector<uint32_t> Y_gpu(num_matches);
    std::vector<uint32_t> M_gpu(num_matches * mmx::N_META);
    clEnqueueReadBuffer(gpu.queue, Y_buf, CL_TRUE, 0, num_matches * sizeof(uint32_t), Y_gpu.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(gpu.queue, M_out_buf, CL_TRUE, 0, num_matches * mmx::N_META * sizeof(uint32_t), M_gpu.data(), 0, nullptr, nullptr);

    // Print results
    std::cout << "\nTable hash results (k22):" << std::endl;
    for(uint32_t i = 0; i < num_matches; ++i) {
        uint32_t Y = Y_gpu[i] & kmask;
        std::cout << "  pair " << i << " (L=" << LR[i*2] << ", R=" << LR[i*2+1] << ") → Y=" << Y << std::endl;
    }

    // Verify: Y values in range
    bool all_ok = true;
    for(uint32_t i = 0; i < num_matches; ++i) {
        if(Y_gpu[i] >= (1u << ksize)) {
            std::cout << "  FAIL: Y value out of range" << std::endl;
            all_ok = false;
        }
    }

    // Cleanup
    clReleaseMemObject(M_buf);
    clReleaseMemObject(LR_buf);
    clReleaseMemObject(Y_buf);
    clReleaseMemObject(M_out_buf);

    if(!all_ok) {
        std::cout << "=== TESTS FAILED ===" << std::endl;
        return 1;
    }

    std::cout << "\nKernel executes correctly. All Y values in range." << std::endl;
    std::cout << "=== TEST PASSED ===" << std::endl;
    return 0;
}
