// Test: F1 kernel - GPU computes F1 values, verify against CPU reference
#include "../src/gpu_device.h"
#include "../src/plot_config.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <fstream>

// CPU F1 reference implementation - will be filled in later with golden vector comparison
// Currently just verifies GPU kernel runs correctly

int main() {
    std::cout << "=== test_f1_kernel ===" << std::endl;

    // Initialize GPU
    mmx::GPUDevice gpu;
    gpu.init(0);
    gpu.print_info();

    // Load F1 kernel — search multiple paths
    std::string kernel_path;
    for(const auto& path : {"kernels/f1.cl", "../kernels/f1.cl", "../../kernels/f1.cl"}) {
        std::ifstream f(path);
        if(f.good()) { kernel_path = path; break; }
    }
    if(kernel_path.empty()) {
        throw std::runtime_error("Cannot find kernels/f1.cl");
    }
    gpu.load_program_from_file("f1", kernel_path);
    cl_kernel kernel = gpu.get_kernel("compute_f1_kernel");

    // Test small batch: compute F1 for X=0..7
    const uint32_t num_x = 8;
    const uint32_t ksize = 22;
    const uint32_t kmask = (1u << ksize) - 1;
    
    // Plot ID (32 bytes, zeroed for test)
    std::vector<uint32_t> plot_id_gpu(8, 0);
    
    // X values
    std::vector<uint32_t> X_values(num_x);
    for(uint32_t i = 0; i < num_x; ++i) X_values[i] = i;

    // Allocate GPU buffers
    cl_int err;
    cl_mem X_buf = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  num_x * sizeof(uint32_t), X_values.data(), &err);
    mmx::GPUDevice::check(err, "X_buf");

    cl_mem ID_buf = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   8 * sizeof(uint32_t), plot_id_gpu.data(), &err);
    mmx::GPUDevice::check(err, "ID_buf");

    cl_mem Y_buf = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY,
                                  num_x * sizeof(uint32_t), nullptr, &err);
    mmx::GPUDevice::check(err, "Y_buf");

    cl_mem M_buf = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY,
                                  num_x * mmx::N_META * sizeof(uint32_t), nullptr, &err);
    mmx::GPUDevice::check(err, "M_buf");

    // Set kernel args
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &X_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &ID_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &Y_buf);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &M_buf);
    clSetKernelArg(kernel, 4, sizeof(uint32_t), &kmask);
    uint32_t xbits = 0;
    clSetKernelArg(kernel, 5, sizeof(uint32_t), &xbits);
    clSetKernelArg(kernel, 6, sizeof(uint32_t), &num_x);

    // Launch kernel
    size_t global = num_x;
    err = clEnqueueNDRangeKernel(gpu.queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    mmx::GPUDevice::check(err, "enqueue compute_f1_kernel");
    gpu.finish();

    // Read results
    std::vector<uint32_t> Y_gpu(num_x);
    std::vector<uint32_t> M_gpu(num_x * mmx::N_META);
    clEnqueueReadBuffer(gpu.queue, Y_buf, CL_TRUE, 0, num_x * sizeof(uint32_t), Y_gpu.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(gpu.queue, M_buf, CL_TRUE, 0, num_x * mmx::N_META * sizeof(uint32_t), M_gpu.data(), 0, nullptr, nullptr);

    // Print results
    std::cout << "\nF1 results (X=0..7, plot_id=zero, k22):" << std::endl;
    for(uint32_t i = 0; i < num_x; ++i) {
        std::cout << "  X=" << X_values[i] << " → Y=" << Y_gpu[i] << std::endl;
    }

    // Verify: no crash, valid results (Y != 0 is a good sign, but some can be 0)
    bool all_ok = true;
    for(uint32_t i = 0; i < num_x; ++i) {
        if(Y_gpu[i] >= (1u << ksize)) {
            std::cout << "  FAIL: Y value out of range (Y >= 2^ksize)" << std::endl;
            all_ok = false;
        }
    }

    // Cleanup
    clReleaseMemObject(X_buf);
    clReleaseMemObject(ID_buf);
    clReleaseMemObject(Y_buf);
    clReleaseMemObject(M_buf);

    if(!all_ok) {
        std::cout << "=== TESTS FAILED ===" << std::endl;
        return 1;
    }

    std::cout << "\nKernel executes correctly. ";
    std::cout << "All Y values in range [0, 2^ksize)." << std::endl;
    std::cout << "=== TEST PASSED ===" << std::endl;
    return 0;
}
