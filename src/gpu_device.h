#pragma once

#include <CL/cl.h>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <unordered_map>

namespace mmx {

// OpenCL device abstraction — wraps context, queue, program, kernel management.
// Designed to be the single point of contact with the OpenCL API.
class GPUDevice {
public:
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;

    std::string device_name;
    std::string vendor;
    bool is_amd = false;
    bool is_nvidia = false;
    uint64_t max_mem_alloc = 0;
    uint64_t global_mem_size = 0;
    uint32_t compute_units = 0;
    uint32_t max_workgroup_size = 0;

    // Kernel programs (loaded from .cl files)
    std::unordered_map<std::string, cl_program> programs;
    std::unordered_map<std::string, cl_kernel> kernels;

    GPUDevice() = default;
    ~GPUDevice();

    // Initialize with device index (default 0)
    void init(int device_index = 0);

    // Load and compile a kernel from source string with options
    void load_program(const std::string& name, const std::string& source,
                      const std::string& options = "");

    // Load and compile a kernel from a .cl file
    void load_program_from_file(const std::string& name, const std::string& filepath,
                                const std::string& options = "");

    // Get a compiled kernel by name
    cl_kernel get_kernel(const std::string& name);

    // Check if a kernel is loaded
    bool has_kernel(const std::string& name) const { return kernels.count(name) > 0; }

    // Finish all queued operations
    void finish() { if(queue) clFinish(queue); }

    // Print device info
    void print_info() const;

    // Error checking helper
    static void check(cl_int err, const std::string& msg = "");

    // Get build log for a program
    std::string get_build_log(cl_program program) const;

private:
    void query_device_info();
};

} // namespace mmx
