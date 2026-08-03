#include "gpu_device.h"
#include <fstream>
#include <sstream>
#include <cstring>

namespace mmx {

GPUDevice::~GPUDevice()
{
    for(auto& [name, kernel] : kernels) {
        if(kernel) clReleaseKernel(kernel);
    }
    for(auto& [name, program] : programs) {
        if(program) clReleaseProgram(program);
    }
    if(queue) clReleaseCommandQueue(queue);
    if(context) clReleaseContext(context);
}

void GPUDevice::init(int device_index)
{
    cl_int err;

    // Get platforms
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    check(err, "clGetPlatformIDs count");
    if(num_platforms == 0) throw std::runtime_error("No OpenCL platforms found");

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    check(err, "clGetPlatformIDs");

    // Find a device across all platforms
    int current_index = 0;
    for(cl_uint p = 0; p < num_platforms && !device; ++p) {
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        if(err != CL_SUCCESS || num_devices == 0) continue;

        std::vector<cl_device_id> devices(num_devices);
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
        check(err, "clGetDeviceIDs");

        for(cl_uint d = 0; d < num_devices; ++d) {
            if(current_index == (int)device_index) {
                platform = platforms[p];
                device = devices[d];
                break;
            }
            current_index++;
        }
    }

    if(!device) {
        throw std::runtime_error("GPU device " + std::to_string(device_index) + " not found");
    }

    // Create context
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
    };
    context = clCreateContext(props, 1, &device, nullptr, nullptr, &err);
    check(err, "clCreateContext");

    // Create command queue (in-order by default)
    queue = clCreateCommandQueue(context, device, 0, &err);
    check(err, "clCreateCommandQueue");

    query_device_info();
}

void GPUDevice::query_device_info()
{
    char name_buf[256] = {};
    char vendor_buf[256] = {};

    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name_buf), name_buf, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor_buf), vendor_buf, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc), &max_mem_alloc, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_workgroup_size), &max_workgroup_size, nullptr);

    device_name = name_buf;
    vendor = vendor_buf;

    // Detect vendor
    is_amd = (vendor.find("AMD") != std::string::npos ||
              vendor.find("Advanced Micro") != std::string::npos ||
              device_name.find("gfx") != std::string::npos ||
              device_name.find("Radeon") != std::string::npos ||
              device_name.find("Instinct") != std::string::npos);
    is_nvidia = (vendor.find("NVIDIA") != std::string::npos ||
                 device_name.find("NVIDIA") != std::string::npos ||
                 device_name.find("Tesla") != std::string::npos ||
                 device_name.find("GeForce") != std::string::npos ||
                 device_name.find("RTX") != std::string::npos ||
                 device_name.find("Quadro") != std::string::npos);
}

void GPUDevice::load_program(const std::string& name, const std::string& source,
                             const std::string& options)
{
    cl_int err;
    const char* src = source.c_str();
    size_t len = source.size();

    cl_program program = clCreateProgramWithSource(context, 1, &src, &len, &err);
    check(err, "clCreateProgramWithSource: " + name);

    // Build options: use CL1.2 standard, vendor-specific tweaks
    std::string build_opts = "-cl-std=CL1.2 -Werror";
    if(is_nvidia) {
        // NVIDIA: more registers per thread with smaller workgroups
        build_opts += " -DNVIDIA_GPU=1";
    }
    if(is_amd) {
        build_opts += " -DAMD_GPU=1";
    }
    if(!options.empty()) {
        build_opts += " " + options;
    }

    err = clBuildProgram(program, 1, &device, build_opts.c_str(), nullptr, nullptr);
    if(err != CL_SUCCESS) {
        std::string log = get_build_log(program);
        clReleaseProgram(program);
        throw std::runtime_error("Build failed for '" + name + "': " + log);
    }

    programs[name] = program;
}

void GPUDevice::load_program_from_file(const std::string& name, const std::string& filepath,
                                       const std::string& options)
{
    std::ifstream file(filepath);
    if(!file.is_open()) {
        throw std::runtime_error("Cannot open kernel file: " + filepath);
    }
    std::stringstream ss;
    ss << file.rdbuf();
    load_program(name, ss.str(), options);
}

cl_kernel GPUDevice::get_kernel(const std::string& kernel_name)
{
    auto it = kernels.find(kernel_name);
    if(it != kernels.end()) return it->second;

    // Find which program contains this kernel
    for(auto& [prog_name, program] : programs) {
        cl_int err;
        cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
        if(err == CL_SUCCESS) {
            kernels[kernel_name] = kernel;
            return kernel;
        }
    }

    throw std::runtime_error("Kernel not found: " + kernel_name);
}

std::string GPUDevice::get_build_log(cl_program program) const
{
    size_t log_size = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::string log(log_size, '\0');
    if(log_size > 0) {
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
    }
    return log;
}

void GPUDevice::check(cl_int err, const std::string& msg)
{
    if(err != CL_SUCCESS) {
        std::string full_msg = "OpenCL error " + std::to_string(err);
        if(!msg.empty()) full_msg += ": " + msg;
        throw std::runtime_error(full_msg);
    }
}

void GPUDevice::print_info() const
{
    std::cout << "GPU: " << device_name << std::endl;
    std::cout << "  Vendor: " << vendor << std::endl;
    std::cout << "  AMD: " << (is_amd ? "yes" : "no") << "  NVIDIA: " << (is_nvidia ? "yes" : "no") << std::endl;
    std::cout << "  Compute units: " << compute_units << std::endl;
    std::cout << "  Max workgroup: " << max_workgroup_size << std::endl;
    std::cout << "  Global memory: " << global_mem_size / (1024*1024*1024) << " GB" << std::endl;
    std::cout << "  Max alloc: " << max_mem_alloc / (1024*1024*1024) << " GB" << std::endl;
}

} // namespace mmx
