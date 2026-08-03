#include "gpu_manager.h"

namespace mmx {

void GPUManager::init(int device_index, int num_devices) {
    // Get platform
    cl_platform_id plat;
    cl_uint np;
    cl_int err = clGetPlatformIDs(1, &plat, &np);
    GPUDevice::check(err, "get platform");

    // Get all devices
    cl_uint nd;
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd);
    GPUDevice::check(err, "count devices");

    std::vector<cl_device_id> all_devs(nd);
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, nd, all_devs.data(), nullptr);
    GPUDevice::check(err, "get devices");

    // Determine how many to use
    if(num_devices <= 0) num_devices = nd - device_index;
    num_devices = std::min(num_devices, (int)(nd - device_index));

    devices.clear();
    for(int i = 0; i < num_devices; i++) {
        int dev_idx = device_index + i;
        auto dev = std::make_unique<GPUDevice>();
        dev->init(dev_idx);
        devices.push_back(std::move(dev));
        std::cout << "[GPU] Added device " << dev_idx << " as GPU " << i << std::endl;
    }
}

} // namespace mmx
