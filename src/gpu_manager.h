#pragma once

#include "gpu_device.h"
#include <vector>
#include <memory>

namespace mmx {

// Manages multiple GPU devices for parallel processing.
// Distributes work across N GPUs, handling context creation and synchronization.
class GPUManager {
public:
    GPUManager() = default;
    ~GPUManager() = default;

    // Initialize N devices starting at device_index
    // If num_devices=0, uses all available devices
    void init(int device_index = 0, int num_devices = 0);

    // Number of devices
    size_t size() const { return devices.size(); }

    // Access a specific device
    GPUDevice& operator[](size_t i) { return *devices[i]; }
    const GPUDevice& operator[](size_t i) const { return *devices[i]; }

    // Get device for a given bucket (round-robin)
    GPUDevice& get_device_for_bucket(uint32_t bucket_idx) {
        return *devices[bucket_idx % devices.size()];
    }

    // Finish all devices
    void finish_all() {
        for(auto& dev : devices) dev->finish();
    }

    // Print all device info
    void print_info() const {
        for(size_t i = 0; i < devices.size(); i++) {
            std::cout << "  GPU " << i << ": ";
            devices[i]->print_info();
        }
    }

private:
    std::vector<std::unique_ptr<GPUDevice>> devices;
};

} // namespace mmx
