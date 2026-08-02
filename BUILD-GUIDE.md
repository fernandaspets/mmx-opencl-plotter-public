# OpenCL Plotter — Build Guide (NVIDIA or AMD)

## 1. Install apt dependencies

```bash
sudo apt update
sudo apt install -y \
    cmake g++ gcc build-essential \
    git \
    libssl-dev zlib1g-dev \
    libjemalloc-dev \
    ocl-icd-libopencl1 ocl-icd-opencl-dev \
    opencl-c-headers opencl-clhpp-headers \
    clinfo \
    autoconf automake libtool pkg-config
```

### NVIDIA OpenCL support
Install the NVIDIA CUDA toolkit (includes OpenCL ICD):
```bash
# Ubuntu 24.04 example — adjust repo URL for your distro:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit

# Verify OpenCL sees your GPU:
clinfo | grep "Device Name"
```

### AMD OpenCL support (if on AMD)
```bash
sudo apt install -y rocm-opencl rocm-opencl-dev
```

## 2. Clone and build mmx-node (provides headers + libs)

```bash
cd ~
git clone https://github.com/madMAx43v3r/mmx-node.git
cd mmx-node

# Build secp256k1 dependency (needs autoconf)
cd lib && ./make_all.sh && cd ..

# Configure and build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) mmx_iface mmx_pos mmx_pos_verify uint256_t vnx_base vnx_addons

# Build postool for plot verification:
make -j$(nproc) mmx_postool
```

> **Note:** The cmake may print warnings about missing Qt5, miniupnpc, or jemalloc.
> These are only needed for the full node binary — not for the libs we need.
> The build will succeed without them.

## 3. Clone and build the OpenCL plotter

```bash
cd ~
git clone <your-repo-url> mmx-opencl-plotter
cd mmx-opencl-plotter
git checkout feature/ocl2.0-svm

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DMMX_NODE_DIR=~/mmx-node
make -j$(nproc)
```

## 4. Test

```bash
# Library paths
export LD_LIBRARY_PATH=~/mmx-node/build:~/mmx-node/build/vnx-base:~/mmx-node/build/vnx-addons:$LD_LIBRARY_PATH

# Farmer key + test plot ID (k22, small/fast test)
FK="02292cd11aa18e5f64344cbe6c580249364dfe5a3683adc25446aadcc1b38555d7"
K22_PID="c8a0398a7544ae09657d477d33df0e57ebe02be97a3834f8995d362495f6aee2"

mkdir -p /tmp/plotram
~/mmx-opencl-plotter/build/mmx_opencl_plotter "$K22_PID" "$FK" /tmp/plotram/ --k 22 --no-yield --opt-gpu-meta --opt-gpu-prefix

# Verify the plot
~/mmx-node/build/tools/mmx_postool -f /tmp/plotram/*.plot -n 20 -v
```

## Useful flags

| Flag | Description |
|------|-------------|
| `--k N` | Plot size (18-32) |
| `--no-yield` | Disable GPU display yield (headless) |
| `--opt-gpu-meta` | GPU metadata extraction |
| `--opt-gpu-prefix` | GPU prefix sum |
| `--device N` | Select GPU index (multi-GPU) |
| `--plotid HEX` | Use specific plot ID (for comparison) |

## RAM disk for fastest I/O

```bash
# Use tmpfs (RAM-backed) for plot output — 3GB/s write speed
mkdir -p /dev/shm/plotram
# Pass /dev/shm/plotram/ as output dir instead of /tmp/plotram/
```
