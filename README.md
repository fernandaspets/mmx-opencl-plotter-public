# MMX OpenCL Plotter

MMX plotter using OpenCL for F1 computation and CPU for the F2-F9 pipeline.

Works on AMD gfx1100 (RX 7900 XTX) where HIP/CUDA/ZLUDA produce broken plots due to a hardware-level fp32 bug. OpenCL avoids the bug by using a different compiler path.

This is a hybrid GPU+CPU plotter — F1 runs on GPU, F2-F9 matching/sorting/hashing runs on CPU with GPU-assisted SHA-512 hashing. It is slower than a full GPU plotter but produces valid plots on hardware where the CUDA/HIP plotter cannot.

## How it works

1. **F1 on GPU** (OpenCL) — computes proof-of-space F1 values using SHA-512 + memory-hard function
2. **F2-F9 on CPU** — parallel radix sort + Y,Y+1 matching + GPU-assisted SHA-512 table hashing
3. **Plot file writing** — parallel park generation with bit-stream encoding

## Performance (k26)

| Component | Time |
|-----------|------|
| F1 (GPU) | 51 sec |
| F2-F9 (CPU + GPU hash) | 211 sec |
| Plot writing | 12 sec |
| **Total** | **274 sec** |

## Verification

```
mmx_postool --file plot.plot --iter 20 --verbose
Pass: 336 / 320, 105 %
Fail: 0 / 320, 0 %
Bad plots: None
```

## ⚠️ Limitations

**This is a testing/experimental tool. Do not use for production farming.**

- **Maximum k-size: k26.** Larger k-sizes (k29+) require a chunked per-bucket architecture to fit in RAM/VRAM. The current implementation loads all entries into memory simultaneously, which works for k26 (~67M entries, ~20 GB RAM) but will exhaust RAM and GPU memory for k29+ (536M entries, ~130 GB RAM).
- k29 is the minimum for MMX mainnet farming. This tool cannot produce k29 plots yet.
- The plotter is slower than the CUDA/HIP plotter (which does everything on GPU). It's useful for AMD GPUs where CUDA/HIP produce broken plots.

## Building

Requires a standard [mmx-node](https://github.com/madMAx43v3r/mmx-node) build (any build, no special branch needed) for header files and libraries.

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DMMX_NODE_DIR=/path/to/mmx-node ..
make -j$(nproc)
```

## Usage

```bash
./mmx_opencl_plotter <plot_id_hex> <farmer_key_hex> [output_dir] [options]
```

Options:
- `--k N` — Set plot k-size (default: 26)
- `--ramdisk DIR` — Write plot to tmpfs at DIR first, then copy to output_dir. Eliminates disk I/O bottleneck during plotting.
- `--test` — Run in test mode
- `--limit N` — Limit entries (test mode)

### Direct output (simple)

```bash
./mmx_opencl_plotter $(python3 -c "import os; print(os.urandom(32).hex())") \
    <farmer_public_key_hex> \
    /output/plots/ --k 26
```

### RAM disk output (faster)

Writing to a RAM disk (tmpfs) avoids disk I/O during plotting, which is a major bottleneck for large k-sizes.

```bash
# 1. Mount a tmpfs (one-time, requires root)
sudo mount -t tmpfs -o size=8G tmpfs /mnt/ramdisk

# 2. Plot to RAM disk, auto-copy to final destination after
./mmx_opencl_plotter $(python3 -c "import os; print(os.urandom(32).hex())") \
    <farmer_public_key_hex> \
    /output/plots/ --ramdisk /mnt/ramdisk --k 26
```

The plot is written to `/mnt/ramdisk` (in RAM), then copied to `/output/plots/` and removed from RAM disk.

**RAM disk size guide:**

| k-size | Plot size | RAM disk needed |
|--------|-----------|-----------------|
| k22 | ~250 MB | 512 MB |
| k26 | ~4.7 GB | 8 GB |
| k30 | ~75 GB | 80 GB |

## Verify

Use `mmx_postool` from any mmx-node build:

```bash
mmx_postool --file <plot.plot> --iter 20 --verbose
```

## Files

- `plotter.cpp` — main plotter (F1 GPU + F2-F9 CPU + plot writer)
- `pos_recompute.cl` — OpenCL F1 kernel (SHA-512 + gen_mem_array + calc_mem_hash)
- `table_hash.cl` — OpenCL table hash kernel (SHA-512 of metadata pairs)
- `test_table_hash.cpp` — verifies GPU table hash matches CPU
- `test_pipeline.cpp` — verifies full F1→F9 pipeline for small k values
- `verify_pd.cpp` — traces PD tree for debugging
- `CMakeLists.txt` — build configuration
