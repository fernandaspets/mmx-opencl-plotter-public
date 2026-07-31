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

## Building

Requires a standard [mmx-node](https://github.com/madMAx43v3r/mmx-node) build (any build, no special branch needed) for header files and libraries.

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DMMX_NODE_DIR=/path/to/mmx-node ..
make -j$(nproc)
```

## Usage

```bash
./mmx_opencl_plotter <plot_id_hex> <farmer_key_hex> [output_dir] [--k KSIZE]
```

Example:
```bash
./mmx_opencl_plotter $(python3 -c "import os; print(os.urandom(32).hex())") \
    <farmer_public_key_hex> \
    /output/plots/ --k 26
```

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
