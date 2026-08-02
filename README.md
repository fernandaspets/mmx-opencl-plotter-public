# MMX OpenCL Plotter for AMD GPUs

OpenCL implementation of the MMX proof-of-space plotter, designed for AMD GPUs
(RDNA3 / gfx1100+). Also works on NVIDIA GPUs via OpenCL.

## Status

**Working — beats CUDA reference plotter in proof quality (104.6% over 25 runs).**

| Metric | Our OpenCL | CUDA Reference |
|--------|-----------|----------------|
| Proof quality (k20, 25 runs) | **266.3/320 avg** | 254.6/320 avg |
| Win rate | **17/25** | 1/25 |
| OCL/CUDA ratio | **104.6%** | 100% |

### Supported k-sizes
- k18-k26: flat pipeline (in-memory, fast)
- k18-k32: chunked pipeline (per-bucket, lower memory)

### Features
- F1 computation on GPU (OpenCL SHA-512 + mem_hash)
- F2-F9 table computation with GPU-assisted hashing
- Cross-boundary bucket matching
- Phase 2 compaction (removes unreachable entries, remaps PD)
- Both flat and chunked pipelines

## Performance

### Current Speeds (GPU-resident M_curr pipeline)

| K | Entries | AMD 7900 XTX | NVIDIA P40 | Pass |
|---|---------|-------------|------------|------|
| 22 | 4M  | 2.9s  | 5.3s  | 97.5% |
| 23 | 8M  | 6.4s  | 10.1s | 94.1% |
| 24 | 16M | 13.1s | 20.2s | 96.3% |
| 25 | 32M | 27.4s | ~50s  | 100.6% |

### ⚠️ Cross-Platform Sort Tradeoff (IMPORTANT for future AMD optimization)

**The sort tiebreaker is vendor-specific. This is a known performance tradeoff:**

- **AMD**: uses **Y-only sort** (no metadata tiebreaker). Fast — no M_curr download
  from GPU needed per table. This is the fast path.
- **NVIDIA**: uses **Y+metadata sort** (metadata tiebreaker required). Correct but
  requires downloading M_curr from GPU before each table's sort (~1.8GB/table for k25).
  Without the metadata tiebreaker, NVIDIA's OpenCL compiler produces hash outputs
  where all Y values land in one bucket at T4, causing O(n²) matching hang.

**Why AMD is faster**: AMD skips the M_curr download (saves ~1.8GB × 8 tables = 14.4GB
of PCIe transfers for k25). The Y-only sort is also faster (no element-by-element
metadata comparison for same-Y entries).

**Future AMD speedup opportunity**: If we can make the Y-only sort work on NVIDIA
(e.g., by doing the sort on GPU, or finding an alternative deterministic tiebreaker
that doesn't need M_curr), both platforms could use the fast path. This would save
~2s on k22 and ~5s on k25 on AMD (the larger buffer + clFinish overhead from the
NVIDIA-compatible code path).

**Key files**: The sort logic is in `plotter.cpp` — search for `use_meta_sort`.
GPU vendor is detected via `clGetDeviceInfo(CL_DEVICE_VENDOR)`.

## Build

```bash
cd build && cmake .. && make -j$(nproc)
cp ../f2_f9.cl ../simple_sort.cl ../table_hash.cl ../pos_recompute.cl .
```

Requires: mmx-node built from source (for libmmx, libvnx).

## Usage

```bash
# Flat pipeline (fast, needs RAM)
./mmx_opencl_plotter <plot_id_hex> <farmer_key_hex> <output_dir> --k 20

# Chunked pipeline (lower memory, slightly slower)
./mmx_opencl_plotter <plot_id_hex> <farmer_key_hex> <output_dir> --k 20 --chunked

# With ramdisk (faster I/O)
./mmx_opencl_plotter <plot_id_hex> <farmer_key_hex> /mnt/ramdisk/ --k 20 --chunked --ramdisk /mnt/ramdisk
```

## Testing

```bash
./tests/run_tests.sh           # Full suite (8 tests)
./tests/run_tests.sh --quick   # Skip slow tests
./tests/run_tests.sh --build-only  # Just build, don't run
```

## Architecture

See `docs/ARCHITECTURE.md` for pipeline details.

## Key Innovations

1. **Phase 2 compaction**: Removes unreachable entries from all PD tables,
   remaps positions. Produces better proof coverage than CUDA reference.
2. **Cross-boundary matching**: Finds Y,Y+1 matches across first-level bucket
   boundaries that single-bucket processing misses.
3. **OpenCL F1**: Full GPU F1 computation including custom mem_hash.
4. **AMD gfx1100 support**: Works around AMD OpenCL compiler quirks.

## License

MIT (see LICENSE file if present). Based on madMAx43v3r's mmx-cuda-plotter.
