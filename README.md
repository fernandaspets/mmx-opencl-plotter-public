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
