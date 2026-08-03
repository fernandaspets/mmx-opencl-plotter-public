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

## Branches

| Branch | Status | Description |
|--------|--------|-------------|
| `master` | **Stable** | OpenCL 1.2 — production code, all optimizations below |
| `feature/ocl2.0-svm` | **Experimental** | OpenCL 2.0 SVM + migration APIs for true async DMA |

### Tags

- `v0.4-ocl1.2` — best stable version on master (tagged Aug 2026)

**Rule:** Develop on `feature/ocl2.0-svm`. Only merge to `master` when a feature
is tested and produces correct pass rates on both AMD and NVIDIA.

## Optimization Modules (master branch)

All modules are toggleable via command-line flags and can be combined:

| Flag | Module | Description | AMD | NVIDIA |
|------|--------|-------------|-----|--------|
| `--opt-gpu-prefix` | A | GPU prefix sum (skip sub-count readback) | ✅ 8% | ✅ 1% |
| `--opt-gpu-meta` | B | GPU gather+hash (skip CPU meta extraction) | ✅ 10% | ✅ 7% |
| `--opt-bufpool` | C | Pre-allocated reusable GPU buffers | ✅ 4% | ✅ small |
| `--opt-queues 2` | E | 3-phase pipeline for 2-GPU overlap | ✅ correct | ✅ correct |
| `--opt-pinned` | G | Pinned memory (CL_MEM_ALLOC_HOST_PTR) | ✅ correct | ❌ NVIDIA bug |
| `--opt-async` | D | Non-blocking reads with events | ❌ slower | ❌ NVIDIA bug |
| `--opt-zero-copy` | F | Map/unmap zero-copy | ❌ 3x slower (discrete) | ❌ untested |
| `--timing` | — | Per-step timing for first bucket | ✅ | ✅ |
| `--no-yield` | — | Disable GPU display yield (headless) | ✅ | ✅ |
| `--device N` | — | Select GPU device index | ✅ | ✅ |

### Best combination (AMD 7900 XTX, k20 chunked)

```bash
./mmx_opencl_plotter <pid> <fkey> /dev/shm/plotram/ --k 20 --chunked --no-yield \
  --opt-gpu-meta --opt-gpu-prefix --opt-bufpool
```

**Result: 5.8s (24% faster than 7.6s baseline), 109.375% pass rate**

### Multi-GPU (2x NVIDIA P40)

```bash
./mmx_opencl_plotter <pid> <fkey> /dev/shm/plotram/ --k 20 --chunked --no-yield \
  --opt-gpu-meta --opt-gpu-prefix --opt-bufpool --opt-queues 2
```

**Result: 9.3s, 109.375% pass rate. Both GPUs utilized but no speedup yet**
(blocking reads prevent overlap — fix requires OpenCL 2.0 SVM, see feature branch).

### Known NVIDIA OpenCL 1.2 Limitations

1. **Non-blocking events unreliable** — `clEnqueueReadBuffer(CL_FALSE)` with events
   produces 0 matches on GPU 0. Must use `CL_TRUE` (blocking).
2. **CL_MEM_ALLOC_HOST_PTR + persistent mapping corrupts data** — pinned memory
   writes via mapped pointer produce 94% pass rate instead of 109%.
3. Both issues are NVIDIA OpenCL 1.2 driver limitations. AMD ROCm works correctly.
4. Fix requires OpenCL 2.0+ APIs (SVM, `clEnqueueMigrateMemObjects`) — see
   `feature/ocl2.0-svm` branch.

## Build

```bash
cd build && cmake .. && make -j$(nproc)
cp ../f2_f9.cl ../simple_sort.cl ../table_hash.cl ../pos_recompute.cl .
```

Requires: mmx-node built from source (for libmmx, libvnx).
On AMD: set `LD_LIBRARY_PATH=/opt/rocm/lib`.

## Usage

```bash
# Flat pipeline (fast, needs RAM — up to k26 on 24GB VRAM)
./mmx_opencl_plotter <plot_id_hex> <farmer_key_hex> <output_dir> --k 20

# Chunked pipeline (lower memory, for k27+)
./mmx_opencl_plotter <plot_id_hex> <farmer_key_hex> <output_dir> --k 20 --chunked

# Headless with all optimizations (AMD)
./mmx_opencl_plotter <pid> <fkey> /dev/shm/plotram/ --k 20 --chunked --no-yield \
  --opt-gpu-meta --opt-gpu-prefix --opt-bufpool

# 2-GPU on honeypot
./mmx_opencl_plotter <pid> <fkey> /dev/shm/plotram/ --k 20 --chunked --no-yield \
  --opt-gpu-meta --opt-gpu-prefix --opt-bufpool --opt-queues 2
```

## Testing

```bash
./tests/run_tests.sh           # Full suite (8 tests)
./tests/run_tests.sh --quick   # Skip slow tests
./tests/run_tests.sh --build-only  # Just build, don't run
```

Verify a plot file:
```bash
~/mmx-node/build_opencl/tools/mmx_postool -f <plot.plot> -n 20 -v
```

## Architecture

See `docs/ARCHITECTURE.md` for pipeline details.
See `docs/PD-BUG-ANALYSIS.md` for PD chain analysis.

### Files

| File | Description |
|------|-------------|
| `plotter.cpp` | Main plotter (~3100 lines) |
| `pos_recompute.cl` | F1 kernel (SHA-512 + mem_hash) |
| `table_hash.cl` | F2-F9 hash kernel (SHA-512 of LR pairs) |
| `f2_f9.cl` | GPU kernels: scatter, sort, match, eval |
| `simple_sort.cl` | Bitonic sort kernel |
| `gather_meta.cl` | Module B: gather metadata for hash |
| `prefix_sum.cl` | Module A: GPU prefix sum (Hillis-Steele) |
| `opt_config.h` | Optimization flag configuration |
| `buffer_pool.h` | Module C/G: reusable + pinned GPU buffers |
| `pipeline.h` | Module E: 3-phase pipeline BucketPending struct |

## Key Innovations

1. **Phase 2 compaction**: Removes unreachable entries from all PD tables,
   remaps positions. Produces better proof coverage than CUDA reference.
2. **Cross-boundary matching**: Finds Y,Y+1 matches across first-level bucket
   boundaries that single-bucket processing misses.
3. **OpenCL F1**: Full GPU F1 computation including custom mem_hash.
4. **AMD gfx1100 support**: Works around AMD OpenCL compiler quirks.
5. **Modular optimization framework**: Each optimization is independently
   toggleable and testable. Modules combine additively.

## Development Setup

- **AMD machine**: 7900 XTX (24GB VRAM, 64GB RAM) — primary development
- **NVIDIA machine**: 2x Tesla P40 (48GB VRAM, 251GB RAM) — scale testing
- RAM disk: `/dev/shm/plotram` (tmpfs, 3GB/s write speed)
- Git: LOCAL ONLY for mmx-app. OpenCL plotter has private + public repos.

## License

MIT (see LICENSE file if present). Based on madMAx43v3r's mmx-cuda-plotter.
