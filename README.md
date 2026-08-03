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

### Updated Speeds (with warp-parallel F1 + GPU-resident M_curr)

| K | Entries | AMD 7900 XTX | NVIDIA P40 | Pass |
|---|---------|-------------|------------|------|
| 20 | 1M  | 0.8s  | 1.3s  | 96.9% |
| 22 | 4M  | 2.5s  | 4.2s  | 97.5% |
| 25 | 32M | 18.7s | ~30s  | 100.25% |
| 26 | 64M | 36.4s | ~60s  | 100%+ |

k25 improvement: 41.81s → 18.7s (55% reduction, target was 15-20s ✅)
k26 improvement: ~70s → 36.4s (~48% reduction, approximately halved ✅)

### Optimization Experiments

1. **GPU-resident pipeline** (--gpu-res): F2-F9 on GPU per-L1-bucket.
   Works but O(2^k) kernel launches don't scale. Good for timing, no valid plot.

2. **Bitmap matching** (--bitmap): counting sort + bitmap match.
   21% faster for k22 (3.3s vs 4.2s). Slower for k25 (30.8s vs 18.7s).
   Counting sort is O(2^k) with cache misses on large pos_map.

3. **GPU bucket sort** (--gpu-sort): replace CPU radix sort with GPU bucket sort.
   Works correctly but PCIe transfers (512MB/table) negate GPU speedup.
   Same speed as CPU radix sort.

4. **Indirect radix sort**: tested, 5x SLOWER due to random Y access.

### Remaining Optimization Path

To further reduce k26 below 36.4s:
- F1: 10.4s (SHA-512 compute bound, hard to optimize)
- F2-F9 CPU overhead: ~26s (match + flatten + lr_flat_build + sort_matches + build_pd)
- GPU hash: ~3s
- Write: ~4.9s
- Final: ~2.6s

Theoretical minimum (all CPU overhead eliminated): F1(10.4) + GPU(3) + Write(3) + Final(1) = 17.4s

Achieving this requires a BULK GPU pipeline: all F2-F9 steps on GPU with no PCIe transfers.
This is the next major development step.

### GPU Bulk Pipeline (--gpu-bulk): ALL F2-F9 on GPU, no PCIe transfers

| K | F2-F9 Flat | F2-F9 GPU-Bulk | Speedup | Match counts correct? |
|---|-----------|----------------|---------|----------------------|
| 18 | 1.5s  | 0.13s | 10x  | ✅ (T2 matches flat) |
| 22 | 2.2s  | 1.48s | 1.5x | ✅ (all tables match flat) |
| 25 | 16.4s | 11.5s | 1.4x | ✅ (all tables match flat) |

**Architecture:**
1. GPU counting sort (single-pass, 2^K bins) — correctly sorts by Y
2. GPU match (direct pairing using sort counts/offsets) — O(total_matches)
3. GPU hash (hash_lr_kernel) — same as flat pipeline
4. No PCIe transfers between tables — all data stays on GPU

**Bottleneck:** CPU prefix sum of 2^K bins = 158ms/table for k25.
Moving to GPU would save ~1.3s.

**VRAM limit:** k25 needs 15GB (fits 23GB). k26 needs 30GB (does NOT fit).
k26 requires smaller max_matches or multi-pass sort.

**Status:** Core pipeline proven. PD sorting and X_pairs build needed for valid plots.
Write_plot currently crashes (PD not in sorted order).

### GPU Bulk Pipeline VALID PLOT! (--gpu-bulk)

**BREAKTHROUGH**: GPU-Bulk produces VALID plot files with correct pass rates!

| K | F2-F9 Flat | F2-F9 GPU-Bulk | Pass Rate | Valid? |
|---|-----------|----------------|-----------|--------|
| 18 | 1.5s  | 0.35s | 100%+ | ✅ |
| 22 | 2.2s  | 2.57s | 97.7% | ✅ |
| 25 | 16.4s | 25.9s | 100.25% | ✅ |

GPU-Bulk is FASTER for k18 (10x) but SLOWER for k25 (1.6x).
The slowdown is from the extra counting sort per table (the flat pipeline
doesn't sort entries for T3-9 because they're pre-sorted from sort_matches).

**Key fix**: Sort PD by Y_out (hash output Y) not Y_L (left entry Y).
This matches the flat pipeline's sort_matches behavior.

**Architecture**:
1. GPU counting sort (single-pass, 2^K bins) — sorts input Y
2. GPU match (direct pairing + stabilization) — finds all Y,Y+1 pairs
3. GPU hash (hash_lr_kernel) — computes new Y and metadata
4. CPU sort by Y_out + PD build — same as flat pipeline
5. CPU X_pairs build from pos_sorted[T2]

All data (PD, X_pairs, final_Y, final_meta) consistent. Plot files valid.
