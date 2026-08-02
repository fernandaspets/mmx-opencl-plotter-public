# OpenCL Plotter Speed Log

## Current Best Results

| K | Entries | AMD 7900 XTX | NVIDIA P40 | AMD Pass | NVIDIA Pass |
|---|---------|-------------|------------|----------|-------------|
| 22 | 4M  | **2.49s** | **4.93s** | 97.5% | 97.5% |
| 23 | 8M  | ~5.5s     | **9.44s** | 94.1% | 94.1% |
| 24 | 16M | ~11s      | ~19s      | 96.3% | 96.3% |
| 25 | 32M | **22.02s**| ~45s      | 100.6% | 100.6% |

## Latest Optimizations
1. **Parallel PD reorder + X_pairs**: was sequential, 5.4s for k25 → now parallel
2. **Larger F1 sub-batch**: 16384→65536, saves 1.6s on F1 for k25
3. **Vendor-specific sort**: AMD Y-only (fast), NVIDIA Y+metadata (correct)
4. **GPU-resident M_curr**: swap buffers, no per-table upload/download
5. **Skip SVM pool**: saves 7GB VRAM (was allocated but unused)
6. **Warp-parallel F1**: 32 threads/entry, 3.5x faster F1
7. **hash_table_lr**: GPU reads M_curr directly via P1/P2 indices
8. **Flat M_curr**: eliminate array↔flat memory copies
9. **Skip dedup**: MMX F2-F9 never produces duplicates

## ⚠️ Cross-Platform Sort Tradeoff
See README.md "Performance" section. AMD uses fast Y-only sort.
NVIDIA needs metadata sort (requires M_curr download). Future optimization:
make Y-only sort work on NVIDIA to recover ~2-5s on AMD.
