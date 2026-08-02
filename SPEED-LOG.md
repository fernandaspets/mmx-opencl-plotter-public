# OpenCL Plotter Speed Log

## Current Best Results

| K | Entries | AMD 7900 XTX | NVIDIA P40 | AMD Pass | NVIDIA Pass |
|---|---------|-------------|------------|----------|-------------|
| 22 | 4M | **2.90s** | **5.30s** | 97.5% | 97.5% |
| 23 | 8M | **6.43s** | **10.08s** | 94.06% | 94.06% |
| 24 | 16M | **13.10s** | **20.18s** | 96.25% | 96.25% |
| 25 | 32M | **27.44s** | ~50s (est) | 100.625% | 100.625% |

## Key Architectural Decisions
- **AMD**: Y-only sort (no metadata tiebreaker, no M_curr download) — fastest
- **NVIDIA**: Y+metadata sort (needs M_curr download for sort comparator) — correct
- **GPU-resident M_curr**: both buffers CL_MEM_READ_WRITE, same max capacity, swap between tables
- **VRAM**: ~5GB for k25 (two 1.5x buffers), no SVM pool needed

## Optimization History (k22 AMD)
| Version | Time | What |
|---------|------|------|
| Chunked baseline | 16.6s | Starting point |
| FLAT pipeline | 9.6s | 48 kernel launches vs 12,288 |
| Skip dedup | 7.0s | No meta sort (duplicates never occur) |
| Warp-parallel F1 | 7.0s | F1: 3.0s → 0.9s (32 threads/entry) |
| hash_table_lr | 6.5s | GPU reads M_curr directly |
| Flat M_curr | 4.6s | Eliminate memory copies |
| **GPU-resident M_curr** | **2.9s** | No per-table upload/download (swap buffers) |
