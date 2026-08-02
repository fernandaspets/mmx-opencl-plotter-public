# OpenCL Plotter Speed Log

## Current Best Results (GPU-resident M_curr + warp F1 + hash_table_lr)

| K | Entries | AMD 7900 XTX | NVIDIA P40 | AMD Pass | NVIDIA Pass |
|---|---------|-------------|------------|----------|-------------|
| 22 | 4M | **3.28s** | **5.29s** | 97.5% | 97.5% |
| 23 | 8M | **6.43s** | **10.39s** | 94.06% | 94.06% |
| 24 | 16M | **13.10s** | **20.18s** | 96.25% | 96.25% |
| 25 | 32M | **30.15s** | **~50s** | 100.625% | 100.625% |
| 22 CUDA ref | 4M | — | 1.52s | — | ~100% |

## Optimization History (k22 AMD: 18.4s → 3.28s = 82% reduction)
| Optimization | Time | What |
|-------------|------|------|
| Chunked baseline | 16.6s | Starting point (k22) |
| FLAT pipeline | 9.6s | 48 kernel launches vs 12,288 |
| Skip dedup | 7.0s | No meta sort (duplicates never occur) |
| Warp-parallel F1 | 7.0s | F1: 3.0s → 0.9s (32 threads/entry) |
| hash_table_lr | 6.5s | GPU reads M_curr directly |
| Flat M_curr | 4.6s | Eliminate memory copies |
| Parallel final copy | 4.6s | OpenMP parallel copy |
| **GPU-resident M_curr** | **3.3s** | No per-table upload/download (swap buffers) |

## Key Architecture
- **F1**: 3-kernel warp-parallel pipeline (32 threads/entry, shared memory reduction)
- **F2-F9**: GPU-resident M_curr (swap cl_mem buffers, no PCIe transfer per table)
- **Hash**: hash_table_lr kernel reads M_curr directly via P1/P2 indices
- **Sort**: metadata tiebreaker + M_curr download (small, only for sort comparator)
- **Final**: skip dedup (never duplicates), parallel copy, PD9 reorder
- **VRAM**: ~1.2GB for k24 (both buffers with max capacity), no SVM pool needed

## Bugs Fixed
1. uint32_t underflow in cross-bucket matching (infinite loop on NVIDIA)
2. Buffer overflow: swapped buffers had different sizes (M_out > M_curr capacity)
3. CL_MEM_READ_ONLY/WRITE_ONLY flags: can't swap buffers with different flags
4. Double-move of M_results (zeroed M_curr_flat in fallback mode)
5. CL_MEM_COPY_HOST_PTR reading past end (max_entries > initial_entries)
6. SVM pool allocated but never used (7GB wasted VRAM)
