# OpenCL Plotter Speed Log

## Current Best Results (NO --opt-svm needed, cl_mem only)

### All K Sizes
| K | Entries | AMD 7900 XTX | NVIDIA P40 | Pass (AMD) | Pass (NVIDIA) | VRAM (NVIDIA) |
|---|---------|-------------|------------|------------|---------------|---------------|
| 22 | 4M | **4.63s** | **9.13s** | 97.5% | 97.5% | ~200MB |
| 23 | 8M | **8.57s** | **17.59s** | 94.06% | 94.06% | ~400MB |
| 24 | 16M | **18.0s** | **32.65s** | 96.25% | 96.25% | 222MB |
| 25 | 32M | **41.81s** | **67.06s** | 100.625% | 100.625% | 4MB idle |
| 22 CUDA ref | 4M | — | 1.52s | — | ~100% | — |

## Key VRAM Fix
SVM pool was allocated (7.31GB for k24!) but never used.
Fixed: only allocate if hash_lr_kernel is unavailable.
NVIDIA k24 VRAM: 22GB → 222MB (99% reduction!)
This means we can safely plot much larger k sizes on 24GB GPUs.

## Optimization History (k23 AMD: 18.4s → 8.57s = 53% reduction)
| Optimization | Time | What |
|-------------|------|------|
| Chunked SVM baseline | 18.4s | Starting point |
| FLAT pipeline | 15.1s | 48 kernel launches vs 12,288 |
| Skip dedup | 14.6s | No meta sort (duplicates never occur) |
| Warp-parallel F1 | 13.0s | F1: 6.2s → 1.7s (32 threads/entry) |
| hash_table_lr | 13.0s | GPU reads M_curr directly |
| **Flat M_curr** | **9.35s** | Eliminate memory copies |
| Parallel final copy | **8.57s** | OpenMP parallel copy |
| Skip SVM pool | 8.57s | Saves 7GB VRAM (no speed change) |

## Architecture
3-kernel F1 pipeline (warp-parallel):
1. gen_mem_array_v2: 1 WI/entry → key + mem to global (16K sub-batch, 72MB VRAM)
2. calc_mem_hash_warp: 32 WI/entry → shared mem + manual tree reduction
3. scatter_f1_v2: 1 WI/entry → final SHA-512 → Y, M

F2-F9 flat pipeline:
- CPU radix sort + match (OpenMP parallel)
- GPU hash_table_lr: reads M_curr directly via P1/P2 indices (no CPU extraction)
- Flat M_curr array (no array<->flat conversions)
- Skip dedup (never any duplicates)
- Parallel final copy (OpenMP)
