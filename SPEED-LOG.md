# OpenCL Plotter Speed Log

## Current Best Results

### k22 (4M entries)
| Platform | Time | Pass | 
|----------|------|------|
| AMD 7900 XTX 1GPU | **4.63s** | 97.5% (312/320) |
| NVIDIA P40 1GPU | **9.14s** | 97.5% (312/320) |
| NVIDIA P40 2GPU | ~17s | 97.5% (overhead > gain at k22) |
| CUDA reference | 1.52s | ~100% |

### k23 (8M entries)
| Platform | Time | Pass |
|----------|------|------|
| AMD 7900 XTX 1GPU | **9.35s** | 94.06% (301/320) |
| NVIDIA P40 1GPU | **17.46s** | 94.06% (301/320) |
| CUDA reference | ~3s | ~100% |

## Optimization History (k23 AMD)
| Version | Time | What changed |
|---------|------|--------------|
| Chunked SVM baseline | 18.4s | Starting point |
| FLAT pipeline | 15.1s | 48 kernel launches vs 12,288 |
| Skip dedup entirely | 14.6s | No meta sort + re-sort (saves 3s) |
| Warp-parallel F1 | 15.0s→13.0s | F1: 6.2s → 1.7s (32 threads/entry) |
| hash_table_lr | 13.0s | GPU reads M_curr directly (skip CPU extraction) |
| **Flat M_curr** | **9.35s** | Eliminate 3 memory copies per table (saves 2.4s) |

## k23 AMD Breakdown (9.35s)
| Phase | Time | % |
|-------|------|---|
| F1 (warp-parallel) | 1.65s | 18% |
| F2-F9 (8 tables) | ~6.0s | 64% |
| Final (sort+copy+PD9) | ~1.7s | 18% |

## Key Optimizations
1. **Warp-parallel F1**: 32 work-items per entry, manual tree reduction (F1 3.5x faster)
2. **hash_table_lr**: GPU reads M_curr directly via P1/P2 indices (no CPU meta extraction)
3. **Flat M_curr**: Keep metadata as flat uint32 array (eliminate array<->flat conversions)
4. **Skip dedup**: MMX F2-F9 never produces duplicates (verified, CUDA also skips)
5. **FLAT pipeline**: Process all entries in one kernel launch per table (no per-bucket overhead)
6. **SVM**: Fine-grain zero-copy on AMD (no cl_mem for hash buffers)
