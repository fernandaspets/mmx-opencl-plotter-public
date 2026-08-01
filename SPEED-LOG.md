# OpenCL Plotter Speed Log

## k22 (4M entries)
| Platform | Time | Pass | Notes |
|----------|------|------|-------|
| AMD 7900 XTX 1GPU | 6.53s | 97.5% (312/320) | FLAT+SVM+hash_lr+warpF1+skipdedup |
| NVIDIA P40 1GPU | 12.37s | 97.5% (312/320) | FLAT+SVM+hash_lr+warpF1+skipdedup |
| NVIDIA P40 2GPU | ~17s | 97.5% | Multi-GPU overhead > gain at k22 |
| CUDA reference | 1.52s | ~100% | madMAx CUDA plotter on Tesla P40 |

## k23 (8M entries)
| Platform | Time | Pass | Notes |
|----------|------|------|-------|
| AMD 7900 XTX 1GPU | 12.99s | 94.06% (301/320) | FLAT+SVM+hash_lr+warpF1+skipdedup |
| NVIDIA P40 1GPU | 24.71s | 94.06% (301/320) | FLAT+SVM+hash_lr+warpF1+skipdedup |
| CUDA reference | ~3s | ~100% | madMAx CUDA plotter (estimated) |

## Optimization History (k23 AMD)
| Version | Time | What changed |
|---------|------|--------------|
| Chunked SVM baseline | 18.4s | Starting point |
| FLAT pipeline | 15.1s | 48 kernel launches vs 12,288 |
| Skip dedup sort | 15.1s | Duplicates never occur (was 17.7s with meta sort) |
| Skip dedup entirely | 14.6s | No meta sort + re-sort (saves 3s) |
| Warp-parallel F1 | 18.0s → 15.0s | F1: 6.2s → 1.7s (32 threads/entry) |
| hash_table_lr | 12.99s | GPU reads M_curr directly (skip CPU extraction) |

## k23 AMD Breakdown (12.99s)
| Phase | Time | % |
|-------|------|---|
| F1 (warp-parallel) | 1.65s | 13% |
| F2-F9 (8 tables) | 9.98s | 77% |
| Final (sort+copy+PD9) | 2.54s | 20% |

### Per-table breakdown (k23):
| Table | Time | Matches |
|-------|------|---------|
| T2 | 1.51s | 8.39M |
| T3 | 1.20s | 8.38M |
| T4 | 1.25s | 8.38M |
| T5 | 1.19s | 8.37M |
| T6 | 1.26s | 8.34M |
| T7 | 1.19s | 8.29M |
| T8 | 1.17s | 8.19M |
| T9 | 1.21s | 7.98M |
