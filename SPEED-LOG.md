# OpenCL Plotter Speed Log

## Current Best Results

| K | Entries | AMD 7900 XTX | NVIDIA P40 | CUDA Ref | AMD Pass | NVIDIA Pass |
|---|---------|-------------|------------|----------|----------|-------------|
| 22 | 4M  | **2.40s**  | 4.93s  | —      | 97.5%   | 97.5%   |
| 25 | 32M | **18.72s** | ~38s   | 1.52s* | 100.6%  | 100.6%  |
| 26 | 64M | **36.16s** | 62.1s  | 28.4s  | 105%    | 105%    |

*CUDA reference time is for k20 (not k25) on single Tesla P40.

## Progress This Session
k25: 41.81s → 27.44s → 22.02s → 20.24s → 19.5s → 18.72s (55% reduction)
k26: 39.8s → 39.2s → 36.16s

## k26 Breakdown (36.16s)
- F1: 10.0s (28%) — SHA-512 compute, hard limit
- T2 sort: 0.60s (radix sort of 67M F1 entries)
- T3-T9: ~2.5-3.3s each = ~22s (sort_matches + match + GPU hash + build_pd)
- Final: copy=1.87s + X_pairs=0.9s + download=1.5s = ~4.3s

## Optimizations Applied
1. Parallel sort (replaces radix bucket sort)
2. Remove clFinish between F1 kernels
3. Parallelize matches, entries_map, PD build
4. Eliminate PD reorder + Final sort + PD9 reorder (sort matches at end of table)
5. Parallel flatten LR pairs
6. **Parallel LSD radix sort** for matches (O(n) vs O(n log n))

## NVIDIA P40 Results (with radix sort)
k25 NVIDIA: 33.4s (was 38s), 100.625% pass
k26 NVIDIA: 64.5s (was 62.1s), 105% pass
Note: NVIDIA is slower because metadata sort requires M_curr download (3.8GB/table for k26).
