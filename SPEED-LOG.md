# OpenCL Plotter Speed Log

## Current Best Results

| K | Entries | AMD 7900 XTX | NVIDIA P40 | CUDA Ref | Pass |
|---|---------|-------------|------------|----------|------|
| 22 | 4M  | **2.40s**  | **4.29s** | —      | 97.5%   |
| 25 | 32M | **17.7s**  | ~33s      | —      | 100.6%  |
| 26 | 64M | **36.8s**  | ~64s      | 28.4s  | 105%    |
| 27 | 128M| **76.1s**  | —         | —      | 92.8%   |

## Session Progress
### k25 (target: ≤15s)
41.81s → 27.44s → 22.02s → 20.24s → 19.5s → 18.72s → 18.33s → **17.7s**
(58% total reduction)

### k26
39.8s → 39.2s → 36.16s → **36.8s**

## Optimizations Applied
1. Parallel sort (replaces radix bucket sort)
2. Remove clFinish between F1 kernels
3. Parallelize matches, entries_map, PD build
4. Eliminate PD reorder + Final sort + PD9 reorder (sort matches at end of table)
5. Parallel flatten LR pairs
6. Parallel LSD radix sort for matches (O(n) vs O(n log n))
7. VRAM reduction (1.5x → 1.1x) for k27 support
8. Merged entries_map + PD build into single loop
9. GPU gather tested for Final copy — slower, reverted

## Remaining Gap vs CUDA (k26: 36.8s vs 28.4s = 8.4s)
- F1: 10.0s (CUDA ~2s on Tesla P40 — much faster SHA-512)
- Per-table CPU: ~2.5s × 8 = 20s (sort_matches + match + build_pd + flatten)
- GPU hash: 0.35s × 8 = 2.8s (CUDA overlaps with 3 streams)
- Final: copy + download + X_pairs = ~4s

## What would close the gap
- GPU sort + match (saves ~5s CPU per table loop)
- Multiple OpenCL queues for overlap (saves ~2.8s GPU hash overlap)
- Faster F1 (needs better SHA-512 GPU kernel or faster GPU)
