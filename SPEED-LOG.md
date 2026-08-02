# OpenCL Plotter Speed Log

## Current Best Results

| K | Entries | AMD 7900 XTX | NVIDIA P40 | RTX PRO 6000 BW | CUDA Ref |
|---|---------|-------------|------------|-----------------|----------|
| 22 | 4M  | **2.29s**  | 4.25s      | **3.29s**       | —      |
| 25 | 32M | **17.7s**  | ~33s       | **22.2s**       | —      |
| 26 | 64M | **36.8s**  | ~64s       | **47.0s**       | 28.4s  |
| 27 | 128M| **76.1s**  | —          | —               | —      |

All produce valid plots (94-105% pass rate depending on PID).

## Session History
### k25 optimization (target: ≤15s)
41.81s → 27.44s → 22.02s → 20.24s → 19.5s → 18.72s → 18.33s → 17.7s

### k26
39.8s → 39.2s → 36.16s → 36.8s

### RTX PRO 6000 Blackwell (new)
Initial: k25=25.5s, k26=49.8s
After hash wg=64: k25=22.2s, k26=47.0s

## Optimizations Applied
1. Parallel sort (replaces radix bucket sort)
2. Remove clFinish between F1 kernels
3. Parallelize matches, entries_map, PD build
4. Eliminate PD reorder + Final sort + PD9 reorder
5. Parallel flatten LR pairs
6. Parallel LSD radix sort for matches (O(n) vs O(n log n))
7. VRAM reduction (1.5x → 1.1x) for k27 support
8. Merged entries_map + PD build into single loop
9. **Auto-detect hash work-group size** (NVIDIA=64, AMD=256)
