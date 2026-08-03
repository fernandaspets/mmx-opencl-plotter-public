# Scoreboard — MMX OpenCL Plotter

Last updated: 2026-08-03 08:50 UTC
Branch: feature/production

## k22 Performance

| GPU | Our OCL | CUDA Ref | ocl2 Target | Gap |
|-----|:-------:|:--------:|:-----------:|:---:|
| AMD RX 7900 XTX | **6.77s** | N/A | **2.01s** | 3.4x |
| RTX 6000 Pro | **9.85s** | 4.20s | 2.86s | 3.4x |
| Tesla P40 | **14.59s** | 5.31s | 3.33s | 4.4x |

## Optimizations Active
- Warp-parallel F1 (3-kernel): 0.48s F1
- GPU-resident hash: ~50ms/table
- Skip-sort for t>2: 1.5ms vs 210ms
- Parallel scatter sort (padded per-thread counters)
- Index-based bucket sort
- No M_out download per table

## Remaining Roadmap
1. **GPU sort + match** — eliminate build_ms (250ms/table)
2. **L1 pipeline fix** — per-bucket sync bottleneck (T2 = 6s → 0.2s)
3. **Full GPU pipeline** — match ocl2 (2.01s)
