# OpenCL Plotter Speed Log

## Current Best Results

| K | Entries | AMD 7900 XTX | NVIDIA P40 | AMD Pass | NVIDIA Pass |
|---|---------|-------------|------------|----------|-------------|
| 22 | 4M  | **2.66s**  | **4.93s**  | 97.5%   | 97.5%   |
| 23 | 8M  | ~5.5s      | **9.44s**  | 94.1%   | 94.1%   |
| 24 | 16M | ~11s       | ~19s       | 96.3%   | 96.3%   |
| 25 | 32M | **19.5s**  | ~38s       | 100.6%  | 100.6%  |
| 26 | 64M | **39.5s**  | **62.1s**  | 105%    | 105%    |

## Scaling Analysis
Each +1 k doubles entries. Time scales ~2x per k:
- AMD: 2.66 → 5.5 → 11 → 19.5 → 39.5 (factor ~2x)
- NVIDIA: 4.93 → 9.44 → 19 → 38 → 62 (factor ~2x, slower baseline)

## ⚠️ Cross-Platform Sort Tradeoff
See README.md "Performance" section.
AMD: Y-only sort (fast, no M_curr download).
NVIDIA: Y+metadata sort (correct, requires M_curr download — 3.8GB/table for k26).
Future: make Y-only sort work on NVIDIA to save ~5-15s on k26 NVIDIA.
