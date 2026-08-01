# OpenCL Plotter Architecture

## Overview
Hybrid GPU+CPU plotter for MMX Proof-of-Space. F1 on GPU, F2-F9 on GPU (chunked) or CPU (flat).

## Code Structure (plotter.cpp ~2300 lines)

### Classes/Structs
- `OCL_Plotter` (line ~150): OpenCL device wrapper, kernel management
- `PlotData` (line ~577): Output struct (final_Y, final_meta, PD tree, X_pairs)
- `MemBucketStore` (line ~1251): In-memory bucket store for chunked pipeline
- `PDEntry` (line ~1728): PD entry (Y, sorted_pos, delta)

### Functions
1. `compute_full_pipeline()` (line ~591): FLAT pipeline — all entries in memory, global sort
2. `write_plot()` (line ~944): Write plot file (header, Y parks, meta parks, PD parks, X parks)
3. `compute_f1_chunked()` (line ~1302): F1 generation → bucket store
4. `process_bucket_chunk()` (line ~1360): CPU fallback per-bucket processing (NO PD computation!)
5. `process_bucket_gpu()` (line ~1457): GPU per-bucket processing (scatter→sort→match→hash→PD)
6. `compute_f2_f9_chunked()` (line ~1733): Orchestrates chunked F2-F9
7. `build_plot_data_from_store()` (line ~1841): Convert bucket store → PlotData
8. `main()` (line ~2004): CLI entry point

### Two Pipelines
- **Flat**: `--k N` (no --chunked flag). All entries in memory. Global sort by (Y, metadata).
  Works at ~95% pass rate. Limited by RAM (k26 ≈ 16GB).
- **Chunked**: `--k N --chunked`. One first-level bucket at a time. GPU sort by (Y, orig_pos).
  Currently 0% pass rate. Supports k29+ (uses ~66GB RAM for k29).

## MMX Plot Format
- 9 tables: F1 (2^k entries) → F2-F9 (progressively fewer)
- F1: SHA-512 of plot_id || x_index, memory-hard function
- F2-F9: SHA-512 of (L_meta || R_meta), Y = XOR(hash[0..13]) & kmask
- Matching: Y_R == Y_L + 1 (consecutive Y values)
- PD tree: 7 tables (PD[3]..PD[9]), stores (position_in_prev_table, delta)
- X table: line-point encoded F1 indices from table 2 matches
- Plot file: header + Y parks + meta parks + PD parks (reverse order) + X parks

## Key Constants
- KSIZE: plot k-size (default 26)
- XBITS = KSIZE (xbits = k for uncompressed)
- LOGBUCKETS = 8 (256 first-level buckets)
- LOGBUCKETS2 = KSIZE - LOGBUCKETS - 9 (sub-buckets, requires k >= 18)
- MY_N_META = 14 (metadata uint32s per entry)
- MY_N_META_OUT = 12 (output metadata)
- PARK_SIZE_Y = 8192, PARK_SIZE_PD = 2048, PARK_SIZE_X = 4096
