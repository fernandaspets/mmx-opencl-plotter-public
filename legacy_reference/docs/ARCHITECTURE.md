# OpenCL Plotter Architecture

## Pipeline Overview

### F1 Computation (GPU)
- `pos_recompute.cl`: SHA-512(X || plot_id) → gen_mem_array → calc_mem_hash → final SHA-512
- Runs in batches on GPU, outputs Y values + metadata per F1 entry
- Stored in MemBucketStore organized by first-level bucket (Y >> shift)

### F2-F9 Computation (per-bucket)
For each table t (2 to 9), for each first-level bucket y:
1. **scatter_2** (GPU): Sub-bucket entries by Y's full bits
2. **simple_sort_y** (GPU): Sort within sub-buckets by PY = (Y, local_pos)
3. **match_p1** (GPU): Find Y,Y+1 pairs within and across sub-buckets
4. **gpu_hash_table** (GPU): SHA-512(L_meta || R_meta) → new Y + metadata
5. **cross_boundary_match** (CPU): Find Y,Y+1 pairs across bucket y/y+1 boundary
6. Store results (metadata, PD, X_pairs) in dst MemBucketStore

### Phase 2 Compaction
After building all tables:
1. Mark T9 entries as reachable (they're the proof roots)
2. For T8 down to T2: mark entries referenced by reachable T(t+1) PD entries
3. Remove unreachable entries from each table
4. Remap PD positions to compacted indices
5. Compact X_pairs and final_Y

This matches CUDA plotter's phase 2 behavior and produces better proof coverage.

### Plot File Writing
- Header (vnx format, 4096 bytes aligned)
- Y table: delta-encoded parks (park_size_y = 8192)
- Meta table: k-bit metadata parks (park_size_meta = 256, HDD plots only)
- PD tables: 7 tables (PD[9] to PD[3]), position + delta-encoded parks
- X table: line-point encoded pairs (park_size_x = 2048)

## Sort Order

Entries in each table are sorted by **(Y, pd_all_index)** where pd_all_index is
the insertion order within the bucket store. This matches CUDA's (Y, local_pos)
sort. The sort is consistent across all tables, ensuring PD chain integrity.

## Key Constants

| Constant | Value | Description |
|----------|-------|-------------|
| KSIZE | 18-32 | Plot k-size |
| XBITS | KSIZE | X bits (C0 = no compression) |
| LOGBUCKETS | 8 | First-level bucket count = 256 |
| PARK_SIZE_Y | 8192 | Entries per Y park |
| PARK_SIZE_PD | 2048 | Entries per PD park |
| PARK_SIZE_META | 256 | Entries per meta park |
| DSIZE_ | 5 | PD delta bits (max delta = 31) |
| MEM_HASH_ITER | 256 | Memory-hard iteration count |
