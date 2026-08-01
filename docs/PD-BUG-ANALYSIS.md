# PD (Prover Data) Bug Analysis

## Status: FLAT pipeline works (~95%), CHUNKED pipeline fails (0%)

## Root Cause: Sort Order Inconsistency

### How PD works (from Prover.cpp)
The Prover chains through PD tables to reconstruct proofs:
```
final_index → PD[9][final_index] → (pos8, delta8) → pos8, pos8+delta8
PD[8][pos8] → (pos7, delta7) → pos7, pos7+delta7
...
PD[3][pos2] → (pos1, delta1) → X table lookup → F1 indices
```

PD[t] is indexed by the **Y-sorted position** in table t.
PD[t][i] stores (position_in_table_{t-1}, delta_in_table_{t-1}).

### FLAT pipeline (working)
1. Each table sorted globally by **(Y, metadata)** — deterministic, consistent
2. LR pairs store sorted positions in this global order
3. PD[t] = {sorted_L, sorted_R - sorted_L} in match order
4. PD[t] **reordered** by entries_map: `PD[t][sorted_pos_in_table_t] = old_pd[match_idx]`
5. All tables use same (Y, metadata) sort → positions are consistent across tables

### CHUNKED pipeline (broken)
1. Each bucket sorted on GPU by **(Y, original_pos)** — different tie-breaker!
2. PD position = bucket_offset + within_bucket_sorted_pos (store-order in prev table)
3. pd_all[t] collected from dst store in **insertion order** (hash output order, NOT sorted by Y)
4. build_plot_data_from_store sorts pd_all[t] by **(Y, sorted_pos)** to build PD[t]
5. But sorted_pos was computed with (Y, orig_pos) sort, while PD[t] is indexed by (Y, sorted_pos) sort
6. **For duplicate Y values, these give different positions → PD chain breaks**

### Why duplicate Y matters
For k18: ~30% of first 50 final_Y entries have duplicate Y values.
For each duplicate Y, the sort tie-breaker determines the position.
Different tie-breakers → different positions → wrong PD indices.

### The CUDA reference (Node_phase1.hip)
- scatter_2 stores PY = (Y, local_pos) — local position within first-level bucket
- hybrid_sort_y sorts by full PY = (Y, local_pos)
- match_p1 stores PD = (P_x, delta) where P_x = bucket_offset + local_sorted_pos
- eval_p1_tx passes PD through with PD_0 adjustment: PD_out = PD_0 + PD_in
- **All tables use the same (Y, local_pos) sort → consistent**

### The inconsistency in our chunked pipeline
- GPU sorts within bucket by (Y, orig_pos) → position A
- build_plot_data_from_store sorts pd_all by (Y, sorted_pos) → position B
- For duplicate Y: A != B → PD chain breaks

## Fix Options

### Option A: Sort store entries by Y before collecting pd_all
Sort dst store buckets by Y after process_bucket_gpu, so store order = Y-sorted order.
Then pd_all is naturally in Y-sorted order, and PD positions match.
**Challenge**: The tie-breaker must match the GPU's (Y, orig_pos) sort.

### Option B: Use flat pipeline's PD construction on chunked output
Collect all entries + PD data, sort globally by (Y, metadata), rebuild PD using entries_map.
This is what the flat pipeline does. Just apply it to chunked output.
**Challenge**: Need to remap PD positions from store-order to global-sorted-order.

### Option C: Store orig_pos in PDEntry and sort by (Y, orig_pos)
Add orig_pos to PDEntry, sort pd_all by (Y, orig_pos) instead of (Y, sorted_pos).
This matches the GPU's sort order. No remap needed.
**Challenge**: orig_pos is the local position within the bucket, not global.

### Option D: Build entries_map for chunked pipeline (RECOMMENDED)
Like the flat pipeline, build entries_map[t] = sorted_pos → match_idx.
Then reorder PD[t] by entries_map, same as flat.
This requires knowing the match_idx for each sorted position in table t.
**Challenge**: Need to track match order vs sorted order in the chunked pipeline.

## Additional Bug: process_bucket_chunk has NO PD computation
The CPU fallback (process_bucket_chunk) doesn't call append_pd.
When VRAM check fails, PD entries are (0, 0).
Fix: Add PD computation to process_bucket_chunk (same as process_bucket_gpu).
