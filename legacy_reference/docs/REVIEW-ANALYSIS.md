# Junior Dev Review Analysis — Final Status

## Reviewer 1: calc_mem_hash fixes
- MEM_HASH_ITER 11337 vs 256: **RESOLVED** (was our test bug)
- Register spilling: **NOT AN ISSUE** (verified by test_memhash_multi)
- `% 32` → `& 31`: **APPLIED** (good safety fix)
- `#pragma unroll 1`: **APPLIED** (good practice)
- `volatile`: **NOT NEEDED** (F1 matches 100/100)

## Reviewer 2: Architecture analysis
- "F1 is WRONG (P0)": **INCORRECT** (was stale info, F1 was never broken)
- PD sort tie-breaker (orig_pos): **DIRECTIONALLY RIGHT** but not the actual fix
- process_bucket_chunk missing PD: **VALID** (latent bug, not triggered for k20)
- Duplicate bounds check in table_hash.cl: **FIXED** (good catch)
- atomic_write_bits race: **VALID CONCERN** (not yet addressed, latent)
- SHA-512 convention mismatch: **VALID** (maintenance hazard, not a bug)
- Modularize plotter.cpp: **GOOD ADVICE** (future work)

## What Actually Fixed the Issues
1. **Cross-boundary matching**: Added CPU cross-boundary match function
2. **X_pairs ordering**: Saved X_pairs into dst store alongside metadata/PD
3. **Phase 2 compaction**: Remove unreachable entries, remap PD positions
   → This was the KEY breakthrough (80.2% → 104.6% of CUDA quality)

## Summary
The reviews were helpful for identifying small bugs and pushing toward
CUDA comparison. The main breakthroughs (cross-boundary matching, phase 2
compaction) were found by comparing entry counts and file structures
against the CUDA reference plotter.
