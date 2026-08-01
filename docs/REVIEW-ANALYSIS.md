# Junior Dev Review Analysis (Aug 2026)

Two junior AI developers reviewed the codebase. Here's our assessment of their findings.

## Reviewer 1: calc_mem_hash fixes

### Claim: MEM_HASH_ITER 11337 vs 256
**Status: RESOLVED (before review)**
- CPU (mmx-node config.h): `MEM_HASH_ITER = 256`
- GPU (pos_recompute.cl): `#define MEM_HASH_ITER 256`
- These MATCH. The 11337 confusion was from our test bug (test used wrong value).
- Note: 11337 is NOT the MMX network port — it's just a number we mistakenly used.

### Claim: Register spilling of mem[1024]
**Status: VALID CONCERN but not a correctness issue**
- mem[1024] = 4KB per work-item does cause register spilling on RDNA3.
- test_memhash_multi verified: 10 work-items with work-group=64, all match CPU.
- Spilling may hurt performance at scale but doesn't cause wrong output.
- Worth monitoring when running k29+ with large batch sizes.

### Claim: % 32 vs & 31
**Status: APPLIED ✓**
- Changed `% 32` to `& 31` in calc_mem_hash loop for safety and performance.
- Mathematically identical for unsigned positive integers.

### Claim: volatile + #pragma unroll 1
**Status: #pragma unroll 1 APPLIED ✓, volatile NOT needed**
- `#pragma unroll 1` added to prevent aggressive unrolling of the 256-iteration loop.
- `volatile` is NOT needed — F1 output matches CPU byte-for-byte (100/100).
- The F1 bug they reference was our test bug (wrong iteration count), not a real bug.

## Reviewer 2: Architecture analysis

### Claim: F1 is WRONG (P0 blocker)
**Status: INCORRECT (based on stale info)**
- The "F1 100% wrong" finding was caused by our test using MEM_HASH_ITER=11337
  while the actual value is 256. After fixing the test, F1 matches 100/100.
- F1 is NOT broken. The 98% flat pass rate is real, not coincidental.

### Claim: PD sort tie-breaker (Option A: orig_pos)
**Status: GOOD INSIGHT — aligns with our analysis**
- Suggestion to store `orig_pos` in PDEntry and sort by (Y, orig_pos) is valid.
- This is cleaner than our current (Y, left_pos) sort + remap approach.
- TODO: Consider implementing this approach for the chunked PD fix.

### Claim: process_bucket_chunk missing PD
**Status: KNOWN BUG — documented in PD-BUG-ANALYSIS.md**
- Not yet fixed. Low priority since chunked pipeline has bigger PD issues.

### Claim: Duplicate bounds check in table_hash.cl
**Status: APPLIED ✓ — removed duplicate**

### Claim: atomic_write_bits race in f2_f9.cl
**Status: VALID CONCERN — needs investigation**
- Two 32-bit atomic_or calls for a 64-bit write is not truly atomic.
- Usually safe because different work-items write to different bit offsets.
- But could cause issues at bucket boundaries. TODO: verify.

### Claim: SHA-512 convention mismatch between .cl files
**Status: VALID — confusing but not a bug**
- pos_recompute.cl uses BE ulongs, table_hash.cl uses LE ulongs with bswap.
- Both produce correct results (verified by tests). But should be unified for clarity.

### Claim: Modularize plotter.cpp
**Status: GOOD LONG-TERM ADVICE**
- plotter.cpp is ~2300 lines. Splitting into modules is good practice.
- Low priority — focus on fixing chunked PD first.

## Summary of Applied Fixes
1. ✅ `% 32` → `& 31` in pos_recompute.cl (safety + performance)
2. ✅ `#pragma unroll 1` on calc_mem_hash loop (prevent aggressive unrolling)
3. ✅ Removed duplicate bounds check in table_hash.cl
4. All 7 regression tests still pass after changes.
