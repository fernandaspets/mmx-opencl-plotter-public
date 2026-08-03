# PD Bug Analysis — RESOLVED

## Status: All bugs fixed

### Bug 1: Missing cross-boundary matches (FIXED)
Chunked pipeline only matched within first-level buckets, missing Y,Y+1 pairs
at bucket boundaries. Fixed by adding `cross_boundary_match()` function that
finds entries where Y = boundary-1 in bucket y and Y+1 = boundary in bucket y+1.

### Bug 2: X_pairs ordering mismatch (FIXED)
X_pairs were collected in [within-bucket, cross-boundary] order but pd_all[2]
was collected from dst store in bucket order. Fixed by saving X_pairs into
dst store's x_pairs_buckets alongside metadata and PD, then collecting both
from dst in the same order.

### Bug 3: No phase 2 compaction (FIXED — KEY BREAKTHROUGH)
We kept ALL entries in all tables, including unreachable ones. This caused:
- Larger plot files (20% more entries per table)
- Dead-end PD paths that wasted prover time
- "zero matches at table 9" errors for many challenges

Fixed by implementing phase 2 compaction:
1. Mark reachable entries from T9 down to T2
2. Remove unreachable entries
3. Remap PD positions to compacted indices

Result: OCL/CUDA ratio improved from 80.2% to **104.6%** (beating CUDA).

### Bug 4: Sort tie-breaker for duplicate Y (NOT A BUG)
Initially thought duplicate Y sort was causing issues. The (Y, pd_all_index)
sort is consistent across all tables and matches CUDA's (Y, local_pos).
The "zero matches" errors were caused by the missing compaction, not the sort.
