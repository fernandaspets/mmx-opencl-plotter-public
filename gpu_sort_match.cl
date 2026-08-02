/*
 * gpu_sort_match.cl — GPU sort + match kernels for flat pipeline
 *
 * Module J: Eliminate host overhead by doing match on GPU.
 * The flat_match kernel replaces CPU match + flatten LR steps.
 */

/* ============= Flat GPU Match ============= */

/* Match: find ALL Y,Y+1 pairs from sorted uint64 array.
 * For each entry, scan forward until Y_R > Y_L+1.
 * Emit a match for EVERY entry where Y_R == Y_L+1 (handles duplicates).
 * Uses atomic counter for output position.
 */
__kernel void flat_match(
    __global const ulong* entries,    // [n] sorted by Y, packed as (Y<<32 | original_index)
    __global uint* lr_out,             // [max_matches * 2] output (P1, P2) pairs
    __global volatile uint* count_out,  // [1] atomic counter (init to 0)
    const uint n,
    const uint max_matches)
{
    uint gid = get_global_id(0);
    if(gid >= n) return;
    
    uint YL = (uint)(entries[gid] >> 32);
    
    for(uint j = gid + 1; j < n; j++) {
        uint YR = (uint)(entries[j] >> 32);
        if(YR > YL + 1) break;
        if(YR == YL + 1) {
            uint pos = atomic_inc(count_out);
            if(pos < max_matches) {
                lr_out[pos * 2]     = (uint)(entries[gid]);  // P1 = original index of L
                lr_out[pos * 2 + 1] = (uint)(entries[j]);     // P2 = original index of R
            }
        }
    }
}

/* ============= Pack entries as uint64 ============= */
/* Convert (Y, index) pairs to uint64 (Y<<32 | index) for GPU processing */
__kernel void pack_entries(
    __global const uint* Y_in,        // [n] Y values
    __global ulong* entries_out,       // [n] output (Y<<32 | index)
    const uint n)
{
    uint gid = get_global_id(0);
    if(gid >= n) return;
    entries_out[gid] = ((ulong)Y_in[gid] << 32) | (ulong)gid;
}

/* ============= Build PD in sorted order ============= */
/* After GPU sort of matches, build PD using sorted positions.
 * sorted_entries[k] = (Y_out<<32 | match_idx)
 * LR_pairs[match_idx] = (sorted_L, sorted_R)  — positions in previous table's sorted array
 * PD[k] = {sorted_L, sorted_R - sorted_L}
 */
__kernel void build_pd_gpu(
    __global const ulong* sorted_entries,  // [m] (Y<<32 | match_idx), sorted by Y
    __global const uint2* lr_pairs,        // [m] (sorted_L, sorted_R) in match order
    __global uint2* pd_out,               // [m] output PD
    const uint m)
{
    uint gid = get_global_id(0);
    if(gid >= m) return;
    uint match_idx = (uint)(sorted_entries[gid]);
    uint2 lr = lr_pairs[match_idx];
    pd_out[gid] = (uint2)(lr.x, lr.y - lr.x);
}

/* ============= Extract match indices for entries_map ============= */
/* entries_map[k] = match_idx = sorted_entries[k] & 0xFFFFFFFF */
__kernel void extract_match_idx(
    __global const ulong* sorted_entries,
    __global uint* match_idx_out,
    const uint m)
{
    uint gid = get_global_id(0);
    if(gid >= m) return;
    match_idx_out[gid] = (uint)(sorted_entries[gid]);
}
