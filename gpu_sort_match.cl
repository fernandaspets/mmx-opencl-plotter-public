/*
 * gpu_sort_match.cl — GPU counting sort by full Y value
 *
 * Approach:
 * 1. Scatter: output[Y * max_dup + counter[Y]++] = entry
 * 2. Compact: scan output, collect non-zero entries in order
 *
 * Works for k22-k26 (2^KSIZE * 8 bytes = up to 512MB scatter array).
 * Falls back to CPU for larger k.
 */

/* Scatter entries to output[Y * max_dup + counter[Y]++] */
__kernel void y_scatter(
    __global const ulong* data_in,
    __global ulong* data_out,
    __global volatile uint* counters,
    const uint n,
    const uint ksize,
    const uint max_dup)
{
    uint gid = get_global_id(0);
    if(gid >= n) return;
    ulong entry = data_in[gid];
    uint Y = (uint)(entry >> 32);
    uint pos = atomic_inc(&counters[Y]);
    if(pos < max_dup)
        data_out[Y * max_dup + pos] = entry;
}

/* Compact: collect sorted entries from scattered output.
 * Each WG scans a range of Y values and writes compacted entries.
 */
__kernel void y_compact(
    __global const ulong* scattered,  // [2^ksize * max_dup]
    __global ulong* output,           // [n] compacted sorted entries
    __global const uint* counts,      // [2^ksize] counts per Y
    __global volatile uint* out_pos,  // [1] running output position
    const uint y_start,               // starting Y value for this WG
    const uint y_end,                 // ending Y value (exclusive)
    const uint max_dup)
{
    for(uint y = y_start + get_local_id(0); y < y_end; y += get_local_size(0)) {
        uint cnt = min(counts[y], (uint)max_dup);
        for(uint p = 0; p < cnt; p++) {
            uint pos = atomic_inc(out_pos);
            output[pos] = scattered[y * max_dup + p];
        }
    }
}
