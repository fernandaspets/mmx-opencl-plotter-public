/*
 * gpu_sort.cl — GPU radix sort for (Y, index) pairs
 *
 * 4-pass LSD radix sort on 8-bit digits of Y.
 * Entries packed as uint64: (Y << 32) | index
 */

/* Count entries per digit value for one pass */
__kernel void radix_histogram(
    __global const ulong* data,
    __global uint* hist_global,     // [256] output
    const uint n,
    const uint shift)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    
    __local uint hist_local[256];
    if(lid < 256) hist_local[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(uint i = gid; i < n; i += get_global_size(0)) {
        uint Y = (uint)(data[i] >> 32);
        uint d = (Y >> shift) & 0xFF;
        atomic_inc(&hist_local[d]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(lid < 256) atomic_add(&hist_global[lid], hist_local[lid]);
}

/*
 * Scatter entries to sorted positions.
 * Uses prefix-summed global histogram + per-WG local histograms.
 * Work-groups: launch num_wg = (n + THREADS_PER_WG - 1) / THREADS_PER_WG
 *
 * Each WG:
 *   1. Builds local histogram of its entries
 *   2. Performs local prefix sum
 *   3. Computes output offset = global_prefix[digit] + sum_of_earlier_WGs_hist[digit] + local_prefix
 *   4. Writes entry to output[offset]
 *
 * The WG prefix (sum of earlier WGs' histograms) is computed by:
 *   wg_prefix[digit] = sum_{w < wg_id} hist_wg_local[w][digit]
 * This requires a WG-level prefix, which we compute iteratively.
 *
 * Simpler approach for first version:
 *   1. Each WG scans its assigned entries, counts per digit in local hist
 *   2. Does local prefix sum → local offsets
 *   3. Writes: output[global_offset[digit] + wg_global_offset + local_offset] = entry
 *      where wg_global_offset is computed from global_offset + running per-WG count for this digit
 *
 * Simplest correct approach: use atomic_inc on global offset array.
 * Just 67M atomics × 4 passes = 268M atomics. Should be fast on modern GPUs.
 */

__kernel void radix_scatter(
    __global const ulong* data_in,
    __global ulong* data_out,
    __global volatile uint* counters,  // [256] running counters, initialized to global_offset
    const uint n,
    const uint shift)
{
    const uint gid = get_global_id(0);
    if(gid >= n) return;
    
    ulong entry = data_in[gid];
    uint Y = (uint)(entry >> 32);
    uint d = (Y >> shift) & 0xFF;
    
    uint pos = atomic_inc(&counters[d]);
    data_out[pos] = entry;
}
