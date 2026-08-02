/*
 * gather_meta.cl — GPU gather kernel for Final copy
 *
 * Reorders M_curr from match order to sorted order on GPU.
 * Eliminates random-access CPU copy (5.3s for k27 → ~0.4s).
 *
 * Each work-item processes one entry: reads M_curr[perm[i]*META] and writes to output[i*META].
 */

__kernel void gather_meta(
    __global const uint* M_curr,    // [num_entries * META] metadata in match order
    __global const uint* perm,      // [num_entries] permutation: sorted_pos -> match_idx
    __global uint* output,          // [num_entries * META] output in sorted order
    const uint num_entries,
    const uint META)                // 14
{
    const uint gid = get_global_id(0);
    if(gid >= num_entries) return;
    
    const uint src_idx = perm[gid];
    const uint src_offset = src_idx * META;
    const uint dst_offset = gid * META;
    
    for(uint j = 0; j < META; j++) {
        output[dst_offset + j] = M_curr[src_offset + j];
    }
}
