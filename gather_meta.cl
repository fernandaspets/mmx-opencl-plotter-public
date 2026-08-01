/*
 * gather_meta.cl — Module B variant 2: GPU gather kernel
 *
 * Gathers L_meta and R_meta from M_curr using LR pair indices.
 * This separates scattered reads (gather) from sequential reads (hash).
 * The hash_table_entries kernel then reads the gathered data sequentially.
 *
 * This avoids the AMD codegen bug in hash_table_lr where scattered reads
 * and SHA-512 computation are interleaved in the same kernel.
 */

__kernel void gather_meta(
    __global const uint* M_curr,      // [num_total * N_META] all metadata
    __global const uint* LR_pairs,     // [num_matches * 2] P1, P2 indices
    __global uint* L_meta_out,         // [num_matches * N_META] gathered L metadata
    __global uint* R_meta_out,         // [num_matches * N_META] gathered R metadata
    const uint num_matches,
    const uint num_total_entries,
    const uint N_META)
{
    const uint gid = get_global_id(0);
    if(gid >= num_matches) return;
    
    uint P1 = LR_pairs[gid * 2];
    uint P2 = LR_pairs[gid * 2 + 1];
    
    // Bounds check (prevents out-of-bounds reads)
    if(P1 >= num_total_entries) P1 = 0;
    if(P2 >= num_total_entries) P2 = 0;
    
    // Gather L_meta and R_meta (scattered reads, sequential writes)
    for(uint i = 0; i < N_META; i++) {
        L_meta_out[gid * N_META + i] = M_curr[P1 * N_META + i];
        R_meta_out[gid * N_META + i] = M_curr[P2 * N_META + i];
    }
}
