// GPU match kernel: scan sorted entries for Y, Y+1 pairs
// Outputs: LR_orig (orig_L, orig_R) for GPU hash, LR_sorted (sorted_L, sorted_R) for PD

__kernel void gpu_match_sorted(
    __global const uint* Y_sorted,      // sorted Y values
    __global const uint* pos_sorted,    // sorted positions (original indices into M_curr)
    __global uint* LR_orig,             // output: (orig_L, orig_R) per match — for GPU hash
    __global uint* LR_sorted,            // output: (sorted_L, sorted_R) per match — for PD
    __global uint* match_count,          // output: total matches (atomic, must be zeroed)
    const uint count,                    // number of entries
    const uint kmask,                    // Y mask
    const uint max_matches)              // safety limit
{
    const uint i = get_global_id(0);
    if(i >= count) return;
    
    uint YL = Y_sorted[i] & kmask;
    
    // Scan forward for Y+1 (entries are sorted by Y)
    uint j = i + 1;
    while(j < count && (Y_sorted[j] & kmask) <= YL + 1) {
        if((Y_sorted[j] & kmask) == YL + 1) {
            uint pos = atomic_add(match_count, 1);
            if(pos < max_matches) {
                LR_orig[pos * 2] = pos_sorted[i];     // orig_L
                LR_orig[pos * 2 + 1] = pos_sorted[j]; // orig_R
                LR_sorted[pos * 2] = i;                 // sorted_L
                LR_sorted[pos * 2 + 1] = j;             // sorted_R
            }
        }
        j++;
    }
}
