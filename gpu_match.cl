// GPU match kernel: scan sorted entries for Y, Y+1 pairs
// Output: 4 values per match: sorted_L, sorted_R, orig_L, orig_R

__kernel void gpu_match_sorted(
    __global const uint* Y_sorted,      // sorted Y values
    __global const uint* pos_sorted,    // sorted positions (original indices)
    __global uint* LR_out,              // output: [sorted_L, sorted_R, orig_L, orig_R] per match
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
                LR_out[pos * 4] = i;              // sorted_L
                LR_out[pos * 4 + 1] = j;           // sorted_R
                LR_out[pos * 4 + 2] = pos_sorted[i]; // orig_L
                LR_out[pos * 4 + 3] = pos_sorted[j]; // orig_R
            }
        }
        j++;
    }
}
