// GPU match kernel: direct pairing using sort counts/offsets
// Each work-item processes one Y value and pairs all entries with Y to all with Y+1
// Outputs: LR_orig (for hash), LR_sorted (for PD), Y_L (for sorting PD)

__kernel void gpu_match_sorted(
    __global const uint* Y_sorted,      // sorted Y values (not directly needed)
    __global const uint* pos_sorted,    // sorted positions (original indices into M_curr)
    __global const uint* counts,        // count per Y value (from counting sort)
    __global const uint* offsets,       // offset per Y value (from counting sort)
    __global uint* LR_orig,             // output: (orig_L, orig_R) per match — for GPU hash
    __global uint* LR_sorted,            // output: (sorted_L, sorted_R) per match — for PD
    __global uint* Y_L_out,             // output: Y value of left entry per match — for sorting PD
    __global uint* match_count,         // output: total matches (atomic, must be zeroed)
    const uint kmask,
    const uint max_matches)
{
    const uint Y = get_global_id(0);
    if(Y > kmask) return;
    
    uint count_y = counts[Y];
    if(count_y == 0) return;
    
    uint Y1 = Y + 1;
    if(Y1 > kmask) return;
    
    uint count_y1 = counts[Y1];
    if(count_y1 == 0) return;
    
    // Pair all entries with Y to all entries with Y+1
    uint off_y = offsets[Y];
    uint off_y1 = offsets[Y1];
    
    for(uint i = 0; i < count_y; i++) {
        uint sorted_L = off_y + i;
        uint orig_L = pos_sorted[sorted_L];
        for(uint j = 0; j < count_y1; j++) {
            uint sorted_R = off_y1 + j;
            uint orig_R = pos_sorted[sorted_R];
            uint pos = atomic_add(match_count, 1);
            if(pos < max_matches) {
                LR_orig[pos * 2] = orig_L;
                LR_orig[pos * 2 + 1] = orig_R;
                LR_sorted[pos * 2] = sorted_L;
                LR_sorted[pos * 2 + 1] = sorted_R;
                Y_L_out[pos] = Y;  // Y value of left entry for PD sorting
            }
        }
    }
}
