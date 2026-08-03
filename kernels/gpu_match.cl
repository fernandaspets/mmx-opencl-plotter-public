// GPU match kernel: direct pairing using sort counts/offsets
// Also stabilizes: sorts entries within same Y by original position

__kernel void gpu_match_sorted(
    __global const uint* Y_sorted,      // sorted Y values
    __global uint* pos_sorted,    // sorted positions (original indices) — WILL BE REORDERED for stability
    __global const uint* counts,        // count per Y value (from counting sort)
    __global const uint* offsets,       // offset per Y value (from counting sort)
    __global uint* LR_orig,             // output: (orig_L, orig_R) per match
    __global uint* LR_sorted,            // output: (sorted_L, sorted_R) per match
    __global uint* Y_L_out,             // output: Y value of left entry per match
    __global uint* match_count,         // output: total matches (atomic, must be zeroed)
    const uint kmask,
    const uint max_matches)
{
    const uint Y = get_global_id(0);
    if(Y > kmask) return;

    uint count_y = counts[Y];
    if(count_y == 0) return;

    uint off_y = offsets[Y];

    // Stabilize: sort entries within this Y bucket by original position
    for(uint i = 1; i < count_y; i++) {
        uint key = pos_sorted[off_y + i];
        int j = (int)i - 1;
        while(j >= 0 && pos_sorted[off_y + j] > key) {
            pos_sorted[off_y + j + 1] = pos_sorted[off_y + j];
            j--;
        }
        pos_sorted[off_y + j + 1] = key;
    }

    uint Y1 = Y + 1;
    if(Y1 > kmask) return;

    uint count_y1 = counts[Y1];
    if(count_y1 == 0) return;

    uint off_y1 = offsets[Y1];
    for(uint i = 1; i < count_y1; i++) {
        uint key = pos_sorted[off_y1 + i];
        int j = (int)i - 1;
        while(j >= 0 && pos_sorted[off_y1 + j] > key) {
            pos_sorted[off_y1 + j + 1] = pos_sorted[off_y1 + j];
            j--;
        }
        pos_sorted[off_y1 + j + 1] = key;
    }

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
                Y_L_out[pos] = Y;
            }
        }
    }
}
