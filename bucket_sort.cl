// GPU bucket sort: sort (Y, value) pairs by Y
// Multiple passes for full radix sort

// Count entries per bucket
__kernel void bucket_count(
    __global const uint* Y_in,       // input Y values
    __global uint* bucket_counts,      // output: count per bucket (must be zeroed)
    const uint shift,
    const uint mask,
    const uint num_bins,
    const uint count)
{
    const uint i = get_global_id(0);
    if(i >= count) return;
    uint bucket = (Y_in[i] >> shift) & mask;
    atomic_add(&bucket_counts[bucket], 1);
}

// Scatter entries to sorted positions (unstable within bucket)
__kernel void bucket_scatter(
    __global const uint* Y_in,       // input Y values
    __global const uint* val_in,      // input values (match_idx)
    __global uint* Y_out,            // output: sorted Y values
    __global uint* val_out,           // output: sorted values
    __global const uint* bucket_offsets,  // prefix sum of counts
    __global uint* bucket_atomic,     // copy of offsets (for atomic_add)
    const uint shift,
    const uint mask,
    const uint count)
{
    const uint i = get_global_id(0);
    if(i >= count) return;
    uint bucket = (Y_in[i] >> shift) & mask;
    uint pos = atomic_add(&bucket_atomic[bucket], 1);
    uint global_pos = bucket_offsets[bucket] + pos;
    Y_out[global_pos] = Y_in[i];
    val_out[global_pos] = val_in[i];
}
