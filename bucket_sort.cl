// GPU bucket sort kernel: count entries per bucket
// Each thread computes bucket index and atomic_adds to bucket count
__kernel void bucket_count(
    __global const uint* Y_in,       // input Y values
    __global uint* bucket_counts,      // output: count per bucket
    const uint num_buckets,
    const uint shift,
    const uint count)
{
    const uint i = get_global_id(0);
    if(i >= count) return;
    uint bucket = min(Y_in[i] >> shift, num_buckets - 1);
    atomic_add(&bucket_counts[bucket], 1);
}

// GPU bucket sort kernel: scatter entries to sorted positions
// Uses atomic_add to get position within bucket (unstable within bucket, but OK for matching)
__kernel void bucket_scatter(
    __global const uint* Y_in,       // input Y values
    __global const uint* pos_in,     // input positions (original index)
    __global uint* Y_out,            // output: sorted Y values
    __global uint* pos_out,           // output: sorted positions
    __global uint* bucket_offsets,    // input: prefix sum of bucket counts
    __global uint* bucket_atomic,     // input: copy of bucket_offsets (for atomic_add)
    const uint num_buckets,
    const uint shift,
    const uint count)
{
    const uint i = get_global_id(0);
    if(i >= count) return;
    uint bucket = min(Y_in[i] >> shift, num_buckets - 1);
    uint pos = atomic_add(&bucket_atomic[bucket], 1);
    uint global_pos = bucket_offsets[bucket] + pos;
    Y_out[global_pos] = Y_in[i];
    pos_out[global_pos] = pos_in[i];
}
