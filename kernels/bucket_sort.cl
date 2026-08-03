// GPU counting sort: single-pass sort by Y value
// Each Y value gets its own bin (2^KSIZE bins, filtered by kmask)

// Count entries per Y value
__kernel void count_by_Y(
    __global const uint* Y_in,       // input Y values
    __global volatile uint* counts,   // output: count per Y (must be zeroed)
    const uint kmask,
    const uint count)
{
    const uint i = get_global_id(0);
    if(i >= count) return;
    uint Y = Y_in[i] & kmask;
    atomic_add(&counts[Y], 1);
}

// Scatter entries to sorted positions by Y value
__kernel void scatter_by_Y(
    __global const uint* Y_in,       // input Y values
    __global const uint* val_in,     // input values (original positions)
    __global uint* Y_out,            // output: sorted Y values
    __global uint* val_out,          // output: sorted values
    __global const uint* offsets,    // prefix sum of counts
    __global volatile uint* atomic_buf, // zeroed (for atomic_add to get position)
    const uint kmask,
    const uint count)
{
    const uint i = get_global_id(0);
    if(i >= count) return;
    uint Y = Y_in[i] & kmask;
    uint pos = atomic_add(&atomic_buf[Y], 1);
    uint global_pos = offsets[Y] + pos;
    Y_out[global_pos] = Y_in[i];
    val_out[global_pos] = val_in[i];
}

// Compute offsets from sorted data: offsets[Y] = first position of Y in sorted array
__kernel void compute_offsets(
    __global const uint* Y_sorted,      // sorted Y values
    __global volatile uint* offsets,     // output: offset per Y (must be zeroed)
    const uint kmask,
    const uint count)
{
    const uint i = get_global_id(0);
    if(i >= count) return;

    uint Y = Y_sorted[i] & kmask;

    if(i == 0) {
        offsets[Y] = 0;
        return;
    }

    uint prev = Y_sorted[i-1] & kmask;
    if(Y != prev) {
        offsets[Y] = i;
    }
}


// Prefix sum (inclusive scan) for small arrays (up to 4096 elements)
__kernel void prefix_sum_counts(
    __global volatile uint* data,
    const uint count)
{
    // Simple single-threaded prefix sum for small arrays
    if(get_global_id(0) != 0) return;
    uint sum = 0;
    for(uint i = 0; i < count; i++) {
        uint val = data[i];
        data[i] = sum;
        sum += val;
    }
    // Store total count at data[count] (for convenience)
    if(count < 4096) data[count] = sum;
}
