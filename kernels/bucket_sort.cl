// GPU counting sort: single-pass sort by Y value
// Y values are masked to kmask (2^ksize - 1, up to 2^22 - 1 = 4M bins)

// Count entries per Y value
__kernel void count_by_Y(
    __global const uint* Y_in,
    __global volatile uint* counts,
    const uint count)
{
    const uint i = get_global_id(0);
    if(i >= count) return;
    uint Y = Y_in[i];
    atomic_add(&counts[Y], 1);
}

// Compute offsets from sorted data
__kernel void compute_offsets(
    __global const uint* Y_sorted,
    __global volatile uint* offsets,
    const uint kmask,
    const uint count)
{
    const uint i = get_global_id(0);
    if(i >= count) return;
    uint Y = Y_sorted[i];
    if(i == 0) { offsets[Y] = 0; return; }
    uint prev = Y_sorted[i-1];
    if(Y != prev) offsets[Y] = i;
}

// Scatter entries to sorted positions
__kernel void scatter_by_Y(
    __global const uint* Y_in,
    __global const uint* val_in,
    __global uint* Y_out,
    __global uint* val_out,
    __global const uint* offsets,
    __global volatile uint* atomic_buf,
    const uint count)
{
    const uint i = get_global_id(0);
    if(i >= count) return;
    uint Y = Y_in[i];
    uint pos = atomic_add(&atomic_buf[Y], 1);
    uint global_pos = offsets[Y] + pos;
    Y_out[global_pos] = Y_in[i];
    val_out[global_pos] = val_in[i];
}
