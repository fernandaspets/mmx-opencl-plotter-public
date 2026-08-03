// Simple odd-even sort: load entries into shared memory, sort, write back
// Each work-group processes one sub-bucket. Uses 256 threads.
__kernel void simple_sort_y(
    __global ulong* data,
    __global const uint* bucket_size,
    const uint max_bucket_size,
    const uint num_sub_buckets)
{
    const uint x = get_local_id(0);
    const uint y = get_group_id(1);  // sub-bucket index (dim1)
    
    __local ulong buf[2048];  // max bucket size
    
    const uint n = min(bucket_size[y], max_bucket_size);
    if (n == 0) return;
    
    // Load into shared memory
    for (uint i = x; i < n; i += get_local_size(0)) {
        buf[i] = data[y * max_bucket_size + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Odd-even sort: n passes, alternating even/odd phases
    for (uint p = 0; p < n; p++) {
        // Even phase: compare (0,1), (2,3), (4,5), ...
        // Odd phase: compare (1,2), (3,4), (5,6), ...
        for (uint i = x; i < n - 1; i += get_local_size(0)) {
            if ((i & 1) == (p & 1)) {
                if (buf[i] > buf[i + 1]) {
                    ulong tmp = buf[i];
                    buf[i] = buf[i + 1];
                    buf[i + 1] = tmp;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write back
    for (uint i = x; i < n; i += get_local_size(0)) {
        data[y * max_bucket_size + i] = buf[i];
    }
}
