/*
 * prefix_sum.cl — Module A: GPU prefix sum (Blelloch scan)
 *
 * Computes exclusive prefix sum of sub-bucket counts on GPU,
 * eliminating the need to download sub_cnt to CPU.
 *
 * Input:  sub_cnt[num_sub] — counts per sub-bucket
 * Output: sub_off[num_sub+1] — exclusive prefix sum (sub_off[0]=0, sub_off[i]=sum(sub_cnt[0..i-1]))
 *
 * Uses Hillis-Steele scan (work-efficient for small arrays).
 * num_sub is typically 2048-8192 for k20-k25, fits in one work-group.
 */

__kernel void gpu_prefix_sum(
    __global const uint* input,    // [num_sub] counts
    __global uint* output,         // [num_sub+1] offsets (output[0]=0)
    __local uint* temp,            // [num_sub+1] scratch
    const uint num_sub)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    
    // Load data into local memory
    temp[lid] = (gid < num_sub) ? input[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Hillis-Steele inclusive scan
    for(uint offset = 1; offset < num_sub; offset *= 2) {
        uint val = temp[lid];
        if(lid >= offset) val += temp[lid - offset];
        barrier(CLK_LOCAL_MEM_FENCE);
        temp[lid] = val;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Convert to exclusive scan: output[0] = 0, output[i] = temp[i-1]
    if(gid == 0) {
        output[0] = 0;
    }
    if(gid < num_sub) {
        output[gid + 1] = temp[gid];
    }
}
// GPU parallel prefix sum (Blelloch scan) — not used yet, placeholder
