// Test kernel: compute SHA-512(X_i || plot_id) and output the key
// This isolates the SHA-512 implementation from gen_mem_array and calc_mem_hash

// Include the SHA-512 code from pos_recompute.cl
${SHA512_CODE}

__kernel void test_sha512(
    __global const uint* X_in,
    __global const uint* ID_in,
    __global uint* key_out,   // [16] uint32 = 64 bytes
    const uint num_x)
{
    const uint gid = get_global_id(0);
    if(gid >= num_x) return;
    
    const uint X_i = X_in[gid];
    
    // Build message: X_i (4 bytes) || plot_id (32 bytes) = 36 bytes
    uint msg32[18];
    for(int i = 0; i < 18; i++) msg32[i] = 0;
    msg32[0] = X_i;
    for(int i = 0; i < 8; i++) msg32[1 + i] = ID_in[i];
    
    // Pack into BE ulong array
    ulong msg64[16];
    for(int i = 0; i < 16; i++) msg64[i] = 0;
    pack_uint32_to_be_ulong(msg32, 9, msg64, 16);
    
    // SHA-512
    ulong key_state[8];
    sha512_hash(msg64, 36, key_state);
    
    // Extract as uint32
    uint key32[16];
    extract_uint32_from_sha512(key_state, key32);
    
    // Output
    for(int i = 0; i < 16; i++) key_out[gid * 16 + i] = key32[i];
}
