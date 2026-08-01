// Test kernel: hash one LR pair and compare with CPU reference
// This tests that the GPU SHA-512 + Y/M extraction matches CPU

__kernel void test_hash_single(
    __global const uint* M_curr,    // [2 * 14] two entries
    __global uint* Y_out,           // [1] Y result
    __global uint* M_out,           // [14] M result
    const uint kmask,
    const uint num_total_entries)
{
    const uint gid = get_global_id(0);
    if(gid > 0) return;
    
    uint P1 = 0;
    uint P2 = 1;
    
    if(P1 >= num_total_entries || P2 >= num_total_entries) {
        Y_out[gid] = 0xFFFFFFFF;
        return;
    }
    
    // Copy SHA-512 code from table_hash.cl
    __private uint L[14], R[14];
    for(int i = 0; i < 14; i++) {
        L[i] = M_curr[P1 * 14 + i];
        R[i] = M_curr[P2 * 14 + i];
    }
    
    // Pack and hash (same as hash_table_lr)
    __private ulong msg[16];
    for(int i = 0; i < 16; i++) msg[i] = 0;
    
    uint vals[28];
    for(int i = 0; i < 14; i++) { vals[i] = L[i]; vals[14+i] = R[i]; }
    
    for(int i = 0; i < 14; i++) {
        ulong lo = (ulong)vals[i*2];
        ulong hi = (ulong)vals[i*2+1];
        ulong le_val = lo | (hi << 32);
        msg[i] = ((le_val & 0xFF00000000000000UL) >> 56)
               | ((le_val & 0x00FF000000000000UL) >> 40)
               | ((le_val & 0x0000FF0000000000UL) >> 24)
               | ((le_val & 0x000000FF00000000UL) >> 8)
               | ((le_val & 0x00000000FF000000UL) << 8)
               | ((le_val & 0x0000000000FF0000UL) << 24)
               | ((le_val & 0x000000000000FF00UL) << 40)
               | ((le_val & 0x00000000000000FFUL) << 56);
    }
    
    msg[14] = 0x8000000000000000UL;
    msg[15] = 0;
    
    __private ulong state[8];
    for(int i = 0; i < 8; i++) state[i] = SHA512_INIT[i];
    sha512_block(msg, state);
    
    __private ulong block2[16];
    for(int i = 0; i < 16; i++) block2[i] = 0;
    block2[15] = (ulong)112 * 8;
    sha512_block(block2, state);
    
    uint hash[16];
    for(int i = 0; i < 8; i++) {
        ulong s = state[i];
        uint hi32 = (uint)(s >> 32);
        uint lo32 = (uint)(s & 0xFFFFFFFF);
        hash[2*i]   = ((hi32 & 0xFF) << 24) | ((hi32 & 0xFF00) << 8)
                     | ((hi32 >> 8) & 0xFF00) | ((hi32 >> 24) & 0xFF);
        hash[2*i+1] = ((lo32 & 0xFF) << 24) | ((lo32 & 0xFF00) << 8)
                     | ((lo32 >> 8) & 0xFF00) | ((lo32 >> 24) & 0xFF);
    }
    
    uint Y = 0;
    for(int i = 0; i < 14; i++) Y ^= hash[i];
    Y &= kmask;
    
    Y_out[gid] = Y;
    for(int i = 0; i < 14; i++) M_out[gid * 14 + i] = hash[i] & kmask;
}
