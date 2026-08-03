/*
 * table_hash.cl — OpenCL kernel for MMX table hashing (F2-F9)
 *
 * Each work-item hashes one LR pair:
 *   msg = L_meta || R_meta  (28 uint32s = 112 bytes)
 *   hash = SHA-512(msg)     (16 uint32s = 64 bytes)
 *   Y = XOR(hash[0..13]) & kmask
 *   M[i] = hash[i] & kmask
 *
 * This replaces the CPU SHA-512 bottleneck in the plotter.
 */

/* === SHA-512 === */

__constant ulong SHA512_K[80] = {
	0x428a2f98d728ae22UL, 0x7137449123ef65cdUL, 0xb5c0fbcfec4d3b2fUL, 0xe9b5dba58189dbbcUL, 0x3956c25bf348b538UL,
	0x59f111f1b605d019UL, 0x923f82a4af194f9bUL, 0xab1c5ed5da6d8118UL, 0xd807aa98a3030242UL, 0x12835b0145706fbeUL,
	0x243185be4ee4b28cUL, 0x550c7dc3d5ffb4e2UL, 0x72be5d74f27b896fUL, 0x80deb1fe3b1696b1UL, 0x9bdc06a725c71235UL,
	0xc19bf174cf692694UL, 0xe49b69c19ef14ad2UL, 0xefbe4786384f25e3UL, 0x0fc19dc68b8cd5b5UL, 0x240ca1cc77ac9c65UL,
	0x2de92c6f592b0275UL, 0x4a7484aa6ea6e483UL, 0x5cb0a9dcbd41fbd4UL, 0x76f988da831153b5UL, 0x983e5152ee66dfabUL,
	0xa831c66d2db43210UL, 0xb00327c898fb213fUL, 0xbf597fc7beef0ee4UL, 0xc6e00bf33da88fc2UL, 0xd5a79147930aa725UL,
	0x06ca6351e003826fUL, 0x142929670a0e6e70UL, 0x27b70a8546d22ffcUL, 0x2e1b21385c26c926UL, 0x4d2c6dfc5ac42aedUL,
	0x53380d139d95b3dfUL, 0x650a73548baf63deUL, 0x766a0abb3c77b2a8UL, 0x81c2c92e47edaee6UL, 0x92722c851482353bUL,
	0xa2bfe8a14cf10364UL, 0xa81a664bbc423001UL, 0xc24b8b70d0f89791UL, 0xc76c51a30654be30UL, 0xd192e819d6ef5218UL,
	0xd69906245565a910UL, 0xf40e35855771202aUL, 0x106aa07032bbd1b8UL, 0x19a4c116b8d2d0c8UL, 0x1e376c085141ab53UL,
	0x2748774cdf8eeb99UL, 0x34b0bcb5e19b48a8UL, 0x391c0cb3c5c95a63UL, 0x4ed8aa4ae3418acbUL, 0x5b9cca4f7763e373UL,
	0x682e6ff3d6b2b8a3UL, 0x748f82ee5defb2fcUL, 0x78a5636f43172f60UL, 0x84c87814a1f0ab72UL, 0x8cc702081a6439ecUL,
	0x90befffa23631e28UL, 0xa4506cebde82bde9UL, 0xbef9a3f7b2c67915UL, 0xc67178f2e372532bUL, 0xca273eceea26619cUL,
	0xd186b8c721c0c207UL, 0xeada7dd6cde0eb1eUL, 0xf57d4f7fee6ed178UL, 0x06f067aa72176fbaUL, 0x0a637dc5a2c898a6UL,
	0x113f9804bef90daeUL, 0x1b710b35131c471bUL, 0x28db77f523047d84UL, 0x32caab7b40c72493UL, 0x3c9ebe0a15c9bebcUL,
	0x431d67c49c100d4cUL, 0x4cc5d4becb3e42b6UL, 0x597f299cfc657e2aUL, 0x5fcb6fab3ad6faecUL, 0x6c44198c4a475817UL
};

__constant ulong SHA512_INIT[8] = {
	0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL, 0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
	0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL, 0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL
};

ulong rotr64(ulong x, int n) { return (x >> n) | (x << (64 - n)); }

void sha512_block(__private ulong* msg, __private ulong* state)
{
	__private ulong w[80];
	for(int i = 0; i < 16; i++) w[i] = msg[i];
	for(int i = 16; i < 80; i++) {
		ulong s0 = rotr64(w[i-15], 1) ^ rotr64(w[i-15], 8) ^ (w[i-15] >> 7);
		ulong s1 = rotr64(w[i-2], 19) ^ rotr64(w[i-2], 61) ^ (w[i-2] >> 6);
		w[i] = w[i-16] + s0 + w[i-7] + s1;
	}
	ulong a=state[0], b=state[1], c=state[2], d=state[3];
	ulong e=state[4], f=state[5], g=state[6], h=state[7];
	for(int i = 0; i < 80; i++) {
		ulong S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
		ulong ch = (e & f) ^ ((~e) & g);
		ulong t1 = h + S1 + ch + SHA512_K[i] + w[i];
		ulong S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
		ulong maj = (a & b) ^ (a & c) ^ (b & c);
		ulong t2 = S0 + maj;
		h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
	}
	state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
	state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

/*
 * Hash one LR pair.
 * L_meta and R_meta are arrays of 14 uint32s each.
 * msg = L_meta[0..13] || R_meta[0..13] = 28 uint32s = 112 bytes
 * We need to pack them as big-endian ulong words for SHA-512.
 *
 * CPU does: hash_512_t tmp(&msg, sizeof(msg)) where msg is uint32[28].
 * The hash_512_t constructor reads the 112 bytes as big-endian and packs into
 * 14 ulong words (each holding 2 uint32s in big-endian order).
 *
 * SHA-512 of 112 bytes: 1 block (128 bytes with padding).
 */
__kernel void hash_table_entries(
    __global const uint* L_meta_in,   // [num_entries * 14]
    __global const uint* R_meta_in,   // [num_entries * 14]
    __global uint* Y_out,             // [num_entries]
    __global uint* M_out,             // [num_entries * 14]
    const uint kmask,
    const uint num_entries)
{
    const uint gid = get_global_id(0);
    if(gid >= num_entries) return;
    
    // Load L_meta and R_meta (14 uint32s each)
    __private uint L[14], R[14];
    for(int i = 0; i < 14; i++) {
        L[i] = L_meta_in[gid * 14 + i];
        R[i] = R_meta_in[gid * 14 + i];
    }
    
    // Pack msg as big-endian ulong words for SHA-512
    // CPU: memcpy(&msg[i], meta_ptr, 4) — this reads uint32s as little-endian
    //       then hash_512_t treats the byte buffer as big-endian for SHA-512.
    // 
    // Actually, MMX hash_512_t(const void* data, size_t size) reads raw bytes
    // and packs them into ulong words in big-endian order (WriteBE64).
    // The msg is uint32[28] stored in memory as little-endian uint32s.
    // So byte layout is: msg[0] LE bytes, msg[1] LE bytes, ...
    // When read as big-endian ulong: first ulong = bytes[0..7] in BE
    //   = msg[0] low byte, msg[0] 2nd byte, ..., msg[0] high byte, msg[1] low byte, ...
    //   = bytes_to_be64(msg[0] le bytes || msg[1] le bytes)
    //
    // For SHA-512 we need the message as big-endian ulong words.
    // Each ulong holds 8 bytes. Two uint32s (LE) → one ulong (BE):
    //   ulong = bswap64((ulong)L[i] | ((ulong)L[i+1] << 32))
    // where bswap64 reverses byte order to get big-endian.
    
    __private ulong msg[16];  // 16 ulong = 128 bytes (112 data + 16 padding)
    for(int i = 0; i < 16; i++) msg[i] = 0;
    
    // Pack 28 uint32s (14 L + 14 R) into 14 ulong words (big-endian)
    uint vals[28];
    for(int i = 0; i < 14; i++) { vals[i] = L[i]; vals[14+i] = R[i]; }
    
    for(int i = 0; i < 14; i++) {
        // Two LE uint32s → one BE ulong
        ulong lo = (ulong)vals[i*2];
        ulong hi = (ulong)vals[i*2+1];
        ulong le_val = lo | (hi << 32);
        // Byte-swap to big-endian
        msg[i] = ((le_val & 0xFF00000000000000UL) >> 56)
               | ((le_val & 0x00FF000000000000UL) >> 40)
               | ((le_val & 0x0000FF0000000000UL) >> 24)
               | ((le_val & 0x000000FF00000000UL) >> 8)
               | ((le_val & 0x00000000FF000000UL) << 8)
               | ((le_val & 0x0000000000FF0000UL) << 24)
               | ((le_val & 0x000000000000FF00UL) << 40)
               | ((le_val & 0x00000000000000FFUL) << 56);
    }
    
    // SHA-512 of 112 bytes requires 2 blocks (256 bytes total with padding)
    // Block 1: data[0..111] + 0x80 + zeros to 128 bytes
    //   msg[0..13] = data (14 ulong = 112 bytes)
    //   msg[14] = 0x8000000000000000 (byte 112 = 0x80, rest zeros)
    //   msg[15] = 0
    msg[14] = 0x8000000000000000UL;
    msg[15] = 0;
    
    __private ulong state[8];
    for(int i = 0; i < 8; i++) state[i] = SHA512_INIT[i];
    sha512_block(msg, state);
    
    // Block 2: all zeros except last ulong = bit length (big-endian)
    __private ulong block2[16];
    for(int i = 0; i < 16; i++) block2[i] = 0;
    block2[15] = (ulong)112 * 8;  // bit length = 896, already big-endian
    sha512_block(block2, state);
    
    // Extract uint32s from state (same as CPU: memcpy from big-endian bytes)
    // CPU does: memcpy(&hash[i], tmp.data() + i*4, 4) where tmp.data() is BE bytes
    // State is LE, so we need to extract uint32s from BE representation
    // hash[i] = bytes[4*i..4*i+3] as LE uint32
    // BE bytes of state[j] → state[j] bytes reversed = LE bytes
    // So hash[2*j] = low 32 bits of bswap64(state[j])
    //     hash[2*j+1] = high 32 bits of bswap64(state[j])
    // Actually simpler: hash bytes are state serialized as BE.
    // hash[0] = first 4 bytes of state[0] in BE = (state[0] >> 32) read as LE uint32
    //         = bswap32(state[0] >> 32)
    // hash[1] = next 4 bytes = bswap32(state[0] & 0xFFFFFFFF)
    
    uint hash[16];
    for(int i = 0; i < 8; i++) {
        ulong s = state[i];
        // BE bytes of state[i]: high byte first
        // hash[2*i] = bytes[4*i..4*i+3] as LE uint32 = bswap32(high 32 bits)
        // hash[2*i+1] = bytes[4*i+4..4*i+7] as LE uint32 = bswap32(low 32 bits)
        uint hi32 = (uint)(s >> 32);
        uint lo32 = (uint)(s & 0xFFFFFFFF);
        // bswap32
        hash[2*i]   = ((hi32 & 0xFF) << 24) | ((hi32 & 0xFF00) << 8)
                     | ((hi32 >> 8) & 0xFF00) | ((hi32 >> 24) & 0xFF);
        hash[2*i+1] = ((lo32 & 0xFF) << 24) | ((lo32 & 0xFF00) << 8)
                     | ((lo32 >> 8) & 0xFF00) | ((lo32 >> 24) & 0xFF);
    }
    
    // Compute Y = XOR(hash[0..13]) & kmask
    uint Y = 0;
    for(int i = 0; i < 14; i++) Y ^= hash[i];
    Y &= kmask;
    
    // Compute M[i] = hash[i] & kmask
    Y_out[gid] = Y;
    for(int i = 0; i < 14; i++) {
        M_out[gid * 14 + i] = hash[i] & kmask;
    }
}

/*
 * Optimized kernel: hash using M_curr + LR pairs directly.
 * Eliminates CPU meta extraction step.
 * Each work-item loads L_meta = M_curr[P1 * 14 + i], R_meta = M_curr[P2 * 14 + i].
 */
__kernel void hash_table_lr(
    __global const uint* M_curr,    // [num_total_entries * 14] all metadata
    __global const uint* LR_pairs,  // [num_matches * 2] P1, P2 indices
    __global uint* Y_out,           // [num_matches]
    __global uint* M_out,           // [num_matches * 14]
    const uint kmask,
    const uint num_matches,
    const uint num_total_entries)
{
    const uint gid = get_global_id(0);
    if(gid >= num_matches) return;
    
    uint P1 = LR_pairs[gid * 2];
    uint P2 = LR_pairs[gid * 2 + 1];
    
    if(P1 >= num_total_entries || P2 >= num_total_entries) {
        Y_out[gid] = 0xFFFFFFFF;
        for(int i = 0; i < 14; i++) M_out[gid * 14 + i] = 0;
        return;
    }
    
    // Load L_meta and R_meta directly from M_curr
    __private uint L[14], R[14];
    for(int i = 0; i < 14; i++) {
        L[i] = M_curr[P1 * 14 + i];
        R[i] = M_curr[P2 * 14 + i];
    }
    
    // Pack msg as big-endian ulong words for SHA-512
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
    for(int i = 0; i < 14; i++) {
        M_out[gid * 14 + i] = hash[i] & kmask;
    }
}
