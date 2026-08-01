/*
 * table_hash.cl — OpenCL kernel for MMX table hashing (F2-F9)
 *
 * Optimized to match CUDA reference (rocm_sha512):
 * - Optimized Ch/Maj functions (fewer ops per round)
 * - bswap64 as separate function (compiler optimizes to native instruction)
 * - Message stored as LE uint32 pairs, bswap inside chunk
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

/* Optimized bitwise functions (from CUDA reference — fewer ops than naive) */
inline ulong sha512_Ch(ulong x, ulong y, ulong z) { return z ^ (x & (y ^ z)); }
inline ulong sha512_Maj(ulong x, ulong y, ulong z) { return (x & y) | (z & (x | y)); }
inline ulong sha512_S0(ulong x) { return rotr64(x, 28) ^ rotr64(x, 34) ^ rotr64(x, 39); }
inline ulong sha512_S1(ulong x) { return rotr64(x, 14) ^ rotr64(x, 18) ^ rotr64(x, 41); }
inline ulong sha512_s0(ulong x) { return rotr64(x, 1) ^ rotr64(x, 8) ^ (x >> 7); }
inline ulong sha512_s1(ulong x) { return rotr64(x, 19) ^ rotr64(x, 61) ^ (x >> 6); }

/* Byte-swap 64-bit. Compiler should optimize to native instruction on AMD/NVIDIA. */
inline ulong bswap64(ulong x) {
	return ((x & 0x00000000000000FFUL) << 56)
	     | ((x & 0x000000000000FF00UL) << 40)
	     | ((x & 0x0000000000FF0000UL) << 24)
	     | ((x & 0x00000000FF000000UL) << 8)
	     | ((x & 0x000000FF00000000UL) >> 8)
	     | ((x & 0x0000FF0000000000UL) >> 24)
	     | ((x & 0x00FF000000000000UL) >> 40)
	     | ((x & 0xFF00000000000000UL) >> 56);
}

/* SHA-512 single block. msg = 16 ulong words in LE (bswap to BE on load).
 * Matches CUDA: w[i] = bswap64(msg[i]) */
void sha512_block(__private const ulong* msg, __private ulong* state)
{
	__private ulong w[80];
	for(int i = 0; i < 16; i++) w[i] = bswap64(msg[i]);
	for(int i = 16; i < 80; i++) {
		w[i] = w[i-16] + sha512_s0(w[i-15]) + w[i-7] + sha512_s1(w[i-2]);
	}
	ulong a=state[0], b=state[1], c=state[2], d=state[3];
	ulong e=state[4], f=state[5], g=state[6], h=state[7];
	for(int i = 0; i < 80; i++) {
		ulong t1 = h + sha512_S1(e) + sha512_Ch(e, f, g) + SHA512_K[i] + w[i];
		ulong t2 = sha512_S0(a) + sha512_Maj(a, b, c);
		h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
	}
	state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
	state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

/*
 * Hash one LR pair.
 * L_meta and R_meta are arrays of 14 uint32s each (LE, native layout).
 * msg = L_meta[0..13] || R_meta[0..13] = 28 uint32s = 112 bytes
 * SHA-512 of 112 bytes needs 2 blocks (256 bytes with padding).
 *
 * Packing: 28 uint32s → 14 ulong words (LE, just pair adjacent uint32s).
 * Block 1: msg[0..13] = data, msg[14] = 0x80 padding (LE), msg[15] = 0
 * Block 2: all zeros except last ulong = bit_length (896) in LE
 * bswap64 inside sha512_block converts LE→BE.
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
    
    /* Pack 28 uint32s into 14 ulong words (LE — just pair adjacent uint32s, no bswap) */
    __private ulong msg[16];
    for(int i = 0; i < 16; i++) msg[i] = 0;
    
    /* msg[0..6] = L_meta (7 ulong = 14 uint32 = 56 bytes) */
    for(int i = 0; i < 7; i++) {
        msg[i] = (ulong)L_meta_in[gid * 14 + i*2] | ((ulong)L_meta_in[gid * 14 + i*2+1] << 32);
    }
    /* msg[7..13] = R_meta (7 ulong = 14 uint32 = 56 bytes) */
    for(int i = 0; i < 7; i++) {
        msg[7 + i] = (ulong)R_meta_in[gid * 14 + i*2] | ((ulong)R_meta_in[gid * 14 + i*2+1] << 32);
    }
    
    /* Block 1 padding: 0x80 at byte 112 = LSB of msg[14] (LE) */
    /* After bswap64: 0x80 → MSB = 0x8000000000000000 (correct BE padding) */
    msg[14] = 0x80;
    msg[15] = 0;
    
    __private ulong state[8];
    for(int i = 0; i < 8; i++) state[i] = SHA512_INIT[i];
    sha512_block(msg, state);
    
    /* Block 2: all zeros except bit length at end.
     * Bit length = 112 * 8 = 896. Store as LE (bswap64 inside block converts to BE). */
    __private ulong block2[16];
    for(int i = 0; i < 16; i++) block2[i] = 0;
    block2[15] = 896;  /* LE: 0x0000000000000380 → bswap64 → 0x8003000000000000?
                          NO! bswap64(896) = bswap64(0x380) should give BE of 896.
                          Actually: 896 = 0x380. LE bytes: 80 03 00 00 00 00 00 00
                          bswap64 → 00 00 00 00 00 00 03 80 = 0x0000000000000380
                          Wait, that IS 896! Because bswap(bswap(x)) = x.
                          We need w[15] = 0x380 (BE of 896).
                          w[15] = bswap64(block2[15]) = bswap64(896) = bswap64(0x380)
                          bswap64(0x0000000000000380) = 0x8003000000000000
                          That's NOT 0x380! So we need block2[15] = bswap64(896). */
    /* Actually: we need w[15] = 896 in BE = 0x0000000000000380.
     * w[15] = bswap64(block2[15]). So block2[15] = bswap64(0x380) = 0x8003000000000000.
     * But 0x8003000000000000 bswapped back = 0x380. Yes.
     * So: block2[15] = bswap64(896) */
    block2[15] = bswap64(896);
    sha512_block(block2, state);
    
    /* Extract uint32s: state is LE, SHA-512 output bytes are BE.
     * hash bytes = bswap64(state[i]) as bytes → extract uint32s */
    uint hash[16];
    for(int i = 0; i < 8; i++) {
        ulong be = bswap64(state[i]);  /* LE state → BE bytes */
        hash[2*i]   = (uint)be;         /* first 4 BE bytes as LE uint32 */
        hash[2*i+1] = (uint)(be >> 32); /* next 4 BE bytes as LE uint32 */
    }
    
    /* Y = XOR(hash[0..13]) & kmask */
    uint Y = 0;
    for(int i = 0; i < 14; i++) Y ^= hash[i];
    Y &= kmask;
    
    Y_out[gid] = Y;
    for(int i = 0; i < 14; i++) {
        M_out[gid * 14 + i] = hash[i] & kmask;
    }
}

/*
 * Optimized kernel: hash using M_curr + LR pairs directly.
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
    
    __private ulong msg[16];
    for(int i = 0; i < 16; i++) msg[i] = 0;
    
    for(int i = 0; i < 7; i++) {
        msg[i] = (ulong)M_curr[P1 * 14 + i*2] | ((ulong)M_curr[P1 * 14 + i*2+1] << 32);
        msg[7 + i] = (ulong)M_curr[P2 * 14 + i*2] | ((ulong)M_curr[P2 * 14 + i*2+1] << 32);
    }
    
    msg[14] = 0x80;
    msg[15] = 0;
    
    __private ulong state[8];
    for(int i = 0; i < 8; i++) state[i] = SHA512_INIT[i];
    sha512_block(msg, state);
    
    __private ulong block2[16];
    for(int i = 0; i < 16; i++) block2[i] = 0;
    block2[15] = bswap64(896);
    sha512_block(block2, state);
    
    uint hash[16];
    for(int i = 0; i < 8; i++) {
        ulong be = bswap64(state[i]);
        hash[2*i]   = (uint)be;
        hash[2*i+1] = (uint)(be >> 32);
    }
    
    uint Y = 0;
    for(int i = 0; i < 14; i++) Y ^= hash[i];
    Y &= kmask;
    
    Y_out[gid] = Y;
    for(int i = 0; i < 14; i++) {
        M_out[gid * 14 + i] = hash[i] & kmask;
    }
}
