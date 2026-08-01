/*
 * f1_warp.cl — Warp-parallel F1 computation for MMX plotting
 *
 * 3-kernel pipeline matching CUDA reference:
 * 1. gen_mem_array: 1 work-item/entry → key + mem[1024] to global memory
 * 2. calc_mem_hash: 32 work-items/entry (subgroup) → hash[32] using shared memory
 * 3. scatter_f1: 1 work-item/entry → final SHA-512(key||hash) → Y, M
 *
 * This eliminates the register spill from mem[1024] in private memory.
 */

/* === SHA-512 (shared with table_hash.cl) === */

__constant ulong SHA512_K_F1[80] = {
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

__constant ulong SHA512_INIT_F1[8] = {
	0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL, 0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
	0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL, 0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL
};

ulong rotr64_f1(ulong x, int n) { return (x >> n) | (x << (64 - n)); }

inline ulong bswap64_f1(ulong x) {
	return ((x & 0x00000000000000FFUL) << 56)
	     | ((x & 0x000000000000FF00UL) << 40)
	     | ((x & 0x0000000000FF0000UL) << 24)
	     | ((x & 0x00000000FF000000UL) << 8)
	     | ((x & 0x000000FF00000000UL) >> 8)
	     | ((x & 0x0000FF0000000000UL) >> 24)
	     | ((x & 0x00FF000000000000UL) >> 40)
	     | ((x & 0xFF00000000000000UL) >> 56);
}

uint bswap32_f1(uint x) {
	return (x >> 24) | ((x >> 8) & 0xFF00) | ((x << 8) & 0xFF0000) | (x << 24);
}

uint rotl32_f1(uint v, int bits) {
	if(bits == 0) return v;
	return (v << bits) | (v >> (32 - bits));
}

void sha512_block_f1(__private const ulong* msg, __private ulong* state)
{
	__private ulong w[80];
	for(int i = 0; i < 16; i++) w[i] = bswap64_f1(msg[i]);
	for(int i = 16; i < 80; i++) {
		ulong s0 = rotr64_f1(w[i-15], 1) ^ rotr64_f1(w[i-15], 8) ^ (w[i-15] >> 7);
		ulong s1 = rotr64_f1(w[i-2], 19) ^ rotr64_f1(w[i-2], 61) ^ (w[i-2] >> 6);
		w[i] = w[i-16] + s0 + w[i-7] + s1;
	}
	ulong a=state[0], b=state[1], c=state[2], d=state[3];
	ulong e=state[4], f=state[5], g=state[6], h=state[7];
	for(int i = 0; i < 80; i++) {
		ulong S1 = rotr64_f1(e, 14) ^ rotr64_f1(e, 18) ^ rotr64_f1(e, 41);
		ulong ch = g ^ (e & (f ^ g));  /* optimized Ch */
		ulong t1 = h + S1 + ch + SHA512_K_F1[i] + w[i];
		ulong S0 = rotr64_f1(a, 28) ^ rotr64_f1(a, 34) ^ rotr64_f1(a, 39);
		ulong maj = (a & b) | (c & (a | b));  /* optimized Maj */
		ulong t2 = S0 + maj;
		h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
	}
	state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
	state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

/*
 * SHA-512 hash of a message stored as LE ulong array.
 * data must have room for padding (17 extra bytes, multiple of 16 ulong).
 * Output state is LE (caller extracts with bswap).
 */
void sha512_hash_f1(__private ulong* data, const uint length, __private ulong* out)
{
	const ulong num_bits = (ulong)length * 8;
	const uint total_bytes = length + 17;
	const uint num_chunks = (total_bytes + 127) / 128;

	/* Padding: set 0x80 bit after last data byte */
	uint byte_idx = length / 8;
	uint bit_shift = (7 - (length % 8)) * 8;
	data[byte_idx] |= ((ulong)0x80 << bit_shift);

	/* Bit length in last ulong of last chunk (LE, bswap inside block) */
	data[num_chunks * 16 - 1] = bswap64_f1(num_bits);

	for(int i = 0; i < 8; i++) out[i] = SHA512_INIT_F1[i];
	for(uint i = 0; i < num_chunks; i++) {
		sha512_block_f1(data + i * 16, out);
	}
}

void extract_uint32_f1(__private const ulong* sha_out, __private uint* state32)
{
	for(int i = 0; i < 8; i++) {
		ulong be = bswap64_f1(sha_out[i]);
		state32[2*i]   = (uint)be;
		state32[2*i+1] = (uint)(be >> 32);
	}
}

void pack_uint32_to_be_ulong_f1(__private const uint* msg32, int num32, __private ulong* msg64, int num64)
{
	for(int i = 0; i < num64; i++) msg64[i] = 0;
	for(int i = 0; i < num32 / 2; i++) {
		msg64[i] = ((ulong)bswap32_f1(msg32[2*i]) << 32) | bswap32_f1(msg32[2*i+1]);
	}
	if(num32 % 2) {
		msg64[num32/2] = (ulong)bswap32_f1(msg32[num32-1]) << 32;
	}
}

/* === MMXPOS hash round === */
#define MMXPOS_HASHROUND_F1(a, b, c, d) \
	a = a + b;                       \
	d = rotl32_f1(d ^ a, 16);        \
	c = c + d;                       \
	b = rotl32_f1(b ^ c, 12);        \
	a = a + b;                       \
	d = rotl32_f1(d ^ a, 8);         \
	c = c + d;                       \
	b = rotl32_f1(b ^ c, 7);

__constant uint MEM_HASH_INIT_F1[16] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174
};

#define MEM_HASH_ITER_F1 256
#define N_META_F1 14

/* === Kernel 1: gen_mem_array === */
/* 1 work-item per entry. Outputs key[16] and mem[1024] to global memory. */
/* mem_buf layout: mem[(row * batch_size + entry) * 32 + elem]  (matching CUDA) */
/* key_buf layout: key[entry * 16 + i] */
__kernel void gen_mem_array_kernel(
    __global uint* mem_out,     // [batch_size * 1024] device memory
    __global uint* key_out,     // [batch_size * 16] device memory
    __global const uint* id_in, // [8] plot_id as uint32 array
    const uint batch_size,      // number of entries in this batch
    const uint x_base)          // base X value (global offset)
{
    const uint entry = get_global_id(0);
    if(entry >= batch_size) return;
    
    const uint X_i = x_base + entry;
    
    /* Step 1: key = SHA-512(X_i || plot_id) */
    uint msg32_key[18];
    for(int i = 0; i < 18; i++) msg32_key[i] = 0;
    msg32_key[0] = X_i;
    for(int i = 0; i < 8; i++) msg32_key[1 + i] = id_in[i];
    
    ulong msg64_key[16];
    pack_uint32_to_be_ulong_f1(msg32_key, 9, msg64_key, 16);
    
    ulong key_state[8];
    sha512_hash_f1(msg64_key, 36, key_state);
    
    uint key32[16];
    extract_uint32_f1(key_state, key32);
    
    /* Write key to global memory */
    for(int i = 0; i < 16; i++) {
        key_out[entry * 16 + i] = key32[i];
    }
    
    /* Step 2: gen_mem_array */
    uint state[32];
    for(int i = 0; i < 16; i++) state[i] = key32[i];
    for(int i = 0; i < 16; i++) state[16 + i] = MEM_HASH_INIT_F1[i];
    
    uint b = 0, c = 0;
    
    /* mem_size = 1024 = 32 * 32, so 32 iterations of 32 elements */
    for(uint i = 0; i < 32; i++) {
        for(int j = 0; j < 4; j++) {
            for(int k = 0; k < 16; k++) {
                MMXPOS_HASHROUND_F1(state[k], b, c, state[16 + k]);
            }
        }
        /* Write row i to global memory: mem[(i * batch_size + entry) * 32 + elem] */
        for(int k = 0; k < 32; k++) {
            mem_out[(i * batch_size + entry) * 32 + k] = state[k];
        }
    }
}

/* === Kernel 2: calc_mem_hash (warp-parallel) === */
/* 32 work-items per entry (one subgroup). 4 subgroups per work-group (128 work-items). */
/* Uses shared memory: lmem[subgroup][32 * 32] = 4KB per subgroup, 16KB per work-group. */
#pragma OPENCL_EXTENSION cl_khr_subgroups : enable

__kernel void calc_mem_hash_kernel(
    __global const uint* mem_in,   // [batch_size * 1024] device memory
    __global uint* hash_out,       // [batch_size * 32] device memory
    const uint batch_size,
    const uint num_iter)
{
    const uint sgid = get_sub_group_local_id();  /* 0..31, element index within entry */
    const uint entry = get_global_id(0) / 32;    /* which entry this subgroup handles */
    if(entry >= batch_size) return;
    
    const uint N = 32;
    
    /* Shared memory: 4 subgroups × 32 rows × 32 elements = 4096 uint32 = 16KB */
    __local uint lmem[4][32 * 32];
    
    /* Determine which subgroup slot we're in (0..3 within the work-group) */
    const uint wg_local = get_local_id(0);  /* 0..127 */
    const uint sg_slot = wg_local / 32;     /* 0..3 */
    
    /* Load mem for this entry into shared memory.
     * Each thread loads 32 values (one per row).
     * lmem[sg_slot][row * 32 + sgid] = mem_in[(row * batch_size + entry) * 32 + sgid] */
    for(int row = 0; row < 32; row++) {
        lmem[sg_slot][row * 32 + sgid] = mem_in[(row * batch_size + entry) * 32 + sgid];
    }
    
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    
    /* Initial state = last row of lmem */
    uint state = lmem[sg_slot][(N - 1) * 32 + sgid];
    
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int iter = 0; iter < num_iter; iter++)
    {
        /* Each thread rotates its state element by its index */
        uint sum = rotl32_f1(state, sgid % 32);
        
        /* Warp-level reduce_add using sub_group_reduce_add */
        sum = sub_group_reduce_add(sum);
        
        /* Thread 0 computes dir, broadcast to all */
        uint dir = 0;
        if(sgid == 0) {
            dir = sum + (sum << 11) + (sum << 22);
        }
        dir = sub_group_broadcast(dir, 0);
        
        const uint bits = (dir >> 22) % 32u;
        const uint offset = (dir >> 27) % 32u;
        
        /* Update state */
        state += rotl32_f1(lmem[sg_slot][offset * 32 + (iter + sgid) % N], bits) ^ sum;
        
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        
        /* Atomic XOR state back into lmem */
        atomic_xor(&lmem[sg_slot][offset * 32 + sgid], state);
        
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    /* Write hash output */
    hash_out[entry * 32 + sgid] = state;
}

/* === Kernel 3: scatter_f1 === */
/* 1 work-item per entry. Final SHA-512(key || hash) → Y, M */
__kernel void scatter_f1_kernel(
    __global const uint* key_in,     // [batch_size * 16]
    __global const uint* hash_in,    // [batch_size * 32]
    __global uint* Y_out,            // [total_entries]
    __global uint* M_out,            // [total_entries * N_META_F1]
    const uint kmask,
    const uint batch_size,
    const uint x_base,
    const uint total_entries)
{
    const uint entry = get_global_id(0);
    if(entry >= batch_size) return;
    
    const uint global_entry = x_base + entry;
    
    /* Load key[16] and hash[32] */
    uint msg32_final[48];  /* 16 + 32 = 48 uint32 = 192 bytes */
    for(int i = 0; i < 16; i++) msg32_final[i] = key_in[entry * 16 + i];
    for(int i = 0; i < 32; i++) msg32_final[16 + i] = hash_in[entry * 32 + i];
    
    /* SHA-512(msg, 192 bytes) → 2 chunks */
    ulong msg64_final[32];
    pack_uint32_to_be_ulong_f1(msg32_final, 48, msg64_final, 32);
    
    ulong final_state[8];
    sha512_hash_f1(msg64_final, 192, final_state);
    
    uint hash16[16];
    extract_uint32_f1(final_state, hash16);
    
    /* Y = XOR(hash[0..13]) & kmask */
    uint Y = 0;
    for(int i = 0; i < N_META_F1; i++) Y ^= hash16[i];
    Y &= kmask;
    
    Y_out[global_entry] = Y;
    for(int i = 0; i < N_META_F1; i++) {
        M_out[global_entry * N_META_F1 + i] = hash16[i] & kmask;
    }
}
