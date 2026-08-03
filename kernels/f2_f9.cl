/*
 * f2_f9.cl — OpenCL kernels for MMX PoSpace tables 2-9
 * Ported from CUDA/HIP plotter (Node_phase1.hip) to OpenCL.
 * Works on AMD gfx1100 where HIP/CUDA produce broken results.
 */

// Build-time defines (passed via -D):
// KSIZE, LOGBUCKETS, LOGBUCKETS2, N_META, N_META_OUT, N_TABLE
// DSIZE_, PSIZE_, PDSIZE, X2SIZE, XBITS
// HYBRID_SORT_LOG_THREADS, NUM_THREADS
// KMASK, DMASK

// SHA-512 constants
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

// Big-endian byte swap
ulong bswap64(ulong x) {
    return ((ulong)(uint)(x >> 56)) |
           ((ulong)((uint)(x >> 40) & 0xFF00)) |
           ((ulong)((uint)(x >> 24) & 0xFF0000)) |
           ((ulong)((uint)(x >> 8) & 0xFF000000)) |
           ((ulong)((uint)(x & 0xFF000000)) << 8) |
           ((ulong)((uint)(x & 0xFF0000)) << 24) |
           ((ulong)((uint)(x & 0xFF00)) << 40) |
           ((ulong)((uint)(x & 0xFF)) << 56);
}

ulong rotr64(ulong x, uint c) {
    return (x >> c) | (x << (64 - c));
}

ulong sha512_Ch(ulong x, ulong y, ulong z) { return z ^ (x & (y ^ z)); }
ulong sha512_Maj(ulong x, ulong y, ulong z) { return (x & y) | (z & (x | y)); }
ulong sha512_S0(ulong x) { return rotr64(x, 28) ^ rotr64(x, 34) ^ rotr64(x, 39); }
ulong sha512_S1(ulong x) { return rotr64(x, 14) ^ rotr64(x, 18) ^ rotr64(x, 41); }
ulong sha512_s0(ulong x) { return rotr64(x, 1) ^ rotr64(x, 8) ^ (x >> 7); }
ulong sha512_s1(ulong x) { return rotr64(x, 19) ^ rotr64(x, 61) ^ (x >> 6); }

void sha512_chunk(const ulong* msg, ulong* state)
{
    ulong w[80];
    for (int i = 0; i < 16; i++) {
        w[i] = bswap64(msg[i]);
    }

    ulong a = state[0], b = state[1], c = state[2], d = state[3];
    ulong e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 80; i++) {
        if (i >= 16) {
            w[i] = w[i-16] + sha512_s0(w[i-15]) + w[i-7] + sha512_s1(w[i-2]);
        }
        ulong temp1 = h + sha512_S1(e) + sha512_Ch(e, f, g) + SHA512_K[i] + w[i];
        ulong temp2 = sha512_S0(a) + sha512_Maj(a, b, c);
        h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

/*
 * SHA-512 hash of a message.
 * msg: ulong buffer, must be zero-initialized, with at least 17 extra bytes.
 *      Contains `length` bytes of input data (in big-endian ulong words).
 * length: number of BYTES of input.
 * hash: output 8x ulong (64 bytes), will be in big-endian.
 */
void sha512(ulong* msg, const uint length, ulong* hash)
{
    const ulong num_bits = (ulong)length * 8;
    const uint total_bytes = length + 17;
    const uint num_chunks = (total_bytes + 127) / 128;

    // Padding: set the 0x80 byte after the message
    msg[length / 8] |= ((ulong)0x80 << ((length % 8) * 8));

    // Length field at end of last chunk (big-endian)
    msg[num_chunks * 16 - 1] = bswap64(num_bits);

    for (int i = 0; i < 8; i++) {
        hash[i] = SHA512_INIT[i];
    }
    for (uint i = 0; i < num_chunks; i++) {
        sha512_chunk(msg + i * 16, hash);
    }
    for (int i = 0; i < 8; i++) {
        hash[i] = bswap64(hash[i]);
    }
}

// ============================================================================
// Bit operations (ported from rocm_encoding.h)
// ============================================================================

ulong read_bits(__global const ulong* src, const ulong bit_offset, const uint num_bits)
{
    uint count = 0;
    ulong offset = bit_offset;
    ulong result = 0;
    while (count < num_bits) {
        const uint shift = offset % 64;
        const uint bits = min(num_bits - count, 64 - shift);
        const ulong value = src[offset / 64] >> shift;
        result |= value << count;
        count += bits;
        offset += bits;
    }
    if (num_bits < 64) {
        result &= ((ulong)1 << num_bits) - 1;
    }
    return result;
}

void atomic_write_bits(__global ulong* dst, ulong value, const ulong bit_offset, const uint num_bits)
{
    if (num_bits < 64) {
        value &= ((ulong)1 << num_bits) - 1;
    }
    const uint shift = bit_offset % 64;
    const uint free_bits = 64 - shift;

    // Split 64-bit atomic_or into two 32-bit operations (OpenCL 1.2 has no 64-bit atomics)
    __global volatile uint* dst32 = (__global volatile uint*)(dst + bit_offset / 64);
    ulong shifted = value << shift;
    atomic_or(dst32, (uint)shifted);
    atomic_or(dst32 + 1, (uint)(shifted >> 32));

    if (free_bits < num_bits) {
        __global volatile uint* dst32b = (__global volatile uint*)(dst + bit_offset / 64 + 1);
        ulong shifted2 = value >> free_bits;
        atomic_or(dst32b, (uint)shifted2);
        atomic_or(dst32b + 1, (uint)(shifted2 >> 32));
    }
}

// encode_symbol: returns (value, num_bits) packed as uint2
uint2 encode_symbol(const uint sym)
{
    switch (sym) {
        case 0: return (uint2)(0, 2);
        case 1: return (uint2)(1, 2);
        case 2: return (uint2)(2, 2);
        case 3: return (uint2)(0b11 | (0 << 2), 4);
        case 4: return (uint2)(0b11 | (1 << 2), 4);
        case 5: return (uint2)(0b11 | (2 << 2), 4);
        case 6: return (uint2)(0b1111 | (0 << 4), 6);
        case 7: return (uint2)(0b1111 | (1 << 4), 6);
        case 8: return (uint2)(0b1111 | (2 << 4), 6);
    }
    const uint index = sym / 3;
    const uint mod = sym % 3;
    uint out = (uint)(-1) >> (32 - 2 * index);
    out |= mod << (2 * index);
    return (uint2)(out, 2 * index + 2);
}

// ============================================================================
// Kernel: scatter_2 — bucket entries by sub-bucket index
// ============================================================================
// Input: Y_in (optional) or C_in (metadata), bucket_size_in
// Output: PY_out (Y << (64-KSIZE)) | local_pos, bucket_size_out
// Each work-item processes one entry.

__kernel void scatter_2(
    __global ulong* PY_out,
    __global uint* bucket_size_out,
    __global const uint* Y_in,         // may be NULL (pass 0)
    __global const uint* C_in,
    const uint bucket_size_in,
    const uint max_bucket_size_2,
    const uint bucket_offset)          // global index offset for this bucket
{
    const uint x = get_global_id(0);
    if (x >= bucket_size_in) return;

    uint Y_i = 0;
    if (Y_in) {
        Y_i = Y_in[x];
    } else {
        for (int i = 0; i < N_META; i++) {
            Y_i ^= C_in[x * N_META + i];
        }
        Y_i &= KMASK;
    }

    // For chunked processing: use masked LOGBUCKETS2-bit sub-bucket index
    const uint index = (Y_i >> (KSIZE - LOGBUCKETS - LOGBUCKETS2)) & ((1u << LOGBUCKETS2) - 1);
    const uint pos = atomic_add((__global volatile uint*)(bucket_size_out + index), 1);
    if (pos < max_bucket_size_2) {
        const uint j = index * max_bucket_size_2 + pos;
        PY_out[j] = ((ulong)Y_i << (64 - KSIZE)) | (ulong)(x + bucket_offset);
    }
}

// ============================================================================
// Kernel: calc_offset_sum — prefix sum of bucket counts
// ============================================================================
// Simple prefix sum over num_buckets_2 entries. For small num_buckets_2 (e.g. 2^15=32K),
// a single work-group can handle it.

__kernel void calc_offset_sum(
    __global uint* offset_out,
    __global const uint* count_in,
    const uint width,
    const uint with_total)
{
    const uint k = get_local_id(0);
    const uint num_items = get_local_size(0);

    __local uint local_count[1024];  // max LOGBUCKETS2
    __local uint prefix[1024];

    // Load counts into local memory
    for (uint i = k; i < width; i += num_items) {
        local_count[i] = count_in[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Hillis-Steele scan
    prefix[k] = (k < width) ? local_count[k] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = 1; offset < width; offset *= 2) {
        uint temp = prefix[k];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (k >= offset && k < width) {
            prefix[k] += prefix[k - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // offset_out[0] = 0, offset_out[i] = sum of count_in[0..i-1]
    if (k == 0) {
        offset_out[0] = 0;
    }
    if (k < width) {
        offset_out[k + 1] = prefix[k];
    }
    if (with_total && k == 0) {
        offset_out[width] = prefix[width - 1] + local_count[width - 1];
    }
}

// ============================================================================
// Kernel: hybrid_sort_y — sort entries within sub-buckets
// ============================================================================
// Uses shared memory insertion sort. Each work-group processes one sub-bucket.

#ifndef NUM_THREADS
#define NUM_THREADS (1 << HYBRID_SORT_LOG_THREADS)
#endif

#ifndef MAX_LOCAL_SIZE
#define MAX_LOCAL_SIZE ((2 * (1 << (KSIZE - LOGBUCKETS - LOGBUCKETS2))) / NUM_THREADS + 24)
#endif

__kernel void hybrid_sort_y(
    __global ulong* data,
    __global const uint* bucket_size,
    const uint max_bucket_size,
    const uint num_sub_buckets)
{
    const uint x = get_local_id(0);
    const uint y = get_group_id(1);  // sub-bucket index

    if (y >= num_sub_buckets) return;

    __local ulong buffer[MAX_LOCAL_SIZE * (NUM_THREADS + 1)];
    __local uint count[NUM_THREADS];
    // Buffer is laid out as buffer[j * NUM_THREADS + index] to match CUDA
    // We access it as buffer[j * (NUM_THREADS+1) + index] for alignment

    count[x] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint size = min(bucket_size[y], max_bucket_size);

    // Scatter entries into thread-local bins
    for (uint i = x; i < size; i += NUM_THREADS) {
        const ulong PY = data[y * max_bucket_size + i];
        const uint index = (PY >> (64 - LOGBUCKETS - LOGBUCKETS2 - HYBRID_SORT_LOG_THREADS)) & (NUM_THREADS - 1);
        const uint j = atomic_add((__local volatile uint*)(count + index), 1);
        if (j < MAX_LOCAL_SIZE) {
            buffer[j * (NUM_THREADS + 1) + index] = PY;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Insertion sort within each thread's bin
    // buffer[x * (NUM_THREADS+1) + i] is the i-th element for thread x
    {
        const uint n = min(count[x], (uint)MAX_LOCAL_SIZE);
        for (int i = 1; i < n; i++) {
            const ulong key = buffer[i * (NUM_THREADS + 1) + x];
            int j;
            for (j = i - 1; j >= 0 && buffer[j * (NUM_THREADS + 1) + x] > key; j--) {
                buffer[(j + 1) * (NUM_THREADS + 1) + x] = buffer[j * (NUM_THREADS + 1) + x];
            }
            buffer[(j + 1) * (NUM_THREADS + 1) + x] = key;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Write back sorted data
    uint off = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        const uint len = min(count[i], (uint)MAX_LOCAL_SIZE);
        if (x < len) {
            data[y * max_bucket_size + off + x] = buffer[x * (NUM_THREADS + 1) + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // ensure all reads from buffer before next off
        off += len;
    }
}

// ============================================================================
// Kernel: match_p1 — find Y,Y+1 pairs within sorted sub-buckets
// ============================================================================
// Each work-group processes one sub-bucket. Work-items scan for Y,Y+1 matches.
// Also checks the first entry of the NEXT sub-bucket for cross-boundary matches.

__kernel void match_p1(
    __global uint2* LR_out,
    __global uint* PD_out,
    __global uint* num_matches,
    __global const ulong* PY_in,
    __global const uint* bucket_size,
    __global const uint* bucket_offset,
    const uint num_sub_buckets,
    const uint max_bucket_size,
    const uint max_total_matches,
    const uint write_pd
)
{
    const uint k = get_local_id(0);
    const uint x = get_global_id(0);   // entry index within sub-bucket
    const uint y = get_global_id(1);   // sub-bucket index

    __local uint2 LR_tmp[512];
    __local uint PD_tmp[512];
    __local uint count;
    __local uint global_offset;

    const uint size = min(bucket_size[y], max_bucket_size);

    if (k == 0) {
        count = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Early exit if all threads are out of bounds
    if (x >= size) {
        // Still need to participate in barriers
    }

    const uint P_x = bucket_offset[y] + x;
    const uint next_size = (y + 1 < num_sub_buckets) ? bucket_size[y + 1] : 0;

    if (x < size) {
        const ulong PY_L = PY_in[y * max_bucket_size + x];
        const uint YL = PY_L >> (64 - KSIZE);
        const uint PL = (uint)PY_L;

        for (uint i = x + 1; ; i++) {
            const bool is_next = (i >= size);
            const uint j = is_next ? (i - size) : i;

            if (is_next && j >= next_size) break;

            const ulong PY_R = PY_in[(y + (is_next ? 1 : 0)) * max_bucket_size + j];
            const uint YR = PY_R >> (64 - KSIZE);
            const uint PR = (uint)PY_R;

            if (YR == YL + 1) {
                const uint pos = atomic_add((__local volatile uint*)&count, 1);
                if (pos < 512) {
                    LR_tmp[pos] = (uint2)(PL, PR);
                    if (write_pd) {
                        PD_tmp[pos] = (P_x << DSIZE_) | (i - x - 1);
                    }
                }
            } else if (YR > YL) {
                break;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (k == 0) {
        count = min(count, (uint)512);
        global_offset = atomic_add((__global volatile uint*)num_matches, count);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint offset = global_offset;
    for (uint i = k; i < count; i += get_local_size(0)) {
        const uint pos = offset + i;
        if (pos < max_total_matches) {
            LR_out[pos] = LR_tmp[i];
            if (write_pd) {
                PD_out[pos] = PD_tmp[i];
            }
        }
    }
}

// ============================================================================
// Kernel: eval_p1_tx — hash pairs (SHA-512), scatter to new buckets, write PD
// ============================================================================
// Each work-item processes one match (LR pair).
// Hashes the metadata of both entries, computes new Y, scatters to new bucket.

__kernel void eval_p1_tx(
    __global uint* Y_out,              // may be NULL (pass 0)
    __global uint* C_out,              // metadata output
    __global ulong* PD_out,    // PD/X2 bit-packed output
    __global uint* bucket_size, // new bucket sizes
    __global const uint* C_in,         // input metadata
    __global const uint* PD_in,        // input PD (from match_p1)
    __global const uint* X_in,         // input X (for table 2, pass 0 otherwise)
    __global const uint2* LR_in,       // match pairs
    __global const uint* num_found,    // number of matches
    const ulong PD_0,                  // base PD offset for this bucket
    const uint max_bucket_size,
    const uint x2size_arg,
    const uint xbits_arg,
    const uint table,                  // current table number (2..9)
    const uint write_y,                // 1 if Y_out should be written
    const uint write_c,                // 1 if C_out should be written
    const uint has_pd_in,              // 1 if PD_in is valid (t >= 3)
    const uint has_x_in                // 1 if X_in is valid (t == 2)
)
{
    const uint x = get_global_id(0);
    if (x >= num_found[0]) return;

    const uint2 LR_i = LR_in[x];
    const uint P_1 = LR_i.x;
    const uint P_2 = LR_i.y;

    // Hash C_in[P_1] || C_in[P_2] using SHA-512
    // Pack pairs of uint32 metadata into ulong64s (little-endian, matching CUDA reinterpret cast)
    // Total message: 2 * N_META * 4 = 2 * N_META * 4 bytes
    ulong msg64[64] = {};
    for (int i = 0; i < N_META; i++) {
        msg64[i / 2] |= (ulong)C_in[P_1 * N_META + i] << ((i % 2) * 32);
        msg64[(N_META + i) / 2] |= (ulong)C_in[P_2 * N_META + i] << (((N_META + i) % 2) * 32);
    }
    ulong hash[8] = {};
    sha512(msg64, 2 * N_META * 4, hash);

    // Extract new Y and metadata
    // SHA-512 produces 8 uint64s = 16 uint32s. CUDA stores as uint32_t[16].
    // We extract uint32s: even index = lower 32 of hash[i/2], odd = upper 32.
    uint Y_new = 0;
    for (int i = 0; i < N_META; i++) {
        uint h32 = (i % 2 == 0) ? (uint)hash[i/2] : (uint)(hash[i/2] >> 32);
        Y_new ^= h32;
    }
    Y_new &= KMASK;

    const uint index = Y_new >> (KSIZE - LOGBUCKETS);
    if ((index >> LOGBUCKETS) == 0) {
        const uint pos = atomic_add((__global volatile uint*)(bucket_size + index), 1);
        if (pos < max_bucket_size) {
            const ulong j = (ulong)index * max_bucket_size + pos;

            if (write_y && Y_out) {
                Y_out[j] = Y_new;
            }
            if (write_c && C_out) {
                if (table < N_TABLE) {
                    for (int i = 0; i < N_META; i++) {
                        uint h32 = (i % 2 == 0) ? (uint)hash[i/2] : (uint)(hash[i/2] >> 32);
                        C_out[j * N_META + i] = h32 & KMASK;
                    }
                } else {
                    for (int i = 0; i < N_META_OUT; i++) {
                        uint h32 = (i % 2 == 0) ? (uint)hash[i/2] : (uint)(hash[i/2] >> 32);
                        C_out[j * N_META_OUT + i] = h32 & KMASK;
                    }
                }
            }
            if (has_pd_in && PD_in) {
                atomic_write_bits(PD_out, PD_0 + PD_in[x], j * PDSIZE, PDSIZE);
            }
            if (has_x_in && X_in) {
                atomic_write_bits(PD_out, X_in[P_1] >> (KSIZE - xbits_arg), j * x2size_arg, XBITS);
                atomic_write_bits(PD_out, X_in[P_2] >> (KSIZE - xbits_arg), j * x2size_arg + XBITS, XBITS);
            }
        }
    }
}

// ============================================================================
// Kernel: write_pd — remap PD to sorted order (for final tables)
// ============================================================================

__kernel void write_pd_kernel(
    __global ulong* PD_out,
    __global const ulong* PD_in,
    __global const ulong* PY_in,
    __global const uint* bucket_size,
    __global const uint* bucket_offset,
    const uint max_bucket_size)
{
    const uint x = get_global_id(0);
    const uint y = get_group_id(1);

    if (x >= min(bucket_size[y], max_bucket_size)) return;

    const uint offset = bucket_offset[y] + x;
    const uint P_i = (uint)PY_in[y * max_bucket_size + x];

    const ulong PD = read_bits(PD_in, (ulong)P_i * PDSIZE, PDSIZE);
    atomic_write_bits(PD_out, PD, (ulong)offset * PDSIZE, PDSIZE);
}

// ============================================================================
// Kernel: write_x2 — write X pairs to X table (table 3)
// ============================================================================

__kernel void write_x2_kernel(
    __global ulong* X_out,
    __global const ulong* X_in,
    __global const ulong* PY_in,
    __global const uint* bucket_size,
    __global const uint* bucket_offset,
    const uint max_bucket_size,
    const uint x2size_arg,
    const uint xbits_arg2)
{
    const uint x = get_global_id(0);
    const uint y = get_group_id(1);

    if (x >= min(bucket_size[y], max_bucket_size)) return;

    const uint offset = bucket_offset[y] + x;
    const uint P_i = (uint)PY_in[y * max_bucket_size + x];

    const uint X_1 = read_bits(X_in, (ulong)P_i * x2size_arg, xbits_arg2);
    const uint X_2 = read_bits(X_in, (ulong)P_i * x2size_arg + xbits_arg2, xbits_arg2);

    atomic_write_bits(X_out, X_1, (ulong)offset * x2size_arg, xbits_arg2);
    atomic_write_bits(X_out, X_2, (ulong)offset * x2size_arg + xbits_arg2, xbits_arg2);
}

// ============================================================================
// Kernel: write_y — write Y values to Y table (for final tables)
// ============================================================================

__kernel void write_y_kernel(
    __global uint* Y_out,
    __global const ulong* PY_in,
    __global const uint* bucket_size,
    __global const uint* bucket_offset,
    const uint max_bucket_size)
{
    const uint x = get_global_id(0);
    const uint y = get_group_id(1);

    if (x >= min(bucket_size[y], max_bucket_size)) return;

    const uint offset = bucket_offset[y] + x;
    Y_out[offset] = (uint)(PY_in[y * max_bucket_size + x] >> (64 - KSIZE));
}

// ============================================================================
// Kernel: write_meta — write metadata to meta table (for final tables, HDD plots)
// ============================================================================

__kernel void write_meta_kernel(
    __global uint* M_out,
    __global const uint* M_in,
    __global const ulong* PY_in,
    __global const uint* bucket_size,
    __global const uint* bucket_offset,
    const uint max_bucket_size)
{
    const uint x = get_global_id(0);
    const uint y = get_group_id(1);

    if (x >= min(bucket_size[y], max_bucket_size)) return;

    const uint offset = bucket_offset[y] + x;
    const uint P_i = (uint)PY_in[y * max_bucket_size + x];

    for (int i = 0; i < N_META_OUT; i++) {
        M_out[offset * N_META_OUT + i] = M_in[P_i * N_META_OUT + i];
    }
}

// ============================================================================
// Kernel: memset_u32 — zero a uint buffer
// ============================================================================

__kernel void memset_u32(__global uint* data, const uint value, const ulong count)
{
    const ulong x = get_global_id(0);
    if (x < count) {
        data[x] = value;
    }
}

// ============================================================================
// Kernel: memset_ulong — zero a ulong buffer
// ============================================================================

__kernel void memset_ulong(__global ulong* data, const ulong value, const ulong count)
{
    const ulong x = get_global_id(0);
    if (x < count) {
        data[x] = value;
    }
}
