#pragma once
// =============================================================================
// alphanumeric_blake3_core.cuh
//
// Shared BLAKE3-92 word-path core for the Alphanumeric CUDA miner.
//
// This single source is compiled two ways:
//   * by nvcc, where the hash functions become __host__ __device__
//     __forceinline__ and the per-hash path runs inside alpha_scan_kernel;
//   * by a plain host C++ compiler (tests/host_kat_test.cpp), so the exact
//     device math can be verified against the independent byte-path reference
//     (alphanumeric_ref_hash.hpp) and the official blake3 library without a
//     GPU present.
//
// Algorithm facts encoded here (unchanged from the verified kernel):
//   92-byte header = 23 LE u32 words:
//     w0        block_number
//     w1..w8    previous_hash
//     w9..w10   timestamp
//     w11..w12  nonce            <- the ONLY per-nonce words
//     w13..w14  difficulty
//     w15..w22  merkle_root
//   BLAKE3 over 92 bytes = one chunk, two compressions:
//     block 0: w0..w15,  block_len 64, flags CHUNK_START
//     block 1: w16..w22 + 9 zero words, block_len 28, flags CHUNK_END|ROOT
//
// Per-job precompute (stage 6):
//   In compress-0 round 0, the four column G mixes use m0..m7 and the
//   diagonal G mixes use (m8,m9) (m10,m11) (m12,m13) (m14,m15). The nonce
//   only occupies m11/m12, so six of the eight round-0 G mixes are
//   nonce-independent. alpha_build_job_pre() applies those six on the host
//   once per job; the kernel finishes round 0 with just the two
//   nonce-carrying diagonals and then runs rounds 1..6.
//
// First-word fast path (stage 6):
//   The share test needs only the first big-endian output word to reject
//   virtually every nonce. Root output word h0 = v0 ^ v8, and in the final
//   round v0 is finalized by diagonal G(0,5,10,15) while v8 is finalized by
//   diagonal G(2,7,8,13). alpha_hash92_head() therefore skips the other two
//   final-round diagonals and the h1..h7 output XORs; alpha_hash92_tail()
//   completes them, from the retained register state, only for the rare
//   candidate (b0 <= target word 0) or when a full 32-byte hash is required.
// =============================================================================

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
  #define ALPHA_CORE_FN __host__ __device__ __forceinline__
  #define ALPHA_UNROLL _Pragma("unroll")
#else
  #define ALPHA_CORE_FN static inline
  #define ALPHA_UNROLL
#endif

namespace alpha_core {

constexpr uint32_t BLAKE3_CHUNK_START = 1u;
constexpr uint32_t BLAKE3_CHUNK_END   = 2u;
constexpr uint32_t BLAKE3_ROOT        = 8u;
constexpr uint32_t BLAKE3_BLOCK0_LEN  = 64u;
constexpr uint32_t BLAKE3_BLOCK1_LEN  = 28u;

// Kept out of __constant__ memory on purpose: every use site below is either
// folded at compile time (host precompute) or loaded uniformly through the
// kernel-parameter constant bank, which profiles identically.
ALPHA_CORE_FN uint32_t alpha_iv(int i) {
    constexpr uint32_t IV[8] = {
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
        0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u,
    };
    return IV[i];
}

ALPHA_CORE_FN uint32_t alpha_rotr32(uint32_t x, uint32_t n) {
#if defined(__CUDA_ARCH__)
    return __funnelshift_r(x, x, n);   // compiles to a single SHF.R
#else
    return (x >> n) | (x << (32u - n));
#endif
}

ALPHA_CORE_FN uint32_t alpha_bswap32(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __byte_perm(x, 0, 0x0123);
#else
    return (x >> 24) | ((x >> 8) & 0x0000FF00u) | ((x << 8) & 0x00FF0000u) | (x << 24);
#endif
}

ALPHA_CORE_FN void alpha_g(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
                           uint32_t mx, uint32_t my) {
    a = a + b + mx;
    d = alpha_rotr32(d ^ a, 16);
    c = c + d;
    b = alpha_rotr32(b ^ c, 12);
    a = a + b + my;
    d = alpha_rotr32(d ^ a, 8);
    c = c + d;
    b = alpha_rotr32(b ^ c, 7);
}

// -----------------------------------------------------------------------------
// Per-job precomputed words.
//   v0[16]  : compress-0 state after the six nonce-independent round-0 G mixes
//             (all four columns + diagonals G(0,5,10,15) and G(3,4,9,14)).
//   m0[16]  : block-0 message words; slots 11/12 are zero placeholders, the
//             nonce words are injected per hash and never read from here.
//   m1[7]   : block-1 message words (words 7..15 of block 1 are zero).
//   target[8]: big-endian target words; byte-lexicographic compare on the
//             32-byte hash == word compare on these.
// -----------------------------------------------------------------------------
struct AlphaJobPre {
    uint32_t v0[16];
    uint32_t m0[16];
    uint32_t m1[7];
    uint32_t target[8];
};

// Message selectors. The index is always a compile-time literal below, so the
// ternaries fold: nonce slots become the nl/nh registers, block-1 slots >= 7
// become literal zero (the adds fold away), everything else is a uniform
// constant-bank read of the kernel parameter.
template <int I>
ALPHA_CORE_FN uint32_t alpha_m0_sel(const AlphaJobPre& jw, uint32_t nl, uint32_t nh) {
    return I == 11 ? nl : (I == 12 ? nh : jw.m0[I]);
}
template <int I>
ALPHA_CORE_FN uint32_t alpha_m1_sel(const AlphaJobPre& jw) {
    return I < 7 ? jw.m1[I] : 0u;
}

// Build the per-job constants. header_words are the 23 LE header words with
// w11/w12 (nonce) ignored; their m0 slots are stored as zero.
ALPHA_CORE_FN void alpha_build_job_pre(const uint32_t header_words[23],
                                       const uint32_t target_be_words[8],
                                       AlphaJobPre& jw) {
    ALPHA_UNROLL
    for(int i = 0; i < 16; ++i) jw.m0[i] = header_words[i];
    jw.m0[11] = 0u;
    jw.m0[12] = 0u;
    ALPHA_UNROLL
    for(int i = 0; i < 7; ++i) jw.m1[i] = header_words[16 + i];
    ALPHA_UNROLL
    for(int i = 0; i < 8; ++i) jw.target[i] = target_be_words[i];

    // compress-0 initial state (cv = IV, counter = 0, block_len = 64,
    // flags = CHUNK_START), exactly as the full compression would set it.
    uint32_t v[16];
    ALPHA_UNROLL
    for(int i = 0; i < 8; ++i) {
        v[i]     = alpha_iv(i);
        v[i + 8] = (i < 4) ? alpha_iv(i) : 0u;
    }
    v[12] = 0u;
    v[13] = 0u;
    v[14] = BLAKE3_BLOCK0_LEN;
    v[15] = BLAKE3_CHUNK_START;

    // Round 0, the six nonce-independent G mixes (identity schedule):
    // columns m0..m7, then diagonals (m8,m9) and (m14,m15). Diagonals
    // (m10,m11) and (m12,m13) carry the nonce and are left for the device.
    alpha_g(v[0], v[4], v[8],  v[12], jw.m0[0],  jw.m0[1]);
    alpha_g(v[1], v[5], v[9],  v[13], jw.m0[2],  jw.m0[3]);
    alpha_g(v[2], v[6], v[10], v[14], jw.m0[4],  jw.m0[5]);
    alpha_g(v[3], v[7], v[11], v[15], jw.m0[6],  jw.m0[7]);
    alpha_g(v[0], v[5], v[10], v[15], jw.m0[8],  jw.m0[9]);
    alpha_g(v[3], v[4], v[9],  v[14], jw.m0[14], jw.m0[15]);

    ALPHA_UNROLL
    for(int i = 0; i < 16; ++i) jw.v0[i] = v[i];
}

#define ALPHA_M0(i) (alpha_m0_sel<(i)>(jw, nl, nh))
#define ALPHA_M1(i) (alpha_m1_sel<(i)>(jw))

// compress-0 from the precomputed round-0 partial state. Produces the 8
// chaining words (out[8..15] of a compression are never needed for a
// single-chunk hash).
ALPHA_CORE_FN void alpha_compress0_from_pre(const AlphaJobPre& jw,
                                            uint32_t nl, uint32_t nh,
                                            uint32_t cv[8]) {
    uint32_t v[16];
    ALPHA_UNROLL
    for(int i = 0; i < 16; ++i) v[i] = jw.v0[i];

    // Finish round 0: the two nonce-carrying diagonals.
    // round 0
    alpha_g(v[1], v[6], v[11], v[12], ALPHA_M0(10), ALPHA_M0(11));
    alpha_g(v[2], v[7], v[8], v[13], ALPHA_M0(12), ALPHA_M0(13));
    // round 1
    alpha_g(v[0], v[4], v[8], v[12], ALPHA_M0(2), ALPHA_M0(6));
    alpha_g(v[1], v[5], v[9], v[13], ALPHA_M0(3), ALPHA_M0(10));
    alpha_g(v[2], v[6], v[10], v[14], ALPHA_M0(7), ALPHA_M0(0));
    alpha_g(v[3], v[7], v[11], v[15], ALPHA_M0(4), ALPHA_M0(13));
    alpha_g(v[0], v[5], v[10], v[15], ALPHA_M0(1), ALPHA_M0(11));
    alpha_g(v[1], v[6], v[11], v[12], ALPHA_M0(12), ALPHA_M0(5));
    alpha_g(v[2], v[7], v[8], v[13], ALPHA_M0(9), ALPHA_M0(14));
    alpha_g(v[3], v[4], v[9], v[14], ALPHA_M0(15), ALPHA_M0(8));
    // round 2
    alpha_g(v[0], v[4], v[8], v[12], ALPHA_M0(3), ALPHA_M0(4));
    alpha_g(v[1], v[5], v[9], v[13], ALPHA_M0(10), ALPHA_M0(12));
    alpha_g(v[2], v[6], v[10], v[14], ALPHA_M0(13), ALPHA_M0(2));
    alpha_g(v[3], v[7], v[11], v[15], ALPHA_M0(7), ALPHA_M0(14));
    alpha_g(v[0], v[5], v[10], v[15], ALPHA_M0(6), ALPHA_M0(5));
    alpha_g(v[1], v[6], v[11], v[12], ALPHA_M0(9), ALPHA_M0(0));
    alpha_g(v[2], v[7], v[8], v[13], ALPHA_M0(11), ALPHA_M0(15));
    alpha_g(v[3], v[4], v[9], v[14], ALPHA_M0(8), ALPHA_M0(1));
    // round 3
    alpha_g(v[0], v[4], v[8], v[12], ALPHA_M0(10), ALPHA_M0(7));
    alpha_g(v[1], v[5], v[9], v[13], ALPHA_M0(12), ALPHA_M0(9));
    alpha_g(v[2], v[6], v[10], v[14], ALPHA_M0(14), ALPHA_M0(3));
    alpha_g(v[3], v[7], v[11], v[15], ALPHA_M0(13), ALPHA_M0(15));
    alpha_g(v[0], v[5], v[10], v[15], ALPHA_M0(4), ALPHA_M0(0));
    alpha_g(v[1], v[6], v[11], v[12], ALPHA_M0(11), ALPHA_M0(2));
    alpha_g(v[2], v[7], v[8], v[13], ALPHA_M0(5), ALPHA_M0(8));
    alpha_g(v[3], v[4], v[9], v[14], ALPHA_M0(1), ALPHA_M0(6));
    // round 4
    alpha_g(v[0], v[4], v[8], v[12], ALPHA_M0(12), ALPHA_M0(13));
    alpha_g(v[1], v[5], v[9], v[13], ALPHA_M0(9), ALPHA_M0(11));
    alpha_g(v[2], v[6], v[10], v[14], ALPHA_M0(15), ALPHA_M0(10));
    alpha_g(v[3], v[7], v[11], v[15], ALPHA_M0(14), ALPHA_M0(8));
    alpha_g(v[0], v[5], v[10], v[15], ALPHA_M0(7), ALPHA_M0(2));
    alpha_g(v[1], v[6], v[11], v[12], ALPHA_M0(5), ALPHA_M0(3));
    alpha_g(v[2], v[7], v[8], v[13], ALPHA_M0(0), ALPHA_M0(1));
    alpha_g(v[3], v[4], v[9], v[14], ALPHA_M0(6), ALPHA_M0(4));
    // round 5
    alpha_g(v[0], v[4], v[8], v[12], ALPHA_M0(9), ALPHA_M0(14));
    alpha_g(v[1], v[5], v[9], v[13], ALPHA_M0(11), ALPHA_M0(5));
    alpha_g(v[2], v[6], v[10], v[14], ALPHA_M0(8), ALPHA_M0(12));
    alpha_g(v[3], v[7], v[11], v[15], ALPHA_M0(15), ALPHA_M0(1));
    alpha_g(v[0], v[5], v[10], v[15], ALPHA_M0(13), ALPHA_M0(3));
    alpha_g(v[1], v[6], v[11], v[12], ALPHA_M0(0), ALPHA_M0(10));
    alpha_g(v[2], v[7], v[8], v[13], ALPHA_M0(2), ALPHA_M0(6));
    alpha_g(v[3], v[4], v[9], v[14], ALPHA_M0(4), ALPHA_M0(7));
    // round 6
    alpha_g(v[0], v[4], v[8], v[12], ALPHA_M0(11), ALPHA_M0(15));
    alpha_g(v[1], v[5], v[9], v[13], ALPHA_M0(5), ALPHA_M0(0));
    alpha_g(v[2], v[6], v[10], v[14], ALPHA_M0(1), ALPHA_M0(9));
    alpha_g(v[3], v[7], v[11], v[15], ALPHA_M0(8), ALPHA_M0(6));
    alpha_g(v[0], v[5], v[10], v[15], ALPHA_M0(14), ALPHA_M0(10));
    alpha_g(v[1], v[6], v[11], v[12], ALPHA_M0(2), ALPHA_M0(12));
    alpha_g(v[2], v[7], v[8], v[13], ALPHA_M0(3), ALPHA_M0(4));
    alpha_g(v[3], v[4], v[9], v[14], ALPHA_M0(7), ALPHA_M0(13));

    ALPHA_UNROLL
    for(int i = 0; i < 8; ++i) cv[i] = v[i] ^ v[i + 8];
}

// compress-1 (root compression) through the fast subset of the final round:
// all four columns plus the two diagonals that finalize v0 and v8. Returns
// h0 = v0 ^ v8, the only word the common rejection path needs. v[16] retains
// the register state so alpha_hash92_tail() can finish without recomputing.
ALPHA_CORE_FN uint32_t alpha_hash92_head(const AlphaJobPre& jw,
                                         uint32_t nl, uint32_t nh,
                                         uint32_t v[16]) {
    uint32_t cv[8];
    alpha_compress0_from_pre(jw, nl, nh, cv);

    ALPHA_UNROLL
    for(int i = 0; i < 8; ++i) {
        v[i]     = cv[i];
        v[i + 8] = (i < 4) ? alpha_iv(i) : 0u;
    }
    v[12] = 0u;
    v[13] = 0u;
    v[14] = BLAKE3_BLOCK1_LEN;
    v[15] = BLAKE3_CHUNK_END | BLAKE3_ROOT;

    // round 0
    alpha_g(v[0], v[4], v[8], v[12], ALPHA_M1(0), ALPHA_M1(1));
    alpha_g(v[1], v[5], v[9], v[13], ALPHA_M1(2), ALPHA_M1(3));
    alpha_g(v[2], v[6], v[10], v[14], ALPHA_M1(4), ALPHA_M1(5));
    alpha_g(v[3], v[7], v[11], v[15], ALPHA_M1(6), ALPHA_M1(7));
    alpha_g(v[0], v[5], v[10], v[15], ALPHA_M1(8), ALPHA_M1(9));
    alpha_g(v[1], v[6], v[11], v[12], ALPHA_M1(10), ALPHA_M1(11));
    alpha_g(v[2], v[7], v[8], v[13], ALPHA_M1(12), ALPHA_M1(13));
    alpha_g(v[3], v[4], v[9], v[14], ALPHA_M1(14), ALPHA_M1(15));
    // round 1
    alpha_g(v[0], v[4], v[8], v[12], ALPHA_M1(2), ALPHA_M1(6));
    alpha_g(v[1], v[5], v[9], v[13], ALPHA_M1(3), ALPHA_M1(10));
    alpha_g(v[2], v[6], v[10], v[14], ALPHA_M1(7), ALPHA_M1(0));
    alpha_g(v[3], v[7], v[11], v[15], ALPHA_M1(4), ALPHA_M1(13));
    alpha_g(v[0], v[5], v[10], v[15], ALPHA_M1(1), ALPHA_M1(11));
    alpha_g(v[1], v[6], v[11], v[12], ALPHA_M1(12), ALPHA_M1(5));
    alpha_g(v[2], v[7], v[8], v[13], ALPHA_M1(9), ALPHA_M1(14));
    alpha_g(v[3], v[4], v[9], v[14], ALPHA_M1(15), ALPHA_M1(8));
    // round 2
    alpha_g(v[0], v[4], v[8], v[12], ALPHA_M1(3), ALPHA_M1(4));
    alpha_g(v[1], v[5], v[9], v[13], ALPHA_M1(10), ALPHA_M1(12));
    alpha_g(v[2], v[6], v[10], v[14], ALPHA_M1(13), ALPHA_M1(2));
    alpha_g(v[3], v[7], v[11], v[15], ALPHA_M1(7), ALPHA_M1(14));
    alpha_g(v[0], v[5], v[10], v[15], ALPHA_M1(6), ALPHA_M1(5));
    alpha_g(v[1], v[6], v[11], v[12], ALPHA_M1(9), ALPHA_M1(0));
    alpha_g(v[2], v[7], v[8], v[13], ALPHA_M1(11), ALPHA_M1(15));
    alpha_g(v[3], v[4], v[9], v[14], ALPHA_M1(8), ALPHA_M1(1));
    // round 3
    alpha_g(v[0], v[4], v[8], v[12], ALPHA_M1(10), ALPHA_M1(7));
    alpha_g(v[1], v[5], v[9], v[13], ALPHA_M1(12), ALPHA_M1(9));
    alpha_g(v[2], v[6], v[10], v[14], ALPHA_M1(14), ALPHA_M1(3));
    alpha_g(v[3], v[7], v[11], v[15], ALPHA_M1(13), ALPHA_M1(15));
    alpha_g(v[0], v[5], v[10], v[15], ALPHA_M1(4), ALPHA_M1(0));
    alpha_g(v[1], v[6], v[11], v[12], ALPHA_M1(11), ALPHA_M1(2));
    alpha_g(v[2], v[7], v[8], v[13], ALPHA_M1(5), ALPHA_M1(8));
    alpha_g(v[3], v[4], v[9], v[14], ALPHA_M1(1), ALPHA_M1(6));
    // round 4
    alpha_g(v[0], v[4], v[8], v[12], ALPHA_M1(12), ALPHA_M1(13));
    alpha_g(v[1], v[5], v[9], v[13], ALPHA_M1(9), ALPHA_M1(11));
    alpha_g(v[2], v[6], v[10], v[14], ALPHA_M1(15), ALPHA_M1(10));
    alpha_g(v[3], v[7], v[11], v[15], ALPHA_M1(14), ALPHA_M1(8));
    alpha_g(v[0], v[5], v[10], v[15], ALPHA_M1(7), ALPHA_M1(2));
    alpha_g(v[1], v[6], v[11], v[12], ALPHA_M1(5), ALPHA_M1(3));
    alpha_g(v[2], v[7], v[8], v[13], ALPHA_M1(0), ALPHA_M1(1));
    alpha_g(v[3], v[4], v[9], v[14], ALPHA_M1(6), ALPHA_M1(4));
    // round 5
    alpha_g(v[0], v[4], v[8], v[12], ALPHA_M1(9), ALPHA_M1(14));
    alpha_g(v[1], v[5], v[9], v[13], ALPHA_M1(11), ALPHA_M1(5));
    alpha_g(v[2], v[6], v[10], v[14], ALPHA_M1(8), ALPHA_M1(12));
    alpha_g(v[3], v[7], v[11], v[15], ALPHA_M1(15), ALPHA_M1(1));
    alpha_g(v[0], v[5], v[10], v[15], ALPHA_M1(13), ALPHA_M1(3));
    alpha_g(v[1], v[6], v[11], v[12], ALPHA_M1(0), ALPHA_M1(10));
    alpha_g(v[2], v[7], v[8], v[13], ALPHA_M1(2), ALPHA_M1(6));
    alpha_g(v[3], v[4], v[9], v[14], ALPHA_M1(4), ALPHA_M1(7));
    // round 6
    alpha_g(v[0], v[4], v[8], v[12], ALPHA_M1(11), ALPHA_M1(15));
    alpha_g(v[1], v[5], v[9], v[13], ALPHA_M1(5), ALPHA_M1(0));
    alpha_g(v[2], v[6], v[10], v[14], ALPHA_M1(1), ALPHA_M1(9));
    alpha_g(v[3], v[7], v[11], v[15], ALPHA_M1(8), ALPHA_M1(6));
    alpha_g(v[0], v[5], v[10], v[15], ALPHA_M1(14), ALPHA_M1(10));
    alpha_g(v[2], v[7], v[8], v[13], ALPHA_M1(3), ALPHA_M1(4));

    return v[0] ^ v[8];
}

// Complete the final round (the two deferred diagonals) and emit all eight
// little-endian output words. Only runs for candidates and full-hash callers.
ALPHA_CORE_FN void alpha_hash92_tail(const AlphaJobPre& jw, uint32_t v[16],
                                     uint32_t h[8]) {
    // round 6
    alpha_g(v[1], v[6], v[11], v[12], ALPHA_M1(2), ALPHA_M1(12));
    alpha_g(v[3], v[4], v[9], v[14], ALPHA_M1(7), ALPHA_M1(13));

    ALPHA_UNROLL
    for(int i = 0; i < 8; ++i) h[i] = v[i] ^ v[i + 8];
}

#undef ALPHA_M0
#undef ALPHA_M1

// Full 32-byte word-path hash for one nonce (self-test / verify / candidate).
ALPHA_CORE_FN void alpha_hash92_words(const AlphaJobPre& jw, uint64_t nonce,
                                      uint32_t h[8]) {
    uint32_t v[16];
    (void)alpha_hash92_head(jw, (uint32_t)nonce, (uint32_t)(nonce >> 32), v);
    alpha_hash92_tail(jw, v, h);
}

// Byte-lexicographic hash <= target_be, split the way the kernel uses it:
// word 0 decides for virtually every nonce; the tail runs on a word-0 tie.
ALPHA_CORE_FN bool alpha_words_le_target_tail(const uint32_t h[8],
                                              const uint32_t target[8]) {
    ALPHA_UNROLL
    for(int i = 1; i < 8; ++i) {
        const uint32_t bi = alpha_bswap32(h[i]);
        if(bi < target[i]) return true;
        if(bi > target[i]) return false;
    }
    return true; // full equality counts as <=
}

ALPHA_CORE_FN bool alpha_hash_words_le_target(const uint32_t h[8],
                                              const uint32_t target[8]) {
    const uint32_t b0 = alpha_bswap32(h[0]);
    if(b0 < target[0]) return true;
    if(b0 > target[0]) return false;
    return alpha_words_le_target_tail(h, target);
}

// 23 LE header words from the typed Alphanumeric job fields. w11/w12 are the
// nonce and are emitted as zero (per-hash injection).
ALPHA_CORE_FN void alpha_header_words_from_fields(uint32_t block_number,
                                                  const uint8_t previous_hash[32],
                                                  uint64_t timestamp,
                                                  uint64_t difficulty,
                                                  const uint8_t merkle_root[32],
                                                  uint32_t w[23]) {
    auto le32 = [](const uint8_t* p) -> uint32_t {
        return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) |
               ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    };
    w[0] = block_number;
    ALPHA_UNROLL
    for(int i = 0; i < 8; ++i) w[1 + i] = le32(previous_hash + i * 4);
    w[9]  = (uint32_t)timestamp;
    w[10] = (uint32_t)(timestamp >> 32);
    w[11] = 0u;
    w[12] = 0u;
    w[13] = (uint32_t)difficulty;
    w[14] = (uint32_t)(difficulty >> 32);
    ALPHA_UNROLL
    for(int i = 0; i < 8; ++i) w[15 + i] = le32(merkle_root + i * 4);
}

ALPHA_CORE_FN void alpha_target_words_from_be_bytes(const uint8_t target_be[32],
                                                    uint32_t t[8]) {
    ALPHA_UNROLL
    for(int i = 0; i < 8; ++i) {
        const uint8_t* p = target_be + i * 4;
        t[i] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
               ((uint32_t)p[2] << 8) | (uint32_t)p[3];
    }
}

} // namespace alpha_core
