#pragma once
// =============================================================================
// alphanumeric_ref_hash.hpp
//
// Independent host-side byte-path reference for the Alphanumeric 92-byte
// BLAKE3 header hash. This is the straightforward implementation (byte
// arrays, per-round message permutation loop) that live pool shares were
// validated against; it deliberately shares no code with the optimized word
// path in alphanumeric_blake3_core.cuh so the two can check each other.
// Cross-checked against the official blake3 library over the exact 92-byte
// header serialization (see tests/cross_check_blake3.py).
//
// Header serialization (all little-endian):
//   u32  block_number
//   [32] previous_hash raw
//   u64  timestamp
//   u64  nonce
//   u64  difficulty
//   [32] merkle_root raw
// =============================================================================

#include <array>
#include <cstdint>
#include <cstring>

namespace alpha_ref {

constexpr uint32_t CHUNK_START = 1u;
constexpr uint32_t CHUNK_END   = 2u;
constexpr uint32_t ROOT        = 8u;

constexpr uint32_t IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u,
};

inline uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}

inline uint32_t load32_le(const uint8_t* p) {
    return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

inline void store32_le(uint8_t* p, uint32_t x) {
    p[0] = (uint8_t)x;
    p[1] = (uint8_t)(x >> 8);
    p[2] = (uint8_t)(x >> 16);
    p[3] = (uint8_t)(x >> 24);
}

inline void store64_le(uint8_t* p, uint64_t x) {
    for(int i = 0; i < 8; ++i) p[i] = (uint8_t)(x >> (8 * i));
}

inline void g(uint32_t v[16], int a, int b, int c, int d, uint32_t mx, uint32_t my) {
    v[a] = v[a] + v[b] + mx;
    v[d] = rotr32(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = rotr32(v[b] ^ v[c], 12);
    v[a] = v[a] + v[b] + my;
    v[d] = rotr32(v[d] ^ v[a], 8);
    v[c] = v[c] + v[d];
    v[b] = rotr32(v[b] ^ v[c], 7);
}

inline void round_fn(uint32_t v[16], const uint32_t m[16]) {
    g(v, 0, 4, 8, 12, m[0], m[1]);
    g(v, 1, 5, 9, 13, m[2], m[3]);
    g(v, 2, 6, 10, 14, m[4], m[5]);
    g(v, 3, 7, 11, 15, m[6], m[7]);
    g(v, 0, 5, 10, 15, m[8], m[9]);
    g(v, 1, 6, 11, 12, m[10], m[11]);
    g(v, 2, 7, 8, 13, m[12], m[13]);
    g(v, 3, 4, 9, 14, m[14], m[15]);
}

inline void permute(uint32_t m[16]) {
    const uint32_t p[16] = {
        m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13],
        m[1], m[11], m[12], m[5], m[9], m[14], m[15], m[8]
    };
    for(int i = 0; i < 16; ++i) m[i] = p[i];
}

inline void compress(const uint32_t cv[8], const uint32_t block_words[16],
                     uint64_t counter, uint32_t block_len, uint32_t flags,
                     uint32_t out[16]) {
    uint32_t v[16];
    uint32_t m[16];
    for(int i = 0; i < 8; ++i) {
        v[i] = cv[i];
        v[i + 8] = IV[i];
    }
    v[12] = (uint32_t)counter;
    v[13] = (uint32_t)(counter >> 32);
    v[14] = block_len;
    v[15] = flags;
    for(int i = 0; i < 16; ++i) m[i] = block_words[i];

    for(int r = 0; r < 7; ++r) {
        round_fn(v, m);
        if(r != 6) permute(m);
    }
    for(int i = 0; i < 8; ++i) {
        out[i] = v[i] ^ v[i + 8];
        out[i + 8] = v[i + 8] ^ cv[i];
    }
}

inline void words_from_block(const uint8_t block[64], uint32_t words[16]) {
    for(int i = 0; i < 16; ++i) words[i] = load32_le(block + i * 4);
}

// BLAKE3 over exactly 92 bytes: one chunk, two compressions.
inline void blake3_hash_92(const uint8_t input[92], uint8_t out_hash[32]) {
    uint32_t cv[8];
    uint32_t words[16];
    uint32_t comp[16];
    uint8_t block[64];

    for(int i = 0; i < 8; ++i) cv[i] = IV[i];

    std::memcpy(block, input, 64);
    words_from_block(block, words);
    compress(cv, words, 0, 64, CHUNK_START, comp);
    for(int i = 0; i < 8; ++i) cv[i] = comp[i];

    std::memset(block, 0, 64);
    std::memcpy(block, input + 64, 28);
    words_from_block(block, words);
    compress(cv, words, 0, 28, CHUNK_END | ROOT, comp);
    for(int i = 0; i < 8; ++i) store32_le(out_hash + i * 4, comp[i]);
}

inline void build_header92(uint8_t header[92], uint32_t block_number,
                           const std::array<uint8_t, 32>& previous_hash,
                           uint64_t timestamp, uint64_t nonce, uint64_t difficulty,
                           const std::array<uint8_t, 32>& merkle_root) {
    store32_le(header + 0, block_number);
    std::memcpy(header + 4, previous_hash.data(), 32);
    store64_le(header + 36, timestamp);
    store64_le(header + 44, nonce);
    store64_le(header + 52, difficulty);
    std::memcpy(header + 60, merkle_root.data(), 32);
}

inline std::array<uint8_t, 32> hash92_fields(uint32_t block_number,
                                             const std::array<uint8_t, 32>& previous_hash,
                                             uint64_t timestamp, uint64_t nonce,
                                             uint64_t difficulty,
                                             const std::array<uint8_t, 32>& merkle_root) {
    uint8_t header[92];
    build_header92(header, block_number, previous_hash, timestamp, nonce, difficulty, merkle_root);
    std::array<uint8_t, 32> out{};
    blake3_hash_92(header, out.data());
    return out;
}

} // namespace alpha_ref
