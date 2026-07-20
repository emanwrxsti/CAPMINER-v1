#pragma once
#include "capstash_job.hpp"
#include <array>
#include <cstdint>
#include <string>

namespace whirlpool {

uint32_t parse_hex_u32_be_string_to_le_value(const std::string& hex);

// Format extranonce2 as a hex string of exactly `size` bytes (2*size hex
// chars), zero-padded. For size==4 this equals "%08x". The same string MUST be
// used both when building the coinbase and when submitting, so the pool
// reconstructs an identical coinbase.
std::string format_extranonce2(uint32_t extranonce2, int size);

// The Stratum coinbase transaction hex: coinb1 || extranonce1 || extranonce2
// || coinb2. Exposed so diagnostics can log the EXACT bytes that get hashed
// into the merkle root (build_merkle_root calls this).
std::string build_coinbase_hex(
    const StratumJob& job,
    const std::string& extranonce1,
    uint32_t extranonce2,
    int extranonce2_size = 4
);

std::array<uint8_t, 32> build_merkle_root(
    const StratumJob& job,
    const std::string& extranonce1,
    uint32_t extranonce2,
    int extranonce2_size = 4
);

std::array<uint8_t, 80> build_header80(
    const StratumJob& job,
    const std::array<uint8_t, 32>& merkle_root,
    uint32_t nonce
);

std::array<uint8_t, 32> cap_pow_hash_header80(const std::array<uint8_t, 80>& header80);

// Decode compact "nBits" into the 256-bit NETWORK target, stored little-endian
// (the same byte order used for share-target comparison and for the folded
// PoW hash). Used to detect when a found share also satisfies the network
// target, i.e. a real block.
std::array<uint8_t, 32> network_target_from_nbits_le(uint32_t nbits);

// hash <= target, both 32-byte little-endian.
bool hash_leq_target_le(const std::array<uint8_t, 32>& hash, const std::array<uint8_t, 32>& target);

}
