#include "capstash_pow.hpp"
#include "sha256_simple.hpp"
#include "../crypto/whirlpool.h"
#include <stdexcept>
#include <cstring>
#include <cstdio>

namespace whirlpool {

static void write_le32(uint8_t* out, uint32_t x) {
    out[0] = uint8_t(x & 0xff);
    out[1] = uint8_t((x >> 8) & 0xff);
    out[2] = uint8_t((x >> 16) & 0xff);
    out[3] = uint8_t((x >> 24) & 0xff);
}

uint32_t parse_hex_u32_be_string_to_le_value(const std::string& hex) {
    if(hex.size() != 8) throw std::runtime_error("bad u32 hex");
    return static_cast<uint32_t>(std::stoul(hex, nullptr, 16));
}

static void reverse32(std::array<uint8_t, 32>& a) {
    for(int i = 0; i < 16; ++i) std::swap(a[i], a[31 - i]);
}

std::string format_extranonce2(uint32_t extranonce2, int size) {
    if(size < 1) size = 4;
    if(size > 8) size = 8;            // we vary only the low 32 bits
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%0*x", size * 2, extranonce2);
    return std::string(buf);
}

std::string build_coinbase_hex(
    const StratumJob& job,
    const std::string& extranonce1,
    uint32_t extranonce2,
    int extranonce2_size
) {
    return job.coinb1 + extranonce1 + format_extranonce2(extranonce2, extranonce2_size) + job.coinb2;
}

std::array<uint8_t, 32> build_merkle_root(
    const StratumJob& job,
    const std::string& extranonce1,
    uint32_t extranonce2,
    int extranonce2_size
) {
    std::string coinbase_hex = build_coinbase_hex(job, extranonce1, extranonce2, extranonce2_size);
    auto coinbase = hex_to_bytes(coinbase_hex);

    auto root = double_sha256(coinbase);

    for(const auto& branch_hex : job.merkle_branch) {
        auto branch_vec = hex_to_bytes(branch_hex);
        if(branch_vec.size() != 32) throw std::runtime_error("bad merkle branch size");

        std::array<uint8_t, 64> concat{};
        std::memcpy(concat.data(), root.data(), 32);
        std::memcpy(concat.data() + 32, branch_vec.data(), 32);
        root = double_sha256_64(concat);
    }

    return root;
}

std::array<uint8_t, 80> build_header80(
    const StratumJob& job,
    const std::array<uint8_t, 32>& merkle_root,
    uint32_t nonce
) {
    std::array<uint8_t, 80> h{};
    auto prev = hex_to_bytes(job.prevhash);
    if(prev.size() != 32) throw std::runtime_error("bad prevhash");

    // Stratum sends prevhash word-swapped relative to the serialized header.
    // Miningcore builds the notify value with PreviousBlockhash.ReverseByteOrder()
    // (byte-swap each 32-bit word, then reverse the whole buffer) but validates
    // against a header whose prevhash is uint256.Parse(PreviousBlockhash) (a plain
    // little-endian reversal). The net transform the miner must apply to the
    // notify prevhash to recover the header field is: reverse the bytes within
    // each 4-byte word, keeping word order. Without this the pool reconstructs a
    // different header and rejects every otherwise-valid share.
    for(int w = 0; w < 8; ++w) {
        std::swap(prev[w * 4 + 0], prev[w * 4 + 3]);
        std::swap(prev[w * 4 + 1], prev[w * 4 + 2]);
    }

    uint32_t version = parse_hex_u32_be_string_to_le_value(job.version);
    uint32_t nbits = parse_hex_u32_be_string_to_le_value(job.nbits);
    uint32_t ntime = parse_hex_u32_be_string_to_le_value(job.ntime);

    write_le32(h.data() + 0, version);
    std::memcpy(h.data() + 4, prev.data(), 32);
    std::memcpy(h.data() + 36, merkle_root.data(), 32);
    write_le32(h.data() + 68, ntime);
    write_le32(h.data() + 72, nbits);
    write_le32(h.data() + 76, nonce);

    return h;
}

std::array<uint8_t, 32> cap_pow_hash_header80(const std::array<uint8_t, 80>& header80) {
    unsigned char wh[WHIRLPOOL512_OUTPUT_SIZE];
    Whirlpool512(header80.data(), 80, wh);

    std::array<uint8_t, 32> out{};
    for(int i = 0; i < 32; ++i) {
        out[i] = wh[i] ^ wh[i + 32];
    }
    return out;
}

std::array<uint8_t, 32> network_target_from_nbits_le(uint32_t nbits) {
    // Bitcoin-style compact form: top byte = exponent (size in bytes), low 3
    // bytes = mantissa. target = mantissa * 256^(exponent-3). Valid PoW targets
    // never set the 0x00800000 sign bit, so we mask it off. We build directly
    // in little-endian (index 0 = least-significant byte) to match the hash and
    // share-target representation.
    std::array<uint8_t, 32> le{};
    uint32_t exp  = nbits >> 24;
    uint32_t mant = nbits & 0x007fffffu;
    if(mant == 0) return le;

    int shift = (int)exp - 3;   // byte offset of the 3 mantissa bytes
    for(int i = 0; i < 3; ++i) {
        int pos = shift + i;
        uint8_t b = (uint8_t)((mant >> (8 * i)) & 0xffu);
        if(pos >= 0 && pos < 32) le[(size_t)pos] = b;
        // Bytes below index 0 are shifted out (exponent < 3); bytes at/above 32
        // would overflow a 256-bit target and are dropped (clamped).
    }
    return le;
}

bool hash_leq_target_le(const std::array<uint8_t, 32>& h, const std::array<uint8_t, 32>& t) {
    for(int i = 31; i >= 0; --i) {            // index 31 is most significant in LE
        if(h[(size_t)i] != t[(size_t)i]) return h[(size_t)i] < t[(size_t)i];
    }
    return true;                              // equal -> hash <= target
}

}
