#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <array>

namespace whirlpool {
std::array<uint8_t, 32> sha256_once(const uint8_t* data, size_t len);
std::array<uint8_t, 32> double_sha256(const std::vector<uint8_t>& data);
std::array<uint8_t, 32> double_sha256_64(const std::array<uint8_t, 64>& data);
}
