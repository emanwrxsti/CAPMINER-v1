#pragma once
#include <array>
#include <string>
#include <cstdint>
bool hash_meets_target_le(const std::array<uint8_t,32>& hash, const std::string& target_hex_be);
