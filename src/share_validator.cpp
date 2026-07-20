#include "share_validator.hpp"
#include <vector>

bool hash_meets_target_le(const std::array<uint8_t,32>& hash_le, const std::string& target_hex_be) {
    // Convert target big-endian hex to comparable bytes; hash is little-endian from Bitcoin-like display.
    if(target_hex_be.size()!=64) return false;
    std::array<uint8_t,32> target_be{};
    for(size_t i=0;i<32;i++) target_be[i]=(uint8_t)std::stoul(target_hex_be.substr(i*2,2), nullptr, 16);
    for(size_t i=0;i<32;i++) {
        uint8_t h = hash_le[31-i];
        uint8_t t = target_be[i];
        if(h < t) return true;
        if(h > t) return false;
    }
    return true;
}
