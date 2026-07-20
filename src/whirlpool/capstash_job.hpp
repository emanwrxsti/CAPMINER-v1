#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace whirlpool {
struct StratumJob {
    std::string job_id;
    std::string prevhash;
    std::string coinb1;
    std::string coinb2;
    std::vector<std::string> merkle_branch;
    std::string version;
    std::string nbits;
    std::string ntime;
    bool clean_jobs = false;
    std::string target;
};

bool parse_notify(const std::string& line, StratumJob& out);
std::vector<uint8_t> hex_to_bytes(const std::string& hex);
std::string bytes_to_hex(const uint8_t* data, size_t len);
std::string address_to_scriptpubkey(const std::string& capstash_address);
}
