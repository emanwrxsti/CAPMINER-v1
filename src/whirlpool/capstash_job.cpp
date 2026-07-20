#include "capstash_job.hpp"
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <regex>

namespace whirlpool {

std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    if(hex.size() % 2) throw std::runtime_error("bad hex length");
    std::vector<uint8_t> out(hex.size() / 2);
    for(size_t i = 0; i < out.size(); ++i)
        out[i] = static_cast<uint8_t>(std::stoul(hex.substr(i * 2, 2), nullptr, 16));
    return out;
}

std::string bytes_to_hex(const uint8_t* data, size_t len) {
    std::ostringstream os;
    for(size_t i = 0; i < len; i++)
        os << std::hex << std::setw(2) << std::setfill('0') << (int)data[i];
    return os.str();
}

bool parse_notify(const std::string& line, StratumJob& out) {
    if(line.find("mining.notify") == std::string::npos)
        return false;

    std::regex re(R"CAP("params"\s*:\s*\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*,\s*\[([^\]]*)\]\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(true|false))CAP");

    std::smatch m;
    if(!std::regex_search(line, m, re))
        return false;

    out.job_id = m[1];
    out.prevhash = m[2];
    out.coinb1 = m[3];
    out.coinb2 = m[4];
    out.version = m[6];
    out.nbits = m[7];
    out.ntime = m[8];
    out.clean_jobs = (m[9] == "true");
    out.merkle_branch.clear();
    std::string branches = m[5];
    std::regex branch_re(R"CAP("([^"]+)")CAP");
    auto begin = std::sregex_iterator(branches.begin(), branches.end(), branch_re);
    auto end = std::sregex_iterator();
    for(auto it = begin; it != end; ++it) {
        out.merkle_branch.push_back((*it)[1]);
    }

    return true;
}

std::string address_to_scriptpubkey(const std::string&) {
    throw std::runtime_error("address_to_scriptpubkey not needed for pool mining yet");
}

}
