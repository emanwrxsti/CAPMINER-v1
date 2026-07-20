#include "alphanumeric_runner.hpp"
#include "alphanumeric_cuda_backend.hpp"
#include "../logger.hpp"
#include "../stats.hpp"
#include "../stratum_client.hpp"
#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cctype>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <map>
#include <limits>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace alphanumeric {

#if defined(CAPMINER_GPU_BACKEND_HIP)
static constexpr const char* kGpuBackendName = "HIP/ROCm";
#elif defined(CAPMINER_GPU_BACKEND_CUDA)
static constexpr const char* kGpuBackendName = "CUDA";
#else
static constexpr const char* kGpuBackendName = "GPU";
#endif

struct AlphaJob {
    std::string job_id;
    uint32_t block_number = 0;
    std::array<uint8_t, 32> previous_hash{};
    uint64_t timestamp = 0;
    uint64_t difficulty = 0;       // Consensus header difficulty used inside the 92-byte header.
    std::array<uint8_t, 32> merkle_root{};
    // Consensus/network target derived from the header difficulty.
    std::array<uint8_t, 32> network_target_be{};
    // Raw target supplied by mining.notify. A target different from the
    // consensus target is treated as an explicit pool share target.
    std::array<uint8_t, 32> notify_target_be{};
    // Effective target used by CUDA for share discovery.
    std::array<uint8_t, 32> target_be{};
    bool has_target = false;
    bool targets_initialized = false;
    bool explicit_share_target = false;
    bool local_scaling_disabled = false;
    bool clean_jobs = false;
};

static std::string hex_bytes(const unsigned char* p, size_t n) {
    static const char* h = "0123456789abcdef";
    std::string out;
    out.reserve(n * 2);
    for(size_t i = 0; i < n; ++i) {
        out.push_back(h[p[i] >> 4]);
        out.push_back(h[p[i] & 0x0f]);
    }
    return out;
}

static std::string hex64_le(uint64_t v) {
    std::ostringstream os;
    os << std::hex << std::nouppercase << std::setfill('0') << std::setw(16) << v;
    return os.str();
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for(char ch : s) {
        switch(ch) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if(static_cast<unsigned char>(ch) < 0x20) {
                    char b[7];
                    std::snprintf(b, sizeof(b), "\\u%04x", (unsigned char)ch);
                    out += b;
                } else {
                    out.push_back(ch);
                }
        }
    }
    return out;
}

static std::string trim_copy(std::string s) {
    auto is_ws = [](unsigned char c){ return std::isspace(c) != 0; };
    while(!s.empty() && is_ws((unsigned char)s.front())) s.erase(s.begin());
    while(!s.empty() && is_ws((unsigned char)s.back())) s.pop_back();
    return s;
}

static uint64_t unix_now_seconds() {
    const auto now = std::chrono::system_clock::now();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count());
}

static std::vector<std::string> split_json_array_top_level(const std::string& body) {
    std::vector<std::string> out;
    std::string cur;
    bool in_str = false;
    bool esc = false;
    int depth = 0;
    for(char ch : body) {
        if(in_str) {
            cur.push_back(ch);
            if(esc) { esc = false; continue; }
            if(ch == '\\') { esc = true; continue; }
            if(ch == '"') in_str = false;
            continue;
        }
        if(ch == '"') { in_str = true; cur.push_back(ch); continue; }
        if(ch == '[' || ch == '{') { depth++; cur.push_back(ch); continue; }
        if(ch == ']' || ch == '}') { depth--; cur.push_back(ch); continue; }
        if(ch == ',' && depth == 0) {
            out.push_back(trim_copy(cur));
            cur.clear();
            continue;
        }
        cur.push_back(ch);
    }
    if(!cur.empty() || !body.empty()) out.push_back(trim_copy(cur));
    return out;
}

static std::string unquote_json_string(std::string s) {
    s = trim_copy(std::move(s));
    if(s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        s = s.substr(1, s.size() - 2);
    }
    std::string out;
    out.reserve(s.size());
    bool esc = false;
    for(char ch : s) {
        if(esc) {
            switch(ch) {
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                default: out.push_back(ch); break;
            }
            esc = false;
        } else if(ch == '\\') {
            esc = true;
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

static bool parse_bool_token(const std::string& token) {
    std::string s = trim_copy(token);
    for(char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s == "true" || s == "1";
}

static uint64_t parse_u64_token(std::string token) {
    token = unquote_json_string(std::move(token));
    token = trim_copy(std::move(token));
    if(token.empty()) return 0;
    int base = 10;
    if(token.size() > 2 && token[0] == '0' && (token[1] == 'x' || token[1] == 'X')) base = 16;
    return static_cast<uint64_t>(std::stoull(token, nullptr, base));
}

static bool parse_hex32(std::string hex, std::array<uint8_t, 32>& out) {
    hex = unquote_json_string(std::move(hex));
    hex = trim_copy(std::move(hex));
    if(hex.rfind("0x", 0) == 0 || hex.rfind("0X", 0) == 0) hex = hex.substr(2);
    if(hex.size() != 64) return false;
    for(size_t i = 0; i < 32; ++i) {
        const std::string b = hex.substr(i * 2, 2);
        out[i] = static_cast<uint8_t>(std::stoul(b, nullptr, 16));
    }
    return true;
}

static bool token_is_hex32(std::string token) {
    token = unquote_json_string(std::move(token));
    token = trim_copy(std::move(token));
    if(token.rfind("0x", 0) == 0 || token.rfind("0X", 0) == 0) token = token.substr(2);
    if(token.size() != 64) return false;
    for(char c : token) {
        if(!std::isxdigit(static_cast<unsigned char>(c))) return false;
    }
    return true;
}

static bool try_parse_u64_token(const std::string& token, uint64_t& out) {
    try {
        out = parse_u64_token(token);
        return true;
    } catch(const std::exception&) {
        return false;
    }
}

static bool token_is_boolish(std::string token) {
    token = trim_copy(std::move(token));
    for(char& c : token) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return token == "true" || token == "false" || token == "1" || token == "0";
}

static std::string object_field_raw(const std::string& obj, const std::vector<std::string>& names) {
    for(const auto& name : names) {
        std::regex re("\\\"" + name + "\\\"\\s*:\\s*(\\\"(?:[^\\\"\\\\]|\\\\.)*\\\"|true|false|null|-?[0-9]+(?:\\.[0-9]+)?|\\{[^}]*\\}|\\[[^]]*\\])");
        std::smatch m;
        if(std::regex_search(obj, m, re)) return m[1];
    }
    return {};
}

static std::string object_field_string(const std::string& obj, const std::vector<std::string>& names) {
    std::string raw = object_field_raw(obj, names);
    if(raw.empty()) return {};
    return unquote_json_string(raw);
}

static uint64_t object_field_u64(const std::string& obj, const std::vector<std::string>& names, uint64_t def = 0) {
    std::string raw = object_field_raw(obj, names);
    if(raw.empty() || raw == "null") return def;
    return parse_u64_token(raw);
}

static bool object_field_bool(const std::string& obj, const std::vector<std::string>& names, bool def = false) {
    std::string raw = object_field_raw(obj, names);
    if(raw.empty() || raw == "null") return def;
    return parse_bool_token(raw);
}

static std::array<uint8_t, 32> alpha_target_from_difficulty(uint64_t difficulty) {
    std::array<uint8_t, 32> target{};
    const uint64_t shift = difficulty / 16ull;
    if(difficulty == 0) {
        target.fill(0xff);
        return target;
    }
    if(shift >= 256) return target;
    // MAX_TARGET >> shift, where MAX_TARGET = 2^256 - 1.
    // Big-endian fixed-width bytes: first `shift` bits are zero, remaining bits are one.
    for(uint64_t bit = shift; bit < 256; ++bit) {
        const size_t byte_index = static_cast<size_t>(bit / 8);
        const int bit_in_byte = 7 - static_cast<int>(bit % 8);
        target[byte_index] |= static_cast<uint8_t>(1u << bit_in_byte);
    }
    return target;
}

static bool hash_le_target_be(const unsigned char hash[32], const std::array<uint8_t, 32>& target) {
    for(size_t i = 0; i < 32; ++i) {
        if(hash[i] < target[i]) return true;
        if(hash[i] > target[i]) return false;
    }
    return true;
}

static int compare_target_be(const std::array<uint8_t, 32>& a,
                             const std::array<uint8_t, 32>& b) {
    for(size_t i = 0; i < a.size(); ++i) {
        if(a[i] < b[i]) return -1;
        if(a[i] > b[i]) return 1;
    }
    return 0;
}

// Scale a 256-bit big-endian target by 1 / pool_difficulty.  Alphanumeric's
// consensus target is MAX_TARGET >> (headerDifficulty / 16). Miningcore's
// Stratum difficulty is a multiplier on that base target: values below 1 make
// pool shares easier, while values above 1 make them harder.
//
// The arithmetic is integer-only (8 x 32-bit limbs plus one carry limb) so it
// behaves identically on MSVC and nvcc and never loses the high target bits to
// floating-point rounding.
static std::array<uint8_t, 32> scale_target_for_pool_difficulty(
    const std::array<uint8_t, 32>& base_target, double pool_difficulty) {
    if(!(pool_difficulty > 0.0) || !std::isfinite(pool_difficulty))
        pool_difficulty = 1.0;

    constexpr uint32_t SCALE = 1000000u;
    double den_d = pool_difficulty * static_cast<double>(SCALE);
    if(den_d < 1.0) den_d = 1.0;
    if(den_d > static_cast<double>(std::numeric_limits<uint32_t>::max()))
        den_d = static_cast<double>(std::numeric_limits<uint32_t>::max());
    const uint32_t denominator = static_cast<uint32_t>(den_d + 0.5);
    const uint32_t numerator = SCALE;

    // Convert target to eight little-endian 32-bit limbs.
    uint32_t in[8]{};
    for(int limb = 0; limb < 8; ++limb) {
        const int off = 28 - limb * 4;
        in[limb] = (static_cast<uint32_t>(base_target[off + 0]) << 24) |
                   (static_cast<uint32_t>(base_target[off + 1]) << 16) |
                   (static_cast<uint32_t>(base_target[off + 2]) << 8)  |
                   (static_cast<uint32_t>(base_target[off + 3]));
    }

    // Multiply by SCALE into a 288-bit value.
    uint32_t wide[9]{};
    uint64_t carry = 0;
    for(int i = 0; i < 8; ++i) {
        const uint64_t product = static_cast<uint64_t>(in[i]) * numerator + carry;
        wide[i] = static_cast<uint32_t>(product);
        carry = product >> 32;
    }
    wide[8] = static_cast<uint32_t>(carry);

    // Divide the 288-bit value by the scaled difficulty.
    uint32_t quotient[9]{};
    uint64_t remainder = 0;
    for(int i = 8; i >= 0; --i) {
        const uint64_t cur = (remainder << 32) | wide[i];
        quotient[i] = static_cast<uint32_t>(cur / denominator);
        remainder = cur % denominator;
    }

    std::array<uint8_t, 32> out{};
    if(quotient[8] != 0) {
        out.fill(0xff); // Easier than MAX_TARGET: saturate instead of wrapping.
        return out;
    }

    for(int limb = 0; limb < 8; ++limb) {
        const uint32_t v = quotient[limb];
        const int off = 28 - limb * 4;
        out[off + 0] = static_cast<uint8_t>(v >> 24);
        out[off + 1] = static_cast<uint8_t>(v >> 16);
        out[off + 2] = static_cast<uint8_t>(v >> 8);
        out[off + 3] = static_cast<uint8_t>(v);
    }
    return out;
}

static void apply_pool_share_target(AlphaJob& job, double pool_difficulty,
                                    bool disable_local_scaling = false) {
    job.network_target_be = alpha_target_from_difficulty(job.difficulty);
    if(!job.targets_initialized) {
        job.notify_target_be = job.has_target ? job.target_be : job.network_target_be;
        job.targets_initialized = true;
    }

    // A notify target that differs from the consensus target is already an
    // explicit share target supplied by the pool. Trust it and do not scale it
    // a second time. The affected Miningcore endpoint sends the consensus
    // target unchanged, so in that case mining.set_difficulty must be applied
    // locally.
    job.explicit_share_target = job.has_target &&
        compare_target_be(job.notify_target_be, job.network_target_be) != 0;
    job.local_scaling_disabled = disable_local_scaling;
    if(disable_local_scaling) {
        job.target_be = job.has_target ? job.notify_target_be : job.network_target_be;
    } else {
        job.target_be = job.explicit_share_target
            ? job.notify_target_be
            : scale_target_for_pool_difficulty(job.network_target_be, pool_difficulty);
    }
}

static double target_work_diff(const std::array<uint8_t, 32>& target) {
    // Dashboard effective hashrate uses accepted * work_diff * 2^32 / time.
    // Convert this target to that work-difficulty unit: 2^224 / (target + 1).
    long double value = 0.0L;
    for(uint8_t b : target) value = value * 256.0L + static_cast<long double>(b);
    if(value < 0.0L) return 0.0;
    const long double work = std::ldexp(1.0L, 224) / (value + 1.0L);
    return static_cast<double>(work);
}

static bool is_set_difficulty_method_line(const std::string& line) {
    return std::regex_search(line, std::regex(
        R"CAP("method"\s*:\s*"mining\.set_difficulty")CAP"));
}

static bool parse_object_job(const std::string& obj, AlphaJob& out) {
    AlphaJob j;
    j.job_id = object_field_string(obj, {"job_id", "jobId", "id"});
    j.block_number = static_cast<uint32_t>(object_field_u64(obj, {"block_number", "blockNumber", "number", "height", "index"}));
    j.timestamp = object_field_u64(obj, {"timestamp", "time", "ntime"});
    j.difficulty = object_field_u64(obj, {"difficulty", "block_difficulty", "blockDifficulty", "header_difficulty", "headerDifficulty"});
    j.clean_jobs = object_field_bool(obj, {"clean_jobs", "cleanJobs", "clean"}, false);

    if(j.job_id.empty()) return false;
    if(!parse_hex32(object_field_string(obj, {"previous_hash", "previousHash", "prev_hash", "prevHash", "prevhash"}), j.previous_hash)) return false;
    if(!parse_hex32(object_field_string(obj, {"merkle_root", "merkleRoot", "merkle"}), j.merkle_root)) return false;

    const std::string target_hex = object_field_string(obj, {"target", "target_be", "targetBe", "share_target", "shareTarget"});
    if(!target_hex.empty()) {
        if(!parse_hex32(target_hex, j.target_be)) return false;
        j.has_target = true;
    }

    if(j.timestamp == 0 || j.difficulty == 0) return false;
    if(!j.has_target) j.target_be = alpha_target_from_difficulty(j.difficulty);
    out = j;
    return true;
}

// Supported Alphanumeric notify arrays. Miningcore variants seen in the field:
//  A) [job_id, block_number, previous_hash, timestamp, difficulty, merkle_root, target, clean_jobs]
//  B) [job_id, block_number, previous_hash, merkle_root, timestamp, difficulty, clean_jobs]
//  C) [job_id, block_number, previous_hash, timestamp, merkle_root, difficulty, clean_jobs]
//
// The flexible parser below first tries the documented format, then infers the
// positions of merkle_root/timestamp/difficulty/target from token shape. This is
// intentional: Miningcore custom coins often send compact notify arrays, and the
// first live Alpha pool returned 7 params with the merkle root before timestamp.
static bool parse_array_job(const std::string& params_body, AlphaJob& out) {
    auto p = split_json_array_top_level(params_body);
    if(p.size() < 6) return false;

    AlphaJob j;
    j.job_id = unquote_json_string(p[0]);
    uint64_t block_number_u64 = 0;
    if(!try_parse_u64_token(p[1], block_number_u64)) return false;
    j.block_number = static_cast<uint32_t>(block_number_u64);
    if(!parse_hex32(p[2], j.previous_hash)) return false;

    // Live Miningcore Alpha compact format:
    // [job_id, height, previous_hash, merkle_root, difficulty, target, clean]
    // It does NOT include a timestamp. For this coin timestamp is part of the
    // 92-byte PoW header, so the miner must choose a Unix timestamp and submit
    // that exact timestamp back with the nonce.
    if(p.size() >= 7 && token_is_hex32(p[3])) {
        uint64_t diff = 0;
        std::array<uint8_t, 32> merkle{};
        std::array<uint8_t, 32> target{};
        if(parse_hex32(p[3], merkle) && try_parse_u64_token(p[4], diff) && parse_hex32(p[5], target)) {
            j.merkle_root = merkle;
            j.difficulty = diff;
            j.target_be = target;
            j.has_target = true;
            j.timestamp = unix_now_seconds();
            j.clean_jobs = parse_bool_token(p[6]);
            if(j.job_id.empty() || j.timestamp == 0 || j.difficulty == 0) return false;
            out = j;
            return true;
        }
    }

    // First try the original documented order. Use non-throwing parse so a
    // merkle-root token in p3 does not abort flexible detection.
    if(p.size() >= 6) {
        uint64_t ts = 0, diff = 0;
        std::array<uint8_t, 32> merkle{};
        if(try_parse_u64_token(p[3], ts) && try_parse_u64_token(p[4], diff) && parse_hex32(p[5], merkle)) {
            j.timestamp = ts;
            j.difficulty = diff;
            j.merkle_root = merkle;
            if(p.size() >= 7) {
                std::string t = trim_copy(p[6]);
                if(t != "null" && t != "\"\"" && !t.empty()) {
                    if(token_is_boolish(t)) {
                        j.clean_jobs = parse_bool_token(t);
                    } else if(parse_hex32(t, j.target_be)) {
                        j.has_target = true;
                    } else {
                        return false;
                    }
                }
            }
            if(p.size() >= 8) j.clean_jobs = parse_bool_token(p[7]);
            if(j.job_id.empty() || j.timestamp == 0 || j.difficulty == 0) return false;
            if(!j.has_target) j.target_be = alpha_target_from_difficulty(j.difficulty);
            out = j;
            return true;
        }
    }

    // Flexible detection for compact Miningcore custom notify arrays.
    bool have_merkle = false;
    bool have_timestamp = false;
    bool have_difficulty = false;

    for(size_t i = 3; i < p.size(); ++i) {
        std::string tok = trim_copy(p[i]);
        if(tok.empty() || tok == "null" || tok == "\"\"") continue;

        if(token_is_boolish(tok)) {
            j.clean_jobs = parse_bool_token(tok);
            continue;
        }

        if(token_is_hex32(tok)) {
            std::array<uint8_t, 32> bytes{};
            if(!parse_hex32(tok, bytes)) return false;
            if(!have_merkle) {
                j.merkle_root = bytes;
                have_merkle = true;
            } else if(!j.has_target) {
                j.target_be = bytes;
                j.has_target = true;
            }
            continue;
        }

        uint64_t v = 0;
        if(!try_parse_u64_token(tok, v)) continue;

        // Current Alpha chain timestamps are normal Unix seconds (~1.78B).
        // Difficulty is bounded by consensus to a few thousand points.
        if(v >= 1'000'000'000ull && !have_timestamp) {
            j.timestamp = v;
            have_timestamp = true;
        } else if(!have_difficulty) {
            j.difficulty = v;
            have_difficulty = true;
        } else if(!have_timestamp && v >= 1'000'000'000ull) {
            j.timestamp = v;
            have_timestamp = true;
        }
    }

    if(j.job_id.empty() || !have_merkle || !have_difficulty || j.difficulty == 0) {
        return false;
    }
    if(!have_timestamp || j.timestamp == 0) {
        // Miningcore Alpha jobs may omit timestamp. Pick an absolute Unix
        // timestamp locally and submit the same timestamp with the nonce.
        j.timestamp = unix_now_seconds();
    }

    if(!j.has_target) j.target_be = alpha_target_from_difficulty(j.difficulty);
    out = j;
    return true;
}

static bool is_notify_method_line(const std::string& line) {
    // Do not treat the subscribe response as a job. Miningcore subscribe replies
    // contain the literal strings "mining.notify" and "mining.set_difficulty"
    // inside the result array, but they are not notifications and have no params.
    return std::regex_search(line, std::regex(R"CAP("method"\s*:\s*"(mining\.notify|alphanumeric\.notify)")CAP"));
}

static std::string notify_params_body(const std::string& line) {
    std::smatch m;
    std::regex arr_re(R"CAP("params"\s*:\s*\[(.*)\]\s*(?:,\s*"id"|\}))CAP");
    if(std::regex_search(line, m, arr_re)) return m[1];
    return {};
}

static std::string describe_notify_shape(const std::string& line) {
    std::string body = notify_params_body(line);
    if(body.empty()) return "notify has no JSON array params";
    auto p = split_json_array_top_level(body);
    std::ostringstream os;
    os << "notify params count=" << p.size();
    for(size_t i = 0; i < p.size() && i < 10; ++i) {
        std::string token = p[i];
        if(token.size() > 96) token = token.substr(0, 96) + "...";
        os << " p" << i << "=" << token;
    }
    if(p.size() >= 9 && p[4].size() > 0 && p[4][0] == '[') {
        os << " | looks like standard Bitcoin/Miningcore Stratum notify: "
              "[job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean]. "
              "That is not the Alphanumeric custom 92-byte-header job format.";
    }
    return os.str();
}

static bool parse_alpha_notify(const std::string& line, AlphaJob& out) {
    if(!is_notify_method_line(line))
        return false;

    // Object style, easiest to implement in a Miningcore custom job manager:
    // {"method":"mining.notify","params":[{"job_id":"...", ...}]}
    std::smatch mo;
    std::regex obj_re(R"CAP("params"\s*:\s*\[\s*(\{.*\})\s*\])CAP");
    if(std::regex_search(line, mo, obj_re)) {
        return parse_object_job(mo[1], out);
    }

    // Raw object as params, accepted too:
    // {"method":"mining.notify","params":{"job_id":"...", ...}}
    std::regex obj_re2(R"CAP("params"\s*:\s*(\{.*\})\s*(?:,\s*"id"|\}))CAP");
    if(std::regex_search(line, mo, obj_re2)) {
        return parse_object_job(mo[1], out);
    }

    // Array style:
    // {"method":"mining.notify","params":["job",123,"prev",1800000000,464,"merkle","target",true]}
    std::regex arr_re(R"CAP("params"\s*:\s*\[(.*)\]\s*(?:,\s*"id"|\}))CAP");
    if(std::regex_search(line, mo, arr_re)) {
        try {
            return parse_array_job(mo[1], out);
        } catch(const std::exception& e) {
            log_line(std::string("Alphanumeric notify array parse error: ") + e.what());
            return false;
        }
    }
    return false;
}

static bool selftest(int dev) {
    std::array<uint8_t, 32> prev{};
    std::array<uint8_t, 32> merkle{};
    for(int i = 0; i < 32; ++i) {
        prev[i] = static_cast<uint8_t>(0x11 + i);
        merkle[i] = static_cast<uint8_t>(0xa0 + i);
    }
    const uint32_t block_number = 123456;
    const uint64_t timestamp = 1800000000ULL;
    const uint64_t nonce = 987654321ULL;
    const uint64_t difficulty = 464ULL;

    auto cpu = alphanumeric_cpu_hash92(block_number, prev, timestamp, nonce, difficulty, merkle);
    unsigned char gpu[32]{};
    if(!alphanumeric_cuda_hash92(dev, block_number, prev, timestamp, nonce, difficulty, merkle, gpu)) {
        log_line("Alphanumeric " + std::string(kGpuBackendName) + " self-test failed: GPU hash call failed");
        return false;
    }

    const bool ok = std::memcmp(cpu.data(), gpu, 32) == 0;
    log_line(std::string("Alphanumeric ") + kGpuBackendName + " self-test nonce=" + std::to_string(nonce) +
             " cpu=" + hex_bytes(cpu.data(), 8) + " gpu=" + hex_bytes(gpu, 8) +
             (ok ? " OK" : " MISMATCH"));
    return ok;
}

static std::string make_submit_params_for(const std::string& fmt,
                                          const std::string& user,
                                          const AlphaJob& job,
                                          const AlphanumericCudaResult& r) {
    const std::string nonce_dec = std::to_string(r.nonce);
    const std::string nonce_hex = hex64_le(r.nonce);
    const std::string hash_hex = hex_bytes(r.hash, 32);

    if(fmt == "compact") {
        return "[\"" + json_escape(user) + "\",\"" + json_escape(job.job_id) + "\"," + nonce_dec + "]";
    }
    if(fmt == "compact-hex") {
        return "[\"" + json_escape(user) + "\",\"" + json_escape(job.job_id) + "\",\"" + nonce_hex + "\"]";
    }
    if(fmt == "compact-decstr") {
        // Miningcore Alpha accepts [worker, jobId, nonce, timestamp?].
        // The timestamp is REQUIRED for block candidates: it is part of the 92-byte
        // PoW header. If we omit it, the pool rebuilds the header with its current
        // server time, so a miner-side block hash can become only a pool share.
        return "[\"" + json_escape(user) + "\",\"" + json_escape(job.job_id) + "\",\"" +
               nonce_dec + "\",\"" + std::to_string(job.timestamp) + "\"]";
    }
    if(fmt == "extended") {
        return "[\"" + json_escape(user) + "\",\"" + json_escape(job.job_id) + "\"," +
               nonce_dec + "," + std::to_string(job.timestamp) + "," +
               std::to_string(job.difficulty) + ",\"" + hash_hex + "\"]";
    }
    if(fmt == "extended-hex") {
        return "[\"" + json_escape(user) + "\",\"" + json_escape(job.job_id) + "\",\"" +
               nonce_hex + "\",\"" + std::to_string(job.timestamp) + "\",\"" +
               std::to_string(job.difficulty) + "\",\"" + hash_hex + "\"]";
    }
    if(fmt == "bitcoin" || fmt == "miningcore") {
        // Standard Miningcore/Bitcoin-style Stratum v1:
        // [worker, job_id, extranonce2, ntime, nonce].
        // Alphanumeric does not use extranonce2; keep it empty. Nonce is sent
        // as 16 lowercase hex chars for the traditional Stratum path.
        return "[\"" + json_escape(user) + "\",\"" + json_escape(job.job_id) + "\",\"\",\"" +
               std::to_string(job.timestamp) + "\",\"" + nonce_hex + "\"]";
    }
    if(fmt == "miningcore-dec") {
        return "[\"" + json_escape(user) + "\",\"" + json_escape(job.job_id) + "\",\"\",\"" +
               std::to_string(job.timestamp) + "\",\"" + nonce_dec + "\"]";
    }
    if(fmt == "object-camel") {
        return "[{\"user\":\"" + json_escape(user) +
               "\",\"jobId\":\"" + json_escape(job.job_id) +
               "\",\"nonce\":" + nonce_dec +
               ",\"nonceHex\":\"" + nonce_hex +
               "\",\"timestamp\":" + std::to_string(job.timestamp) +
               ",\"difficulty\":" + std::to_string(job.difficulty) +
               ",\"hash\":\"" + hash_hex + "\"}]";
    }
    if(fmt == "object-raw-camel") {
        return "{\"user\":\"" + json_escape(user) +
               "\",\"jobId\":\"" + json_escape(job.job_id) +
               "\",\"nonce\":" + nonce_dec +
               ",\"nonceHex\":\"" + nonce_hex +
               "\",\"timestamp\":" + std::to_string(job.timestamp) +
               ",\"difficulty\":" + std::to_string(job.difficulty) +
               ",\"hash\":\"" + hash_hex + "\"}";
    }

    // object / object-snake: array with a single snake_case object.
    return "[{\"user\":\"" + json_escape(user) +
           "\",\"job_id\":\"" + json_escape(job.job_id) +
           "\",\"nonce\":" + nonce_dec +
           ",\"nonce_hex\":\"" + nonce_hex +
           "\",\"timestamp\":" + std::to_string(job.timestamp) +
           ",\"difficulty\":" + std::to_string(job.difficulty) +
           ",\"hash\":\"" + hash_hex + "\"}]";
}

static std::vector<std::string> submit_formats_for_config(const MinerConfig& cfg) {
    const std::string fmt = cfg.alpha_submit_format.empty() ? "auto" : cfg.alpha_submit_format;
    if(fmt == "auto" || fmt == "probe") {
        // Submit the same valid nonce using the common Miningcore/custom parser
        // shapes. This is deliberately noisy but is the fastest way to identify
        // the server-side submit contract without rebuilding the miner after each
        // guess. Once one format accepts, rerun with that exact format.
        return {
            "miningcore",       // [user, job, ex2, ntime, nonce_hex]
            "miningcore-dec",   // [user, job, ex2, ntime, nonce_dec]
            "compact",          // [user, job, nonce_dec]
            "compact-decstr",   // [user, job, "nonce_dec"]
            "compact-hex",      // [user, job, "nonce_hex"]
            "extended",         // [user, job, nonce_dec, timestamp, difficulty, hash]
            "extended-hex",     // [user, job, nonce_hex, timestamp, difficulty, hash]
            "object",           // [{user, job_id, nonce, ...}]
            "object-camel",     // [{user, jobId, nonce, ...}]
            "object-raw-camel"  // {user, jobId, nonce, ...}
        };
    }
    if(fmt == "bitcoin") return {"miningcore"};
    return {fmt};
}


static int run_benchmark(const MinerConfig& cfg, int dev) {
    if(cfg.bench_sweep) {
        log_line("Alphanumeric benchmark sweep (threads=" + std::to_string(cfg.threads) +
                 " blocks_per_sm=" + std::to_string(cfg.blocks_per_sm) +
                 " seconds/config=" + std::to_string(cfg.bench_seconds) + ")");
        log_line("  batch          GH/s  avg_batch_ms  launches/s  accounting");
        bool any_fail = false;
        for(int lg = 24; lg <= 28; ++lg) {
            const AlphanumericBenchResult br = alphanumeric_cuda_benchmark_batch(
                dev, cfg.threads, cfg.blocks_per_sm, cfg.bench_seconds, 1ull << lg);
            if(!br.ok) {
                log_line("  2^" + std::to_string(lg) + "  BENCH FAILED (GPU backend error above)");
                any_fail = true;
                continue;
            }
            char msg[200];
            std::snprintf(msg, sizeof(msg), "  2^%-4d  %10.3f  %12.2f  %10.1f  %s",
                          lg, br.hashes_per_sec / 1e9, br.avg_batch_ms,
                          br.launches_per_sec, br.accounting_ok ? "OK" : "MISMATCH");
            log_line(msg);
            if(!br.accounting_ok) any_fail = true;
        }
        return any_fail ? 1 : 0;
    }

    const AlphanumericBenchResult br = alphanumeric_cuda_benchmark_batch(
        dev, cfg.threads, cfg.blocks_per_sm, cfg.bench_seconds, 1ull << cfg.bench_batch_log2);
    if(!br.ok) {
        log_line("Alphanumeric benchmark failed (GPU backend error above).");
        return 1;
    }
    char msg[240];
    std::snprintf(msg, sizeof(msg),
                  "Alphanumeric BLAKE3-92 benchmark: %.3f GH/s (%.0f MH/s)  batch=2^%d  "
                  "avg_batch=%.2f ms  launches/s=%.1f  accounting=%s",
                  br.hashes_per_sec / 1e9, br.hashes_per_sec / 1e6, cfg.bench_batch_log2,
                  br.avg_batch_ms, br.launches_per_sec, br.accounting_ok ? "OK" : "MISMATCH");
    log_line(msg);
    return br.accounting_ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// --verify N: on-device correctness suite (exit code 0 pass, 4 fail).
//
// Phase 1  N random single-nonce GPU hashes against the CPU byte-path
//          reference; the first four iterations pin the nonce edge cases
//          0, 2^32-1, 2^32 and 2^64-1.
// Phase 2  Planted-share scans. For a small random range the CPU computes
//          every hash and takes the byte-lexicographic minimum; with that
//          minimum as the target the GPU scan must report found=true with
//          exactly that nonce and hash (hash == target counts as a share,
//          and no other nonce in the range can qualify). A second scan of
//          the same range with target = minimum - 1 must report found=false
//          and account for every single nonce. Range starts include 0, a
//          range crossing the 2^32 boundary, and a range wrapping 2^64.
// Phase 3  Exact work accounting with an impossible all-zero target for
//          counts that exercise the partial-warp/stride tails.
// ---------------------------------------------------------------------------
static bool byte_dec_32(std::array<uint8_t, 32>& t) {
    for(int i = 31; i >= 0; --i) {
        if(t[i] != 0) { t[i]--; return true; }
        t[i] = 0xff;
    }
    return false; // was all-zero: nothing below it
}

static int run_verify(const MinerConfig& cfg, int dev) {
    const int n = cfg.alpha_verify;
    std::random_device rd;
    const uint64_t seed = (static_cast<uint64_t>(rd()) << 32) ^ rd();
    std::mt19937_64 rng(seed);
    log_line("Alphanumeric --verify " + std::to_string(n) +
             " starting (seed=" + std::to_string(seed) + ")");

    auto rand_bytes = [&](std::array<uint8_t, 32>& a) {
        for(auto& b : a) b = static_cast<uint8_t>(rng());
    };

    MinerStats stats;
    uint64_t p1_fail = 0;

    for(int i = 0; i < n; ++i) {
        std::array<uint8_t, 32> prev{};
        std::array<uint8_t, 32> merkle{};
        rand_bytes(prev);
        rand_bytes(merkle);
        const uint32_t bn = static_cast<uint32_t>(rng());
        const uint64_t ts = rng();
        const uint64_t dif = rng();
        uint64_t nonce = 0;
        switch(i) {
            case 0: nonce = 0; break;
            case 1: nonce = 0xffffffffULL; break;
            case 2: nonce = 0x100000000ULL; break;
            case 3: nonce = ~0ULL; break;
            default: nonce = rng(); break;
        }
        const auto cpu = alphanumeric_cpu_hash92(bn, prev, ts, nonce, dif, merkle);
        unsigned char gpu[32]{};
        if(!alphanumeric_cuda_hash92(dev, bn, prev, ts, nonce, dif, merkle, gpu) ||
           std::memcmp(cpu.data(), gpu, 32) != 0) {
            ++p1_fail;
            log_line("VERIFY FAIL phase1 i=" + std::to_string(i) +
                     " nonce=" + std::to_string(nonce) +
                     " cpu=" + hex_bytes(cpu.data(), 8) + " gpu=" + hex_bytes(gpu, 8));
            if(p1_fail > 8) break;
        }
        if((i + 1) % 500 == 0) {
            log_line("  phase1 " + std::to_string(i + 1) + "/" + std::to_string(n));
        }
    }
    log_line("Alphanumeric verify phase1 (GPU vs CPU single hash): " +
             std::to_string(n) + " vectors, " + std::to_string(p1_fail) + " failures");

    uint64_t p2_fail = 0;
    const int scans = std::max(4, std::min(64, n / 50));
    for(int s_idx = 0; s_idx < scans; ++s_idx) {
        std::array<uint8_t, 32> prev{};
        std::array<uint8_t, 32> merkle{};
        rand_bytes(prev);
        rand_bytes(merkle);
        const uint32_t bn = static_cast<uint32_t>(rng());
        const uint64_t ts = rng() & 0xffffffffffULL;
        const uint64_t dif = rng() & 0xffffffffULL;
        const uint64_t count = 4096 + (rng() % 61441); // 4096..65536
        uint64_t start = 0;
        switch(s_idx) {
            case 0: start = 0; break;
            case 1: start = 0x100000000ULL - 2048; break; // crosses the 32-bit line
            case 2: start = ~0ULL - 4095; break;          // wraps past 2^64
            default: start = rng(); break;
        }

        // CPU argmin over the exact range (uint64 wrap intended).
        std::array<uint8_t, 32> best{};
        uint64_t best_nonce = 0;
        for(uint64_t i = 0; i < count; ++i) {
            const uint64_t nonce = start + i;
            const auto h = alphanumeric_cpu_hash92(bn, prev, ts, nonce, dif, merkle);
            if(i == 0 || std::memcmp(h.data(), best.data(), 32) < 0) {
                best = h;
                best_nonce = nonce;
            }
        }

        AlphanumericCudaResult r{};
        if(!alphanumeric_cuda_scan92(dev, bn, prev, ts, dif, merkle, best, start, count,
                                     r, stats, cfg.threads, cfg.blocks_per_sm)) {
            ++p2_fail;
            log_line("VERIFY FAIL phase2 scan=" + std::to_string(s_idx) + " (GPU backend error)");
            continue;
        }
        const bool hit_ok = r.found && r.nonce == best_nonce &&
                            std::memcmp(r.hash, best.data(), 32) == 0 &&
                            r.hashes_scanned > 0 && r.hashes_scanned <= count;
        if(!hit_ok) {
            ++p2_fail;
            log_line("VERIFY FAIL phase2 scan=" + std::to_string(s_idx) +
                     " start=" + std::to_string(start) + " count=" + std::to_string(count) +
                     " found=" + std::to_string(r.found ? 1 : 0) +
                     " nonce=" + std::to_string(r.nonce) +
                     " expect=" + std::to_string(best_nonce) +
                     " scanned=" + std::to_string(r.hashes_scanned));
        }

        std::array<uint8_t, 32> miss = best;
        if(byte_dec_32(miss)) {
            AlphanumericCudaResult r2{};
            if(!alphanumeric_cuda_scan92(dev, bn, prev, ts, dif, merkle, miss, start, count,
                                         r2, stats, cfg.threads, cfg.blocks_per_sm)) {
                ++p2_fail;
                log_line("VERIFY FAIL phase2-miss scan=" + std::to_string(s_idx) + " (GPU backend error)");
                continue;
            }
            if(r2.found || r2.hashes_scanned != count) {
                ++p2_fail;
                log_line("VERIFY FAIL phase2-miss scan=" + std::to_string(s_idx) +
                         " found=" + std::to_string(r2.found ? 1 : 0) +
                         " scanned=" + std::to_string(r2.hashes_scanned) +
                         "/" + std::to_string(count));
            }
        }
    }
    log_line("Alphanumeric verify phase2 (planted-share scans): " + std::to_string(scans) +
             " ranges, " + std::to_string(p2_fail) + " failures");

    uint64_t p3_fail = 0;
    {
        std::array<uint8_t, 32> prev{};
        std::array<uint8_t, 32> merkle{};
        std::array<uint8_t, 32> zero_target{};
        rand_bytes(prev);
        rand_bytes(merkle);
        const uint64_t counts[3] = {1ull, 65543ull, 1ull << 22};
        for(const uint64_t c : counts) {
            AlphanumericCudaResult r{};
            const bool ok = alphanumeric_cuda_scan92(dev, 42, prev, 1800000000ULL, 464ULL,
                                                     merkle, zero_target, rng(), c, r, stats,
                                                     cfg.threads, cfg.blocks_per_sm);
            if(!ok || r.found || r.hashes_scanned != c) {
                ++p3_fail;
                log_line("VERIFY FAIL phase3 count=" + std::to_string(c) +
                         " found=" + std::to_string(r.found ? 1 : 0) +
                         " scanned=" + std::to_string(r.hashes_scanned));
            }
        }
    }
    log_line("Alphanumeric verify phase3 (exact work accounting): " +
             std::to_string(p3_fail) + " failures");

    const uint64_t total = p1_fail + p2_fail + p3_fail;
    log_line(total == 0 ? "Alphanumeric --verify PASS"
                        : "Alphanumeric --verify FAIL (" + std::to_string(total) + " failures)");
    return total == 0 ? 0 : 4;
}

static uint64_t initial_chunk_from_intensity(int intensity) {
    if(intensity < 1) intensity = 1;
    if(intensity > 30) intensity = 30;
    return 1ull << intensity;
}

int run(const MinerConfig& cfg) {
    using namespace std::chrono;

    if(!cfg.cuda) {
        log_line("Alphanumeric mode requires the compiled GPU backend; do not disable it.");
        return 2;
    }

    const int dev = cfg.devices.empty() ? 0 : cfg.devices[0];
    alphanumeric_cuda_list_devices();
    log_line("Alphanumeric " + std::string(kGpuBackendName) + " device: " + alphanumeric_cuda_device_name(dev));

    if(!selftest(dev)) {
        log_line("FATAL: GPU BLAKE3-92 output does not match CPU reference. Refusing to mine.");
        return 3;
    }

    if(cfg.alpha_verify > 0) return run_verify(cfg, dev);
    if(cfg.benchmark) return run_benchmark(cfg, dev);

    MinerStats stats;
    stats.set_context(cfg.pool, cfg.worker);
    stats.set_gpu_name(alphanumeric_cuda_device_name(dev));
    std::thread(dashboard_loop, std::ref(stats), dev, cfg.quiet_dashboard).detach();

    StratumClient sc;
    if(!sc.connect_to(cfg.pool)) { log_line("stratum connect failed"); return 4; }
    if(!sc.login(cfg.wallet, cfg.worker, cfg.pass)) { log_line("stratum login failed"); return 5; }

    std::mutex submit_format_mu;
    std::map<int, std::string> submit_format_by_id;
    std::atomic<bool> disable_local_target_scaling{false};

    sc.set_submit_result_handler([&](int id, bool accepted, const std::string& reason){
        std::string fmt;
        {
            std::lock_guard<std::mutex> lk(submit_format_mu);
            auto it = submit_format_by_id.find(id);
            if(it != submit_format_by_id.end()) {
                fmt = it->second;
                submit_format_by_id.erase(it);
            }
        }
        const std::string tag = fmt.empty() ? "" : (" format=" + fmt);
        if(accepted) {
            stats.accepted++;
            log_line("Alphanumeric share ACCEPTED (id=" + std::to_string(id) + tag + ")");
            if(cfg.alpha_submit_format == "auto" || cfg.alpha_submit_format == "probe" || cfg.alpha_submit_format.empty()) {
                log_line("ACCEPTED submit format found: " + fmt + "  <-- rerun with --alpha-submit-format " + fmt);
            }
        } else {
            stats.rejected++;
            log_line("Alphanumeric share REJECTED (id=" + std::to_string(id) + tag + ") reason=" + reason);

            std::string lower_reason = reason;
            for(char& c : lower_reason)
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if(lower_reason.find("low difficulty") != std::string::npos ||
               lower_reason.find("low-difficulty") != std::string::npos) {
                if(!disable_local_target_scaling.exchange(true, std::memory_order_acq_rel)) {
                    log_line("Pool rejected a locally scaled target; falling back to the exact mining.notify target");
                }
            }
        }
    });

    std::mutex job_mu;
    AlphaJob current_job;
    bool have_job = false;
    std::atomic<uint64_t> job_seq{0};
    std::atomic<bool> cancel{false};
    std::atomic<bool> running{true};
    std::atomic<double> pool_difficulty{1.0};

    std::thread net([&]{
        sc.read_loop([&](const std::string& line){
            if(cfg.debug_shares) {
                log_line("POOL RX: " + line);
            }

            // StratumClient parses mining.set_difficulty before invoking this
            // callback. Apply the new pool multiplier immediately, even if the
            // pool does not send a replacement notify at the same moment.
            if(is_set_difficulty_method_line(line)) {
                const double new_diff = sc.difficulty();
                pool_difficulty.store(new_diff, std::memory_order_release);
                stats.difficulty.store(new_diff);
                bool updated = false;
                AlphaJob updated_job;
                {
                    std::lock_guard<std::mutex> lk(job_mu);
                    if(have_job) {
                        apply_pool_share_target(
                            current_job, new_diff,
                            disable_local_target_scaling.load(std::memory_order_acquire));
                        stats.work_diff.store(target_work_diff(current_job.target_be));
                        updated_job = current_job;
                        updated = true;
                    }
                }
                if(updated) {
                    job_seq.fetch_add(1, std::memory_order_acq_rel);
                    cancel.store(true, std::memory_order_release);
                    // Reach into a batch already running on the GPU (drains in ~us).
                    alphanumeric_cuda_request_cancel();
                    log_line("Alphanumeric share target updated: pool_diff=" +
                             std::to_string(new_diff) +
                             " network_target=" + hex_bytes(updated_job.network_target_be.data(), 8) + "..." +
                             " share_target=" + hex_bytes(updated_job.target_be.data(), 8) + "...");
                }
                return;
            }

            AlphaJob j;
            if(!parse_alpha_notify(line, j)) {
                if(is_notify_method_line(line)) {
                    log_line("Alphanumeric notify received but could not parse it. RAW: " + line.substr(0, 1800));
                    log_line("Notify shape: " + describe_notify_shape(line));
                    log_line("Expected custom Alpha params: [job_id, block_number, previous_hash, timestamp, difficulty, merkle_root, target, clean_jobs] or one object with those fields.");
                }
                return;
            }
            const double pd = pool_difficulty.load(std::memory_order_acquire);
            apply_pool_share_target(
                j, pd, disable_local_target_scaling.load(std::memory_order_acquire));
            {
                std::lock_guard<std::mutex> lk(job_mu);
                current_job = j;
                have_job = true;
            }
            job_seq.fetch_add(1, std::memory_order_acq_rel);
            cancel.store(true, std::memory_order_release);
            // Reach into a batch already running on the GPU (drains in ~us).
            alphanumeric_cuda_request_cancel();
            stats.difficulty.store(pd);
            // Effective work per accepted share in units compatible with the
            // dashboard's 2^32-based estimator.
            stats.work_diff.store(target_work_diff(j.target_be));
            log_line("Alphanumeric job id=" + j.job_id +
                     " block=" + std::to_string(j.block_number) +
                     " timestamp=" + std::to_string(j.timestamp) +
                     " header_difficulty=" + std::to_string(j.difficulty) +
                     " pool_diff=" + std::to_string(pd) +
                     " network_target=" + hex_bytes(j.network_target_be.data(), 8) + "..." +
                     " share_target=" + hex_bytes(j.target_be.data(), 8) + "..." +
                     " target_source=" +
                         (j.local_scaling_disabled ? "notify-fallback" :
                          (j.explicit_share_target ? "notify" : "set_difficulty")));
        });
        running.store(false, std::memory_order_release);
    });

    log_line("waiting for Alphanumeric mining.notify jobs from Miningcore...");
    log_line("expected notify object: job_id, block_number/height, previous_hash, timestamp, difficulty, merkle_root, optional target");

    AlphaJob job;
    bool local_have = false;
    uint64_t next_nonce = 0;
    uint64_t chunk = initial_chunk_from_intensity(cfg.intensity);
    const uint64_t min_chunk = 1ull << 16;
    const uint64_t max_chunk = 1ull << 28;
    if(chunk < min_chunk) chunk = min_chunk;
    if(chunk > max_chunk) chunk = max_chunk;

    const std::string user = cfg.wallet + "." + cfg.worker;
    auto last_heartbeat = steady_clock::now();
    uint64_t heartbeat_hashes = stats.hashes.load();

    while(running.load(std::memory_order_acquire)) {
        if(cancel.load(std::memory_order_acquire) || !local_have) {
            std::lock_guard<std::mutex> lk(job_mu);
            if(have_job) {
                job = current_job;
                local_have = true;
                next_nonce = 0;
                cancel.store(false, std::memory_order_release);
            }
        }
        if(!local_have) {
            std::this_thread::sleep_for(milliseconds(10));
            continue;
        }

        const uint64_t my_seq = job_seq.load(std::memory_order_acquire);
        const uint64_t count = chunk;
        AlphanumericCudaResult r{};
        auto t0 = steady_clock::now();
        if(!alphanumeric_cuda_scan92(dev, job.block_number, job.previous_hash, job.timestamp,
                                     job.difficulty, job.merkle_root, job.target_be,
                                     next_nonce, count, r, stats,
                                     cfg.threads, cfg.blocks_per_sm)) {
            log_line(std::string("GPU scan failed; resetting ") + kGpuBackendName + " device before retry");
            if(!alphanumeric_cuda_reset(dev))
                log_line("GPU backend reset failed; another retry will be attempted");
            std::this_thread::sleep_for(milliseconds(250));
            continue;
        }

        const auto heartbeat_now = steady_clock::now();
        if(duration_cast<seconds>(heartbeat_now - last_heartbeat).count() >= 30) {
            const uint64_t total_hashes = stats.hashes.load();
            const double secs = duration_cast<duration<double>>(heartbeat_now - last_heartbeat).count();
            const double hps = secs > 0.0
                ? static_cast<double>(total_hashes - heartbeat_hashes) / secs
                : 0.0;
            char hb[240];
            std::snprintf(hb, sizeof(hb),
                          "Alphanumeric GPU heartbeat: %.2f GH/s job_seq=%llu chunk=%llu pool_diff=%.6f",
                          hps / 1e9,
                          static_cast<unsigned long long>(my_seq),
                          static_cast<unsigned long long>(chunk),
                          pool_difficulty.load(std::memory_order_acquire));
            log_line(hb);
            last_heartbeat = heartbeat_now;
            heartbeat_hashes = total_hashes;
        }

        if(cfg.batch_ms > 0) {
            double ms = duration_cast<duration<double, std::milli>>(steady_clock::now() - t0).count();
            if(ms > 0.5) {
                uint64_t nc = static_cast<uint64_t>((double)count * (double)cfg.batch_ms / ms);
                if(nc < min_chunk) nc = min_chunk;
                if(nc > max_chunk) nc = max_chunk;
                chunk = nc;
            }
        }

        if(r.found) {
            if(my_seq != job_seq.load(std::memory_order_acquire) || cancel.load(std::memory_order_acquire)) {
                if(cfg.debug_shares) log_line("stale Alphanumeric share skipped job=" + job.job_id);
            } else if(!hash_le_target_be(r.hash, job.target_be)) {
                log_line("BUG: GPU backend returned a hash above target; refusing submit");
            } else {
                if(hash_le_target_be(r.hash, job.network_target_be)) {
                    stats.blocks_found++;
                    log_line("Alphanumeric BLOCK CANDIDATE: share also meets network target job=" + job.job_id);
                }

                const auto formats = submit_formats_for_config(cfg);
                bool sent_any = false;
                for(const auto& fmt : formats) {
                    const std::string params = make_submit_params_for(fmt, user, job, r);
                    int sid = sc.submit_raw("mining.submit", params, cfg.debug_shares);
                    if(sid >= 0) {
                        {
                            std::lock_guard<std::mutex> lk(submit_format_mu);
                            submit_format_by_id[sid] = fmt;
                        }
                        stats.submitted++;
                        sent_any = true;
                        log_line("submitted Alphanumeric share id=" + std::to_string(sid) +
                                 " format=" + fmt +
                                 " job=" + job.job_id +
                                 " nonce=" + std::to_string(r.nonce) +
                                 " timestamp=" + std::to_string(job.timestamp) +
                                 " difficulty=" + std::to_string(job.difficulty) +
                                 " hash=" + hex_bytes(r.hash, 8) + "...");
                    }
                }
                if(sent_any) stats.shares_found++;
            }
            next_nonce = r.nonce + 1;
        } else {
            next_nonce += count;
        }
    }

    sc.close();
    if(net.joinable()) net.join();
    return 0;
}

} // namespace alphanumeric
