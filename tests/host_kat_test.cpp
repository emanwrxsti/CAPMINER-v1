// =============================================================================
// host_kat_test.cpp
//
// Host-compiled verification for the Alphanumeric CUDA miner. Compiles the
// EXACT device math (alphanumeric_blake3_core.cuh takes its plain-C++ path
// when __CUDACC__ is not defined) and checks it against the independent
// byte-path reference plus pure-logic models of the kernel's loop control.
//
// Build:  g++ -O2 -std=c++20 -I ../src -o host_kat_test host_kat_test.cpp
//         cl /O2 /std:c++20 /EHsc /I ..\src host_kat_test.cpp   (Windows)
// Run:    ./host_kat_test                 (full suite, exit 0 = pass)
//         ./host_kat_test --dump-blake3 <184 hex chars>    (92-byte header)
//         ./host_kat_test --hash-fields BN PREVHEX TS NONCE DIFF MERKLEHEX
//
// The two CLI modes exist for tests/cross_check_blake3.py, which drives this
// binary against the official blake3 Python library.
// =============================================================================

#include "alphanumeric/alphanumeric_blake3_core.cuh"
#include "alphanumeric/alphanumeric_ref_hash.hpp"

#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using alpha_core::AlphaJobPre;

static int g_failures = 0;

#define CHECK(cond, ...) do { \
    if(!(cond)) { \
        ++g_failures; \
        std::printf("FAIL %s:%d  ", __FILE__, __LINE__); \
        std::printf(__VA_ARGS__); \
        std::printf("\n"); \
        if(g_failures > 25) { std::printf("too many failures, aborting\n"); return 1; } \
    } \
} while(0)

static std::string hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    s.reserve(n * 2);
    for(size_t i = 0; i < n; ++i) {
        s.push_back(d[p[i] >> 4]);
        s.push_back(d[p[i] & 15]);
    }
    return s;
}

static bool unhex(const std::string& s, uint8_t* out, size_t n) {
    if(s.size() != n * 2) return false;
    auto nib = [](char c) -> int {
        if(c >= '0' && c <= '9') return c - '0';
        if(c >= 'a' && c <= 'f') return c - 'a' + 10;
        if(c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for(size_t i = 0; i < n; ++i) {
        const int hi = nib(s[i * 2]);
        const int lo = nib(s[i * 2 + 1]);
        if(hi < 0 || lo < 0) return false;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return true;
}

// Word-path hash of a full field set, via the same route the kernel takes:
// header words -> AlphaJobPre (host precompute) -> head+tail with injected nonce.
static void word_path_hash(uint32_t bn, const std::array<uint8_t, 32>& prev,
                           uint64_t ts, uint64_t nonce, uint64_t diff,
                           const std::array<uint8_t, 32>& merkle,
                           uint8_t out[32]) {
    uint32_t w[23];
    alpha_core::alpha_header_words_from_fields(bn, prev.data(), ts, diff, merkle.data(), w);
    uint32_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    AlphaJobPre jw{};
    alpha_core::alpha_build_job_pre(w, t, jw);
    uint32_t h[8];
    alpha_core::alpha_hash92_words(jw, nonce, h);
    for(int i = 0; i < 8; ++i) {
        out[i * 4 + 0] = (uint8_t)h[i];
        out[i * 4 + 1] = (uint8_t)(h[i] >> 8);
        out[i * 4 + 2] = (uint8_t)(h[i] >> 16);
        out[i * 4 + 3] = (uint8_t)(h[i] >> 24);
    }
}

// -----------------------------------------------------------------------------
// Test 1: shipped self-test vector, word path vs byte path.
// -----------------------------------------------------------------------------
static int test_selftest_vector() {
    std::array<uint8_t, 32> prev{};
    std::array<uint8_t, 32> merkle{};
    for(int i = 0; i < 32; ++i) {
        prev[i] = (uint8_t)(0x11 + i);
        merkle[i] = (uint8_t)(0xa0 + i);
    }
    const auto ref = alpha_ref::hash92_fields(123456u, prev, 1800000000ULL,
                                              987654321ULL, 464ULL, merkle);
    uint8_t wp[32];
    word_path_hash(123456u, prev, 1800000000ULL, 987654321ULL, 464ULL, merkle, wp);
    CHECK(std::memcmp(ref.data(), wp, 32) == 0,
          "selftest vector mismatch ref=%s word=%s",
          hex(ref.data(), 32).c_str(), hex(wp, 32).c_str());
    std::printf("test 1  selftest vector            %s\n", hex(wp, 16).c_str());
    return 0;
}

// -----------------------------------------------------------------------------
// Test 2: field serialization. alpha_header_words_from_fields must match the
// reference byte serializer word-for-word (nonce slots zero).
// -----------------------------------------------------------------------------
static int test_field_serialization(std::mt19937_64& rng) {
    for(int it = 0; it < 2000; ++it) {
        std::array<uint8_t, 32> prev{};
        std::array<uint8_t, 32> merkle{};
        for(auto& b : prev) b = (uint8_t)rng();
        for(auto& b : merkle) b = (uint8_t)rng();
        const uint32_t bn = (uint32_t)rng();
        const uint64_t ts = rng();
        const uint64_t diff = rng();

        uint8_t header[92];
        alpha_ref::build_header92(header, bn, prev, ts, 0ULL, diff, merkle);
        uint32_t expect[23];
        for(int i = 0; i < 23; ++i) expect[i] = alpha_ref::load32_le(header + i * 4);

        uint32_t got[23];
        alpha_core::alpha_header_words_from_fields(bn, prev.data(), ts, diff, merkle.data(), got);
        CHECK(std::memcmp(expect, got, sizeof(expect)) == 0,
              "field serialization mismatch it=%d", it);
        if(g_failures) return 1;
    }
    std::printf("test 2  field serialization        2000 random field sets\n");
    return 0;
}

// -----------------------------------------------------------------------------
// Test 3: 20,000 random word-path vs byte-path hashes (200 jobs x 100 nonces),
// with the four nonce edge cases pinned in every job.
// -----------------------------------------------------------------------------
static int test_random_kats(std::mt19937_64& rng) {
    const uint64_t edge_nonces[4] = {0ULL, 0xffffffffULL, 0x100000000ULL, ~0ULL};
    for(int job = 0; job < 200; ++job) {
        std::array<uint8_t, 32> prev{};
        std::array<uint8_t, 32> merkle{};
        for(auto& b : prev) b = (uint8_t)rng();
        for(auto& b : merkle) b = (uint8_t)rng();
        const uint32_t bn = (uint32_t)rng();
        const uint64_t ts = rng();
        const uint64_t diff = rng();

        uint32_t w[23];
        alpha_core::alpha_header_words_from_fields(bn, prev.data(), ts, diff, merkle.data(), w);
        uint32_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        AlphaJobPre jw{};
        alpha_core::alpha_build_job_pre(w, t, jw);
        CHECK(jw.m0[11] == 0 && jw.m0[12] == 0, "nonce slots must be zero in pre");

        for(int k = 0; k < 100; ++k) {
            const uint64_t nonce = (k < 4) ? edge_nonces[k] : rng();
            uint32_t h[8];
            alpha_core::alpha_hash92_words(jw, nonce, h);
            uint8_t wp[32];
            for(int i = 0; i < 8; ++i) {
                wp[i * 4 + 0] = (uint8_t)h[i];
                wp[i * 4 + 1] = (uint8_t)(h[i] >> 8);
                wp[i * 4 + 2] = (uint8_t)(h[i] >> 16);
                wp[i * 4 + 3] = (uint8_t)(h[i] >> 24);
            }
            const auto ref = alpha_ref::hash92_fields(bn, prev, ts, nonce, diff, merkle);
            CHECK(std::memcmp(ref.data(), wp, 32) == 0,
                  "KAT mismatch job=%d k=%d nonce=%" PRIu64 " ref=%s word=%s",
                  job, k, nonce, hex(ref.data(), 16).c_str(), hex(wp, 16).c_str());
            if(g_failures) return 1;
        }
    }
    std::printf("test 3  word path vs byte path     20000 vectors (200 jobs x 100 nonces)\n");
    return 0;
}

// -----------------------------------------------------------------------------
// Test 4: head/tail split. head() must return exactly h[0] of the full hash,
// and tail() must complete the identical remaining words from head()'s
// retained state.
// -----------------------------------------------------------------------------
static int test_head_tail_split(std::mt19937_64& rng) {
    for(int it = 0; it < 5000; ++it) {
        std::array<uint8_t, 32> prev{};
        std::array<uint8_t, 32> merkle{};
        for(auto& b : prev) b = (uint8_t)rng();
        for(auto& b : merkle) b = (uint8_t)rng();
        uint32_t w[23];
        alpha_core::alpha_header_words_from_fields((uint32_t)rng(), prev.data(), rng(),
                                                   rng(), merkle.data(), w);
        uint32_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        AlphaJobPre jw{};
        alpha_core::alpha_build_job_pre(w, t, jw);

        const uint64_t nonce = rng();
        uint32_t full[8];
        alpha_core::alpha_hash92_words(jw, nonce, full);

        uint32_t v[16];
        const uint32_t h0 = alpha_core::alpha_hash92_head(jw, (uint32_t)nonce,
                                                          (uint32_t)(nonce >> 32), v);
        CHECK(h0 == full[0], "head h0 %08x != full h[0] %08x it=%d", h0, full[0], it);
        uint32_t h[8];
        alpha_core::alpha_hash92_tail(jw, v, h);
        CHECK(std::memcmp(h, full, sizeof(full)) == 0, "tail completion mismatch it=%d", it);
        if(g_failures) return 1;
    }
    std::printf("test 4  head/tail split            5000 vectors\n");
    return 0;
}

// -----------------------------------------------------------------------------
// Test 5: target comparison. The word compare (fast word-0 test + tie tail)
// must equal byte-lexicographic memcmp(hash, target) <= 0 in every case,
// including exact equality, +/-1 neighbours, and forced word-0 ties.
// -----------------------------------------------------------------------------
static int test_target_compare(std::mt19937_64& rng) {
    auto bump = [](std::array<uint8_t, 32>& a, int dir) -> bool {
        for(int i = 31; i >= 0; --i) {
            if(dir > 0) {
                if(a[i] != 0xff) { a[i]++; return true; }
                a[i] = 0;
            } else {
                if(a[i] != 0) { a[i]--; return true; }
                a[i] = 0xff;
            }
        }
        return false;
    };

    long cases = 0;
    for(int it = 0; it < 4000; ++it) {
        std::array<uint8_t, 32> hb{};
        for(auto& b : hb) b = (uint8_t)rng();

        // Candidate targets: random, equal, +/-1, and a word-0 tie with a
        // random tail (exercises the two-stage compare split).
        std::array<uint8_t, 32> cands[5];
        cands[0] = hb;
        for(auto& b : cands[1]) b = (uint8_t)rng();
        cands[2] = hb; if(!bump(cands[2], +1)) cands[2] = hb;
        cands[3] = hb; if(!bump(cands[3], -1)) cands[3] = hb;
        cands[4] = hb; for(int i = 4; i < 32; ++i) cands[4][i] = (uint8_t)rng();

        uint32_t hw[8];
        for(int i = 0; i < 8; ++i) {
            hw[i] = ((uint32_t)hb[i * 4]) | ((uint32_t)hb[i * 4 + 1] << 8) |
                    ((uint32_t)hb[i * 4 + 2] << 16) | ((uint32_t)hb[i * 4 + 3] << 24);
        }
        for(const auto& tb : cands) {
            uint32_t tw[8];
            alpha_core::alpha_target_words_from_be_bytes(tb.data(), tw);
            const bool expect = std::memcmp(hb.data(), tb.data(), 32) <= 0;
            const bool got = alpha_core::alpha_hash_words_le_target(hw, tw);
            CHECK(got == expect, "target compare mismatch it=%d hash=%s target=%s",
                  it, hex(hb.data(), 32).c_str(), hex(tb.data(), 32).c_str());
            ++cases;
            if(g_failures) return 1;
        }
    }
    std::printf("test 5  target byte-lex compare    %ld cases\n", cases);
    return 0;
}

// -----------------------------------------------------------------------------
// Test 6: expected-iterations closed form used by the deficit accounting.
// expected(tid) = tid < count ? (count-1-tid)/stride + 1 : 0 must equal the
// brute-force number of loop iterations of for(idx=tid; idx<count; idx+=stride).
// -----------------------------------------------------------------------------
static int test_expected_formula(std::mt19937_64& rng) {
    for(int it = 0; it < 50000; ++it) {
        const uint32_t stride = 1u + (uint32_t)(rng() % (1u << 20));
        const uint32_t count = (uint32_t)(rng() % (1u << 22));
        const uint32_t tid = (uint32_t)(rng() % (stride * 2u + 4u));
        uint32_t brute = 0;
        for(uint64_t idx = tid; idx < count; idx += stride) ++brute;
        const uint32_t formula = (tid < count) ? (count - 1u - tid) / stride + 1u : 0u;
        CHECK(formula == brute, "expected formula tid=%u stride=%u count=%u f=%u b=%u",
              tid, stride, count, formula, brute);
        if(g_failures) return 1;
    }
    std::printf("test 6  expected-iterations form   50000 random (tid,stride,count)\n");
    return 0;
}

// -----------------------------------------------------------------------------
// Test 7: kernel loop-control simulator. Reproduces the exact control flow of
// alpha_scan_kernel (pre-increment poll, check-before-hash, break-on-found
// keeps bailed=false) and verifies, across many random launches with random
// bail points and found events:
//   a) with no early exit every index in [0,count) is hashed exactly once;
//   b) done = poll - bailed matches the number of hashed indices per thread;
//   c) sum(hashed) + sum(deficit) == count  (the accounting identity the
//      host relies on), whether threads bail, find, or run to completion.
// -----------------------------------------------------------------------------
struct SimThread {
    uint64_t hashed = 0;
    uint32_t poll = 0;
    bool bailed = false;
};

static int test_loop_simulator(std::mt19937_64& rng) {
    for(int it = 0; it < 3000; ++it) {
        const uint32_t threads = 32u << (rng() % 5);          // 32..512
        const uint32_t blocks = 1u + (uint32_t)(rng() % 48);
        const uint32_t stride = threads * blocks;
        const uint32_t count = (uint32_t)(rng() % (stride * 40u + 7u));

        // Global bail point: polls >= bail_at read found/gen as "stop" (only
        // observable at poll % 256 == 0). ~0 disables. found_idx: the index
        // whose hash "wins" (break after hashing). ~0 disables.
        const bool with_bail = (rng() & 3u) == 0u;
        const bool with_found = (rng() & 3u) == 1u;
        const uint32_t bail_at = with_bail ? 256u * (1u + (uint32_t)(rng() % 8)) : ~0u;
        const uint64_t found_idx = with_found && count ? rng() % count : ~0ull;

        std::vector<uint8_t> seen(count, 0);
        uint64_t hashed_total = 0;
        uint64_t deficit_total = 0;

        for(uint32_t tid = 0; tid < stride; ++tid) {
            SimThread st;
            for(uint64_t idx = tid; idx < count; idx += stride) {
                ++st.poll;
                if(((st.poll & 255u) == 0u) && st.poll >= bail_at) {
                    st.bailed = true;
                    break;
                }
                // "hash idx"
                CHECK(idx < count, "idx out of range");
                CHECK(seen[idx] == 0, "idx hashed twice: %" PRIu64, idx);
                seen[idx] = 1;
                ++st.hashed;
                if(idx == found_idx) break;   // found: hashed, bailed stays false
            }
            const uint32_t done = st.poll - (st.bailed ? 1u : 0u);
            CHECK(done == st.hashed, "done bookkeeping tid=%u done=%u hashed=%" PRIu64,
                  tid, done, st.hashed);
            const uint32_t expected = (tid < count) ? (count - 1u - tid) / stride + 1u : 0u;
            CHECK(expected >= done, "done exceeds expected tid=%u", tid);
            deficit_total += expected - done;
            hashed_total += st.hashed;
            if(g_failures) return 1;
        }

        CHECK(hashed_total + deficit_total == count,
              "accounting identity it=%d hashed=%" PRIu64 " deficit=%" PRIu64 " count=%u",
              it, hashed_total, deficit_total, count);
        if(!with_bail && !with_found) {
            CHECK(deficit_total == 0, "full scan must have zero deficit it=%d", it);
            for(uint32_t i = 0; i < count; ++i) {
                if(!seen[i]) { CHECK(false, "index %u never hashed", i); break; }
            }
        }
        if(g_failures) return 1;
    }
    std::printf("test 7  kernel loop simulator      3000 launches (coverage + accounting)\n");
    return 0;
}

// -----------------------------------------------------------------------------
// Test 8: slice planner. Reproduces scan92's >2^31 splitting and checks that
// the slices exactly partition [start, start+count) modulo 2^64 with every
// slice <= 2^31, including ranges that wrap past 2^64.
// -----------------------------------------------------------------------------
static int test_slice_planner(std::mt19937_64& rng) {
    const uint64_t SLICE_MAX = 1ull << 31;
    for(int it = 0; it < 20000; ++it) {
        uint64_t count;
        switch(it & 3) {
            case 0: count = rng() % (1ull << 20); break;
            case 1: count = (1ull << 31) + (rng() % (1ull << 20)) - (1ull << 19); break;
            case 2: count = rng() % (1ull << 34); break;
            default: count = rng() % (1ull << 36); break;
        }
        uint64_t start = (it % 5 == 0) ? (~0ull - (rng() % (1ull << 33))) : rng();

        uint64_t s = start;
        uint64_t remaining = count;
        uint64_t covered = 0;
        uint64_t expect_next = start;
        while(remaining != 0) {
            const uint64_t slice = remaining < SLICE_MAX ? remaining : SLICE_MAX;
            CHECK(slice >= 1 && slice <= SLICE_MAX, "slice size");
            CHECK(s == expect_next, "slice start discontinuity it=%d", it);
            expect_next = s + slice;    // u64 wrap intended
            covered += slice;
            s += slice;
            remaining -= slice;
            if(g_failures) return 1;
        }
        CHECK(covered == count, "coverage it=%d covered=%" PRIu64 " count=%" PRIu64,
              it, covered, count);
        CHECK(s == start + count, "end nonce it=%d", it);
        if(g_failures) return 1;
    }
    std::printf("test 8  slice planner              20000 ranges incl. 2^64 wrap\n");
    return 0;
}

// -----------------------------------------------------------------------------
// CLI modes for the python cross-check
// -----------------------------------------------------------------------------
static int mode_dump_blake3(const std::string& hex_header) {
    uint8_t header[92];
    if(!unhex(hex_header, header, 92)) {
        std::fprintf(stderr, "need exactly 184 hex chars\n");
        return 2;
    }
    uint8_t out[32];
    alpha_ref::blake3_hash_92(header, out);
    std::printf("%s\n", hex(out, 32).c_str());
    return 0;
}

static int mode_hash_fields(int argc, char** argv) {
    if(argc != 8) {
        std::fprintf(stderr, "--hash-fields BN PREVHEX TS NONCE DIFF MERKLEHEX\n");
        return 2;
    }
    const uint32_t bn = (uint32_t)std::strtoull(argv[2], nullptr, 10);
    std::array<uint8_t, 32> prev{};
    std::array<uint8_t, 32> merkle{};
    if(!unhex(argv[3], prev.data(), 32)) { std::fprintf(stderr, "bad prev hex\n"); return 2; }
    const uint64_t ts = std::strtoull(argv[4], nullptr, 10);
    const uint64_t nonce = std::strtoull(argv[5], nullptr, 10);
    const uint64_t diff = std::strtoull(argv[6], nullptr, 10);
    if(!unhex(argv[7], merkle.data(), 32)) { std::fprintf(stderr, "bad merkle hex\n"); return 2; }

    const auto ref = alpha_ref::hash92_fields(bn, prev, ts, nonce, diff, merkle);
    uint8_t wp[32];
    word_path_hash(bn, prev, ts, nonce, diff, merkle, wp);
    if(std::memcmp(ref.data(), wp, 32) != 0) {
        std::fprintf(stderr, "INTERNAL MISMATCH ref=%s word=%s\n",
                     hex(ref.data(), 32).c_str(), hex(wp, 32).c_str());
        return 3;
    }
    std::printf("%s\n", hex(wp, 32).c_str());
    return 0;
}

int main(int argc, char** argv) {
    if(argc >= 2 && std::string(argv[1]) == "--dump-blake3") {
        if(argc != 3) { std::fprintf(stderr, "--dump-blake3 <hex92>\n"); return 2; }
        return mode_dump_blake3(argv[2]);
    }
    if(argc >= 2 && std::string(argv[1]) == "--hash-fields") {
        return mode_hash_fields(argc, argv);
    }

    uint64_t seed = 0x1c3a5f7e9b2d4c68ull;
    if(argc >= 2) seed = std::strtoull(argv[1], nullptr, 0);
    std::mt19937_64 rng(seed);
    std::printf("host_kat_test seed=0x%" PRIx64 "\n", seed);

    if(test_selftest_vector()) return 1;
    if(test_field_serialization(rng)) return 1;
    if(test_random_kats(rng)) return 1;
    if(test_head_tail_split(rng)) return 1;
    if(test_target_compare(rng)) return 1;
    if(test_expected_formula(rng)) return 1;
    if(test_loop_simulator(rng)) return 1;
    if(test_slice_planner(rng)) return 1;

    if(g_failures == 0) {
        std::printf("ALL HOST TESTS PASSED\n");
        return 0;
    }
    std::printf("%d FAILURES\n", g_failures);
    return 1;
}
