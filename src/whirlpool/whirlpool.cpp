#include "whirlpool.hpp"
#include "../stratum_client.hpp"
#include "../logger.hpp"
#include "../stats.hpp"
#include "../cuda_backend.hpp"
#include "capstash_job.hpp"
#include "capstash_pow.hpp"
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <array>
#include <string>
#include <unordered_set>

namespace whirlpool {

static std::array<uint8_t, 32> make_share_target(double difficulty) {
    if(difficulty <= 0.0) difficulty = 1.0;

    // Standard Bitcoin-style diff1 target:
    // 00000000ffff0000000000000000000000000000000000000000000000000000
    // Stored little-endian for uint256 comparison.
    std::array<uint8_t, 32> target{};
    target[26] = 0xff;
    target[27] = 0xff;

    uint64_t div = static_cast<uint64_t>(std::ceil(difficulty));
    if(div < 1) div = 1;

    uint64_t rem = 0;
    for(int i = 31; i >= 0; --i) {
        uint64_t cur = (rem << 8) | target[i];
        target[i] = static_cast<uint8_t>(cur / div);
        rem = cur % div;
    }
    return target;
}

// Bitcoin-style difficulty of a 32-byte little-endian hash: Diff1 / hashValue,
// where Diff1 = 0xFFFF * 2^208. Used only for diagnostics (comparing what the
// miner thinks a share is worth against what the pool scored it).
static double share_difficulty_of(const unsigned char h[32]) {
    double v = 0.0;
    for(int i = 31; i >= 0; --i) v = v * 256.0 + (double)h[i];
    if(v <= 0.0) return 0.0;
    return std::ldexp(65535.0, 208) / v;   // (0xFFFF << 208) / v
}

int run_stratum(const MinerConfig& cfg) {
    using namespace std::chrono;

    MinerStats stats;
    const int dev = cfg.devices.empty() ? 0 : cfg.devices[0];

    if(cfg.cuda) cuda_list_devices();
    if(cfg.cuda) cuda_print_kernel_info(dev, cfg.threads);

    stats.set_context(cfg.pool, cfg.worker);
    stats.set_gpu_name(cuda_device_name(dev));

    // Dashboard runs on its own thread; the mining thread never waits on it.
    std::thread(dashboard_loop, std::ref(stats), dev, cfg.quiet_dashboard).detach();

    StratumClient sc;
    if(!sc.connect_to(cfg.pool)) { log_line("stratum connect failed"); return 2; }
    if(!sc.login(cfg.wallet, cfg.worker, cfg.pass)) { log_line("stratum login failed"); return 3; }

    // Accepted/rejected are driven ONLY by correlated mining.submit responses.
    // On a "low difficulty" reject we parse the difficulty the POOL scored our
    // share at. Comparing that to the difficulty the miner computed for the same
    // share (logged at submit time, same id) tells us the root cause:
    //   * constant pool/miner ratio  -> a fixed share multiplier
    //   * random ratio (and pool score ~1e-9 with a normal share rate)
    //                                 -> the pool is hashing a DIFFERENT header
    //                                    (a header-construction mismatch).
    sc.set_submit_result_handler([&](int id, bool accepted, const std::string& reason){
        if(accepted) {
            stats.accepted++;
            log_line("share ACCEPTED (id=" + std::to_string(id) + ")");
            return;
        }
        stats.rejected++;
        double scored = 0.0;
        auto lp = reason.find('(');
        if(lp != std::string::npos) scored = std::strtod(reason.c_str() + lp + 1, nullptr);
        char b[200];
        std::snprintf(b, sizeof(b), "share REJECTED (id=%d) pool_scored_diff=%.4g reason=%s",
                      id, scored, reason.c_str());
        log_line(b);
    });

    // --- Shared state between the network thread and the mining thread --------
    std::mutex               job_mu;          // guards job_current / have_job
    StratumJob               job_current;
    bool                     have_job = false;
    std::atomic<uint64_t>    current_job_seq{0};   // ++ on every mining.notify
    std::atomic<bool>        cancel_current_job{false};
    std::atomic<bool>        dedup_clear_req{false};
    std::atomic<bool>        running{true};

    // Network thread: parse jobs, bump the job generation, request cancellation
    // of in-flight work. It never blocks on the GPU, so the socket never backs
    // up and the miner always sees the freshest job.
    std::thread net([&]{
        sc.read_loop([&](const std::string& line){
            StratumJob j;
            if(!parse_notify(line, j)) return;
            if(cfg.debug_shares) log_line("RAW NOTIFY: " + line.substr(0, 1000));
            {
                std::lock_guard<std::mutex> lk(job_mu);
                job_current = j;
                have_job = true;
            }
            if(j.clean_jobs) dedup_clear_req.store(true);   // tip moved: drop dedup keys
            current_job_seq.fetch_add(1);                   // (B) new generation
            cancel_current_job.store(true);                 // (B) cancel previous work
            log_line("job id=" + j.job_id + " nbits=" + j.nbits + " ntime=" + j.ntime +
                     " clean=" + (j.clean_jobs ? "1" : "0"));
        });
        running.store(false);   // connection closed -> stop mining
    });

    // The subscribe reply (which carries extranonce1) is parsed by the network
    // thread above. Wait for it before snapshotting extranonce1 -- otherwise the
    // coinbase is built with an empty extranonce1, producing a merkle root the
    // pool can't reproduce, so every share is rejected as "low difficulty".
    {
        int waited_ms = 0;
        while(running.load() && !sc.subscribed() && waited_ms < 15000) {
            std::this_thread::sleep_for(milliseconds(5));
            waited_ms += 5;
        }
        if(!sc.subscribed())
            log_line("WARNING: no subscribe reply (extranonce1) yet; mining may be rejected");
    }
    const std::string en1      = sc.extranonce1();
    const int         en2_size = sc.extranonce2_size();
    log_line("using extranonce1=" + en1 + " extranonce2_size=" + std::to_string(en2_size));

    // --- Mining thread (this thread) -----------------------------------------
    bool verified_once = false;
    std::unordered_set<std::string> submitted_keys;   // owned solely by this thread

    StratumJob job;                 // local snapshot of the job being scanned
    bool       local_have = false;
    std::string cur_job_id;
    uint32_t   cur_en2   = 0;
    uint64_t   next_nonce = 0;       // 0 .. 2^32 ; uint64 so the wrap test is exact
    double     cur_diff  = -1.0;
    std::array<uint8_t, 32> target{};
    bool       need_setup = false;

    auto clamp_u64 = [](uint64_t v, uint64_t lo, uint64_t hi){ return v < lo ? lo : (v > hi ? hi : v); };

    // Per-launch nonce count. Start from intensity but cap so a single launch
    // stays responsive and never overflows uint32. uint64 shift avoids UB.
    uint64_t chunk;
    {
        int it = cfg.intensity;
        uint64_t c = (it >= 31) ? (1ull << 31) : (1ull << it);
        chunk = clamp_u64(c, (1u << 16), (1u << 30));
    }

    const uint64_t NONCE_SPAN = (1ull << 32);

    log_line("waiting for CapStash mining.notify jobs...");

    while(running.load()) {
        // (B) Pick up the latest job whenever work was cancelled or we have none.
        if(cancel_current_job.load() || !local_have) {
            std::lock_guard<std::mutex> lk(job_mu);
            if(have_job) {
                job        = job_current;
                cur_job_id = job.job_id;
                local_have = true;
                cur_en2    = 0;          // (C) new job starts fresh
                next_nonce = 0;
                need_setup = true;
                cancel_current_job.store(false);
                if(dedup_clear_req.exchange(false)) submitted_keys.clear();  // (B) clean job
            }
        }
        if(!local_have) { std::this_thread::sleep_for(milliseconds(2)); continue; }

        if(submitted_keys.size() > 10000) submitted_keys.clear();   // (B) cap memory

        // (C) Difficulty change with no new job: update the target for future
        // scans but keep scanning from next_nonce (do not rewind / resubmit).
        double d = sc.difficulty();
        if(d != cur_diff) {
            cur_diff = d;
            target = make_share_target(d);
            need_setup = true;
            stats.difficulty.store(d);
            stats.work_diff.store(d);
        }

        if(need_setup) {
            auto merkle = build_merkle_root(job, en1, cur_en2, en2_size);
            auto header = build_header80(job, merkle, 0);
            if(!cuda_setup_job(dev, header, target)) {
                log_line("cuda_setup_job failed");
                std::this_thread::sleep_for(milliseconds(5));
                continue;
            }
            need_setup = false;

            if(!verified_once) {
                auto cpu = cap_pow_hash_header80(header);
                unsigned char gpu[32]{};
                cuda_hash_one_nonce(dev, 0, gpu);
                bool ok = (std::memcmp(cpu.data(), gpu, 32) == 0);
                log_line(std::string("verify nonce0 cpu=") + bytes_to_hex(cpu.data(), 8) +
                         " gpu=" + bytes_to_hex(gpu, 8) + (ok ? " OK" : " MISMATCH"));
                if(!ok) { log_line("FATAL: GPU hash != CPU hash; refusing to mine"); sc.close(); break; }

                // One-time breakdown of the header we actually hash (verbose),
                // so a header mismatch with the pool is inspectable.
                // prevhash(display) should equal the daemon's getbestblockhash.
                if(cfg.debug_shares) {
                    const uint8_t* H = header.data();
                    unsigned char ph_disp[32];
                    for(int b = 0; b < 32; ++b) ph_disp[b] = H[4 + 31 - b];
                    log_line("HEADER job=" + job.job_id + " ver=" + job.version + " nbits=" + job.nbits +
                             " ntime=" + job.ntime + " en2=" + format_extranonce2(cur_en2, en2_size) +
                             " merkle_branches=" + std::to_string(job.merkle_branch.size()));
                    log_line("  header80         : " + bytes_to_hex(H, 80));
                    log_line("  version   [0:4]  : " + bytes_to_hex(H + 0, 4));
                    log_line("  prevhash  [4:36] : " + bytes_to_hex(H + 4, 32));
                    log_line("  prevhash(display): " + bytes_to_hex(ph_disp, 32) + "  <- vs daemon getbestblockhash");
                    log_line("  merkle    [36:68]: " + bytes_to_hex(H + 36, 32));
                    log_line("  ntime     [68:72]: " + bytes_to_hex(H + 68, 4));
                    log_line("  nbits     [72:76]: " + bytes_to_hex(H + 72, 4));
                }
                verified_once = true;
            }
        }

        // One bounded GPU launch on the current job. We capture the generation
        // BEFORE launching so a job that arrives mid-scan invalidates the result.
        uint64_t my_seq    = current_job_seq.load();
        uint64_t remaining = NONCE_SPAN - next_nonce;
        uint32_t count     = (uint32_t)(remaining < chunk ? remaining : chunk);

        auto t0 = steady_clock::now();
        CudaResult r{};
        if(!cuda_mine_batch(dev, (uint32_t)next_nonce, count, r, stats, cfg.threads, cfg.blocks_per_sm)) {
            std::this_thread::sleep_for(milliseconds(5));
            continue;
        }

        // (A) Auto-size the next launch toward --batch-ms for responsiveness
        // without starving the GPU. Keeps the loop checking for new jobs often.
        if(cfg.batch_ms > 0) {
            double ms = duration_cast<duration<double, std::milli>>(steady_clock::now() - t0).count();
            if(ms > 0.5) {
                uint64_t nc = (uint64_t)((double)count * (double)cfg.batch_ms / ms);
                chunk = clamp_u64(nc, (1u << 16), (1u << 30));
            }
        }

        if(r.found) {
            char noncehex[9];
            std::snprintf(noncehex, sizeof(noncehex), "%08x", r.nonce);
            std::string ex2hex = format_extranonce2(cur_en2, en2_size);
            std::string key    = job.job_id + ":" + ex2hex + ":" + job.ntime + ":" + noncehex;

            // (B) Never submit a share from an old job. If a new generation
            // arrived (or cancellation was requested) while this batch ran, the
            // share is stale -- drop it.
            if(my_seq != current_job_seq.load() || cancel_current_job.load()) {
                if(cfg.debug_shares)
                    log_line("stale share skipped job=" + job.job_id + " nonce=" + noncehex);
            } else if(submitted_keys.count(key)) {
                if(cfg.debug_shares)
                    log_line("duplicate share skipped (" + key + ")");
            } else {
                // Does this share also meet the NETWORK target (nBits)? -> block.
                uint32_t nbits_val = parse_hex_u32_be_string_to_le_value(job.nbits);
                auto net_target = network_target_from_nbits_le(nbits_val);
                std::array<uint8_t, 32> hsh{};
                std::memcpy(hsh.data(), r.hash, 32);
                if(hash_leq_target_le(hsh, net_target)) {
                    stats.blocks_found++;
                    log_line("BLOCK FOUND: share also meets network target");
                }

                if(cfg.debug_shares) {
                    std::string cb = build_coinbase_hex(job, en1, cur_en2, en2_size);
                    std::string cb_first = cb.substr(0, std::min<size_t>(64, cb.size()));
                    std::string cb_last  = cb.size() > 64 ? cb.substr(cb.size() - 64) : cb;
                    auto dbg_merkle  = build_merkle_root(job, en1, cur_en2, en2_size);
                    auto dbg_header  = build_header80(job, dbg_merkle, r.nonce);
                    auto dbg_cpuhash = cap_pow_hash_header80(dbg_header);
                    bool selftest_ok = (std::memcmp(dbg_cpuhash.data(), r.hash, 32) == 0);

                    log_line("==================== SHARE DEBUG ====================");
                    log_line("extranonce1     : " + en1);
                    log_line("extranonce2     : " + ex2hex + "  (pool extranonce2_size=" +
                             std::to_string(en2_size) + ")");
                    log_line("nonce           : " + std::string(noncehex));
                    log_line("coinbase len    : " + std::to_string(cb.size()) +
                             " hex chars (" + std::to_string(cb.size() / 2) + " bytes)");
                    log_line("coinbase first64: " + cb_first);
                    log_line("coinbase last64 : " + cb_last);
                    log_line("merkle root     : " + bytes_to_hex(dbg_merkle.data(), 32));
                    log_line("header80        : " + bytes_to_hex(dbg_header.data(), 80));
                    log_line("  prevhash [4:36]: " + bytes_to_hex(dbg_header.data() + 4, 32));
                    {
                        unsigned char disp[32];
                        for(int b = 0; b < 32; ++b) disp[b] = *(dbg_header.data() + 4 + 31 - b);
                        log_line("  prevhash(display): " + bytes_to_hex(disp, 32) +
                                 "  <- compare to daemon getbestblockhash");
                    }
                    log_line("CPU hash (submit): " + bytes_to_hex(dbg_cpuhash.data(), 32));
                    log_line("GPU hash (found) : " + bytes_to_hex(r.hash, 32));
                    log_line(std::string("SELF-TEST        : ") +
                             (selftest_ok ? "OK" : "MISMATCH (bug is miner-side)"));
                    log_line("====================================================");
                }

                std::string user = cfg.wallet + "." + cfg.worker;
                int sid = sc.submit_share(user, job.job_id, ex2hex, job.ntime, noncehex, cfg.debug_shares);
                if(sid >= 0) {
                    submitted_keys.insert(key);   // (B) record only after sending
                    stats.submitted++;
                    stats.shares_found++;
                    // Diagnostic: what the miner thinks this share is worth. Pair
                    // this with the pool's "pool_scored_diff" for the same id. A
                    // constant ratio => share multiplier; a random ratio with
                    // pool scores ~1e-9 => the pool is hashing a different header.
                    if(cfg.debug_shares) {
                        char b[160];
                        std::snprintf(b, sizeof(b), "submit id=%d miner_diff=%.4g (need %.4g) nonce=%s",
                                      sid, share_difficulty_of(r.hash), sc.difficulty(), noncehex);
                        log_line(b);
                    }
                }
            }
        }

        next_nonce += count;
        if(next_nonce >= NONCE_SPAN) {
            // (C) Nonce space exhausted for this extranonce2 -> roll to the next
            // one (new coinbase/merkle/header) instead of reusing 0 forever.
            cur_en2++;
            next_nonce = 0;
            need_setup = true;
        }
    }

    sc.close();
    if(net.joinable()) net.join();
    return 0;
}
}

static int run_benchmark(const MinerConfig& cfg) {
    const int dev = cfg.devices.empty() ? 0 : cfg.devices[0];
    cuda_list_devices();
    log_line("CUDA benchmark (synthetic job, no pool) -- " +
             std::to_string(cfg.bench_seconds) + "s per config");
    cuda_print_kernel_info(dev, cfg.threads);

    double best = 0.0; int bt = 0, bb = 0;
    if(cfg.threads != 256 || cfg.blocks_per_sm != 24) {
        // User pinned a config: benchmark exactly that one.
        best = cuda_benchmark(dev, cfg.threads, cfg.blocks_per_sm, cfg.bench_seconds);
        bt = cfg.threads; bb = cfg.blocks_per_sm;
    } else {
        // Sweep common configs to find the best for this GPU.
        const int tset[] = {128, 256, 512};
        const int bset[] = {8, 12, 16, 24, 32};
        for(int t : tset) for(int b : bset) {
            double h = cuda_benchmark(dev, t, b, cfg.bench_seconds);
            if(h > best) { best = h; bt = t; bb = b; }
        }
    }
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "best: %.3f GH/s at --threads %d --blocks-per-sm %d", best / 1e9, bt, bb);
    log_line(msg);
    return 0;
}

int run(const MinerConfig& cfg) {
    if(cfg.benchmark) return run_benchmark(cfg);
    return whirlpool::run_stratum(cfg);
}
