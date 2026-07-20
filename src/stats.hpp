#pragma once
#include <atomic>
#include <cstdint>
#include <string>
#include <mutex>
#include <chrono>

// All counters are atomic so the mining/network threads can update them while
// the dashboard thread reads them. Display context (pool/worker/gpu name) is
// set rarely and guarded by a small mutex.
struct MinerStats {
    // Hot counters.
    std::atomic<uint64_t> hashes{0};        // total hashes tried (drives hashrate)
    std::atomic<uint64_t> shares_found{0};  // local candidates that met the share target
    std::atomic<uint64_t> submitted{0};     // mining.submit messages actually sent
    std::atomic<uint64_t> accepted{0};      // submit responses with result=true
    std::atomic<uint64_t> rejected{0};      // submit responses with result=false (or error)
    std::atomic<uint64_t> blocks_found{0};  // submitted hash also met the NETWORK target (nBits)
    std::atomic<uint64_t> blocks_won{0};    // confirmed by pool/daemon (rarely signalled on stratum v1)

    std::atomic<double>   difficulty{1.0};  // current pool share difficulty (display)
    std::atomic<double>   work_diff{1.0};    // Bitcoin-difficulty of accepted shares
                                             // (= pool difficulty * learned multiplier)

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    // Display-only context. Guarded by mu.
    std::mutex  mu;
    std::string pool;
    std::string worker;
    std::string gpu_name = "GPU";

    void set_context(const std::string& p, const std::string& w) {
        std::lock_guard<std::mutex> lk(mu); pool = p; worker = w;
    }
    void set_gpu_name(const std::string& g) {
        std::lock_guard<std::mutex> lk(mu); gpu_name = g;
    }
    void get_context(std::string& p, std::string& w, std::string& g) {
        std::lock_guard<std::mutex> lk(mu); p = pool; w = worker; g = gpu_name;
    }
};

// Dashboard refresh loop, prints every 5 seconds.
//   nvml_index : GPU index used for NVML telemetry (power/temp/fan).
//   quiet      : if true, redraw a clean fixed dashboard (clear screen each
//                refresh) and suppress normal per-line logging to stdout.
void dashboard_loop(MinerStats& s, int nvml_index, bool quiet);
