#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct MinerConfig {
    std::string pool;
    std::string algo = "capstash"; // capstash or alphanumeric
    std::string wallet;
    std::string worker = "rig1";
    std::string pass = "x";
    std::vector<int> devices;
    int intensity = 24;
    bool cuda = true;
    bool opencl = true;
    std::string log_file;
    bool quiet_dashboard = false;   // redraw a clean fixed dashboard, suppress log scroll
    bool debug_shares = false;      // print full header/coinbase/merkle debug per share
    int  threads = 256;             // CUDA threads per block
    int  blocks_per_sm = 24;        // resident blocks per SM (grid-stride saturation)
    int  batch_ms = 15;             // target wall-time per GPU launch (responsiveness)
    bool benchmark = false;         // run CUDA throughput benchmark, then exit
    double bench_seconds = 3.0;     // seconds per benchmark config
    int  bench_batch_log2 = 24;     // benchmark nonces per launch = 2^N (alphanumeric)
    bool bench_sweep = false;       // benchmark a batch-size sweep 2^24..2^28, then exit
    int  alpha_verify = 0;          // run N GPU-vs-CPU verification vectors, then exit
    std::string alpha_submit_format = "auto"; // auto/probe, miningcore/bitcoin, compact, extended, object variants
};

MinerConfig parse_args(int argc, char** argv);
void print_usage();
