#include "config.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cctype>

static std::vector<int> parse_devices(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, ',')) {
        if(!item.empty()) out.push_back(std::stoi(item));
    }
    return out;
}

void print_usage() {
    std::cout <<
R"(capminer --pool stratum+tcp://host:port --wallet WALLET --worker rig1 [options]

Required:
  --pool       Stratum URL, example stratum+tcp://us.icminers.com:PORT
  --wallet     Wallet address

Options:
  --algo       Mining engine: capstash or alphanumeric, default capstash
  --coin       Alias for --algo
  --worker     Worker name, default rig1
  --pass       Stratum password, default x
  --devices    Comma list, example 0,1
  --intensity  GPU launch intensity (nonces/launch = 2^intensity), default 24
  --batch-ms   Target milliseconds per GPU launch, default 15 (auto-sizes work)
  --threads    GPU threads per block, default 256
  --blocks-per-sm  Resident blocks per SM/CU, default 24
  --debug-shares   Print full header/coinbase/merkle debug for every share
  --benchmark      Run a GPU throughput benchmark (no pool needed) and exit
  --bench-seconds  Seconds per benchmark config, default 3
  --bench-batch-log2 N  Alphanumeric benchmark batch = 2^N nonces/launch, default 24
  --bench-sweep    Alphanumeric: benchmark batch sizes 2^24..2^28 and print a table
  --verify N       Alphanumeric: run N random GPU-vs-CPU hash checks plus
                   planted-share scan and accounting tests on the GPU, then exit
  --alpha-submit-format  Alphanumeric submit params: auto/probe, miningcore, miningcore-dec, compact, compact-hex, compact-decstr, extended, extended-hex, object, object-camel, object-raw-camel; default auto
  --no-cuda    Disable the compiled CUDA/HIP GPU backend
  --no-hip     Alias for --no-cuda in an AMD HIP build
  --no-opencl  Disable legacy OpenCL detection
  --log-file   Write log output to file
  --quiet-dashboard  Show only the status dashboard (clean, redraws in place)
)";
}

MinerConfig parse_args(int argc, char** argv) {
    MinerConfig cfg;
    for(int i=1;i<argc;i++) {
        std::string a = argv[i];
        auto need = [&](const char* name)->std::string {
            if(i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if(a=="--pool") cfg.pool = need("--pool");
        else if(a=="--algo" || a=="--coin") cfg.algo = need(a.c_str());
        else if(a=="--wallet") cfg.wallet = need("--wallet");
        else if(a=="--worker") cfg.worker = need("--worker");
        else if(a=="--pass") cfg.pass = need("--pass");
        else if(a=="--devices") cfg.devices = parse_devices(need("--devices"));
        else if(a=="--intensity") cfg.intensity = std::stoi(need("--intensity"));
        else if(a=="--batch-ms") cfg.batch_ms = std::stoi(need("--batch-ms"));
        else if(a=="--threads") cfg.threads = std::stoi(need("--threads"));
        else if(a=="--blocks-per-sm") cfg.blocks_per_sm = std::stoi(need("--blocks-per-sm"));
        else if(a=="--debug-shares") cfg.debug_shares = true;
        else if(a=="--benchmark") cfg.benchmark = true;
        else if(a=="--bench-seconds") cfg.bench_seconds = std::stod(need("--bench-seconds"));
        else if(a=="--bench-batch-log2") cfg.bench_batch_log2 = std::stoi(need("--bench-batch-log2"));
        else if(a=="--bench-sweep") cfg.bench_sweep = true;
        else if(a=="--verify") cfg.alpha_verify = std::stoi(need("--verify"));
        else if(a=="--alpha-submit-format") cfg.alpha_submit_format = need("--alpha-submit-format");
        else if(a=="--no-cuda" || a=="--no-hip" || a=="--no-gpu") cfg.cuda = false;
        else if(a=="--no-opencl") cfg.opencl = false;
        else if(a=="--log-file") cfg.log_file = need("--log-file");
        else if(a=="--quiet-dashboard") cfg.quiet_dashboard = true;
        else if(a=="--help" || a=="-h") { print_usage(); std::exit(0); }
        else throw std::runtime_error("unknown arg: " + a);
    }
    for(char& c : cfg.algo) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if(cfg.algo != "capstash" && cfg.algo != "alphanumeric") {
        throw std::runtime_error("--algo must be capstash or alphanumeric");
    }
    if(cfg.bench_sweep) cfg.benchmark = true;   // sweep is a benchmark mode
    if(cfg.alpha_verify < 0) cfg.alpha_verify = 0;
    if(cfg.bench_batch_log2 < 16) { cfg.bench_batch_log2 = 16; std::cerr << "warning: --bench-batch-log2 raised to 16\n"; }
    if(cfg.bench_batch_log2 > 31) { cfg.bench_batch_log2 = 31; std::cerr << "warning: --bench-batch-log2 capped at 31 (one launch slice)\n"; }
    const bool offline_mode = cfg.benchmark || cfg.alpha_verify > 0;
    if(!offline_mode) {
        if(cfg.pool.empty()) throw std::runtime_error("--pool is required");
        if(cfg.wallet.empty()) throw std::runtime_error("--wallet is required");
    }
    // Intensity sets nonces-per-launch = 2^intensity, but a single launch is
    // chunked and capped internally (see whirlpool.cpp) so it can never overflow
    // uint32 nonce arithmetic. Allow a wide range; the miner clamps the chunk.
    // Intensity sets nonces-per-launch = 2^intensity, but a single launch is
    // chunk-auto-sized to --batch-ms anyway, so very high values do nothing.
    if(cfg.intensity < 1)  { cfg.intensity = 1;  std::cerr << "warning: --intensity raised to 1\n"; }
    if(cfg.intensity > 40) {
        std::cerr << "warning: --intensity " << cfg.intensity << " is above the cap (40); clamping. "
                     "Note: chunks auto-size to --batch-ms, so values above ~22 have no effect on throughput.\n";
        cfg.intensity = 40;
    } else if(cfg.intensity > 30) {
        std::cerr << "warning: --intensity " << cfg.intensity
                  << " has no real effect because chunks auto-size to --batch-ms (default 15 ms). "
                     "Hashrate is set by the kernel and GPU, not by this flag.\n";
    }
    if(cfg.batch_ms < 0)   cfg.batch_ms = 0;
    if(cfg.threads < 32)   cfg.threads = 32;
    if(cfg.threads > 1024) cfg.threads = 1024;
    if(cfg.blocks_per_sm < 1)   cfg.blocks_per_sm = 1;
    if(cfg.blocks_per_sm > 256) cfg.blocks_per_sm = 256;
    if(cfg.devices.size() > 1) {
        std::cerr << "warning: this build mines on the first --devices entry only; "
                     "run one capminer process per GPU for multi-GPU systems\n";
    }
    for(char& c : cfg.alpha_submit_format) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if(cfg.alpha_submit_format != "auto" && cfg.alpha_submit_format != "probe" &&
       cfg.alpha_submit_format != "miningcore" && cfg.alpha_submit_format != "miningcore-dec" &&
       cfg.alpha_submit_format != "object" && cfg.alpha_submit_format != "object-camel" &&
       cfg.alpha_submit_format != "object-raw-camel" &&
       cfg.alpha_submit_format != "extended" && cfg.alpha_submit_format != "extended-hex" &&
       cfg.alpha_submit_format != "compact" && cfg.alpha_submit_format != "compact-hex" &&
       cfg.alpha_submit_format != "compact-decstr" && cfg.alpha_submit_format != "bitcoin") {
        throw std::runtime_error("--alpha-submit-format must be auto/probe, miningcore, miningcore-dec, bitcoin, object, object-camel, object-raw-camel, extended, extended-hex, compact, compact-hex, or compact-decstr");
    }
    return cfg;
}
