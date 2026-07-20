#include "config.hpp"
#include "logger.hpp"
#include "whirlpool/whirlpool.hpp"
#include "alphanumeric/alphanumeric_runner.hpp"
#include <iostream>

int main(int argc, char** argv) {
    try {
        MinerConfig cfg = parse_args(argc, argv);
        log_init(cfg.log_file);
        console_init();
        if(cfg.quiet_dashboard) log_set_quiet(true);
        print_banner();
        log_line("capminer starting: Windows/Linux NVIDIA/AMD GPU-only, no dev fee, no CPU mining, no persistence");
        if(cfg.algo == "alphanumeric") {
            return alphanumeric::run(cfg);
        }
        if(!cfg.cuda && !cfg.opencl) {
            log_line("both CUDA and OpenCL disabled; nothing to mine with");
            return 1;
        }
        return run(cfg);
    } catch(const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n\n";
        print_usage();
        return 1;
    }
}
