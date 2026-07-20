#include "stats.hpp"
#include "logger.hpp"
#include "nvml_telemetry.hpp"
#include <thread>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <vector>
#include <string>

#ifdef _WIN32
#include <io.h>
#define CAP_ISATTY() (_isatty(_fileno(stdout)) != 0)
#else
#include <unistd.h>
#define CAP_ISATTY() (isatty(fileno(stdout)) != 0)
#endif

// ---- ANSI palette ---------------------------------------------------------
namespace ui {
    static const char* RST = "\x1b[0m";
    static const char* B   = "\x1b[1m";
    static const char* DIM = "\x1b[2m";
    static const char* CY  = "\x1b[96m";  // bright cyan  (labels / frame)
    static const char* GN  = "\x1b[92m";  // bright green (good numbers)
    static const char* YL  = "\x1b[93m";  // bright yellow(power/temp)
    static const char* RD  = "\x1b[91m";  // bright red   (rejects)
    static const char* WT  = "\x1b[97m";  // bright white (values)
    static const char* MG  = "\x1b[95m";  // magenta      (verify)
    static const char* GY  = "\x1b[90m";  // grey         (timestamps)
    static const char* EOL = "\x1b[K";    // clear to end of line
}

static const int    PANEL_W   = 60;
static const size_t TAIL_LINES = 12;

static std::string fmt_hashrate(double hps) {
    char buf[64];
    if(hps >= 1e9)      std::snprintf(buf, sizeof(buf), "%.2f GH/s", hps / 1e9);
    else if(hps >= 1e6) std::snprintf(buf, sizeof(buf), "%.2f MH/s", hps / 1e6);
    else if(hps >= 1e3) std::snprintf(buf, sizeof(buf), "%.2f KH/s", hps / 1e3);
    else                std::snprintf(buf, sizeof(buf), "%.2f H/s",  hps);
    return std::string(buf);
}

static std::string fmt_uptime(long long secs) {
    if(secs < 0) secs = 0;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02lld:%02lld:%02lld",
                  secs/3600, (secs%3600)/60, secs%60);
    return std::string(buf);
}

static std::string rep(const char* unit, int n) {
    std::string s; s.reserve((size_t)n * 3);
    for(int i=0;i<n;++i) s += unit;
    return s;
}
static std::string pad(std::string s, size_t w) {
    if(s.size() < w) s.append(w - s.size(), ' ');
    return s;
}

// One "Label  value" cell: coloured label (cyan) + coloured value.
static std::string cell(const std::string& label, const std::string& value,
                        const char* vcolor, size_t labw, size_t valw) {
    std::string s = ui::CY + pad(label, labw) + ui::RST + vcolor + pad(value, valw) + ui::RST;
    return s;
}

// Colour a captured log line by keyword, then truncate to the panel width.
static std::string colorize_log(const std::string& line) {
    const char* col = ui::WT;
    if(line.find("ACCEPTED") != std::string::npos)      col = ui::GN;
    else if(line.find("REJECT")   != std::string::npos) col = ui::RD;
    else if(line.find("verify")   != std::string::npos) col = ui::MG;
    else if(line.find("job id")   != std::string::npos) col = ui::CY;
    else if(line.find("difficulty")!=std::string::npos) col = ui::YL;

    // timestamp (first 19 chars "YYYY-MM-DD HH:MM:SS") dimmed, rest coloured.
    std::string ts, rest;
    if(line.size() > 20) { ts = line.substr(0,19); rest = line.substr(19); }
    else                 { rest = line; }
    std::string body = rest;
    size_t maxbody = 74;                       // keep within a typical 80-col window
    if(body.size() > maxbody) body = body.substr(0, maxbody-1) + "\xE2\x80\xA6"; // ...
    std::string out = " ";
    if(!ts.empty()) out += std::string(ui::GY) + ts + ui::RST;
    out += col + body + ui::RST;
    return out;
}

void dashboard_loop(MinerStats& s, int nvml_index, bool quiet) {
    using namespace std::chrono;
    using namespace ui;

    const bool tty = CAP_ISATTY();

    NvmlTelemetry nvml;
    nvml.init(nvml_index < 0 ? 0u : (unsigned int)nvml_index);

    uint64_t last_hashes = s.hashes.load();
    auto     last_t      = steady_clock::now();
    bool     first       = true;

    const std::string barflat(PANEL_W, '=');   // plain bar for non-tty fallback

    while(true) {
        std::this_thread::sleep_for(seconds(5));

        auto   now = steady_clock::now();
        double dt  = duration_cast<duration<double>>(now - last_t).count();
        if(dt <= 0.0) dt = 1e-9;
        uint64_t h   = s.hashes.load();
        double   hps = (double)(h - last_hashes) / dt;
        last_hashes  = h; last_t = now;

        long long up = duration_cast<seconds>(now - s.start).count();
        double eff = (up > 0)
            ? (double)s.accepted.load() * s.work_diff.load() * 4294967296.0 / (double)up
            : 0.0;

        std::string pool, worker, gpu;
        s.get_context(pool, worker, gpu);

        GpuTelemetry t = nvml.sample();
        std::string power = t.has_power ? (std::to_string((t.power_mw + 500)/1000) + " W") : "N/A";
        std::string temp  = t.has_temp  ? (std::to_string(t.temp_c) + " C") : "N/A";
        std::string fan   = t.has_fan   ? (std::to_string(t.fan_pct) + " %") : "N/A";

        char diffbuf[32];
        std::snprintf(diffbuf, sizeof(diffbuf), "%.6f", s.difficulty.load());

        // -------- non-interactive (piped to file): plain block, logs scroll --
        if(!tty) {
            std::ostringstream os;
            os << barflat << "\n"
               << "ICMINERS CAPMINER | Created by eman@icminers.com | 0% Dev Fee\n"
               << barflat << "\n"
               << "Pool: "       << (pool.empty()?"-":pool) << "\n"
               << "Worker: "     << (worker.empty()?"-":worker) << "\n"
               << "GPU: "        << gpu << "\n"
               << "Uptime: "     << fmt_uptime(up) << "\n"
               << "Difficulty: " << diffbuf << "\n"
               << "Hashrate: "   << fmt_hashrate(hps) << "\n"
               << "Effective: "  << fmt_hashrate(eff) << " (accepted)\n"
               << "Shares: accepted=" << s.accepted.load()
                      << " rejected=" << s.rejected.load()
                      << " submitted=" << s.submitted.load() << "\n"
               << "Blocks/Won: " << s.blocks_found.load() << " / " << s.blocks_won.load() << "\n"
               << "Power: " << power << "  Temp: " << temp << "  Fan: " << fan << "\n"
               << barflat << "\n";
            console_block(os.str(), /*clear_first=*/false);
            continue;
        }

        // -------- interactive: fixed colour TUI, banner pinned at the top -----
        // First draw: take over the screen and silence raw log scrolling (the
        // captured lines are shown in the activity tail instead).
        if(first) log_set_quiet(true);

        const std::string top = rep("\xE2\x95\x90", PANEL_W); // heavy double rule
        std::ostringstream os;
        if(first) os << "\x1b[2J";          // one-time full clear
        os << "\x1b[H";                      // home -> redraw in place (no flicker)

        os << CY << top << RST << EOL << "\n";
        os << " " << B << YL << "ICMINERS CAPMINER" << RST
           << DIM << WT << "  |  " << RST
           << CY  << "Created by eman@icminers.com" << RST
           << DIM << WT << "  |  " << RST
           << B   << GN << "0% Dev Fee" << RST << EOL << "\n";
        os << CY << top << RST << EOL << "\n";

        os << " " << cell("Pool",   pool.empty()?"-":pool, WT, 11, 0) << EOL << "\n";
        os << " " << cell("Worker", worker.empty()?"-":worker, WT, 11, 0) << EOL << "\n";
        os << " " << cell("GPU",    gpu, WT, 11, 0) << EOL << "\n";
        os << " " << cell("Uptime", fmt_uptime(up), WT, 11, 16)
                  << cell("Difficulty", diffbuf, WT, 12, 0) << EOL << "\n";
        os << " " << cell("Hashrate", fmt_hashrate(hps), GN, 11, 16)
                  << cell("Effective", fmt_hashrate(eff), GN, 12, 0) << EOL << "\n";

        {
            uint64_t acc = s.accepted.load(), rej = s.rejected.load(), sub = s.submitted.load();
            os << " " << CY << pad("Shares", 11) << RST
               << WT << "accepted " << RST << GN << acc << RST
               << WT << "  rejected " << RST << (rej? RD : GY) << rej << RST
               << WT << "  submitted " << RST << WT << sub << RST << EOL << "\n";
        }
        {
            std::string bw = std::to_string(s.blocks_found.load()) + " / " + std::to_string(s.blocks_won.load());
            os << " " << cell("Blocks/Won", bw, WT, 11, 0) << EOL << "\n";
        }
        os << " " << cell("Power", power, YL, 11, 16)
                  << CY << pad("Temp",6) << RST << YL << pad(temp,9) << RST
                  << CY << pad("Fan",5) << RST << WT << fan << RST << EOL << "\n";

        // activity tail
        if(!quiet) {
            std::string lbl = " recent activity ";
            int dash = PANEL_W - (int)lbl.size(); if(dash < 0) dash = 0;
            os << GY << rep("\xE2\x94\x80", 1) << RST << DIM << lbl << RST
               << GY << rep("\xE2\x94\x80", dash > 1 ? dash-1 : 0) << RST << EOL << "\n";
            std::vector<std::string> recent = log_recent(TAIL_LINES);
            for(const auto& ln : recent) os << colorize_log(ln) << EOL << "\n";
            // pad to a stable height so the panel doesn't jump
            for(size_t i = recent.size(); i < TAIL_LINES; ++i) os << EOL << "\n";
        }

        os << "\x1b[J";                       // erase anything below the panel
        console_block(os.str(), /*clear_first=*/false);
        first = false;
    }
}
