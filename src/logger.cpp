#include "logger.hpp"
#include <fstream>
#include <iostream>
#include <mutex>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <deque>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

static std::mutex        g_mu;
static std::ofstream     g_file;
static std::atomic<bool> g_quiet{false};
static std::deque<std::string> g_ring;            // recent lines for the dashboard tail
static const size_t      g_ring_max = 256;

void log_init(const std::string& file) {
    if(!file.empty()) g_file.open(file, std::ios::app);
}

void console_init() {
#ifdef _WIN32
    // UTF-8 so box-drawing / bullet glyphs in the dashboard render correctly.
    SetConsoleOutputCP(CP_UTF8);
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    if(h && h != INVALID_HANDLE_VALUE) {
        DWORD mode = 0;
        if(GetConsoleMode(h, &mode)) {
            SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
    }
#endif
}

void log_set_quiet(bool quiet) { g_quiet.store(quiet); }

static std::string timestamp_now() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream os;
    os << std::put_time(&tm, "%F %T");
    return os.str();
}

void log_line(const std::string& msg) {
    std::lock_guard<std::mutex> lk(g_mu);
    std::string line = timestamp_now() + " " + msg;
    if(!g_quiet.load()) std::cout << line << std::endl;
    if(g_file) g_file << line << std::endl;
    g_ring.push_back(line);
    if(g_ring.size() > g_ring_max) g_ring.pop_front();
}

std::vector<std::string> log_recent(size_t n) {
    std::lock_guard<std::mutex> lk(g_mu);
    std::vector<std::string> out;
    size_t start = (g_ring.size() > n) ? (g_ring.size() - n) : 0;
    for(size_t i = start; i < g_ring.size(); ++i) out.push_back(g_ring[i]);
    return out;
}

void console_block(const std::string& text, bool clear_first) {
    std::lock_guard<std::mutex> lk(g_mu);
    if(clear_first) {
        // VT: clear screen + scrollback, move cursor home. Ignored harmlessly
        // by terminals that don't understand it.
        std::cout << "\x1b[2J\x1b[3J\x1b[H";
    }
    std::cout << text << std::flush;
}

void print_banner() {
    std::lock_guard<std::mutex> lk(g_mu);
    std::ostringstream os;
    os << "ICMINERS CAPMINER\n"
       << "Created by eman@icminers.com\n"
       << "0% Dev Fee\n"
       << "Windows/Linux GPU-only CapMiner\n";
    std::cout << os.str() << std::flush;
    if(g_file) g_file << os.str();
}
