#pragma once
#include <string>
#include <vector>

void log_init(const std::string& file);
void log_line(const std::string& msg);

// Most-recent log lines (already timestamped), oldest-first, for the dashboard
// activity tail. Safe to call from the dashboard thread.
std::vector<std::string> log_recent(size_t n);

// --- Console / dashboard helpers ------------------------------------------
// Best-effort: enable ANSI/VT escape handling on the Windows console so the
// dashboard can clear/redraw. No-op if it isn't supported.
void console_init();

// When quiet is true, log_line() stops writing to stdout (it still writes to
// the --log-file if one is open). The dashboard keeps drawing. Used by
// --quiet-dashboard so the status box stays clean.
void log_set_quiet(bool quiet);

// Print a multi-line block atomically with respect to log_line(). If
// clear_first is true, the screen is cleared and the cursor homed first.
void console_block(const std::string& text, bool clear_first);

// Always prints the ICMINERS banner to stdout (and the log file if open),
// regardless of quiet mode.
void print_banner();
