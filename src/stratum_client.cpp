#include "stratum_client.hpp"
#include "logger.hpp"

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #include <winsock2.h>
  #include <ws2tcpip.h>
  using cap_socket_t = SOCKET;
  using cap_socklen_t = int;
  static constexpr cap_socket_t CAP_INVALID_SOCKET = INVALID_SOCKET;
  static int cap_socket_error() { return WSAGetLastError(); }
  static bool cap_would_retry(int e) { return e == WSAEINTR; }
  static void cap_close_socket(cap_socket_t s) { closesocket(s); }
#else
  #include <sys/types.h>
  #include <sys/socket.h>
  #include <netdb.h>
  #include <unistd.h>
  #include <cerrno>
  using cap_socket_t = int;
  using cap_socklen_t = socklen_t;
  static constexpr cap_socket_t CAP_INVALID_SOCKET = -1;
  static int cap_socket_error() { return errno; }
  static bool cap_would_retry(int e) { return e == EINTR; }
  static void cap_close_socket(cap_socket_t s) { ::close(s); }
#endif

#include <stdexcept>
#include <sstream>
#include <regex>
#include <cstring>

StratumUrl parse_stratum_url(const std::string& url) {
    const std::string prefix = "stratum+tcp://";
    if(url.rfind(prefix, 0) != 0)
        throw std::runtime_error("pool must start with stratum+tcp://");

    const std::string rest = url.substr(prefix.size());
    if(rest.empty()) throw std::runtime_error("pool URL is empty");

    // IPv6 literals must be bracketed: stratum+tcp://[2001:db8::1]:3333
    std::string host;
    std::string port_text;
    if(rest.front() == '[') {
        const auto end = rest.find(']');
        if(end == std::string::npos || end + 1 >= rest.size() || rest[end + 1] != ':')
            throw std::runtime_error("invalid bracketed IPv6 pool URL");
        host = rest.substr(1, end - 1);
        port_text = rest.substr(end + 2);
    } else {
        const auto pos = rest.rfind(':');
        if(pos == std::string::npos)
            throw std::runtime_error("pool URL missing :port");
        host = rest.substr(0, pos);
        port_text = rest.substr(pos + 1);
    }

    if(host.empty() || port_text.empty())
        throw std::runtime_error("pool URL missing host or port");
    const int port = std::stoi(port_text);
    if(port < 1 || port > 65535)
        throw std::runtime_error("pool port must be between 1 and 65535");
    return {host, static_cast<uint16_t>(port)};
}

bool StratumClient::connect_to(const std::string& url) {
    close();
#ifdef _WIN32
    WSADATA wsa{};
    if(WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;
    wsa_started_ = true;
#endif

    const auto su = parse_stratum_url(url);
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    addrinfo* res = nullptr;
    const std::string port = std::to_string(su.port);
    const int gai = getaddrinfo(su.host.c_str(), port.c_str(), &hints, &res);
    if(gai != 0) {
        log_line("getaddrinfo failed for " + su.host + ":" + port);
        close();
        return false;
    }

    cap_socket_t connected = CAP_INVALID_SOCKET;
    for(addrinfo* ai = res; ai; ai = ai->ai_next) {
        cap_socket_t s = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if(s == CAP_INVALID_SOCKET) continue;
        if(::connect(s, ai->ai_addr, static_cast<cap_socklen_t>(ai->ai_addrlen)) == 0) {
            connected = s;
            break;
        }
        cap_close_socket(s);
    }
    freeaddrinfo(res);

    if(connected == CAP_INVALID_SOCKET) {
        close();
        return false;
    }

    sock_ = static_cast<std::intptr_t>(connected);
    log_line("connected to " + su.host + ":" + port);
    return true;
}

bool StratumClient::send_line(const std::string& json) {
    if(sock_ == kInvalidSocket) return false;
    std::string line = json;
    if(line.empty() || line.back() != '\n') line.push_back('\n');

    const cap_socket_t s = static_cast<cap_socket_t>(sock_);
    size_t sent_total = 0;
    while(sent_total < line.size()) {
#ifdef _WIN32
        const int n = ::send(s, line.data() + sent_total,
                             static_cast<int>(line.size() - sent_total), 0);
#else
        const ssize_t n = ::send(s, line.data() + sent_total,
                                 line.size() - sent_total, MSG_NOSIGNAL);
#endif
        if(n > 0) {
            sent_total += static_cast<size_t>(n);
            continue;
        }
        if(n < 0 && cap_would_retry(cap_socket_error())) continue;
        return false;
    }
    return true;
}

bool StratumClient::login(const std::string& wallet, const std::string& worker,
                          const std::string& pass) {
    const std::string user = wallet + "." + worker;
    if(!send_line(R"({"id":1,"method":"mining.subscribe","params":["capminer/1.0"]})"))
        return false;
    std::ostringstream os;
    os << R"({"id":2,"method":"mining.authorize","params":[")"
       << user << R"(",")" << pass << R"("]})";
    return send_line(os.str());
}

void StratumClient::parse_subscribe_response(const std::string& line) {
    if(line.find("\"result\"") == std::string::npos) return;
    std::regex re(R"CAP("result"\s*:\s*\[.*,\s*"([^"]+)"\s*,\s*([0-9]+)\s*\]\s*,\s*"id"\s*:\s*1)CAP");
    std::smatch m;
    if(std::regex_search(line, m, re)) {
        extranonce1_ = m[1];
        extranonce2_size_ = std::stoi(m[2]);
        subscribed_.store(true, std::memory_order_release);
        log_line("stratum subscribe extranonce1=" + extranonce1_ +
                 " extranonce2_size=" + std::to_string(extranonce2_size_));
    }
}

int StratumClient::submit_share(const std::string& user, const std::string& job_id,
                                const std::string& extranonce2, const std::string& ntime,
                                const std::string& nonce, bool verbose) {
    const int id = ++next_id_;
    std::ostringstream os;
    os << R"({"id":)" << id
       << R"(,"method":"mining.submit","params":[")"
       << user << R"(",")" << job_id << R"(",")" << extranonce2
       << R"(",")" << ntime << R"(",")" << nonce << R"("]})";

    {
        std::lock_guard<std::mutex> lk(submit_mu_);
        pending_submits_.insert(id);
    }
    if(verbose) {
        log_line("RAW SUBMIT: " + os.str());
        log_line("submitting share id=" + std::to_string(id) + " job=" + job_id +
                 " ex2=" + extranonce2 + " ntime=" + ntime + " nonce=" + nonce);
        log_line("mining.submit payload: " + os.str());
    }
    if(!send_line(os.str())) {
        std::lock_guard<std::mutex> lk(submit_mu_);
        pending_submits_.erase(id);
        return -1;
    }
    return id;
}

int StratumClient::submit_raw(const std::string& method, const std::string& params_json,
                              bool verbose) {
    const int id = ++next_id_;
    std::ostringstream os;
    os << R"({"id":)" << id << R"(,"method":")" << method
       << R"(","params":)" << params_json << "}";
    {
        std::lock_guard<std::mutex> lk(submit_mu_);
        pending_submits_.insert(id);
    }
    if(verbose) log_line("RAW SUBMIT: " + os.str());
    if(!send_line(os.str())) {
        std::lock_guard<std::mutex> lk(submit_mu_);
        pending_submits_.erase(id);
        return -1;
    }
    return id;
}

void StratumClient::parse_submit_response(const std::string& line) {
    if(line.find("\"method\"") != std::string::npos) return;
    std::smatch m;
    std::regex idre(R"CAP("id"\s*:\s*([0-9]+))CAP");
    if(!std::regex_search(line, m, idre)) return;

    int id = 0;
    try { id = std::stoi(m[1]); } catch(...) { return; }
    bool was_pending = false;
    {
        std::lock_guard<std::mutex> lk(submit_mu_);
        auto it = pending_submits_.find(id);
        if(it != pending_submits_.end()) {
            pending_submits_.erase(it);
            was_pending = true;
        }
    }
    if(!was_pending) return;

    const bool accepted = std::regex_search(line, std::regex(R"CAP("result"\s*:\s*true)CAP"));
    std::string reason;
    if(!accepted) {
        std::smatch em;
        if(std::regex_search(line, em,
            std::regex(R"CAP("error"\s*:\s*\[\s*[0-9-]+\s*,\s*"((?:[^"\\]|\\.)*)")CAP")))
            reason = em[1];
        if(reason.empty()) reason = "raw:" + line.substr(0, 220);
    }
    if(submit_cb_) submit_cb_(id, accepted, reason);
}

void StratumClient::parse_set_difficulty(const std::string& line) {
    if(line.find("mining.set_difficulty") == std::string::npos) return;
    std::regex re(R"CAP("params"\s*:\s*\[\s*([0-9]+(?:\.[0-9]+)?)\s*\])CAP");
    std::smatch m;
    if(std::regex_search(line, m, re)) {
        difficulty_ = std::stod(m[1]);
        log_line("pool difficulty=" + std::to_string(difficulty_.load()));
    }
}

void StratumClient::read_loop(LineHandler on_line) {
    if(sock_ == kInvalidSocket) return;
    const cap_socket_t s = static_cast<cap_socket_t>(sock_);
    std::string buf;
    char tmp[4096];
    while(true) {
#ifdef _WIN32
        const int n = ::recv(s, tmp, static_cast<int>(sizeof(tmp)), 0);
#else
        const ssize_t n = ::recv(s, tmp, sizeof(tmp), 0);
#endif
        if(n > 0) {
            buf.append(tmp, tmp + n);
        } else if(n < 0 && cap_would_retry(cap_socket_error())) {
            continue;
        } else {
            break;
        }

        size_t p = 0;
        while((p = buf.find('\n')) != std::string::npos) {
            auto line = buf.substr(0, p);
            if(!line.empty() && line.back() == '\r') line.pop_back();
            buf.erase(0, p + 1);
            parse_subscribe_response(line);
            parse_set_difficulty(line);
            parse_submit_response(line);
            on_line(line);
        }
    }
}

void StratumClient::close() {
    if(sock_ != kInvalidSocket) {
        cap_close_socket(static_cast<cap_socket_t>(sock_));
        sock_ = kInvalidSocket;
    }
#ifdef _WIN32
    if(wsa_started_) {
        WSACleanup();
        wsa_started_ = false;
    }
#endif
    subscribed_.store(false, std::memory_order_release);
}
