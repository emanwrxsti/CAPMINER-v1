#pragma once
#include <string>
#include <functional>
#include <atomic>
#include <set>
#include <mutex>
#include <cstdint>

struct StratumUrl { std::string host; uint16_t port=0; };
StratumUrl parse_stratum_url(const std::string& url);

class StratumClient {
public:
    ~StratumClient() { close(); }
    StratumClient() = default;
    StratumClient(const StratumClient&) = delete;
    StratumClient& operator=(const StratumClient&) = delete;
    using LineHandler = std::function<void(const std::string&)>;
    // Called when a mining.submit response arrives, correlated by its id.
    using SubmitResultHandler = std::function<void(int id, bool accepted, const std::string& reason)>;

    bool connect_to(const std::string& url);
    bool login(const std::string& wallet, const std::string& worker, const std::string& pass);
    bool send_line(const std::string& json);
    // Returns the JSON-RPC id of the submit (>=0), or -1 if the send failed.
    // When verbose is true, logs the submit line and exact JSON payload.
    int submit_share(const std::string& user, const std::string& job_id, const std::string& extranonce2, const std::string& ntime, const std::string& nonce, bool verbose = false);
    // Generic JSON-RPC submit helper. params_json must already be a JSON array or object string.
    // The returned id is tracked so parse_submit_response can update accepted/rejected counts.
    int submit_raw(const std::string& method, const std::string& params_json, bool verbose = false);
    void read_loop(LineHandler on_line);
    void close();

    void set_submit_result_handler(SubmitResultHandler h) { submit_cb_ = std::move(h); }

    std::string extranonce1() const { return extranonce1_; }
    int extranonce2_size() const { return extranonce2_size_; }
    bool subscribed() const { return subscribed_.load(std::memory_order_acquire); }
    double difficulty() const { return difficulty_; }

private:
    static constexpr std::intptr_t kInvalidSocket = -1;
    std::intptr_t sock_ = kInvalidSocket;
#ifdef _WIN32
    bool wsa_started_ = false;
#endif
    // Subscribe uses id 1, authorize id 2; start submit ids well clear of those.
    std::atomic<int> next_id_{100};

    std::string extranonce1_;
    int extranonce2_size_ = 4;
    std::atomic<bool> subscribed_{false};   // set once the subscribe reply is parsed
    std::atomic<double> difficulty_{1.0};

    std::mutex submit_mu_;
    std::set<int> pending_submits_;
    SubmitResultHandler submit_cb_;

    void parse_subscribe_response(const std::string& line);
    void parse_set_difficulty(const std::string& line);
    void parse_submit_response(const std::string& line);
};
