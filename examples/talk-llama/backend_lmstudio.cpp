#include "backend_lmstudio.h"

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <string>
#include <vector>
#include <functional>
#include <sstream>
#include <utility>
#include <cstring>

using json = nlohmann::json;

namespace {

// --- Helpers for CURL write callbacks ---

// Non-streaming: just append all bytes into a std::string
static size_t write_to_string(char* ptr, size_t size, size_t nmemb, void* userdata) {
    const size_t n = size * nmemb;
    auto* out = static_cast<std::string*>(userdata);
    out->append(ptr, n);
    return n;
}

// Streaming SSE state & parser
struct SseState {
    std::string buffer; // accumulates raw bytes until we find complete lines
    std::function<void(const LLMToken&)> on_token;
};

// Trim helpers
static inline std::string ltrim(const std::string& s) {
    size_t i = s.find_first_not_of(" \t\r\n");
    return (i == std::string::npos) ? std::string() : s.substr(i);
}
static inline std::string rtrim(const std::string& s) {
    size_t i = s.find_last_not_of(" \t\r\n");
    return (i == std::string::npos) ? std::string() : s.substr(0, i + 1);
}
static inline std::string trim(const std::string& s) { return rtrim(ltrim(s)); }

// Parse one SSE "data: ..." line -> emit delta text if present
static void parse_sse_line(const std::string& line, const std::function<void(const LLMToken&)>& on_token) {
    // Expected format: "data: {json}" or "data: [DONE]"
    static const std::string prefix = "data:";
    if (line.size() < prefix.size()) return;
    if (line.compare(0, prefix.size(), prefix) != 0) return;

    std::string payload = trim(line.substr(prefix.size()));
    if (payload == "[DONE]") {
        // End of stream
        LLMToken t; t.is_final = true; on_token(t);
        return;
    }

    try {
        json j = json::parse(payload);

        // OpenAI-compatible SSE delta: choices[0].delta.content
        if (j.contains("choices") && !j["choices"].empty()) {
            const auto& c0 = j["choices"][0];
            if (c0.contains("delta") && c0["delta"].contains("content")) {
                std::string piece = c0["delta"]["content"].get<std::string>();
                if (!piece.empty()) {
                    on_token({piece, false});
                }
                return;
            }
            // Some servers send accumulated "message" objects even in stream
            if (c0.contains("message") && c0["message"].contains("content")) {
                std::string piece = c0["message"]["content"].get<std::string>();
                if (!piece.empty()) {
                    on_token({piece, false});
                }
                return;
            }
        }

        // Fallback: try top-level "choices[0].text" (older-style)
        if (j.contains("choices") && !j["choices"].empty() && j["choices"][0].contains("text")) {
            std::string piece = j["choices"][0]["text"].get<std::string>();
            if (!piece.empty()) on_token({piece, false});
            return;
        }

    } catch (...) {
        // ignore bad lines quietly
    }
}

// Streaming write callback: buffer -> split into lines -> parse "data:" lines
static size_t write_sse(char* ptr, size_t size, size_t nmemb, void* userdata) {
    const size_t n = size * nmemb;
    auto* st = static_cast<SseState*>(userdata);
    st->buffer.append(ptr, n);

    // Process complete lines (SSE is line-oriented, events separated by blank lines)
    size_t pos = 0;
    while (true) {
        size_t nl = st->buffer.find('\n', pos);
        if (nl == std::string::npos) {
            // keep incomplete tail for next chunk
            st->buffer.erase(0, pos);
            break;
        }
        std::string line = st->buffer.substr(pos, nl - pos);
        pos = nl + 1;

        line = trim(line);
        if (line.empty()) continue;

        parse_sse_line(line, st->on_token);
    }

    return n;
}

} // namespace


// ---------------- LMStudio Backend ----------------

class LMStudioBackend : public LLMBackend {
public:
    explicit LMStudioBackend(const LMStudioOpts& o) : opts(o) {}

    bool init() override {
        curl_global_init(CURL_GLOBAL_ALL);
        return true;
    }

    void shutdown() override {
        curl_global_cleanup();
    }

    bool generate(const std::string& prompt,
                  const LLMGenerateParams& p,
                  const std::function<void(const LLMToken&)>& on_token) override
    {
        if (opts.model_id.empty()) {
            on_token({"[LMStudio] model_id not set", false});
            on_token({"", true});
            return false;
        }

        CURL* curl = curl_easy_init();
        if (!curl) {
            on_token({"[LMStudio] curl_easy_init failed", false});
            on_token({"", true});
            return false;
        }

        // Build endpoint
        std::string endpoint = opts.url;
        if (!endpoint.empty() && endpoint.back() == '/') endpoint.pop_back();
        endpoint += "/chat/completions";

        // Headers
        struct curl_slist* headers = nullptr;
        std::string auth = "Authorization: Bearer " + opts.api_key;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        headers = curl_slist_append(headers, auth.c_str());

        // Request body
        json body;
        body["model"]       = opts.model_id;
        body["stream"]      = p.stream;
        body["max_tokens"]  = p.max_tokens;
        body["temperature"] = p.temperature;
        body["top_p"]       = p.top_p;
        body["top_k"]       = p.top_k;
        body["min_p"]       = p.min_p;
        if (p.seed >= 0) body["seed"] = p.seed;
        if (!p.stop.empty()) body["stop"] = p.stop;

        json msgs = json::array();
        if (!p.system_prompt.empty()) {
            msgs.push_back({{"role","system"},{"content",p.system_prompt}});
        }
        msgs.push_back({{"role","user"},{"content",prompt}});
        body["messages"] = msgs;

        std::string body_str = body.dump();

        // CURL basic options
        curl_easy_setopt(curl, CURLOPT_URL, endpoint.c_str());
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, opts.timeout_ms);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_str.c_str());

        bool ok = true;

        if (p.stream) {
            // Streaming: parse SSE
            SseState st;
            st.on_token = on_token;
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_sse);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &st);

            CURLcode rc = curl_easy_perform(curl);
            long http_code = 0; curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

            if (rc != CURLE_OK || http_code / 100 != 2) {
                std::ostringstream oss;
                oss << "[LMStudio HTTP " << http_code << "] " << curl_easy_strerror(rc);
                on_token({oss.str(), false});
                on_token({"", true});
                ok = false;
            } else {
                // If the server didn't send [DONE], ensure we at least emit final
                on_token({"", true});
            }
        } else {
            // Non-streaming: read whole JSON, then emit full text
            std::string resp;
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_string);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);

            CURLcode rc = curl_easy_perform(curl);
            long http_code = 0; curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

            if (rc != CURLE_OK || http_code / 100 != 2) {
                std::ostringstream oss;
                oss << "[LMStudio HTTP " << http_code << "] " << curl_easy_strerror(rc);
                on_token({oss.str(), false});
                on_token({"", true});
                ok = false;
            } else {
                try {
                    json j = json::parse(resp);
                    std::string out;
                    if (j.contains("choices") && !j["choices"].empty()) {
                        const auto &c0 = j["choices"][0];
                        if (c0.contains("message") && c0["message"].contains("content")) {
                            out = c0["message"]["content"].get<std::string>();
                        } else if (c0.contains("text")) {
                            out = c0["text"].get<std::string>();
                        }
                    }
                    if (!out.empty()) on_token({out, false});
                    on_token({"", true});
                } catch (...) {
                    on_token({"[LMStudio parse error]", false});
                    on_token({"", true});
                    ok = false;
                }
            }
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return ok;
    }

private:
    LMStudioOpts opts;
};

// Factory
std::unique_ptr<LLMBackend> make_backend_lmstudio(const LMStudioOpts& opts) {
    return std::make_unique<LMStudioBackend>(opts);
}
