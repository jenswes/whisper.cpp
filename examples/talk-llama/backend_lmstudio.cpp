#include "backend_lmstudio.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <sstream>

using json = nlohmann::json;

class LMStudioBackend : public LLMBackend {
public:
    LMStudioBackend(const LMStudioOpts& o) : opts(o) {}
    bool init() override {
        curl_global_init(CURL_GLOBAL_ALL);
        return true;
    }
    void shutdown() override { curl_global_cleanup(); }

    bool generate(const std::string& prompt,
                  const LLMGenerateParams& p,
                  const std::function<void(const LLMToken&)>& on_token) override {
        if (opts.model_id.empty()) {
            on_token({"[LMStudio] model_id not set", false});
            on_token({"", true});
            return false;
        }
        CURL* curl = curl_easy_init();
        if (!curl) return false;

        std::string endpoint = opts.url;
        if (!endpoint.empty() && endpoint.back() == '/') endpoint.pop_back();
        endpoint += "/chat/completions";

        curl_easy_setopt(curl, CURLOPT_URL, endpoint.c_str());
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, (long)opts.timeout_ms);

        struct curl_slist* headers = nullptr;
        std::string auth = "Authorization: Bearer " + opts.api_key;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        headers = curl_slist_append(headers, auth.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        json body;
        body["model"]       = opts.model_id;
        body["stream"]      = false;                // Skeleton: erst non-stream
        body["max_tokens"]  = p.max_tokens;
        body["temperature"] = p.temperature;
        body["top_p"]       = p.top_p;
        body["top_k"]       = p.top_k;
        body["min_p"]       = p.min_p;
        if (p.seed >= 0) body["seed"] = p.seed;

        json msgs = json::array();
        if (!p.system_prompt.empty())
            msgs.push_back({{"role","system"},{"content",p.system_prompt}});
        msgs.push_back({{"role","user"},{"content",prompt}});
        body["messages"] = msgs;

        std::string body_str = body.dump();
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_str.c_str());

        std::string resp;
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, +[](char* ptr,size_t s,size_t n,void* ud)->size_t{
            auto* out = static_cast<std::string*>(ud); out->append(ptr, s*n); return s*n;
        });
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);

        CURLcode rc = curl_easy_perform(curl);
        long http_code = 0; curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (rc != CURLE_OK || http_code/100 != 2) {
            std::ostringstream oss;
            oss << "[LMStudio HTTP " << http_code << "] " << curl_easy_strerror(rc);
            on_token({oss.str(), false});
            on_token({"", true});
            return false;
        }

        try {
            json j = json::parse(resp);
            std::string out;
            if (j.contains("choices") && !j["choices"].empty()) {
                auto &c0 = j["choices"][0];
                if (c0.contains("message") && c0["message"].contains("content"))
                    out = c0["message"]["content"].get<std::string>();
            }
            if (!out.empty()) on_token({out, false});
            on_token({"", true});
        } catch (...) {
            on_token({"[LMStudio parse error]", false});
            on_token({"", true});
            return false;
        }
        return true;
    }
private:
    LMStudioOpts opts;
};

std::unique_ptr<LLMBackend> make_backend_lmstudio(const LMStudioOpts& opts) {
    return std::make_unique<LMStudioBackend>(opts);
}
