#pragma once
#include <functional>
#include <memory>
#include <string>

struct LLMToken {
    std::string text;
    bool is_final = false;
};

struct LLMGenerateParams {
    int   max_tokens   = 256;
    float temperature  = 0.7f;
    int   top_k        = 40;
    float top_p        = 0.95f;
    float min_p        = 0.05f;
    int   seed         = -1;
    bool  stream       = true;
    std::string system_prompt;
    // OpenAI-kompatible Stop-Sequenzen
    std::vector<std::string> stop;
};

class LLMBackend {
public:
    virtual ~LLMBackend() = default;
    virtual bool init() = 0;
    virtual void shutdown() = 0;
    virtual bool generate(const std::string& prompt,
                          const LLMGenerateParams& params,
                          const std::function<void(const LLMToken&)>& on_token) = 0;
};
