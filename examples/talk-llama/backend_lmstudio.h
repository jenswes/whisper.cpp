#pragma once
#include "llm_backend.h"
#include <memory>
#include <string>

struct LMStudioOpts {
    std::string url      = "http://localhost:1234/v1";
    std::string api_key  = "lm-studio";
    std::string model_id;
    int  timeout_ms      = 60000;
    bool stream          = true;
};

std::unique_ptr<LLMBackend> make_backend_lmstudio(const LMStudioOpts& opts);
