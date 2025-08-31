#pragma once
#include <memory>
#include <string>
#include "llm_backend.h"

// Options for the LM Studio (OpenAI-compatible) local server backend.
struct LMStudioOpts {
    std::string url      = "http://localhost:1234/v1"; // base URL, no trailing slash
    std::string api_key  = "lm-studio";                // default LM Studio key
    std::string model_id;                              // required: model id as shown by LM Studio
	std::vector<std::string> stop;
    bool        stream   = false;                      // enable Server-Sent Events streaming
    long        timeout_ms = 60000;                    // request timeout
};

// Factory: creates an LLMBackend that talks to LM Studio.
std::unique_ptr<LLMBackend> make_backend_lmstudio(const LMStudioOpts& opts);
