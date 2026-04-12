/**
 * marunthagam_jni.cpp
 *
 * JNI bridge between Kotlin (LlamaWrapper) and llama.cpp C API.
 *
 * Responsibilities:
 *   - Load a GGUF model file from external storage
 *   - Run single-turn inference with deterministic sampling (triage use-case)
 *   - Release all llama.cpp resources cleanly (RAII via ScopedContext)
 *   - Forward all errors to Android logcat; return empty string to Kotlin on failure
 *
 * Threading:
 *   - All three exported JNI functions are protected by a single std::mutex.
 *   - The llama_context is NOT thread-safe; callers must not call runInference
 *     concurrently. The mutex prevents this at the JNI boundary.
 *
 * llama.cpp API assumptions (llama.cpp commit circa early-2025 / v0.0.x):
 *   - llama_model_load_from_file()   — loads GGUF, returns llama_model*
 *   - llama_new_context_with_model() — allocates inference context
 *   - llama_tokenize()               — converts string → token array
 *   - llama_batch_get_one()          — wraps token array into llama_batch
 *   - llama_decode()                 — runs forward pass on batch
 *   - llama_get_logits()             — raw logits after last decode
 *   - llama_sampler_chain_init/add/sample/accept — chain-based sampler API
 *     (replaces the older llama_sample_* free-function API)
 *   - llama_token_to_piece()         — converts token id → UTF-8 string piece
 *   - llama_token_eos()              — EOS token id for the loaded model
 *   - llama_model_free() / llama_free() — cleanup
 *
 * If you are using an older llama.cpp that still has the free-function sampler
 * API (llama_sample_top_p, llama_sample_temperature, etc.), replace the
 * llama_sampler_chain_* block with those equivalents.
 */

#include <jni.h>
#include <android/log.h>

#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <cstring>

// llama.cpp public header
#include "llama.h"

// -----------------------------------------------------------------------
// Logging helpers
// -----------------------------------------------------------------------
static constexpr const char* LOG_TAG = "Marunthagam";

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// -----------------------------------------------------------------------
// Module-level state — all access is guarded by g_mutex
// -----------------------------------------------------------------------
namespace {

std::mutex         g_mutex;
llama_model*       g_model   = nullptr;
llama_context*     g_ctx     = nullptr;

// Sampling hyperparameters (low temperature for deterministic triage output)
constexpr float  SAMPLE_TEMPERATURE   = 0.1f;
constexpr float  SAMPLE_TOP_P         = 0.9f;
constexpr float  SAMPLE_REPEAT_PENALTY = 1.1f;

// -----------------------------------------------------------------------
// RAII wrapper: ensures llama_sampler_chain is freed on all exit paths
// -----------------------------------------------------------------------
struct SamplerChainGuard {
    llama_sampler* sampler = nullptr;

    explicit SamplerChainGuard(llama_sampler* s) : sampler(s) {}

    ~SamplerChainGuard() {
        if (sampler) {
            llama_sampler_free(sampler);
            sampler = nullptr;
        }
    }

    // Non-copyable
    SamplerChainGuard(const SamplerChainGuard&) = delete;
    SamplerChainGuard& operator=(const SamplerChainGuard&) = delete;
};

// -----------------------------------------------------------------------
// Helper: convert jstring → std::string; returns empty string on error
// -----------------------------------------------------------------------
std::string jstring_to_std(JNIEnv* env, jstring js) {
    if (!js) return {};
    const char* chars = env->GetStringUTFChars(js, nullptr);
    if (!chars) return {};
    std::string result(chars);
    env->ReleaseStringUTFChars(js, chars);
    return result;
}

} // anonymous namespace

// -----------------------------------------------------------------------
// Exported JNI functions
// Naming convention: Java_<fully_qualified_class_underscores>_<method>
// Class: com.marunthagam.inference.LlamaWrapper
// -----------------------------------------------------------------------
extern "C" {

/**
 * loadModel — loads a GGUF model from the given file-system path.
 *
 * @param model_path_jstring  absolute path, e.g. "/sdcard/marunthagam/gemma4-e4b-q4_k_m.gguf"
 * @return JNI_TRUE on success, JNI_FALSE on failure
 */
JNIEXPORT jboolean JNICALL
Java_com_marunthagam_inference_LlamaWrapper_loadModel(
        JNIEnv*  env,
        jobject  /* obj */,
        jstring  model_path_jstring)
{
    std::lock_guard<std::mutex> lock(g_mutex);

    const std::string model_path = jstring_to_std(env, model_path_jstring);
    if (model_path.empty()) {
        LOGE("loadModel: received null or empty model path");
        return JNI_FALSE;
    }

    // Free any previously loaded model first
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }

    LOGI("loadModel: loading from %s", model_path.c_str());

    // Model load parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU-only on Android (no Vulkan/Metal)

    g_model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!g_model) {
        LOGE("loadModel: llama_model_load_from_file failed for path: %s", model_path.c_str());
        return JNI_FALSE;
    }

    // Context parameters — sized for Gemma 4 E4B on a mid-range Android phone
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = 2048;  // context window; 2048 is safe for triage prompts
    ctx_params.n_threads = 4;     // physical cores on typical ARM big.LITTLE
    ctx_params.n_batch   = 512;   // prompt batch size

    g_ctx = llama_new_context_with_model(g_model, ctx_params);
    if (!g_ctx) {
        LOGE("loadModel: llama_new_context_with_model failed");
        llama_model_free(g_model);
        g_model = nullptr;
        return JNI_FALSE;
    }

    LOGI("loadModel: model loaded successfully");
    return JNI_TRUE;
}

/**
 * runInference — tokenises the prompt and generates up to max_tokens new tokens.
 *
 * Uses a sampler chain: temperature → top-p → repeat-penalty.
 * Stops on EOS token or when max_tokens is reached.
 *
 * @param prompt_jstring  fully-formed Gemma 4 chat-template prompt (UTF-8)
 * @param max_tokens_jint maximum tokens to generate (caller passes 512 for triage)
 * @return generated text as jstring; empty string on any error
 */
JNIEXPORT jstring JNICALL
Java_com_marunthagam_inference_LlamaWrapper_runInference(
        JNIEnv*  env,
        jobject  /* obj */,
        jstring  prompt_jstring,
        jint     max_tokens_jint)
{
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_model || !g_ctx) {
        LOGE("runInference: model not loaded — call loadModel() first");
        return env->NewStringUTF("");
    }

    const std::string prompt = jstring_to_std(env, prompt_jstring);
    if (prompt.empty()) {
        LOGE("runInference: empty prompt");
        return env->NewStringUTF("");
    }

    const int max_tokens = static_cast<int>(max_tokens_jint);
    LOGI("runInference: prompt length=%zu, max_tokens=%d", prompt.size(), max_tokens);

    // ------------------------------------------------------------------
    // Tokenise the prompt
    // ------------------------------------------------------------------
    // Over-allocate; llama_tokenize will fill and return the actual count
    std::vector<llama_token> tokens(prompt.size() + 16);
    const int n_tokens = llama_tokenize(
            g_model,
            prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            /*add_special=*/true,
            /*parse_special=*/false);

    if (n_tokens < 0) {
        // Negative return means the buffer was too small; retry with exact size
        const int required = -n_tokens;
        tokens.resize(static_cast<size_t>(required));
        const int retry = llama_tokenize(
                g_model,
                prompt.c_str(),
                static_cast<int32_t>(prompt.size()),
                tokens.data(),
                required,
                /*add_special=*/true,
                /*parse_special=*/false);
        if (retry < 0) {
            LOGE("runInference: tokenize failed even after resize (required=%d)", required);
            return env->NewStringUTF("");
        }
        tokens.resize(static_cast<size_t>(retry));
    } else {
        tokens.resize(static_cast<size_t>(n_tokens));
    }

    LOGI("runInference: tokenised into %zu tokens", tokens.size());

    // ------------------------------------------------------------------
    // Decode the prompt tokens (prefill)
    // ------------------------------------------------------------------
    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    if (llama_decode(g_ctx, batch) != 0) {
        LOGE("runInference: llama_decode (prefill) failed");
        // Clear KV cache even on failure — the partial prefill left stale state
        // that must not bleed into the next patient's inference run (privacy critical).
        llama_kv_cache_clear(g_ctx);
        return env->NewStringUTF("");
    }

    // ------------------------------------------------------------------
    // Sampler chain: temperature → top-p → repetition penalty
    // The chain API was introduced to replace the legacy free-function samplers.
    // ------------------------------------------------------------------
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler* sampler_chain = llama_sampler_chain_init(chain_params);
    if (!sampler_chain) {
        LOGE("runInference: failed to init sampler chain");
        return env->NewStringUTF("");
    }
    SamplerChainGuard sampler_guard(sampler_chain);

    // Add samplers in order: repetition penalty first, then top-p, then temperature
    llama_sampler_chain_add(sampler_chain,
            llama_sampler_init_penalties(
                    /*n_vocab=*/    llama_model_n_vocab(g_model),
                    /*special_eos_id=*/ llama_token_eos(g_model),
                    /*linefeed_id=*/    llama_token_nl(g_model),
                    /*penalty_last_n=*/ 64,
                    /*penalty_repeat=*/ SAMPLE_REPEAT_PENALTY,
                    /*penalty_freq=*/   0.0f,
                    /*penalty_present=*/0.0f,
                    /*penalize_nl=*/    false,
                    /*ignore_eos=*/     false));

    llama_sampler_chain_add(sampler_chain,
            llama_sampler_init_top_p(SAMPLE_TOP_P, /*min_keep=*/1));

    llama_sampler_chain_add(sampler_chain,
            llama_sampler_init_temp(SAMPLE_TEMPERATURE));

    // ------------------------------------------------------------------
    // Auto-regressive generation loop
    // ------------------------------------------------------------------
    std::string output;
    output.reserve(static_cast<size_t>(max_tokens) * 4);  // UTF-8 Tamil can be ~4 bytes/char

    const llama_token eos_token = llama_token_eos(g_model);

    for (int i = 0; i < max_tokens; ++i) {
        // Sample next token from logits
        const llama_token new_token = llama_sampler_sample(sampler_chain, g_ctx, -1);

        if (new_token == eos_token) {
            LOGI("runInference: EOS reached at step %d", i);
            break;
        }

        // Accept the token into the sampler's penalty history
        llama_sampler_accept(sampler_chain, new_token);

        // Convert token id → UTF-8 piece
        // llama_token_to_piece writes into a caller-provided buffer
        char piece_buf[256] = {0};
        const int piece_len = llama_token_to_piece(
                g_model,
                new_token,
                piece_buf,
                static_cast<int>(sizeof(piece_buf) - 1),
                /*special=*/false);

        if (piece_len > 0) {
            output.append(piece_buf, static_cast<size_t>(piece_len));
        } else if (piece_len < 0) {
            LOGW("runInference: llama_token_to_piece returned %d at step %d", piece_len, i);
        }

        // Decode the new token so its KV cache entry exists for the next step
        llama_batch next_batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(g_ctx, next_batch) != 0) {
            LOGE("runInference: llama_decode failed at generation step %d", i);
            break;
        }
    }

    LOGI("runInference: generated %zu bytes of output", output.size());

    // Clear the KV cache so the next call starts fresh
    llama_kv_cache_clear(g_ctx);

    return env->NewStringUTF(output.c_str());
}

/**
 * freeModel — releases all llama.cpp resources.
 *
 * Safe to call multiple times; subsequent calls are no-ops.
 */
JNIEXPORT void JNICALL
Java_com_marunthagam_inference_LlamaWrapper_freeModel(
        JNIEnv*  /* env */,
        jobject  /* obj */)
{
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_ctx) {
        LOGI("freeModel: freeing context");
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        LOGI("freeModel: freeing model");
        llama_model_free(g_model);
        g_model = nullptr;
    }
    LOGI("freeModel: done");
}

} // extern "C"
