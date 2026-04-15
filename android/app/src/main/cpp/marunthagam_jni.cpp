/**
 * marunthagam_jni.cpp
 *
 * JNI bridge between Kotlin and llama.cpp for offline Android inference.
 */

#include <jni.h>
#include <android/log.h>

#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"

static constexpr const char * LOG_TAG = "Marunthagam";

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

std::mutex     g_mutex;
llama_model  * g_model = nullptr;
llama_context * g_ctx = nullptr;

constexpr float SAMPLE_TEMPERATURE = 0.1f;
constexpr float SAMPLE_TOP_P = 0.9f;
constexpr float SAMPLE_REPEAT_PENALTY = 1.1f;
constexpr int32_t PENALTY_LAST_N = 64;

struct SamplerChainGuard {
    llama_sampler * sampler = nullptr;

    explicit SamplerChainGuard(llama_sampler * value) : sampler(value) {}

    ~SamplerChainGuard() {
        if (sampler) {
            llama_sampler_free(sampler);
            sampler = nullptr;
        }
    }

    SamplerChainGuard(const SamplerChainGuard &) = delete;
    SamplerChainGuard & operator=(const SamplerChainGuard &) = delete;
};

std::string jstring_to_std(JNIEnv * env, jstring input) {
    if (!input) {
        return {};
    }

    const char * chars = env->GetStringUTFChars(input, nullptr);
    if (!chars) {
        return {};
    }

    std::string result(chars);
    env->ReleaseStringUTFChars(input, chars);
    return result;
}

const llama_vocab * current_vocab() {
    return g_model ? llama_model_get_vocab(g_model) : nullptr;
}

void clear_context_memory() {
    if (!g_ctx) {
        return;
    }

    llama_memory_t memory = llama_get_memory(g_ctx);
    if (memory) {
        llama_memory_clear(memory, /* data = */ true);
    }
}

}  // namespace

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_marunthagam_inference_LlamaWrapper_nativeLoadModel(
        JNIEnv * env,
        jobject /* obj */,
        jstring model_path_jstring) {
    std::lock_guard<std::mutex> lock(g_mutex);

    const std::string model_path = jstring_to_std(env, model_path_jstring);
    if (model_path.empty()) {
        LOGE("nativeLoadModel: received null or empty model path");
        return JNI_FALSE;
    }

    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    g_model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!g_model) {
        LOGE("nativeLoadModel: llama_model_load_from_file failed for %s", model_path.c_str());
        return JNI_FALSE;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = 4;
    ctx_params.n_batch = 512;

    g_ctx = llama_init_from_model(g_model, ctx_params);
    if (!g_ctx) {
        LOGE("nativeLoadModel: llama_init_from_model failed");
        llama_model_free(g_model);
        g_model = nullptr;
        return JNI_FALSE;
    }

    LOGI("nativeLoadModel: model loaded successfully");
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_com_marunthagam_inference_LlamaWrapper_nativeRunInference(
        JNIEnv * env,
        jobject /* obj */,
        jstring prompt_jstring,
        jint max_tokens_jint) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_model || !g_ctx) {
        LOGE("nativeRunInference: model not loaded");
        return env->NewStringUTF("");
    }

    const llama_vocab * vocab = current_vocab();
    if (!vocab) {
        LOGE("nativeRunInference: model vocabulary unavailable");
        return env->NewStringUTF("");
    }

    const std::string prompt = jstring_to_std(env, prompt_jstring);
    if (prompt.empty()) {
        LOGE("nativeRunInference: empty prompt");
        return env->NewStringUTF("");
    }

    const int max_tokens = static_cast<int>(max_tokens_jint);
    std::vector<llama_token> tokens(prompt.size() + 16);

    int32_t n_tokens = llama_tokenize(
            vocab,
            prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            /* add_special = */ true,
            /* parse_special = */ false);

    if (n_tokens < 0) {
        const int32_t required = -n_tokens;
        tokens.resize(static_cast<size_t>(required));
        n_tokens = llama_tokenize(
                vocab,
                prompt.c_str(),
                static_cast<int32_t>(prompt.size()),
                tokens.data(),
                required,
                /* add_special = */ true,
                /* parse_special = */ false);
    }

    if (n_tokens <= 0) {
        LOGE("nativeRunInference: tokenization failed");
        return env->NewStringUTF("");
    }

    tokens.resize(static_cast<size_t>(n_tokens));

    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(g_ctx, batch) != 0) {
        LOGE("nativeRunInference: llama_decode prefill failed");
        clear_context_memory();
        return env->NewStringUTF("");
    }

    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler * sampler_chain = llama_sampler_chain_init(chain_params);
    if (!sampler_chain) {
        LOGE("nativeRunInference: sampler chain init failed");
        clear_context_memory();
        return env->NewStringUTF("");
    }
    SamplerChainGuard sampler_guard(sampler_chain);

    llama_sampler_chain_add(
            sampler_chain,
            llama_sampler_init_penalties(
                    PENALTY_LAST_N,
                    SAMPLE_REPEAT_PENALTY,
                    0.0f,
                    0.0f));
    llama_sampler_chain_add(
            sampler_chain,
            llama_sampler_init_top_p(SAMPLE_TOP_P, /* min_keep = */ 1));
    llama_sampler_chain_add(
            sampler_chain,
            llama_sampler_init_temp(SAMPLE_TEMPERATURE));

    const llama_token eos_token = llama_vocab_eos(vocab);
    std::string output;
    output.reserve(static_cast<size_t>(max_tokens) * 4);

    for (int i = 0; i < max_tokens; ++i) {
        const llama_token sampled = llama_sampler_sample(sampler_chain, g_ctx, -1);
        if (sampled == eos_token) {
            break;
        }

        llama_sampler_accept(sampler_chain, sampled);

        char piece_buf[256] = {0};
        const int32_t piece_len = llama_token_to_piece(
                vocab,
                sampled,
                piece_buf,
                static_cast<int32_t>(sizeof(piece_buf) - 1),
                /* lstrip = */ 0,
                /* special = */ false);

        if (piece_len > 0) {
            output.append(piece_buf, static_cast<size_t>(piece_len));
        } else if (piece_len < 0) {
            LOGW("nativeRunInference: token_to_piece returned %d", piece_len);
        }

        llama_token sampled_mutable = sampled;
        llama_batch next_batch = llama_batch_get_one(&sampled_mutable, 1);
        if (llama_decode(g_ctx, next_batch) != 0) {
            LOGE("nativeRunInference: llama_decode generation failed at step %d", i);
            break;
        }
    }

    clear_context_memory();
    return env->NewStringUTF(output.c_str());
}

JNIEXPORT void JNICALL
Java_com_marunthagam_inference_LlamaWrapper_nativeFreeModel(
        JNIEnv * /* env */,
        jobject /* obj */) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_ctx) {
        clear_context_memory();
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
}

}  // extern "C"
