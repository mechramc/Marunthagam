package com.marunthagam.inference

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * LlamaWrapper — singleton Kotlin bridge to the `libmarunthagam.so` JNI layer.
 *
 * All inference calls are dispatched on [Dispatchers.IO] and protected by a
 * [Mutex] so that concurrent coroutines never call the non-thread-safe C++
 * llama_context simultaneously. (The JNI layer also has a std::mutex, but
 * defence-in-depth here prevents blocking an IO thread pool thread for the
 * full duration of inference when a second coroutine tries to acquire.)
 *
 * Usage in Jetpack Compose / ViewModel:
 * ```kotlin
 * // In a ViewModel:
 * viewModelScope.launch {
 *     val loaded = LlamaWrapper.loadModel("/sdcard/marunthagam/gemma4-e4b-q4_k_m.gguf")
 *     if (loaded) {
 *         val result = LlamaWrapper.runInference(prompt)
 *         // result is null on error
 *     }
 * }
 * ```
 *
 * Lifecycle note: call [freeModel] in `onCleared()` of the ViewModel that
 * owns inference, or in `Application.onTerminate()` as a safety net.
 */
object LlamaWrapper {

    private const val TAG = "LlamaWrapper"

    /**
     * Serialises all [loadModel] / [runInference] / [freeModel] calls.
     * Using a coroutine Mutex (not a Java synchronized block) so that
     * suspending while waiting does not block the IO thread pool thread.
     */
    private val inferenceMutex = Mutex()

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    /**
     * Returns true if a model has been successfully loaded and not yet freed.
     *
     * This is a volatile read — no locking required for a simple boolean check.
     * Callers that need a guaranteed-consistent view should hold [inferenceMutex].
     */
    @Volatile
    var isLoaded: Boolean = false
        private set

    // -----------------------------------------------------------------------
    // Native function declarations
    // These map 1-to-1 to the JNI exports in marunthagam_jni.cpp.
    // -----------------------------------------------------------------------

    private external fun nativeLoadModel(modelPath: String): Boolean
    private external fun nativeRunInference(prompt: String, maxTokens: Int): String
    private external fun nativeFreeModel()

    // -----------------------------------------------------------------------
    // Public API — all suspend functions; safe to call from any coroutine scope
    // -----------------------------------------------------------------------

    /**
     * Loads the GGUF model from [modelPath] into memory.
     *
     * This is a long-running operation (~10–30 s for a 5 GB model on a mid-range
     * phone). Always call from a coroutine; never from the main thread.
     *
     * @param modelPath Absolute path to the GGUF file, e.g.
     *                  `/sdcard/marunthagam/gemma4-e4b-q4_k_m.gguf`
     * @return `true` if the model was loaded successfully; `false` on any error.
     *         On failure the previous model (if any) is already freed — the
     *         wrapper is in an unloaded state and [isLoaded] returns false.
     */
    suspend fun loadModel(modelPath: String): Boolean = withContext(Dispatchers.IO) {
        inferenceMutex.withLock {
            Log.i(TAG, "loadModel: starting — path=$modelPath")
            val success = runCatching { nativeLoadModel(modelPath) }
                .onFailure { error ->
                    Log.e(TAG, "loadModel: native call threw exception", error)
                }
                .getOrDefault(false)

            isLoaded = success
            if (success) {
                Log.i(TAG, "loadModel: success")
            } else {
                Log.e(TAG, "loadModel: failed — check logcat for C++ errors")
            }
            success
        }
    }

    /**
     * Runs a single-turn inference pass on the already-loaded model.
     *
     * The KV cache is cleared after each call in the JNI layer, so each
     * invocation is stateless — appropriate for the one-shot triage use-case.
     *
     * @param prompt     Fully-formed Gemma 4 chat-template string (UTF-8).
     *                   Build this via [TriageEngine] rather than constructing
     *                   it manually.
     * @param maxTokens  Maximum number of tokens to generate. Default 512 is
     *                   sufficient for `triage_classify()` structured output.
     * @return Generated text string, or `null` if the model is not loaded or
     *         an error occurs. An empty string from the native layer is
     *         normalised to `null` so callers can use safe-call syntax.
     */
    suspend fun runInference(
        prompt: String,
        maxTokens: Int = 512,
    ): String? = withContext(Dispatchers.IO) {
        inferenceMutex.withLock {
            if (!isLoaded) {
                Log.e(TAG, "runInference: model not loaded")
                return@withLock null
            }

            val raw = runCatching { nativeRunInference(prompt, maxTokens) }
                .onFailure { error ->
                    Log.e(TAG, "runInference: native call threw exception", error)
                }
                .getOrDefault("")

            raw.ifBlank {
                Log.w(TAG, "runInference: received empty output from native layer")
                null
            }
        }
    }

    /**
     * Releases all native llama.cpp resources (model weights + context).
     *
     * After this call [isLoaded] returns false. It is safe to call [loadModel]
     * again after [freeModel] to reload the same or a different model.
     *
     * Idempotent: calling [freeModel] when no model is loaded is a no-op.
     */
    suspend fun freeModel(): Unit = withContext(Dispatchers.IO) {
        inferenceMutex.withLock {
            if (!isLoaded) {
                Log.d(TAG, "freeModel: no model loaded, nothing to do")
                return@withLock
            }
            runCatching { nativeFreeModel() }
                .onFailure { error ->
                    Log.e(TAG, "freeModel: native call threw exception", error)
                }
            isLoaded = false
            Log.i(TAG, "freeModel: model released")
        }
    }

    // -----------------------------------------------------------------------
    // Library initialisation — executed once when the class is first referenced
    // -----------------------------------------------------------------------
    init {
        System.loadLibrary("marunthagam")
    }
}
