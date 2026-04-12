# CMakeLists.txt — Marunthagam JNI native build
#
# Builds libmarunthagam.so, which wraps llama.cpp for offline Gemma 4 E4B
# inference on Android 10+ (API 29+), targeting ARM64-v8a with NEON.
#
# Assumptions:
#   - llama.cpp source tree is cloned/copied to android/app/src/main/cpp/llama.cpp/
#   - NDK r26+ (supports C++17 fully on Android targets)
#   - Only ARM64-v8a ABI is targeted for production; x86_64 may be added for emulator testing

cmake_minimum_required(VERSION 3.22.1)

project(marunthagam_jni
    VERSION 1.0.0
    DESCRIPTION "Marunthagam llama.cpp JNI bridge for offline Tamil health AI"
    LANGUAGES C CXX
)

# ---------------------------------------------------------------------------
# Global C++ standard — C++17 required by llama.cpp and our own RAII patterns
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# ARM64 / NEON optimisation flags
# llama.cpp already enables NEON internally when it detects ANDROID + ARM64,
# but we add the flag here explicitly so our own translation unit also benefits.
# ---------------------------------------------------------------------------
if(ANDROID_ABI STREQUAL "arm64-v8a")
    add_compile_options(
        -march=armv8-a+fp+simd   # baseline for all Android ARM64 devices
        -O3
        -ffast-math              # safe for inference workloads (no IEEE edge cases needed)
    )
    message(STATUS "ARM64-v8a target: NEON / SIMD enabled via -march=armv8-a+fp+simd")
endif()

# ---------------------------------------------------------------------------
# llama.cpp subdirectory
#
# Key CMake variables forwarded to llama.cpp's own CMakeLists:
#   LLAMA_NATIVE=OFF  — do not auto-detect host CPU features (cross-compiling)
#   LLAMA_CURL=OFF    — no curl dependency (fully offline)
#   LLAMA_BUILD_TESTS=OFF
#   LLAMA_BUILD_EXAMPLES=OFF
#   LLAMA_BUILD_SERVER=OFF
#   GGML_METAL=OFF    — Metal is Apple-only
#   GGML_OPENMP=OFF   — OpenMP unavailable in NDK
# ---------------------------------------------------------------------------
set(LLAMA_NATIVE        OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL          OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER  OFF CACHE BOOL "" FORCE)
set(GGML_METAL          OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP         OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

# ---------------------------------------------------------------------------
# Android system libraries
# ---------------------------------------------------------------------------
find_library(LOG_LIB     log)      # __android_log_print
find_library(ANDROID_LIB android)  # AAssetManager (future: bundled assets)

# ---------------------------------------------------------------------------
# JNI shared library: libmarunthagam.so
# ---------------------------------------------------------------------------
add_library(marunthagam SHARED
    marunthagam_jni.cpp
)

target_include_directories(marunthagam PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include   # llama.h
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include  # ggml.h (pulled in transitively but explicit is cleaner)
)

target_link_libraries(marunthagam
    llama          # the llama.cpp static/shared library target
    ${LOG_LIB}     # logcat
    ${ANDROID_LIB} # android system
)

# Strip debug symbols in release to minimise APK size
target_compile_options(marunthagam PRIVATE
    $<$<CONFIG:Release>:-fvisibility=hidden>
)

message(STATUS "Marunthagam JNI target configured: libmarunthagam.so")
