#include "hy3d_gguf.h"
#include "hy3d_model_loader.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace {

std::uint64_t peak_rss_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(
            GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
            sizeof(counters)) == 0) {
        return 0;
    }
    return static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
#else
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
#if defined(__APPLE__)
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024ULL;
#endif
#endif
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << "usage: benchmark_gguf_loader <model.gguf> [first-block] [block-count]\n";
        return 0;
    }
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: benchmark_gguf_loader <model.gguf> [first-block] [block-count]\n";
        return 2;
    }

    const std::string model_path = argv[1];
    std::uint32_t first_block = 0;
    std::uint32_t block_count = 1;
    try {
        if (argc >= 3) {
            first_block = static_cast<std::uint32_t>(std::stoul(argv[2]));
        }
        if (argc >= 4) {
            block_count = static_cast<std::uint32_t>(std::stoul(argv[3]));
        }
    } catch (...) {
        std::cerr << "block arguments must be unsigned integers\n";
        return 2;
    }

    const auto inspect_started = std::chrono::steady_clock::now();
    const auto info = hy3d::inspect_gguf(model_path);
    const auto inspect_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - inspect_started).count();
    if (!info.ok()) {
        std::cerr << "inspect failed: " << info.error() << "\n";
        return 1;
    }

    const auto load_started = std::chrono::steady_clock::now();
    const auto model = hy3d::load_hunyuan_dit_blocks_from_gguf(
        model_path,
        first_block,
        block_count,
        true,
        true,
        true);
    const auto load_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - load_started).count();
    if (!model.ok()) {
        std::cerr << "load failed: " << model.error() << "\n";
        return 1;
    }

    std::cout << "model=" << std::filesystem::absolute(model_path).string() << "\n"
              << "file_bytes=" << std::filesystem::file_size(model_path) << "\n"
              << "metadata_tensors=" << info.value().tensor_infos.size() << "\n"
              << "loaded_tensors=" << model.value().tensor_count() << "\n"
              << "inspect_seconds=" << inspect_seconds << "\n"
              << "load_seconds=" << load_seconds << "\n"
              << "peak_rss_bytes=" << peak_rss_bytes() << "\n";
    return 0;
}
