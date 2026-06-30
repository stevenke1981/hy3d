#include "hy3d_runtime.h"

#include <chrono>
#include <iostream>
#include <string>

int main() {
    constexpr std::size_t tensor_count = 20000;
    constexpr std::size_t lookups = 1000000;

    hy3d::HunyuanDitModel model;
    for (std::size_t i = 0; i < tensor_count; ++i) {
        hy3d::RuntimeTensor tensor;
        tensor.name = "blocks." + std::to_string(i) + ".weight";
        model.add_tensor(std::move(tensor));
    }

    std::size_t found = 0;
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < lookups; ++i) {
        const auto name = "blocks." + std::to_string(i % tensor_count) + ".weight";
        found += model.find_tensor(name) != nullptr ? 1U : 0U;
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    std::cout << "tensors=" << tensor_count << "\n"
              << "lookups=" << lookups << "\n"
              << "found=" << found << "\n"
              << "elapsed_seconds=" << elapsed << "\n"
              << "lookups_per_second=" << static_cast<double>(lookups) / elapsed << "\n";

    if (found != lookups) {
        std::cerr << "lookup correctness gate failed\n";
        return 1;
    }
    if (elapsed > 10.0) {
        std::cerr << "lookup performance regression: benchmark exceeded 10 seconds\n";
        return 2;
    }
    return 0;
}
