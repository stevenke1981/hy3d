#include "hy3d_math.h"
#include "hy3d_runtime.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

hy3d::RuntimeTensor f32_tensor(
    const std::string& name,
    const std::vector<std::uint64_t>& dimensions,
    const std::vector<float>& values) {
    hy3d::RuntimeTensor tensor;
    tensor.name = name;
    tensor.type = hy3d::GgmlType::F32;
    tensor.dimensions = dimensions;
    tensor.bytes.resize(values.size() * sizeof(float));
    std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
    return tensor;
}

bool require_parity(
    const std::string& name,
    const hy3d::Result<std::vector<float>>& actual,
    const std::vector<float>& expected) {
    if (!actual.ok() || actual.value().size() != expected.size()) {
        std::cerr << name << " result shape mismatch\n";
        return false;
    }
    for (std::size_t i = 0; i < actual.value().size(); ++i) {
        if (!std::isfinite(actual.value()[i]) || std::fabs(actual.value()[i] - expected[i]) > 1e-5f) {
            std::cerr << name << " parity mismatch at " << i << ": actual=" << actual.value()[i]
                      << " expected=" << expected[i] << "\n";
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_native_parity <fixture.csv>\n";
        return 2;
    }

    std::ifstream input(argv[1]);
    if (!input) {
        std::cerr << "failed to open reference fixture\n";
        return 2;
    }

    std::unordered_map<std::string, std::vector<float>> rows;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream stream(line);
        std::string name;
        std::getline(stream, name, ',');
        std::string value;
        while (std::getline(stream, value, ',')) {
            rows[name].push_back(std::stof(value));
        }
    }

    const auto output = hy3d::scaled_dot_product_attention(rows["q"], rows["k"], rows["v"], 2, 2, 1, 2);
    if (!require_parity("attention", output, rows["attention_expected"])) {
        return 1;
    }

    hy3d::HunyuanDitModel timestep_model;
    if (!require_parity(
            "timestep",
            timestep_model.project_timestep_conditioning(1.5f, 4),
            rows["timestep_expected"])) {
        return 1;
    }

    hy3d::HunyuanDitModel final_model;
    final_model.add_tensor(f32_tensor(
        "final_layer.linear.weight",
        {2, 2},
        {1.0f, 0.0f, 0.0f, 1.0f}));
    final_model.add_tensor(f32_tensor("final_layer.linear.bias", {2}, {0.0f, 0.0f}));
    if (!require_parity(
            "final_layer",
            final_model.apply_final_layer({0.0f, 0.0f, 1.0f, -1.0f}, 2),
            rows["final_expected"])) {
        return 1;
    }

    hy3d::HunyuanDitModel conditioned;
    conditioned.add_tensor(f32_tensor("blocks.0.attn1.to_q.weight", {2, 2}, {1, 0, 0, 1}));
    conditioned.add_tensor(f32_tensor("blocks.0.attn1.to_k.weight", {2, 2}, {1, 0, 0, 1}));
    conditioned.add_tensor(f32_tensor("blocks.0.attn1.to_v.weight", {2, 2}, {1, 0, 0, 1}));
    conditioned.add_tensor(f32_tensor("blocks.0.attn1.out_proj.weight", {2, 2}, {1, 0, 0, 1}));
    conditioned.add_tensor(f32_tensor("blocks.0.attn2.to_q.weight", {2, 2}, {1, 0, 0, 1}));
    conditioned.add_tensor(f32_tensor("blocks.0.attn2.to_k.weight", {2, 3}, {1, 0, 0, 0, 1, 0}));
    conditioned.add_tensor(f32_tensor("blocks.0.attn2.to_v.weight", {2, 3}, {0, 0, 1, 1, 0, 0}));
    conditioned.add_tensor(f32_tensor("blocks.0.attn2.out_proj.weight", {2, 2}, {1, 0, 0, 1}));
    conditioned.add_tensor(f32_tensor("blocks.0.attn2.q_norm.weight", {2}, {1, 1}));
    conditioned.add_tensor(f32_tensor("blocks.0.attn2.k_norm.weight", {2}, {1, 1}));
    if (!require_parity(
            "conditioned_block",
            conditioned.run_dit_block_conditioned(
                "blocks.0",
                {1, 0, 0, 1},
                2,
                {1, 2, 3},
                1,
                1,
                2),
            rows["conditioned_expected"])) {
        return 1;
    }
    return 0;
}
