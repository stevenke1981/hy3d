#include "hy3d_math.h"
#include "hy3d_runtime.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << "\n";
        std::exit(1);
    }
}

void require_near(float actual, float expected, const char* message) {
    if (std::fabs(actual - expected) > 0.001f) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << "\n";
        std::exit(1);
    }
}

std::vector<std::uint8_t> bytes_from_f32(const std::vector<float>& values) {
    std::vector<std::uint8_t> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

hy3d::RuntimeTensor f32_tensor(
    const std::string& name,
    const std::vector<std::uint64_t>& dimensions,
    const std::vector<float>& values) {
    hy3d::RuntimeTensor tensor;
    tensor.name = name;
    tensor.type = hy3d::GgmlType::F32;
    tensor.dimensions = dimensions;
    tensor.bytes = bytes_from_f32(values);
    return tensor;
}

} // namespace

int main() {
    require_near(hy3d::float_from_f16_bits(0x3C00), 1.0f, "f16 1.0");
    require_near(hy3d::float_from_f16_bits(0x4000), 2.0f, "f16 2.0");
    require_near(hy3d::float_from_f16_bits(0xBC00), -1.0f, "f16 -1.0");
    require_near(hy3d::float_from_f16_bits(0x0000), 0.0f, "f16 zero");

    hy3d::RuntimeTensor weight;
    weight.name = "linear.weight";
    weight.type = hy3d::GgmlType::F32;
    weight.dimensions = {2, 3};
    weight.bytes = bytes_from_f32({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    });

    hy3d::RuntimeTensor bias;
    bias.name = "linear.bias";
    bias.type = hy3d::GgmlType::F32;
    bias.dimensions = {2};
    bias.bytes = bytes_from_f32({0.5f, -0.5f});

    const auto output = hy3d::linear_projection({10.0f, 20.0f, 30.0f}, weight, &bias);
    require(output.ok(), "linear projection should succeed");
    require(output.value().size() == 2, "linear projection output size");
    require_near(output.value()[0], 140.5f, "linear output 0");
    require_near(output.value()[1], 319.5f, "linear output 1");

    hy3d::HunyuanDitModel dit;
    dit.add_tensor(weight);
    dit.add_tensor(bias);
    const auto projected = dit.project_linear("linear.weight", "linear.bias", {10.0f, 20.0f, 30.0f});
    require(projected.ok(), "dit linear projection should succeed");
    require_near(projected.value()[0], 140.5f, "dit linear output 0");
    require_near(projected.value()[1], 319.5f, "dit linear output 1");

    const auto missing = dit.project_linear("missing.weight", "", {1.0f});
    require(!missing.ok(), "missing weight should fail");
    require(missing.error().find("tensor not found") != std::string::npos, "missing tensor error");

    {
        const auto batch = hy3d::linear_projection_batch(
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
            2,
            weight,
            &bias);
        require(batch.ok(), "batched linear projection should succeed");
        require(batch.value().size() == 4, "batched linear output size");
        require_near(batch.value()[0], 14.5f, "batched linear token0 out0");
        require_near(batch.value()[1], 31.5f, "batched linear token0 out1");
        require_near(batch.value()[2], 32.5f, "batched linear token1 out0");
        require_near(batch.value()[3], 76.5f, "batched linear token1 out1");
    }

    {
        auto values = std::vector<float>{1000.0f, 1000.0f};
        const auto softmax = hy3d::softmax(values);
        require(softmax.ok(), "softmax should succeed");
        require_near(softmax.value()[0], 0.5f, "softmax stable 0");
        require_near(softmax.value()[1], 0.5f, "softmax stable 1");
    }

    {
        const auto attended = hy3d::scaled_dot_product_attention(
            {1.0f, 0.0f},
            {1.0f, 0.0f},
            {5.0f, 7.0f},
            1,
            1,
            2);
        require(attended.ok(), "single token attention should succeed");
        require(attended.value().size() == 2, "single token attention output size");
        require_near(attended.value()[0], 5.0f, "single token attention output 0");
        require_near(attended.value()[1], 7.0f, "single token attention output 1");
    }

    {
        const auto attended = hy3d::scaled_dot_product_attention(
            {1.0f, 0.0f, 0.0f, 1.0f},
            {1.0f, 0.0f, 0.0f, 1.0f},
            {10.0f, 20.0f, 30.0f, 40.0f},
            2,
            1,
            2);
        require(attended.ok(), "two token attention should succeed");
        require(attended.value().size() == 4, "two token attention output size");
        require(attended.value()[0] > 10.0f, "two token attention mixes value 0");
        require(attended.value()[0] < 20.0f, "two token attention bounded value 0");
        require(attended.value()[3] > 30.0f, "two token attention mixes value 3");
        require(attended.value()[3] < 40.0f, "two token attention bounded value 3");
    }

    {
        const auto norm = hy3d::layer_norm_batch({1.0f, 3.0f, 2.0f, 4.0f}, 2, nullptr, nullptr);
        require(norm.ok(), "layer norm should succeed");
        require_near(norm.value()[0], -0.999995f, "layer norm token0 dim0");
        require_near(norm.value()[1], 0.999995f, "layer norm token0 dim1");

        const auto norm_weight = f32_tensor("norm.weight", {2}, {2.0f, 3.0f});
        const auto norm_bias = f32_tensor("norm.bias", {2}, {0.5f, -0.5f});
        const auto affine_norm = hy3d::layer_norm_batch({1.0f, 3.0f}, 1, &norm_weight, &norm_bias);
        require(affine_norm.ok(), "affine layer norm should succeed");
        require_near(affine_norm.value()[0], -1.49999f, "affine layer norm dim0");
        require_near(affine_norm.value()[1], 2.49998f, "affine layer norm dim1");

        const auto activated = hy3d::gelu({0.0f, 1.0f});
        require_near(activated[0], 0.0f, "gelu zero");
        require_near(activated[1], 0.841192f, "gelu one");
    }

    return 0;
}
