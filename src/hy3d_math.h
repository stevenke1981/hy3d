#pragma once

#include "hy3d_result.h"
#include "hy3d_runtime.h"

#include <cstdint>
#include <vector>

namespace hy3d {

float float_from_f16_bits(std::uint16_t bits);
Result<std::vector<float>> tensor_to_f32(const RuntimeTensor& tensor);
Result<std::vector<float>> linear_projection(
    const std::vector<float>& input,
    const RuntimeTensor& weight,
    const RuntimeTensor* bias);
Result<std::vector<float>> linear_projection_batch(
    const std::vector<float>& input,
    std::size_t tokens,
    const RuntimeTensor& weight,
    const RuntimeTensor* bias);
Result<std::vector<float>> layer_norm_batch(
    const std::vector<float>& input,
    std::size_t tokens,
    const RuntimeTensor* weight,
    const RuntimeTensor* bias,
    float epsilon = 1.0e-5f);
Result<std::vector<float>> rms_norm_attention_heads(
    const std::vector<float>& input,
    std::size_t tokens,
    std::size_t heads,
    std::size_t head_dim,
    const RuntimeTensor* weight,
    float epsilon = 1.0e-6f);
std::vector<float> gelu(const std::vector<float>& input);
std::vector<float> silu(const std::vector<float>& input);
Result<std::vector<float>> timestep_sincos_embedding(float timestep, std::size_t width);
Result<std::vector<float>> softmax(const std::vector<float>& values);
Result<std::vector<float>> scaled_dot_product_attention(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    std::size_t tokens,
    std::size_t heads,
    std::size_t head_dim);
Result<std::vector<float>> scaled_dot_product_attention(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    std::size_t query_tokens,
    std::size_t key_value_tokens,
    std::size_t heads,
    std::size_t head_dim);

} // namespace hy3d
