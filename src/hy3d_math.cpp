#include "hy3d_math.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

namespace hy3d {
namespace {

Result<std::uint64_t> element_count(const RuntimeTensor& tensor) {
    std::uint64_t count = 1;
    for (const auto dim : tensor.dimensions) {
        if (dim != 0 && count > std::numeric_limits<std::uint64_t>::max() / dim) {
            return Result<std::uint64_t>::failure("tensor element count overflow: " + tensor.name);
        }
        count *= dim;
    }
    return Result<std::uint64_t>::success(count);
}

} // namespace

float float_from_f16_bits(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16u;
    const std::uint32_t exponent = (bits >> 10u) & 0x1Fu;
    const std::uint32_t mantissa = bits & 0x03FFu;

    std::uint32_t out = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            float value = std::ldexp(static_cast<float>(mantissa), -24);
            if (sign != 0) {
                value = -value;
            }
            return value;
        }
    } else if (exponent == 0x1Fu) {
        out = sign | 0x7F800000u | (mantissa << 13u);
    } else {
        const std::uint32_t adjusted_exponent = exponent + (127u - 15u);
        out = sign | (adjusted_exponent << 23u) | (mantissa << 13u);
    }

    float value = 0.0f;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

Result<std::vector<float>> tensor_to_f32(const RuntimeTensor& tensor) {
    const auto count = element_count(tensor);
    if (!count.ok()) {
        return Result<std::vector<float>>::failure(count.error());
    }

    std::vector<float> values(static_cast<std::size_t>(count.value()));
    if (tensor.type == GgmlType::F32) {
        const auto expected = count.value() * sizeof(float);
        if (tensor.bytes.size() != expected) {
            return Result<std::vector<float>>::failure("F32 tensor byte size mismatch: " + tensor.name);
        }
        std::memcpy(values.data(), tensor.bytes.data(), tensor.bytes.size());
        return Result<std::vector<float>>::success(values);
    }

    if (tensor.type == GgmlType::F16) {
        const auto expected = count.value() * sizeof(std::uint16_t);
        if (tensor.bytes.size() != expected) {
            return Result<std::vector<float>>::failure("F16 tensor byte size mismatch: " + tensor.name);
        }
        for (std::size_t i = 0; i < values.size(); ++i) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, tensor.bytes.data() + i * sizeof(std::uint16_t), sizeof(bits));
            values[i] = float_from_f16_bits(bits);
        }
        return Result<std::vector<float>>::success(values);
    }

    return Result<std::vector<float>>::failure("unsupported tensor type: " + tensor.name);
}

Result<std::vector<float>> linear_projection(
    const std::vector<float>& input,
    const RuntimeTensor& weight,
    const RuntimeTensor* bias) {
    if (weight.dimensions.size() != 2) {
        return Result<std::vector<float>>::failure("linear weight must be 2D: " + weight.name);
    }

    const auto out_features = static_cast<std::size_t>(weight.dimensions[0]);
    const auto in_features = static_cast<std::size_t>(weight.dimensions[1]);
    if (input.size() != in_features) {
        return Result<std::vector<float>>::failure("linear input size does not match weight: " + weight.name);
    }

    const auto weight_values = tensor_to_f32(weight);
    if (!weight_values.ok()) {
        return Result<std::vector<float>>::failure(weight_values.error());
    }

    std::vector<float> bias_values;
    if (bias != nullptr) {
        if (bias->dimensions.size() != 1 || bias->dimensions[0] != out_features) {
            return Result<std::vector<float>>::failure("linear bias shape does not match weight: " + bias->name);
        }
        auto converted_bias = tensor_to_f32(*bias);
        if (!converted_bias.ok()) {
            return Result<std::vector<float>>::failure(converted_bias.error());
        }
        bias_values = converted_bias.value();
    }

    std::vector<float> output(out_features, 0.0f);
    for (std::size_t row = 0; row < out_features; ++row) {
        float sum = bias_values.empty() ? 0.0f : bias_values[row];
        for (std::size_t col = 0; col < in_features; ++col) {
            sum += input[col] * weight_values.value()[row * in_features + col];
        }
        output[row] = sum;
    }
    return Result<std::vector<float>>::success(output);
}

Result<std::vector<float>> linear_projection_batch(
    const std::vector<float>& input,
    std::size_t tokens,
    const RuntimeTensor& weight,
    const RuntimeTensor* bias) {
    if (tokens == 0) {
        return Result<std::vector<float>>::failure("batched linear tokens must be positive");
    }
    if (weight.dimensions.size() != 2) {
        return Result<std::vector<float>>::failure("batched linear weight must be 2D: " + weight.name);
    }

    const auto out_features = static_cast<std::size_t>(weight.dimensions[0]);
    const auto in_features = static_cast<std::size_t>(weight.dimensions[1]);
    if (input.size() != tokens * in_features) {
        return Result<std::vector<float>>::failure("batched linear input size does not match tokens * weight input: " + weight.name);
    }

    std::vector<float> output;
    output.reserve(tokens * out_features);
    for (std::size_t token = 0; token < tokens; ++token) {
        const auto begin = input.begin() + static_cast<std::ptrdiff_t>(token * in_features);
        const std::vector<float> row(begin, begin + static_cast<std::ptrdiff_t>(in_features));
        const auto projected = linear_projection(row, weight, bias);
        if (!projected.ok()) {
            return Result<std::vector<float>>::failure(projected.error());
        }
        output.insert(output.end(), projected.value().begin(), projected.value().end());
    }
    return Result<std::vector<float>>::success(output);
}

Result<std::vector<float>> layer_norm_batch(
    const std::vector<float>& input,
    std::size_t tokens,
    const RuntimeTensor* weight,
    const RuntimeTensor* bias,
    float epsilon) {
    if (tokens == 0) {
        return Result<std::vector<float>>::failure("layer norm tokens must be positive");
    }
    if (input.empty() || input.size() % tokens != 0) {
        return Result<std::vector<float>>::failure("layer norm input size must be divisible by tokens");
    }
    const auto features = input.size() / tokens;

    std::vector<float> weight_values(features, 1.0f);
    if (weight != nullptr) {
        if (weight->dimensions.size() != 1 || weight->dimensions[0] != features) {
            return Result<std::vector<float>>::failure("layer norm weight shape mismatch: " + weight->name);
        }
        const auto converted = tensor_to_f32(*weight);
        if (!converted.ok()) {
            return Result<std::vector<float>>::failure(converted.error());
        }
        weight_values = converted.value();
    }

    std::vector<float> bias_values(features, 0.0f);
    if (bias != nullptr) {
        if (bias->dimensions.size() != 1 || bias->dimensions[0] != features) {
            return Result<std::vector<float>>::failure("layer norm bias shape mismatch: " + bias->name);
        }
        const auto converted = tensor_to_f32(*bias);
        if (!converted.ok()) {
            return Result<std::vector<float>>::failure(converted.error());
        }
        bias_values = converted.value();
    }

    std::vector<float> output(input.size(), 0.0f);
    for (std::size_t token = 0; token < tokens; ++token) {
        const auto base = token * features;
        float mean = 0.0f;
        for (std::size_t feature = 0; feature < features; ++feature) {
            mean += input[base + feature];
        }
        mean /= static_cast<float>(features);

        float variance = 0.0f;
        for (std::size_t feature = 0; feature < features; ++feature) {
            const float centered = input[base + feature] - mean;
            variance += centered * centered;
        }
        variance /= static_cast<float>(features);
        const float inv_std = 1.0f / std::sqrt(variance + epsilon);

        for (std::size_t feature = 0; feature < features; ++feature) {
            const float normalized = (input[base + feature] - mean) * inv_std;
            output[base + feature] = normalized * weight_values[feature] + bias_values[feature];
        }
    }
    return Result<std::vector<float>>::success(output);
}

Result<std::vector<float>> rms_norm_attention_heads(
    const std::vector<float>& input,
    std::size_t tokens,
    std::size_t heads,
    std::size_t head_dim,
    const RuntimeTensor* weight,
    float epsilon) {
    if (tokens == 0 || heads == 0 || head_dim == 0) {
        return Result<std::vector<float>>::failure("RMS attention norm dimensions must be positive");
    }
    const auto expected = tokens * heads * head_dim;
    if (input.size() != expected) {
        return Result<std::vector<float>>::failure("RMS attention norm input size mismatch");
    }

    std::vector<float> weight_values(head_dim, 1.0f);
    if (weight != nullptr) {
        if (weight->dimensions.size() != 1 || weight->dimensions[0] != head_dim) {
            return Result<std::vector<float>>::failure("RMS attention norm weight shape mismatch: " + weight->name);
        }
        const auto converted = tensor_to_f32(*weight);
        if (!converted.ok()) {
            return Result<std::vector<float>>::failure(converted.error());
        }
        weight_values = converted.value();
    }

    auto offset = [&](std::size_t token, std::size_t head, std::size_t dim) {
        return ((token * heads + head) * head_dim) + dim;
    };

    std::vector<float> output(input.size(), 0.0f);
    for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t head = 0; head < heads; ++head) {
            float mean_square = 0.0f;
            for (std::size_t dim = 0; dim < head_dim; ++dim) {
                const float value = input[offset(token, head, dim)];
                mean_square += value * value;
            }
            mean_square /= static_cast<float>(head_dim);
            const float inv_rms = 1.0f / std::sqrt(mean_square + epsilon);
            for (std::size_t dim = 0; dim < head_dim; ++dim) {
                output[offset(token, head, dim)] = input[offset(token, head, dim)] * inv_rms * weight_values[dim];
            }
        }
    }
    return Result<std::vector<float>>::success(output);
}

std::vector<float> gelu(const std::vector<float>& input) {
    std::vector<float> output(input.size(), 0.0f);
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    for (std::size_t i = 0; i < input.size(); ++i) {
        const float x = input[i];
        output[i] = 0.5f * x * (1.0f + std::tanh(kSqrtTwoOverPi * (x + 0.044715f * x * x * x)));
    }
    return output;
}

std::vector<float> silu(const std::vector<float>& input) {
    std::vector<float> output(input.size(), 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] / (1.0f + std::exp(-input[i]));
    }
    return output;
}

Result<std::vector<float>> timestep_sincos_embedding(float timestep, std::size_t width) {
    if (width == 0) {
        return Result<std::vector<float>>::failure("timestep embedding width must be positive");
    }
    std::vector<float> output(width, 0.0f);
    const auto half = width / 2;
    if (half == 0) {
        output[0] = timestep;
        return Result<std::vector<float>>::success(output);
    }
    constexpr float kMaxPeriod = 10000.0f;
    const float denominator = static_cast<float>(half);
    for (std::size_t i = 0; i < half; ++i) {
        const float exponent = -std::log(kMaxPeriod) * static_cast<float>(i) / denominator;
        const float angle = timestep * std::exp(exponent);
        output[i] = std::sin(angle);
        output[half + i] = std::cos(angle);
    }
    return Result<std::vector<float>>::success(output);
}

Result<std::vector<float>> softmax(const std::vector<float>& values) {
    if (values.empty()) {
        return Result<std::vector<float>>::failure("softmax input must not be empty");
    }

    const auto max_it = std::max_element(values.begin(), values.end());
    std::vector<float> output(values.size(), 0.0f);
    float sum = 0.0f;
    for (std::size_t i = 0; i < values.size(); ++i) {
        output[i] = std::exp(values[i] - *max_it);
        sum += output[i];
    }
    if (sum == 0.0f) {
        return Result<std::vector<float>>::failure("softmax normalization sum is zero");
    }
    for (auto& value : output) {
        value /= sum;
    }
    return Result<std::vector<float>>::success(output);
}

Result<std::vector<float>> scaled_dot_product_attention(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    std::size_t tokens,
    std::size_t heads,
    std::size_t head_dim) {
    return scaled_dot_product_attention(query, key, value, tokens, tokens, heads, head_dim);
}

Result<std::vector<float>> scaled_dot_product_attention(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    std::size_t query_tokens,
    std::size_t key_value_tokens,
    std::size_t heads,
    std::size_t head_dim) {
    if (query_tokens == 0 || key_value_tokens == 0 || heads == 0 || head_dim == 0) {
        return Result<std::vector<float>>::failure("attention dimensions must be positive");
    }
    const auto query_expected = query_tokens * heads * head_dim;
    const auto key_value_expected = key_value_tokens * heads * head_dim;
    if (query.size() != query_expected || key.size() != key_value_expected || value.size() != key_value_expected) {
        return Result<std::vector<float>>::failure("attention q/k/v sizes do not match token dimensions");
    }

    std::vector<float> output(query_expected, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto offset = [&](std::size_t token, std::size_t head, std::size_t dim) {
        return ((token * heads + head) * head_dim) + dim;
    };

    for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t q_token = 0; q_token < query_tokens; ++q_token) {
            std::vector<float> scores(key_value_tokens, 0.0f);
            for (std::size_t k_token = 0; k_token < key_value_tokens; ++k_token) {
                float dot = 0.0f;
                for (std::size_t dim = 0; dim < head_dim; ++dim) {
                    dot += query[offset(q_token, head, dim)] * key[offset(k_token, head, dim)];
                }
                scores[k_token] = dot * scale;
            }

            const auto weights = softmax(scores);
            if (!weights.ok()) {
                return Result<std::vector<float>>::failure(weights.error());
            }

            for (std::size_t dim = 0; dim < head_dim; ++dim) {
                float sum = 0.0f;
                for (std::size_t k_token = 0; k_token < key_value_tokens; ++k_token) {
                    sum += weights.value()[k_token] * value[offset(k_token, head, dim)];
                }
                output[offset(q_token, head, dim)] = sum;
            }
        }
    }

    return Result<std::vector<float>>::success(output);
}

} // namespace hy3d
