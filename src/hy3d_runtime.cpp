#include "hy3d_runtime.h"

#include "hy3d_math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <utility>

namespace hy3d {
namespace {

std::vector<std::string> mapped_names(const std::string& prefix, const std::string& canonical_suffix) {
    if (canonical_suffix == "attn1_q.weight") {
        return {prefix + ".attn1_q.weight", prefix + ".attn_q.weight", prefix + ".attn1.to_q.weight"};
    }
    if (canonical_suffix == "attn1_k.weight") {
        return {prefix + ".attn1_k.weight", prefix + ".attn_k.weight", prefix + ".attn1.to_k.weight"};
    }
    if (canonical_suffix == "attn1_v.weight") {
        return {prefix + ".attn1_v.weight", prefix + ".attn_v.weight", prefix + ".attn1.to_v.weight"};
    }
    if (canonical_suffix == "attn1_output.weight") {
        return {prefix + ".attn1_output.weight", prefix + ".attn_output.weight", prefix + ".attn1.out_proj.weight"};
    }
    if (canonical_suffix == "attn1_output.bias") {
        return {prefix + ".attn1_output.bias", prefix + ".attn_output.bias", prefix + ".attn1.out_proj.bias"};
    }
    if (canonical_suffix == "attn1_q.bias") {
        return {prefix + ".attn1_q.bias", prefix + ".attn_q.bias", prefix + ".attn1.to_q.bias"};
    }
    if (canonical_suffix == "attn1_k.bias") {
        return {prefix + ".attn1_k.bias", prefix + ".attn_k.bias", prefix + ".attn1.to_k.bias"};
    }
    if (canonical_suffix == "attn1_v.bias") {
        return {prefix + ".attn1_v.bias", prefix + ".attn_v.bias", prefix + ".attn1.to_v.bias"};
    }
    if (canonical_suffix == "attn1_q_norm.weight") {
        return {prefix + ".attn1_q_norm.weight", prefix + ".attn1.q_norm.weight"};
    }
    if (canonical_suffix == "attn1_k_norm.weight") {
        return {prefix + ".attn1_k_norm.weight", prefix + ".attn1.k_norm.weight"};
    }
    if (canonical_suffix == "attn2_q.weight") {
        return {prefix + ".attn2_q.weight", prefix + ".attn2.to_q.weight"};
    }
    if (canonical_suffix == "attn2_k.weight") {
        return {prefix + ".attn2_k.weight", prefix + ".attn2.to_k.weight"};
    }
    if (canonical_suffix == "attn2_v.weight") {
        return {prefix + ".attn2_v.weight", prefix + ".attn2.to_v.weight"};
    }
    if (canonical_suffix == "attn2_output.weight") {
        return {prefix + ".attn2_output.weight", prefix + ".attn2.out_proj.weight"};
    }
    if (canonical_suffix == "attn2_output.bias") {
        return {prefix + ".attn2_output.bias", prefix + ".attn2.out_proj.bias"};
    }
    if (canonical_suffix == "attn2_q.bias") {
        return {prefix + ".attn2_q.bias", prefix + ".attn2.to_q.bias"};
    }
    if (canonical_suffix == "attn2_k.bias") {
        return {prefix + ".attn2_k.bias", prefix + ".attn2.to_k.bias"};
    }
    if (canonical_suffix == "attn2_v.bias") {
        return {prefix + ".attn2_v.bias", prefix + ".attn2.to_v.bias"};
    }
    if (canonical_suffix == "attn2_q_norm.weight") {
        return {prefix + ".attn2_q_norm.weight", prefix + ".attn2.q_norm.weight"};
    }
    if (canonical_suffix == "attn2_k_norm.weight") {
        return {prefix + ".attn2_k_norm.weight", prefix + ".attn2.k_norm.weight"};
    }
    if (canonical_suffix == "attn_q.weight") {
        return {prefix + ".attn_q.weight", prefix + ".attn1.to_q.weight"};
    }
    if (canonical_suffix == "attn_k.weight") {
        return {prefix + ".attn_k.weight", prefix + ".attn1.to_k.weight"};
    }
    if (canonical_suffix == "attn_v.weight") {
        return {prefix + ".attn_v.weight", prefix + ".attn1.to_v.weight"};
    }
    if (canonical_suffix == "attn_output.weight") {
        return {prefix + ".attn_output.weight", prefix + ".attn1.out_proj.weight"};
    }
    if (canonical_suffix == "attn_output.bias") {
        return {prefix + ".attn_output.bias", prefix + ".attn1.out_proj.bias"};
    }
    if (canonical_suffix == "mlp_fc1.weight") {
        return {prefix + ".mlp_fc1.weight", prefix + ".mlp.fc1.weight"};
    }
    if (canonical_suffix == "mlp_fc1.bias") {
        return {prefix + ".mlp_fc1.bias", prefix + ".mlp.fc1.bias"};
    }
    if (canonical_suffix == "mlp_fc2.weight") {
        return {prefix + ".mlp_fc2.weight", prefix + ".mlp.fc2.weight"};
    }
    if (canonical_suffix == "mlp_fc2.bias") {
        return {prefix + ".mlp_fc2.bias", prefix + ".mlp.fc2.bias"};
    }
    return {prefix + "." + canonical_suffix};
}

std::string attention_suffix(const std::string& attention_name, const std::string& suffix) {
    if (attention_name == "attn1") {
        if (suffix == "q.weight") {
            return "attn1_q.weight";
        }
        if (suffix == "q.bias") {
            return "attn1_q.bias";
        }
        if (suffix == "k.weight") {
            return "attn1_k.weight";
        }
        if (suffix == "k.bias") {
            return "attn1_k.bias";
        }
        if (suffix == "v.weight") {
            return "attn1_v.weight";
        }
        if (suffix == "v.bias") {
            return "attn1_v.bias";
        }
        if (suffix == "output.weight") {
            return "attn1_output.weight";
        }
        if (suffix == "output.bias") {
            return "attn1_output.bias";
        }
        if (suffix == "q_norm.weight") {
            return "attn1_q_norm.weight";
        }
        if (suffix == "k_norm.weight") {
            return "attn1_k_norm.weight";
        }
    }
    if (attention_name == "attn2") {
        if (suffix == "q.weight") {
            return "attn2_q.weight";
        }
        if (suffix == "q.bias") {
            return "attn2_q.bias";
        }
        if (suffix == "k.weight") {
            return "attn2_k.weight";
        }
        if (suffix == "k.bias") {
            return "attn2_k.bias";
        }
        if (suffix == "v.weight") {
            return "attn2_v.weight";
        }
        if (suffix == "v.bias") {
            return "attn2_v.bias";
        }
        if (suffix == "output.weight") {
            return "attn2_output.weight";
        }
        if (suffix == "output.bias") {
            return "attn2_output.bias";
        }
        if (suffix == "q_norm.weight") {
            return "attn2_q_norm.weight";
        }
        if (suffix == "k_norm.weight") {
            return "attn2_k_norm.weight";
        }
    }
    return attention_name + "." + suffix;
}

std::vector<float> token_slice(const std::vector<float>& values, std::size_t token, std::size_t features) {
    const auto begin = values.begin() + static_cast<std::ptrdiff_t>(token * features);
    return std::vector<float>(begin, begin + static_cast<std::ptrdiff_t>(features));
}

} // namespace

void HunyuanDitModel::add_tensor(RuntimeTensor tensor) {
    tensors_.push_back(std::move(tensor));
}

std::size_t HunyuanDitModel::tensor_count() const {
    return tensors_.size();
}

bool HunyuanDitModel::has_tensor(const std::string& name) const {
    return find_tensor(name) != nullptr;
}

const RuntimeTensor* HunyuanDitModel::find_tensor(const std::string& name) const {
    const auto it = std::find_if(tensors_.begin(), tensors_.end(), [&](const RuntimeTensor& tensor) {
        return tensor.name == name;
    });
    if (it == tensors_.end()) {
        return nullptr;
    }
    return &(*it);
}

const RuntimeTensor* HunyuanDitModel::find_mapped_tensor(const std::string& prefix, const std::string& canonical_suffix) const {
    for (const auto& name : mapped_names(prefix, canonical_suffix)) {
        const auto* tensor = find_tensor(name);
        if (tensor != nullptr) {
            return tensor;
        }
    }
    return nullptr;
}

Result<std::vector<float>> HunyuanDitModel::project_linear(
    const std::string& weight_name,
    const std::string& bias_name,
    const std::vector<float>& input) const {
    const RuntimeTensor* weight = find_tensor(weight_name);
    if (weight == nullptr) {
        return Result<std::vector<float>>::failure("tensor not found: " + weight_name);
    }

    const RuntimeTensor* bias = nullptr;
    if (!bias_name.empty()) {
        bias = find_tensor(bias_name);
        if (bias == nullptr) {
            return Result<std::vector<float>>::failure("tensor not found: " + bias_name);
        }
    }

    return linear_projection(input, *weight, bias);
}

Result<AttentionQkv> HunyuanDitModel::project_attention_qkv(
    const std::string& prefix,
    const std::vector<float>& input) const {
    const auto* q_weight = find_mapped_tensor(prefix, "attn_q.weight");
    const auto* k_weight = find_mapped_tensor(prefix, "attn_k.weight");
    const auto* v_weight = find_mapped_tensor(prefix, "attn_v.weight");
    if (q_weight == nullptr) {
        return Result<AttentionQkv>::failure("tensor not found: " + prefix + ".attn_q.weight");
    }
    if (k_weight == nullptr) {
        return Result<AttentionQkv>::failure("tensor not found: " + prefix + ".attn_k.weight");
    }
    if (v_weight == nullptr) {
        return Result<AttentionQkv>::failure("tensor not found: " + prefix + ".attn_v.weight");
    }

    const auto q = linear_projection(input, *q_weight, find_mapped_tensor(prefix, "attn_q.bias"));
    if (!q.ok()) {
        return Result<AttentionQkv>::failure(q.error());
    }
    const auto k = linear_projection(input, *k_weight, find_mapped_tensor(prefix, "attn_k.bias"));
    if (!k.ok()) {
        return Result<AttentionQkv>::failure(k.error());
    }
    const auto v = linear_projection(input, *v_weight, find_mapped_tensor(prefix, "attn_v.bias"));
    if (!v.ok()) {
        return Result<AttentionQkv>::failure(v.error());
    }

    AttentionQkv result;
    result.q = q.value();
    result.k = k.value();
    result.v = v.value();
    return Result<AttentionQkv>::success(result);
}

Result<std::vector<float>> HunyuanDitModel::run_attention_block(
    const std::string& prefix,
    const std::vector<float>& input,
    std::size_t tokens,
    std::size_t heads,
    std::size_t head_dim) const {
    return run_cross_attention_block(prefix, input, tokens, input, tokens, heads, head_dim);
}

Result<std::vector<float>> HunyuanDitModel::run_cross_attention_block(
    const std::string& prefix,
    const std::vector<float>& input,
    std::size_t tokens,
    const std::vector<float>& context,
    std::size_t context_tokens,
    std::size_t heads,
    std::size_t head_dim) const {
    const bool is_self_attention = input.data() == context.data() && tokens == context_tokens;
    const std::string attention_name = is_self_attention ? "attn1" : "attn2";
    const auto q_weight = find_mapped_tensor(prefix, attention_suffix(attention_name, "q.weight"));
    const auto k_weight = find_mapped_tensor(prefix, attention_suffix(attention_name, "k.weight"));
    const auto v_weight = find_mapped_tensor(prefix, attention_suffix(attention_name, "v.weight"));
    const auto out_weight = find_mapped_tensor(prefix, attention_suffix(attention_name, "output.weight"));
    if (q_weight == nullptr) {
        return Result<std::vector<float>>::failure("tensor not found: " + prefix + "." + attention_name + ".to_q.weight");
    }
    if (k_weight == nullptr) {
        return Result<std::vector<float>>::failure("tensor not found: " + prefix + "." + attention_name + ".to_k.weight");
    }
    if (v_weight == nullptr) {
        return Result<std::vector<float>>::failure("tensor not found: " + prefix + "." + attention_name + ".to_v.weight");
    }
    if (out_weight == nullptr) {
        return Result<std::vector<float>>::failure("tensor not found: " + prefix + "." + attention_name + ".out_proj.weight");
    }

    const auto q_bias = find_mapped_tensor(prefix, attention_suffix(attention_name, "q.bias"));
    const auto k_bias = find_mapped_tensor(prefix, attention_suffix(attention_name, "k.bias"));
    const auto v_bias = find_mapped_tensor(prefix, attention_suffix(attention_name, "v.bias"));
    const auto out_bias = find_mapped_tensor(prefix, attention_suffix(attention_name, "output.bias"));

    const auto q = linear_projection_batch(input, tokens, *q_weight, q_bias);
    if (!q.ok()) {
        return Result<std::vector<float>>::failure(q.error());
    }
    const auto k = linear_projection_batch(context, context_tokens, *k_weight, k_bias);
    if (!k.ok()) {
        return Result<std::vector<float>>::failure(k.error());
    }
    const auto v = linear_projection_batch(context, context_tokens, *v_weight, v_bias);
    if (!v.ok()) {
        return Result<std::vector<float>>::failure(v.error());
    }

    std::vector<float> q_values = q.value();
    const auto q_norm_weight = find_mapped_tensor(prefix, attention_suffix(attention_name, "q_norm.weight"));
    if (q_norm_weight != nullptr) {
        const auto q_norm = rms_norm_attention_heads(q.value(), tokens, heads, head_dim, q_norm_weight);
        if (!q_norm.ok()) {
            return Result<std::vector<float>>::failure(q_norm.error());
        }
        q_values = q_norm.value();
    }

    std::vector<float> k_values = k.value();
    const auto k_norm_weight = find_mapped_tensor(prefix, attention_suffix(attention_name, "k_norm.weight"));
    if (k_norm_weight != nullptr) {
        const auto k_norm = rms_norm_attention_heads(k.value(), context_tokens, heads, head_dim, k_norm_weight);
        if (!k_norm.ok()) {
            return Result<std::vector<float>>::failure(k_norm.error());
        }
        k_values = k_norm.value();
    }

    const auto attended = scaled_dot_product_attention(
        q_values,
        k_values,
        v.value(),
        tokens,
        context_tokens,
        heads,
        head_dim);
    if (!attended.ok()) {
        return Result<std::vector<float>>::failure(attended.error());
    }

    return linear_projection_batch(attended.value(), tokens, *out_weight, out_bias);
}

Result<std::vector<float>> HunyuanDitModel::run_dit_block(
    const std::string& prefix,
    const std::vector<float>& input,
    std::size_t tokens,
    std::size_t heads,
    std::size_t head_dim) const {
    return run_dit_block_conditioned(prefix, input, tokens, {}, 0, heads, head_dim);
}

Result<std::vector<float>> HunyuanDitModel::run_dit_block_conditioned(
    const std::string& prefix,
    const std::vector<float>& input,
    std::size_t tokens,
    const std::vector<float>& context,
    std::size_t context_tokens,
    std::size_t heads,
    std::size_t head_dim) const {
    if (tokens == 0 || input.empty() || input.size() % tokens != 0) {
        return Result<std::vector<float>>::failure("DiT block input must be non-empty and divisible by tokens");
    }
    if (!context.empty() && (context_tokens == 0 || context.size() % context_tokens != 0)) {
        return Result<std::vector<float>>::failure("DiT block context must be divisible by context tokens");
    }

    const auto norm1 = layer_norm_batch(
        input,
        tokens,
        find_tensor(prefix + ".norm1.weight"),
        find_tensor(prefix + ".norm1.bias"));
    if (!norm1.ok()) {
        return Result<std::vector<float>>::failure(norm1.error());
    }

    const auto attention = run_attention_block(prefix, norm1.value(), tokens, heads, head_dim);
    if (!attention.ok()) {
        return Result<std::vector<float>>::failure(attention.error());
    }
    if (attention.value().size() != input.size()) {
        return Result<std::vector<float>>::failure("DiT attention output size does not match residual input");
    }

    std::vector<float> hidden(input.size(), 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) {
        hidden[i] = input[i] + attention.value()[i];
    }

    if (!context.empty()) {
        const auto norm2_cross = layer_norm_batch(
            hidden,
            tokens,
            find_tensor(prefix + ".norm2.weight"),
            find_tensor(prefix + ".norm2.bias"));
        if (!norm2_cross.ok()) {
            return Result<std::vector<float>>::failure(norm2_cross.error());
        }

        const auto cross_attention = run_cross_attention_block(
            prefix,
            norm2_cross.value(),
            tokens,
            context,
            context_tokens,
            heads,
            head_dim);
        if (!cross_attention.ok()) {
            return Result<std::vector<float>>::failure(cross_attention.error());
        }
        if (cross_attention.value().size() != hidden.size()) {
            return Result<std::vector<float>>::failure("DiT cross-attention output size does not match residual input");
        }
        for (std::size_t i = 0; i < hidden.size(); ++i) {
            hidden[i] += cross_attention.value()[i];
        }
    }

    const auto norm3_weight = find_tensor(prefix + ".norm3.weight");
    const auto norm3_bias = find_tensor(prefix + ".norm3.bias");
    const auto norm2 = layer_norm_batch(
        hidden,
        tokens,
        norm3_weight != nullptr ? norm3_weight : find_tensor(prefix + ".norm2.weight"),
        norm3_weight != nullptr ? norm3_bias : find_tensor(prefix + ".norm2.bias"));
    if (!norm2.ok()) {
        return Result<std::vector<float>>::failure(norm2.error());
    }

    Result<std::vector<float>> feed_forward = Result<std::vector<float>>::failure("feed-forward not run");
    const auto fc1 = find_mapped_tensor(prefix, "mlp_fc1.weight");
    const auto fc2 = find_mapped_tensor(prefix, "mlp_fc2.weight");
    const auto moe_gate = find_tensor(prefix + ".moe.gate.weight");
    if (fc1 != nullptr || fc2 != nullptr) {
        if (fc1 == nullptr) {
            return Result<std::vector<float>>::failure("tensor not found: " + prefix + ".mlp_fc1.weight");
        }
        if (fc2 == nullptr) {
            return Result<std::vector<float>>::failure("tensor not found: " + prefix + ".mlp_fc2.weight");
        }
        const auto mlp_hidden = linear_projection_batch(norm2.value(), tokens, *fc1, find_mapped_tensor(prefix, "mlp_fc1.bias"));
        if (!mlp_hidden.ok()) {
            return Result<std::vector<float>>::failure(mlp_hidden.error());
        }
        const auto activated = gelu(mlp_hidden.value());
        feed_forward = linear_projection_batch(activated, tokens, *fc2, find_mapped_tensor(prefix, "mlp_fc2.bias"));
    } else if (moe_gate != nullptr) {
        feed_forward = run_moe_block(prefix, norm2.value(), tokens, 2);
    } else {
        return Result<std::vector<float>>::success(hidden);
    }
    if (!feed_forward.ok()) {
        return Result<std::vector<float>>::failure(feed_forward.error());
    }
    if (feed_forward.value().size() != hidden.size()) {
        return Result<std::vector<float>>::failure("DiT feed-forward output size does not match residual input");
    }

    std::vector<float> output(hidden.size(), 0.0f);
    for (std::size_t i = 0; i < hidden.size(); ++i) {
        output[i] = hidden[i] + feed_forward.value()[i];
    }
    return Result<std::vector<float>>::success(output);
}

Result<std::vector<float>> HunyuanDitModel::run_dit_blocks_conditioned(
    std::uint32_t first_block,
    std::uint32_t block_count,
    const std::vector<float>& input,
    std::size_t tokens,
    const std::vector<float>& context,
    std::size_t context_tokens,
    std::size_t heads,
    std::size_t head_dim) const {
    if (block_count == 0) {
        return Result<std::vector<float>>::failure("DiT block count must be positive");
    }
    std::vector<float> hidden = input;
    std::vector<std::vector<float>> skip_stack;
    for (std::uint32_t offset = 0; offset < block_count; ++offset) {
        const auto block_index = first_block + offset;
        const auto prefix = std::string("blocks.") + std::to_string(block_index);
        const auto* skip_weight = find_tensor(prefix + ".skip_linear.weight");
        if (skip_weight != nullptr && !skip_stack.empty()) {
            const auto skip = std::move(skip_stack.back());
            skip_stack.pop_back();
            if (skip.size() != hidden.size()) {
                return Result<std::vector<float>>::failure("DiT skip value size does not match hidden state");
            }
            const auto features = hidden.size() / tokens;
            std::vector<float> concatenated;
            concatenated.reserve(hidden.size() * 2);
            for (std::size_t token = 0; token < tokens; ++token) {
                const auto base = token * features;
                concatenated.insert(concatenated.end(), skip.begin() + static_cast<std::ptrdiff_t>(base), skip.begin() + static_cast<std::ptrdiff_t>(base + features));
                concatenated.insert(concatenated.end(), hidden.begin() + static_cast<std::ptrdiff_t>(base), hidden.begin() + static_cast<std::ptrdiff_t>(base + features));
            }
            const auto projected = linear_projection_batch(concatenated, tokens, *skip_weight, find_tensor(prefix + ".skip_linear.bias"));
            if (!projected.ok()) {
                return Result<std::vector<float>>::failure(projected.error());
            }
            const auto normalized = layer_norm_batch(
                projected.value(),
                tokens,
                find_tensor(prefix + ".skip_norm.weight"),
                find_tensor(prefix + ".skip_norm.bias"));
            if (!normalized.ok()) {
                return Result<std::vector<float>>::failure(normalized.error());
            }
            hidden = normalized.value();
        }

        const auto output = run_dit_block_conditioned(prefix, hidden, tokens, context, context_tokens, heads, head_dim);
        if (!output.ok()) {
            return Result<std::vector<float>>::failure(output.error());
        }
        hidden = output.value();
        if (skip_weight == nullptr) {
            skip_stack.push_back(hidden);
        }
    }
    return Result<std::vector<float>>::success(hidden);
}

Result<std::vector<float>> HunyuanDitModel::run_moe_block(
    const std::string& prefix,
    const std::vector<float>& input,
    std::size_t tokens,
    std::size_t top_k) const {
    const auto* gate = find_tensor(prefix + ".moe.gate.weight");
    if (gate == nullptr) {
        return Result<std::vector<float>>::failure("tensor not found: " + prefix + ".moe.gate.weight");
    }
    if (gate->dimensions.size() != 2) {
        return Result<std::vector<float>>::failure("MoE gate must be 2D: " + gate->name);
    }
    const auto experts = static_cast<std::size_t>(gate->dimensions[0]);
    const auto features = static_cast<std::size_t>(gate->dimensions[1]);
    if (experts == 0 || features == 0 || top_k == 0) {
        return Result<std::vector<float>>::failure("MoE dimensions and top-k must be positive");
    }
    if (input.size() != tokens * features) {
        return Result<std::vector<float>>::failure("MoE input size does not match gate width");
    }

    const auto logits = linear_projection_batch(input, tokens, *gate, nullptr);
    if (!logits.ok()) {
        return Result<std::vector<float>>::failure(logits.error());
    }

    std::vector<float> output(input.size(), 0.0f);
    const auto selected_count = std::min(top_k, experts);
    for (std::size_t token = 0; token < tokens; ++token) {
        const auto scores_begin = logits.value().begin() + static_cast<std::ptrdiff_t>(token * experts);
        const std::vector<float> token_logits(scores_begin, scores_begin + static_cast<std::ptrdiff_t>(experts));
        const auto scores = softmax(token_logits);
        if (!scores.ok()) {
            return Result<std::vector<float>>::failure(scores.error());
        }
        std::vector<std::pair<float, std::size_t>> ranked;
        ranked.reserve(experts);
        for (std::size_t expert = 0; expert < experts; ++expert) {
            ranked.emplace_back(scores.value()[expert], expert);
        }
        std::partial_sort(
            ranked.begin(),
            ranked.begin() + static_cast<std::ptrdiff_t>(selected_count),
            ranked.end(),
            [](const auto& left, const auto& right) { return left.first > right.first; });

        const auto token_input = token_slice(input, token, features);
        for (std::size_t rank = 0; rank < selected_count; ++rank) {
            const auto expert = ranked[rank].second;
            const auto expert_prefix = prefix + ".moe.experts." + std::to_string(expert);
            const auto* fc1 = find_tensor(expert_prefix + ".net.0.proj.weight");
            const auto* fc2 = find_tensor(expert_prefix + ".net.2.weight");
            if (fc1 == nullptr || fc2 == nullptr) {
                continue;
            }
            const auto hidden = linear_projection(token_input, *fc1, find_tensor(expert_prefix + ".net.0.proj.bias"));
            if (!hidden.ok()) {
                return Result<std::vector<float>>::failure(hidden.error());
            }
            const auto activated = gelu(hidden.value());
            const auto expert_out = linear_projection(activated, *fc2, find_tensor(expert_prefix + ".net.2.bias"));
            if (!expert_out.ok()) {
                return Result<std::vector<float>>::failure(expert_out.error());
            }
            for (std::size_t feature = 0; feature < features; ++feature) {
                output[token * features + feature] += ranked[rank].first * expert_out.value()[feature];
            }
        }
    }

    const auto* shared_fc1 = find_tensor(prefix + ".moe.shared_experts.net.0.proj.weight");
    const auto* shared_fc2 = find_tensor(prefix + ".moe.shared_experts.net.2.weight");
    if (shared_fc1 != nullptr && shared_fc2 != nullptr) {
        const auto hidden = linear_projection_batch(input, tokens, *shared_fc1, find_tensor(prefix + ".moe.shared_experts.net.0.proj.bias"));
        if (!hidden.ok()) {
            return Result<std::vector<float>>::failure(hidden.error());
        }
        const auto activated = gelu(hidden.value());
        const auto shared_out = linear_projection_batch(activated, tokens, *shared_fc2, find_tensor(prefix + ".moe.shared_experts.net.2.bias"));
        if (!shared_out.ok()) {
            return Result<std::vector<float>>::failure(shared_out.error());
        }
        if (shared_out.value().size() != output.size()) {
            return Result<std::vector<float>>::failure("MoE shared expert output size mismatch");
        }
        for (std::size_t i = 0; i < output.size(); ++i) {
            output[i] += shared_out.value()[i];
        }
    }

    return Result<std::vector<float>>::success(output);
}

Result<std::vector<float>> HunyuanDitModel::project_timestep_conditioning(float timestep, std::size_t width) const {
    const auto embedding = timestep_sincos_embedding(timestep, width);
    if (!embedding.ok()) {
        return Result<std::vector<float>>::failure(embedding.error());
    }

    const auto* fc1 = find_tensor("t_embedder.mlp.0.weight");
    const auto* fc2 = find_tensor("t_embedder.mlp.2.weight");
    if (fc1 == nullptr && fc2 == nullptr) {
        return Result<std::vector<float>>::success(embedding.value());
    }
    if (fc1 == nullptr) {
        return Result<std::vector<float>>::failure("tensor not found: t_embedder.mlp.0.weight");
    }
    if (fc2 == nullptr) {
        return Result<std::vector<float>>::failure("tensor not found: t_embedder.mlp.2.weight");
    }

    const auto hidden = linear_projection(embedding.value(), *fc1, find_tensor("t_embedder.mlp.0.bias"));
    if (!hidden.ok()) {
        return Result<std::vector<float>>::failure(hidden.error());
    }
    const auto activated = gelu(hidden.value());
    return linear_projection(activated, *fc2, find_tensor("t_embedder.mlp.2.bias"));
}

Result<std::string> HunyuanDitModel::plan_forward(
    const std::vector<std::uint64_t>& latent_shape,
    const std::vector<std::uint64_t>& condition_shape) const {
    if (latent_shape.empty()) {
        return Result<std::string>::failure("latent shape must not be empty");
    }
    if (condition_shape.empty()) {
        return Result<std::string>::failure("condition shape must not be empty");
    }
    return Result<std::string>::failure("DiT forward is not implemented yet");
}

Result<DiffusionScheduler> DiffusionScheduler::make_linear(int steps) {
    if (steps <= 0) {
        return Result<DiffusionScheduler>::failure("scheduler steps must be positive");
    }

    DiffusionScheduler scheduler;
    scheduler.timesteps_.reserve(static_cast<std::size_t>(steps));
    for (int step = steps - 1; step >= 0; --step) {
        scheduler.timesteps_.push_back(step);
    }
    return Result<DiffusionScheduler>::success(scheduler);
}

const std::vector<int>& DiffusionScheduler::timesteps() const {
    return timesteps_;
}

Result<SchedulerStep> DiffusionScheduler::step_euler(
    const std::vector<float>& sample,
    const std::vector<float>& model_output,
    float sigma_current,
    float sigma_next) const {
    if (sample.size() != model_output.size()) {
        return Result<SchedulerStep>::failure("Euler scheduler sample and model output sizes must match");
    }
    SchedulerStep step;
    step.sample.resize(sample.size());
    const float delta = sigma_next - sigma_current;
    for (std::size_t i = 0; i < sample.size(); ++i) {
        step.sample[i] = sample[i] + model_output[i] * delta;
    }
    return Result<SchedulerStep>::success(step);
}

Result<std::string> MeshDecoder::decode_latents(const std::vector<std::uint64_t>& latent_shape) const {
    if (latent_shape.empty()) {
        return Result<std::string>::failure("latent shape must not be empty");
    }
    return Result<std::string>::failure("mesh decoder is not implemented yet");
}

Result<DecodedMesh> MeshDecoder::decode_density_grid(
    const std::vector<float>& density,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t depth,
    float threshold) const {
    if (width == 0 || height == 0 || depth == 0) {
        return Result<DecodedMesh>::failure("density grid dimensions must be positive");
    }
    const auto expected = static_cast<std::size_t>(width) * height * depth;
    if (density.size() != expected) {
        return Result<DecodedMesh>::failure("density grid size does not match width * height * depth");
    }

    DecodedMesh mesh;
    auto index = [&](std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        return (static_cast<std::size_t>(z) * height + y) * width + x;
    };
    auto occupied = [&](std::int32_t x, std::int32_t y, std::int32_t z) {
        if (x < 0 || y < 0 || z < 0) {
            return false;
        }
        if (x >= static_cast<std::int32_t>(width) || y >= static_cast<std::int32_t>(height) || z >= static_cast<std::int32_t>(depth)) {
            return false;
        }
        return density[index(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y), static_cast<std::uint32_t>(z))] >= threshold;
    };

    struct Face {
        std::array<float, 3> normal;
        std::array<std::array<float, 3>, 4> corners;
    };

    const std::array<Face, 6> faces = {{
        {{{-1.0f, 0.0f, 0.0f}}, {{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 0.0f}}}},
        {{{1.0f, 0.0f, 0.0f}}, {{{1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 0.0f, 1.0f}}}},
        {{{0.0f, -1.0f, 0.0f}}, {{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}}}},
        {{{0.0f, 1.0f, 0.0f}}, {{{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 0.0f}}}},
        {{{0.0f, 0.0f, -1.0f}}, {{{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}}}},
        {{{0.0f, 0.0f, 1.0f}}, {{{0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 1.0f}}}},
    }};

    std::ostringstream obj;
    const std::array<std::array<std::int32_t, 3>, 6> neighbor_offsets = {{
        {{-1, 0, 0}},
        {{1, 0, 0}},
        {{0, -1, 0}},
        {{0, 1, 0}},
        {{0, 0, -1}},
        {{0, 0, 1}},
    }};

    for (std::uint32_t z = 0; z < depth; ++z) {
        for (std::uint32_t y = 0; y < height; ++y) {
            for (std::uint32_t x = 0; x < width; ++x) {
                if (!occupied(static_cast<std::int32_t>(x), static_cast<std::int32_t>(y), static_cast<std::int32_t>(z))) {
                    continue;
                }
                for (std::size_t face_index = 0; face_index < faces.size(); ++face_index) {
                    const auto& neighbor = neighbor_offsets[face_index];
                    if (occupied(static_cast<std::int32_t>(x) + neighbor[0], static_cast<std::int32_t>(y) + neighbor[1], static_cast<std::int32_t>(z) + neighbor[2])) {
                        continue;
                    }
                    const auto first_vertex = static_cast<std::uint32_t>(mesh.vertices.size() / 3);
                    for (const auto& corner : faces[face_index].corners) {
                        mesh.vertices.push_back(static_cast<float>(x) + corner[0]);
                        mesh.vertices.push_back(static_cast<float>(y) + corner[1]);
                        mesh.vertices.push_back(static_cast<float>(z) + corner[2]);
                    }
                    mesh.indices.insert(mesh.indices.end(), {
                        first_vertex,
                        first_vertex + 1,
                        first_vertex + 2,
                        first_vertex,
                        first_vertex + 2,
                        first_vertex + 3,
                    });
                }
            }
        }
    }

    for (std::size_t i = 0; i < mesh.vertices.size(); i += 3) {
        obj << "v " << mesh.vertices[i] << " " << mesh.vertices[i + 1] << " " << mesh.vertices[i + 2] << "\n";
    }
    for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
        obj << "f " << (mesh.indices[i] + 1) << " " << (mesh.indices[i + 1] + 1) << " " << (mesh.indices[i + 2] + 1) << "\n";
    }
    mesh.obj = obj.str();
    return Result<DecodedMesh>::success(mesh);
}

} // namespace hy3d
