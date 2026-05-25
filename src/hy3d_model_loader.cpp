#include "hy3d_model_loader.h"

#include "hy3d_gguf.h"

#include <algorithm>
#include <utility>

namespace hy3d {
namespace {

bool is_optional_tensor_name(const std::string& name) {
    return name.find(".bias") != std::string::npos || name.find(".norm") != std::string::npos ||
           name.find("_norm") != std::string::npos || name.find(".mlp.") != std::string::npos ||
           name.find(".moe.") != std::string::npos || name.find(".skip_") != std::string::npos ||
           name.find("pooler.") != std::string::npos || name.find("extra_embedder.") != std::string::npos ||
           name.find("t_embedder.") != std::string::npos;
}

Result<HunyuanDitModel> load_named_tensors_from_gguf(const std::string& path, const std::vector<std::string>& names) {
    const auto info = inspect_gguf(path);
    if (!info.ok()) {
        return Result<HunyuanDitModel>::failure(info.error());
    }

    HunyuanDitModel model;
    for (const auto& name : names) {
        const auto tensor_info = std::find_if(
            info.value().tensor_infos.begin(),
            info.value().tensor_infos.end(),
            [&](const GgufTensorInfo& tensor) { return tensor.name == name; });
        if (tensor_info == info.value().tensor_infos.end()) {
            if (is_optional_tensor_name(name)) {
                continue;
            }
            return Result<HunyuanDitModel>::failure("required block tensor not found: " + name);
        }

        const auto bytes = read_gguf_tensor_data(path, info.value(), name);
        if (!bytes.ok()) {
            return Result<HunyuanDitModel>::failure(bytes.error());
        }

        RuntimeTensor runtime_tensor;
        runtime_tensor.name = tensor_info->name;
        runtime_tensor.type = tensor_info->type;
        runtime_tensor.dimensions = tensor_info->dimensions;
        runtime_tensor.bytes = bytes.value();
        model.add_tensor(std::move(runtime_tensor));
    }

    return Result<HunyuanDitModel>::success(std::move(model));
}

} // namespace

std::vector<std::string> hunyuan_dit_block_tensor_names(
    std::uint32_t block_index,
    bool include_mlp,
    bool include_cross_attention,
    bool include_timestep) {
    const auto prefix = std::string("blocks.") + std::to_string(block_index);
    std::vector<std::string> names = {
        prefix + ".norm1.weight",
        prefix + ".norm1.bias",
        prefix + ".norm2.weight",
        prefix + ".norm2.bias",
        prefix + ".attn1.q_norm.weight",
        prefix + ".attn1.k_norm.weight",
        prefix + ".attn1.to_q.weight",
        prefix + ".attn1.to_k.weight",
        prefix + ".attn1.to_v.weight",
        prefix + ".attn1.out_proj.weight",
        prefix + ".attn1.out_proj.bias",
        prefix + ".skip_linear.weight",
        prefix + ".skip_linear.bias",
        prefix + ".skip_norm.weight",
        prefix + ".skip_norm.bias",
    };
    if (include_cross_attention) {
        names.push_back(prefix + ".attn2.q_norm.weight");
        names.push_back(prefix + ".attn2.k_norm.weight");
        names.push_back(prefix + ".attn2.to_q.weight");
        names.push_back(prefix + ".attn2.to_k.weight");
        names.push_back(prefix + ".attn2.to_v.weight");
        names.push_back(prefix + ".attn2.out_proj.weight");
        names.push_back(prefix + ".attn2.out_proj.bias");
    }
    if (include_mlp) {
        names.push_back(prefix + ".norm3.weight");
        names.push_back(prefix + ".norm3.bias");
        names.push_back(prefix + ".mlp.fc1.weight");
        names.push_back(prefix + ".mlp.fc1.bias");
        names.push_back(prefix + ".mlp.fc2.weight");
        names.push_back(prefix + ".mlp.fc2.bias");
        names.push_back(prefix + ".moe.gate.weight");
        names.push_back(prefix + ".moe.shared_experts.net.0.proj.weight");
        names.push_back(prefix + ".moe.shared_experts.net.0.proj.bias");
        names.push_back(prefix + ".moe.shared_experts.net.2.weight");
        names.push_back(prefix + ".moe.shared_experts.net.2.bias");
        for (int expert = 0; expert < 8; ++expert) {
            names.push_back(prefix + ".moe.experts." + std::to_string(expert) + ".net.0.proj.weight");
            names.push_back(prefix + ".moe.experts." + std::to_string(expert) + ".net.0.proj.bias");
            names.push_back(prefix + ".moe.experts." + std::to_string(expert) + ".net.2.weight");
            names.push_back(prefix + ".moe.experts." + std::to_string(expert) + ".net.2.bias");
        }
    }
    if (include_timestep) {
        names.push_back("t_embedder.mlp.0.weight");
        names.push_back("t_embedder.mlp.0.bias");
        names.push_back("t_embedder.mlp.2.weight");
        names.push_back("t_embedder.mlp.2.bias");
    }
    return names;
}

std::vector<std::string> hunyuan_dit_block_range_tensor_names(
    std::uint32_t first_block,
    std::uint32_t block_count,
    bool include_mlp,
    bool include_cross_attention,
    bool include_timestep) {
    std::vector<std::string> names;
    for (std::uint32_t offset = 0; offset < block_count; ++offset) {
        const auto block_names = hunyuan_dit_block_tensor_names(
            first_block + offset,
            include_mlp,
            include_cross_attention,
            include_timestep && offset == 0);
        for (const auto& name : block_names) {
            if (std::find(names.begin(), names.end(), name) == names.end()) {
                names.push_back(name);
            }
        }
    }
    return names;
}

std::vector<std::string> hunyuan_dit_forward_tensor_names(
    std::uint32_t first_block,
    std::uint32_t block_count,
    bool include_mlp,
    bool include_cross_attention,
    bool include_timestep) {
    auto names = hunyuan_dit_block_range_tensor_names(
        first_block,
        block_count,
        include_mlp,
        include_cross_attention,
        include_timestep);
    const std::vector<std::string> forward_names = {
        "x_embedder.weight",
        "x_embedder.bias",
        "final_layer.norm_final.weight",
        "final_layer.norm_final.bias",
        "final_layer.linear.weight",
        "final_layer.linear.bias",
        "pooler.positional_embedding",
        "pooler.q_proj.weight",
        "pooler.q_proj.bias",
        "pooler.k_proj.weight",
        "pooler.k_proj.bias",
        "pooler.v_proj.weight",
        "pooler.v_proj.bias",
        "pooler.c_proj.weight",
        "pooler.c_proj.bias",
        "extra_embedder.0.weight",
        "extra_embedder.0.bias",
        "extra_embedder.2.weight",
        "extra_embedder.2.bias",
    };
    names.insert(names.end(), forward_names.begin(), forward_names.end());
    return names;
}

Result<HunyuanDitModel> load_hunyuan_dit_block_from_gguf(
    const std::string& path,
    std::uint32_t block_index,
    bool include_mlp,
    bool include_cross_attention,
    bool include_timestep) {
    const auto names = hunyuan_dit_block_tensor_names(block_index, include_mlp, include_cross_attention, include_timestep);
    return load_named_tensors_from_gguf(path, names);
}

Result<HunyuanDitModel> load_hunyuan_dit_blocks_from_gguf(
    const std::string& path,
    std::uint32_t first_block,
    std::uint32_t block_count,
    bool include_mlp,
    bool include_cross_attention,
    bool include_timestep) {
    const auto names = hunyuan_dit_block_range_tensor_names(
        first_block,
        block_count,
        include_mlp,
        include_cross_attention,
        include_timestep);
    return load_named_tensors_from_gguf(path, names);
}

Result<HunyuanDitModel> load_hunyuan_dit_forward_from_gguf(
    const std::string& path,
    std::uint32_t first_block,
    std::uint32_t block_count,
    bool include_mlp,
    bool include_cross_attention,
    bool include_timestep) {
    const auto names = hunyuan_dit_forward_tensor_names(
        first_block,
        block_count,
        include_mlp,
        include_cross_attention,
        include_timestep);
    return load_named_tensors_from_gguf(path, names);
}

} // namespace hy3d
