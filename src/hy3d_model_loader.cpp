#include "hy3d_model_loader.h"

#include "hy3d_gguf.h"

#include <algorithm>
#include <utility>

namespace hy3d {

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
    }
    if (include_timestep) {
        names.push_back("t_embedder.mlp.0.weight");
        names.push_back("t_embedder.mlp.0.bias");
        names.push_back("t_embedder.mlp.2.weight");
        names.push_back("t_embedder.mlp.2.bias");
    }
    return names;
}

Result<HunyuanDitModel> load_hunyuan_dit_block_from_gguf(
    const std::string& path,
    std::uint32_t block_index,
    bool include_mlp,
    bool include_cross_attention,
    bool include_timestep) {
    const auto info = inspect_gguf(path);
    if (!info.ok()) {
        return Result<HunyuanDitModel>::failure(info.error());
    }

    HunyuanDitModel model;
    const auto names = hunyuan_dit_block_tensor_names(block_index, include_mlp, include_cross_attention, include_timestep);
    for (const auto& name : names) {
        const auto tensor_info = std::find_if(
            info.value().tensor_infos.begin(),
            info.value().tensor_infos.end(),
            [&](const GgufTensorInfo& tensor) { return tensor.name == name; });
        if (tensor_info == info.value().tensor_infos.end()) {
            if (name.find(".bias") != std::string::npos || name.find(".norm") != std::string::npos ||
                name.find("_norm") != std::string::npos) {
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

} // namespace hy3d
