#include "hy3d_runtime.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#undef assert
#define assert(expr)                                                                                                   \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            std::cerr << "assertion failed: " #expr << "\n";                                                          \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

namespace {

void require_near(float actual, float expected, const char* message) {
    if (std::fabs(actual - expected) > 0.001f) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << "\n";
        std::exit(1);
    }
}

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

} // namespace

int main() {
    hy3d::RuntimeTensor tensor;
    tensor.name = "blocks.0.attn1.to_q.weight";
    tensor.type = hy3d::GgmlType::F16;
    tensor.dimensions = {2048, 2048};
    tensor.bytes.resize(2048);

    hy3d::HunyuanDitModel dit;
    dit.add_tensor(tensor);
    assert(dit.tensor_count() == 1);
    assert(dit.has_tensor("blocks.0.attn1.to_q.weight"));
    assert(!dit.has_tensor("blocks.0.attn1.to_k.weight"));

    const auto plan = dit.plan_forward({1, 2048}, {1, 1024});
    assert(!plan.ok());
    assert(plan.error().find("DiT forward is not implemented yet") != std::string::npos);

    hy3d::RuntimeTensor q_weight;
    q_weight = f32_tensor("blk.0.attn_q.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});

    hy3d::RuntimeTensor k_weight = q_weight;
    k_weight.name = "blk.0.attn_k.weight";

    hy3d::RuntimeTensor v_weight = q_weight;
    v_weight.name = "blk.0.attn_v.weight";

    dit.add_tensor(q_weight);
    dit.add_tensor(k_weight);
    dit.add_tensor(v_weight);
    const auto qkv = dit.project_attention_qkv("blk.0", {3.0f, 4.0f});
    assert(qkv.ok());
    assert(qkv.value().q == std::vector<float>({3.0f, 4.0f}));
    assert(qkv.value().k == std::vector<float>({3.0f, 4.0f}));
    assert(qkv.value().v == std::vector<float>({3.0f, 4.0f}));

    hy3d::RuntimeTensor out_weight = q_weight;
    out_weight.name = "blk.0.attn_output.weight";
    dit.add_tensor(out_weight);

    const auto attention = dit.run_attention_block("blk.0", {1.0f, 0.0f, 0.0f, 1.0f}, 2, 1, 2);
    assert(attention.ok());
    assert(attention.value().size() == 4);
    require_near(attention.value()[0], 0.6697615f, "attention block token0 dim0");
    require_near(attention.value()[1], 0.3302384f, "attention block token0 dim1");
    require_near(attention.value()[2], 0.3302384f, "attention block token1 dim0");
    require_near(attention.value()[3], 0.6697615f, "attention block token1 dim1");

    const auto block = dit.run_dit_block("blk.0", {1.0f, 0.0f, 0.0f, 1.0f}, 2, 1, 2);
    assert(block.ok());
    assert(block.value().size() == 4);

    dit.add_tensor(f32_tensor("blk.0.mlp_fc1.weight", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}));
    dit.add_tensor(f32_tensor("blk.0.mlp_fc2.weight", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}));
    const auto block_with_mlp = dit.run_dit_block("blk.0", {1.0f, 0.0f, 0.0f, 1.0f}, 2, 1, 2);
    assert(block_with_mlp.ok());
    assert(block_with_mlp.value().size() == 4);

    const auto schedule = hy3d::DiffusionScheduler::make_linear(4);
    assert(schedule.ok());
    assert(schedule.value().timesteps() == std::vector<int>({3, 2, 1, 0}));
    const auto stepped = schedule.value().step_euler({1.0f, 2.0f}, {0.5f, -1.0f}, 1.0f, 0.25f);
    assert(stepped.ok());
    require_near(stepped.value().sample[0], 0.625f, "Euler step dim0");
    require_near(stepped.value().sample[1], 2.75f, "Euler step dim1");

    hy3d::MeshDecoder decoder;
    const auto mesh = decoder.decode_latents({1, 64, 64, 64});
    assert(!mesh.ok());
    assert(mesh.error().find("mesh decoder is not implemented yet") != std::string::npos);
    const auto decoded = decoder.decode_density_grid({1.0f}, 1, 1, 1, 0.5f);
    assert(decoded.ok());
    assert(decoded.value().vertices.size() == 72);
    assert(decoded.value().indices.size() == 36);
    assert(decoded.value().obj.find("v 0 0 0") != std::string::npos);
    assert(decoded.value().obj.find("f 1 2 3") != std::string::npos);

    hy3d::HunyuanDitModel official_names;
    official_names.add_tensor(f32_tensor("blocks.0.attn1.to_q.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}));
    official_names.add_tensor(f32_tensor("blocks.0.attn1.to_k.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}));
    official_names.add_tensor(f32_tensor("blocks.0.attn1.to_v.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}));
    official_names.add_tensor(f32_tensor("blocks.0.attn1.out_proj.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}));
    official_names.add_tensor(f32_tensor("blocks.0.attn1.q_norm.weight", {2}, {1.0f, 1.0f}));
    official_names.add_tensor(f32_tensor("blocks.0.attn1.k_norm.weight", {2}, {1.0f, 1.0f}));
    official_names.add_tensor(f32_tensor("blocks.0.mlp.fc1.weight", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}));
    official_names.add_tensor(f32_tensor("blocks.0.mlp.fc2.weight", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}));
    assert(official_names.find_mapped_tensor("blocks.0", "attn_q.weight") != nullptr);
    assert(official_names.find_mapped_tensor("blocks.0", "attn1_q_norm.weight") != nullptr);
    assert(official_names.find_mapped_tensor("blocks.0", "mlp_fc1.weight") != nullptr);
    const auto official_block = official_names.run_dit_block("blocks.0", {1.0f, 0.0f, 0.0f, 1.0f}, 2, 1, 2);
    assert(official_block.ok());
    assert(official_block.value().size() == 4);

    hy3d::HunyuanDitModel conditioned;
    conditioned.add_tensor(f32_tensor("blocks.0.attn1.to_q.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}));
    conditioned.add_tensor(f32_tensor("blocks.0.attn1.to_k.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}));
    conditioned.add_tensor(f32_tensor("blocks.0.attn1.to_v.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}));
    conditioned.add_tensor(f32_tensor("blocks.0.attn1.out_proj.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}));
    conditioned.add_tensor(f32_tensor("blocks.0.attn2.to_q.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}));
    conditioned.add_tensor(f32_tensor("blocks.0.attn2.to_k.weight", {2, 3}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}));
    conditioned.add_tensor(f32_tensor("blocks.0.attn2.to_v.weight", {2, 3}, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f}));
    conditioned.add_tensor(f32_tensor("blocks.0.attn2.out_proj.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}));
    conditioned.add_tensor(f32_tensor("blocks.0.attn2.q_norm.weight", {2}, {1.0f, 1.0f}));
    conditioned.add_tensor(f32_tensor("blocks.0.attn2.k_norm.weight", {2}, {1.0f, 1.0f}));
    assert(conditioned.find_mapped_tensor("blocks.0", "attn2_q.weight") != nullptr);
    const auto conditioned_block = conditioned.run_dit_block_conditioned(
        "blocks.0",
        {1.0f, 0.0f, 0.0f, 1.0f},
        2,
        {1.0f, 2.0f, 3.0f},
        1,
        1,
        2);
    assert(conditioned_block.ok());
    assert(conditioned_block.value().size() == 4);

    conditioned.add_tensor(f32_tensor("t_embedder.mlp.0.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}));
    conditioned.add_tensor(f32_tensor("t_embedder.mlp.0.bias", {2}, {0.0f, 0.0f}));
    conditioned.add_tensor(f32_tensor("t_embedder.mlp.2.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}));
    conditioned.add_tensor(f32_tensor("t_embedder.mlp.2.bias", {2}, {0.0f, 0.0f}));
    const auto timestep = conditioned.project_timestep_conditioning(1.0f, 2);
    assert(timestep.ok());
    assert(timestep.value().size() == 2);

    return 0;
}
