#pragma once

#include "hy3d_gguf.h"
#include "hy3d_result.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hy3d {

struct RuntimeTensor {
    std::string name;
    GgmlType type = GgmlType::F32;
    std::vector<std::uint64_t> dimensions;
    std::vector<std::uint8_t> bytes;
};

struct AttentionQkv {
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
};

struct SchedulerStep {
    std::vector<float> sample;
};

struct DecodedMesh {
    std::vector<float> vertices;
    std::vector<std::uint32_t> indices;
    std::string obj;
};

class HunyuanDitModel {
public:
    void add_tensor(RuntimeTensor tensor);
    [[nodiscard]] std::size_t tensor_count() const;
    [[nodiscard]] bool has_tensor(const std::string& name) const;
    [[nodiscard]] const RuntimeTensor* find_tensor(const std::string& name) const;
    [[nodiscard]] const RuntimeTensor* find_mapped_tensor(
        const std::string& prefix,
        const std::string& canonical_suffix) const;

    Result<std::vector<float>> project_linear(
        const std::string& weight_name,
        const std::string& bias_name,
        const std::vector<float>& input) const;

    Result<AttentionQkv> project_attention_qkv(
        const std::string& prefix,
        const std::vector<float>& input) const;

    Result<std::vector<float>> run_attention_block(
        const std::string& prefix,
        const std::vector<float>& input,
        std::size_t tokens,
        std::size_t heads,
        std::size_t head_dim) const;

    Result<std::vector<float>> run_cross_attention_block(
        const std::string& prefix,
        const std::vector<float>& input,
        std::size_t tokens,
        const std::vector<float>& context,
        std::size_t context_tokens,
        std::size_t heads,
        std::size_t head_dim) const;

    Result<std::vector<float>> run_dit_block(
        const std::string& prefix,
        const std::vector<float>& input,
        std::size_t tokens,
        std::size_t heads,
        std::size_t head_dim) const;

    Result<std::vector<float>> run_dit_block_conditioned(
        const std::string& prefix,
        const std::vector<float>& input,
        std::size_t tokens,
        const std::vector<float>& context,
        std::size_t context_tokens,
        std::size_t heads,
        std::size_t head_dim) const;

    Result<std::vector<float>> run_dit_blocks_conditioned(
        std::uint32_t first_block,
        std::uint32_t block_count,
        const std::vector<float>& input,
        std::size_t tokens,
        const std::vector<float>& context,
        std::size_t context_tokens,
        std::size_t heads,
        std::size_t head_dim) const;

    Result<std::vector<float>> run_moe_block(
        const std::string& prefix,
        const std::vector<float>& input,
        std::size_t tokens,
        std::size_t top_k = 2) const;

    Result<std::vector<float>> project_timestep_conditioning(float timestep, std::size_t width) const;

    Result<std::string> plan_forward(
        const std::vector<std::uint64_t>& latent_shape,
        const std::vector<std::uint64_t>& condition_shape) const;

private:
    std::vector<RuntimeTensor> tensors_;
};

class DiffusionScheduler {
public:
    static Result<DiffusionScheduler> make_linear(int steps);

    [[nodiscard]] const std::vector<int>& timesteps() const;
    Result<SchedulerStep> step_euler(
        const std::vector<float>& sample,
        const std::vector<float>& model_output,
        float sigma_current,
        float sigma_next) const;

private:
    std::vector<int> timesteps_;
};

class MeshDecoder {
public:
    Result<std::string> decode_latents(const std::vector<std::uint64_t>& latent_shape) const;
    Result<DecodedMesh> decode_density_grid(
        const std::vector<float>& density,
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t depth,
        float threshold) const;
};

} // namespace hy3d
