#include "hy3d_commands.h"
#include "hy3d_backend.h"
#include "hy3d_gguf.h"
#include "hy3d_model_loader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace hy3d {
namespace {

hy3d::Result<std::vector<float>> read_f32_file(const std::string& path, std::size_t expected_values) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return hy3d::Result<std::vector<float>>::failure("failed to open f32 input file: " + path);
    }
    in.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    if (size != expected_values * sizeof(float)) {
        return hy3d::Result<std::vector<float>>::failure("f32 input byte size mismatch: " + path);
    }
    std::vector<float> values(expected_values, 0.0f);
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(size));
    if (!in) {
        return hy3d::Result<std::vector<float>>::failure("failed to read f32 input file: " + path);
    }
    return hy3d::Result<std::vector<float>>::success(std::move(values));
}

} // namespace

int run_help_command(const CliOptions&) {
    std::cout << help_text();
    return 0;
}

int run_inspect_command(const CliOptions& options) {
        const auto info = hy3d::inspect_gguf(options.model_path);
        if (!info.ok()) {
            std::cerr << "error: " << info.error() << "\n";
            return 1;
        }
        std::cout << hy3d::format_gguf_info(info.value());
        return 0;
}

int run_tensor_command(const CliOptions& options) {
        const auto info = hy3d::inspect_gguf(options.model_path);
        if (!info.ok()) {
            std::cerr << "error: " << info.error() << "\n";
            return 1;
        }
        const auto bytes = hy3d::read_gguf_tensor_data(options.model_path, info.value(), options.tensor_name);
        if (!bytes.ok()) {
            std::cerr << "error: " << bytes.error() << "\n";
            return 1;
        }

        const auto it = std::find_if(
            info.value().tensor_infos.begin(),
            info.value().tensor_infos.end(),
            [&](const hy3d::GgufTensorInfo& tensor) { return tensor.name == options.tensor_name; });
        std::cout << "tensor: " << options.tensor_name << "\n";
        if (it != info.value().tensor_infos.end()) {
            std::cout << "type: " << static_cast<std::uint32_t>(it->type) << "\n";
            std::cout << "shape:";
            for (const auto dim : it->dimensions) {
                std::cout << " " << dim;
            }
            std::cout << "\n";
            std::cout << "byte_size: " << it->byte_size << "\n";
        }
        const auto preview_size = std::min<std::uint64_t>(options.max_bytes, bytes.value().size());
        std::cout << "data_preview_hex:";
        for (std::uint64_t i = 0; i < preview_size; ++i) {
            std::cout << " " << std::hex << std::setw(2) << std::setfill('0')
                      << static_cast<int>(bytes.value()[static_cast<std::size_t>(i)]);
        }
        std::cout << std::dec << "\n";
        return 0;
}

int run_dit_block_command(const CliOptions& options) {
        const auto model = hy3d::load_hunyuan_dit_blocks_from_gguf(
            options.model_path,
            static_cast<std::uint32_t>(options.block_index),
            static_cast<std::uint32_t>(options.block_count),
            !options.no_mlp,
            !options.no_cross_attn,
            !options.no_timestep);
        if (!model.ok()) {
            std::cerr << "error: " << model.error() << "\n";
            return 1;
        }

        const std::string prefix = "blocks." + std::to_string(options.block_index);
        std::cout << "dit_block: " << prefix << "\n";
        std::cout << "block_count: " << options.block_count << "\n";
        std::cout << "loaded_tensors: " << model.value().tensor_count() << "\n";
        std::cout << "tokens: " << options.tokens << "\n";
        std::cout << "heads: " << options.heads << "\n";
        std::cout << "head_dim: " << options.head_dim << "\n";
        std::cout << "include_mlp: " << (options.no_mlp ? "false" : "true") << "\n";
        std::cout << "include_cross_attn: " << (options.no_cross_attn ? "false" : "true") << "\n";
        std::cout << "include_timestep: " << (options.no_timestep ? "false" : "true") << "\n";
        if (!options.no_cross_attn) {
            std::cout << "context_tokens: " << options.context_tokens << "\n";
            std::cout << "context_dim: " << options.context_dim << "\n";
        }
        if (options.dry_run) {
            std::cout << "dry_run: ok\n";
            return 0;
        }

        const auto width = static_cast<std::size_t>(options.heads) * static_cast<std::size_t>(options.head_dim);
        std::vector<float> input(static_cast<std::size_t>(options.tokens) * width, 0.0f);
        auto block_tokens = static_cast<std::size_t>(options.tokens);
        if (!options.no_timestep) {
            const auto timestep = model.value().project_timestep_conditioning(options.timestep, width);
            if (!timestep.ok()) {
                std::cerr << "error: " << timestep.error() << "\n";
                return 1;
            }
            if (timestep.value().size() != width) {
                std::cerr << "error: timestep conditioning width does not match heads * head_dim\n";
                return 1;
            }
            std::vector<float> conditioned;
            conditioned.reserve(input.size() + timestep.value().size());
            conditioned.insert(conditioned.end(), timestep.value().begin(), timestep.value().end());
            conditioned.insert(conditioned.end(), input.begin(), input.end());
            input = std::move(conditioned);
            ++block_tokens;
            std::cout << "timestep: " << options.timestep << "\n";
        }

        hy3d::Result<std::vector<float>> output = hy3d::Result<std::vector<float>>::failure("not run");
        if (options.no_cross_attn) {
            output = model.value().run_dit_blocks_conditioned(
                static_cast<std::uint32_t>(options.block_index),
                static_cast<std::uint32_t>(options.block_count),
                input,
                block_tokens,
                {},
                0,
                static_cast<std::size_t>(options.heads),
                static_cast<std::size_t>(options.head_dim));
        } else {
            std::vector<float> context(
                static_cast<std::size_t>(options.context_tokens) * static_cast<std::size_t>(options.context_dim),
                0.0f);
            output = model.value().run_dit_blocks_conditioned(
                static_cast<std::uint32_t>(options.block_index),
                static_cast<std::uint32_t>(options.block_count),
                input,
                block_tokens,
                context,
                static_cast<std::size_t>(options.context_tokens),
                static_cast<std::size_t>(options.heads),
                static_cast<std::size_t>(options.head_dim));
        }
        if (!output.ok()) {
            std::cerr << "error: " << output.error() << "\n";
            return 1;
        }
        std::cout << "block_tokens: " << block_tokens << "\n";
        double checksum = 0.0;
        for (const auto value : output.value()) {
            checksum += std::fabs(static_cast<double>(value));
        }
        std::cout << "output_values: " << output.value().size() << "\n";
        std::cout << "l1_checksum: " << checksum << "\n";
        return 0;
}

int run_dit_forward_command(const CliOptions& options) {
        const auto model = hy3d::load_hunyuan_dit_forward_from_gguf(
            options.model_path,
            static_cast<std::uint32_t>(options.block_index),
            static_cast<std::uint32_t>(options.block_count),
            true,
            true,
            true);
        if (!model.ok()) {
            std::cerr << "error: " << model.error() << "\n";
            return 1;
        }

        std::cout << "dit_forward: blocks." << options.block_index << "\n";
        std::cout << "block_count: " << options.block_count << "\n";
        std::cout << "loaded_tensors: " << model.value().tensor_count() << "\n";
        std::cout << "latent_tokens: " << options.tokens << "\n";
        std::cout << "latent_dim: " << options.latent_dim << "\n";
        std::cout << "context_tokens: " << options.context_tokens << "\n";
        std::cout << "context_dim: " << options.context_dim << "\n";
        if (options.dry_run) {
            std::cout << "dry_run: ok\n";
            return 0;
        }

        const auto latent_values = static_cast<std::size_t>(options.tokens) * static_cast<std::size_t>(options.latent_dim);
        const auto context_values = static_cast<std::size_t>(options.context_tokens) * static_cast<std::size_t>(options.context_dim);
        std::vector<float> latents(latent_values, 0.0f);
        std::vector<float> context(context_values, 0.0f);
        if (!options.latent_path.empty()) {
            auto loaded = read_f32_file(options.latent_path, latent_values);
            if (!loaded.ok()) {
                std::cerr << "error: " << loaded.error() << "\n";
                return 1;
            }
            latents = loaded.take_value();
        }
        if (!options.context_path.empty()) {
            auto loaded = read_f32_file(options.context_path, context_values);
            if (!loaded.ok()) {
                std::cerr << "error: " << loaded.error() << "\n";
                return 1;
            }
            context = loaded.take_value();
        }

        const auto output = model.value().run_dit_forward_scaffold(
            static_cast<std::uint32_t>(options.block_index),
            static_cast<std::uint32_t>(options.block_count),
            latents,
            static_cast<std::size_t>(options.tokens),
            context,
            static_cast<std::size_t>(options.context_tokens),
            options.timestep,
            static_cast<std::size_t>(options.heads),
            static_cast<std::size_t>(options.head_dim));
        if (!output.ok()) {
            std::cerr << "error: " << output.error() << "\n";
            return 1;
        }
        double checksum = 0.0;
        for (const auto value : output.value()) {
            checksum += std::fabs(static_cast<double>(value));
        }
        std::cout << "output_values: " << output.value().size() << "\n";
        std::cout << "l1_checksum: " << checksum << "\n";
        return 0;
}

int run_generate_command(const CliOptions& options) {
        hy3d::GenerateRequest request;
        request.backend = options.backend;
        request.device = options.device;
        request.quality = options.quality;
        request.image_path = options.image_path;
        request.output_path = options.output_path;
        request.model_path = options.model_path;
        request.steps = options.steps;
        request.seed = options.seed;
        request.low_vram = options.low_vram;
        request.dry_run = options.dry_run;
        request.no_rembg = options.no_rembg;

        const auto result = hy3d::run_generate(request);
        if (!result.ok()) {
            std::cerr << "error: " << result.error() << "\n";
            return 1;
        }
        return result.value();
}

int run_texture_command(const CliOptions& options) {
        hy3d::TextureRequest request;
        request.backend = options.backend;
        request.device = options.device;
        request.mesh_path = options.mesh_path;
        request.image_path = options.image_path;
        request.output_path = options.output_path;
        request.model_path = options.model_path;
        request.max_views = options.max_views;
        request.resolution = options.resolution;
        request.dry_run = options.dry_run;
        request.no_remesh = options.no_remesh;

        const auto result = hy3d::run_texture(request);
        if (!result.ok()) {
            std::cerr << "error: " << result.error() << "\n";
            return 1;
        }
        return result.value();
}

int run_command(const CliOptions& options) {
    switch (options.command) {
    case CommandKind::Help:
        return run_help_command(options);
    case CommandKind::Inspect:
        return run_inspect_command(options);
    case CommandKind::Tensor:
        return run_tensor_command(options);
    case CommandKind::DitBlock:
        return run_dit_block_command(options);
    case CommandKind::DitForward:
        return run_dit_forward_command(options);
    case CommandKind::Generate:
        return run_generate_command(options);
    case CommandKind::Texture:
        return run_texture_command(options);
    }

    std::cerr << "error: unhandled command\n";
    return 2;
}

} // namespace hy3d
