#include "hy3d_backend.h"
#include "hy3d_cli.h"
#include "hy3d_gguf.h"
#include "hy3d_model_loader.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::vector<std::string> args;
    args.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }

    const auto parsed = hy3d::parse_args(args);
    if (!parsed.ok()) {
        std::cerr << "error: " << parsed.error() << "\n\n" << hy3d::help_text();
        return 2;
    }

    const auto& options = parsed.value();
    if (options.command == hy3d::CommandKind::Help) {
        std::cout << hy3d::help_text();
        return 0;
    }

    if (options.command == hy3d::CommandKind::Inspect) {
        const auto info = hy3d::inspect_gguf(options.model_path);
        if (!info.ok()) {
            std::cerr << "error: " << info.error() << "\n";
            return 1;
        }
        std::cout << hy3d::format_gguf_info(info.value());
        return 0;
    }

    if (options.command == hy3d::CommandKind::Tensor) {
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

    if (options.command == hy3d::CommandKind::DitBlock) {
        const auto model = hy3d::load_hunyuan_dit_block_from_gguf(
            options.model_path,
            static_cast<std::uint32_t>(options.block_index),
            !options.no_mlp,
            !options.no_cross_attn,
            !options.no_timestep);
        if (!model.ok()) {
            std::cerr << "error: " << model.error() << "\n";
            return 1;
        }

        const std::string prefix = "blocks." + std::to_string(options.block_index);
        std::cout << "dit_block: " << prefix << "\n";
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
            output = model.value().run_dit_block(
                prefix,
                input,
                block_tokens,
                static_cast<std::size_t>(options.heads),
                static_cast<std::size_t>(options.head_dim));
        } else {
            std::vector<float> context(
                static_cast<std::size_t>(options.context_tokens) * static_cast<std::size_t>(options.context_dim),
                0.0f);
            output = model.value().run_dit_block_conditioned(
                prefix,
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

    if (options.command == hy3d::CommandKind::Generate) {
        hy3d::GenerateRequest request;
        request.backend = options.backend;
        request.device = options.device;
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

    if (options.command == hy3d::CommandKind::Texture) {
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

    std::cerr << "error: unhandled command\n";
    return 2;
}
