#include "hy3d_cli.h"

#include <cstdint>
#include <sstream>

namespace hy3d {
namespace {

bool is_help_flag(const std::string& arg) {
    return arg == "--help" || arg == "-h" || arg == "help";
}

Result<std::string> require_value(const std::vector<std::string>& args, std::size_t index, const std::string& option) {
    if (index + 1 >= args.size()) {
        return Result<std::string>::failure("missing value for " + option);
    }
    return Result<std::string>::success(args[index + 1]);
}

bool is_quality(const std::string& quality) {
    return quality == "smoke" || quality == "draft" || quality == "normal" ||
        quality == "character-normal" || quality == "final";
}

int steps_for_quality(const std::string& quality) {
    if (quality == "smoke") {
        return 5;
    }
    if (quality == "draft") {
        return 10;
    }
    if (quality == "character-normal") {
        return 40;
    }
    if (quality == "final") {
        return 50;
    }
    return 30;
}

} // namespace

std::string help_text() {
    std::ostringstream out;
    out << "Hunyuan3D GGUF C++ CLI\n"
        << "\n"
        << "Usage:\n"
        << "  hy3d --help\n"
        << "  hy3d inspect --model <file.gguf>\n"
        << "  hy3d --inspect <file.gguf>\n"
        << "  hy3d tensor --model <file.gguf> --name <tensor> [--bytes N]\n"
        << "  hy3d dit-block --model <file.gguf> [--block N] [--block-count N] [--tokens N] [--context-tokens N] [--context-dim N] [--timestep N] [--heads N] [--head-dim N] [--no-cross-attn] [--no-timestep] [--no-mlp] [--dry-run]\n"
        << "  hy3d dit-forward --model <file.gguf> [--block N] [--block-count N] [--latent-tokens N] [--latent-dim N] [--context-tokens N] [--context-dim N] [--latent-bin f32.bin] [--context-bin f32.bin] [--timestep N] [--dry-run]\n"
        << "  hy3d generate --backend python --image <input.png> --out <output.glb> [--model-path <path>] [--quality smoke|draft|normal|character-normal|final] [--steps N]\n"
        << "  hy3d texture --backend python --mesh <input.glb> --image <input.png> --out <textured.glb> [--model-path <path>] [--resolution 512] [--max-views 6]\n"
        << "\n"
        << "Commands:\n"
        << "  inspect     Inspect GGUF header and tensor metadata.\n"
        << "  tensor      Read a named tensor's metadata and leading bytes.\n"
        << "  dit-block   Load and run one mapped Hunyuan3D DiT block from GGUF.\n"
        << "  dit-forward Run native DiT x_embedder -> blocks -> final_layer scaffold.\n"
        << "  generate    Generate through a selected backend.\n"
        << "  texture     Generate PBR texture through the Python paint backend.\n"
        << "\n"
        << "Quality presets:\n"
        << "  smoke       5 steps, useful for CUDA/backend checks.\n"
        << "  draft       10 steps, quick preview.\n"
        << "  normal      30 steps, default quality. Explicit --steps overrides presets.\n"
        << "  character-normal 40 steps, better for single-image character references.\n"
        << "  final       50 steps, slower final pass.\n"
        << "\n"
        << "Backends:\n"
        << "  python      Bridge to scripts/run_python_backend.ps1.\n"
        << "  native      Reserved for future native Hunyuan3D GGUF inference.\n";
    return out.str();
}

Result<CliOptions> parse_args(const std::vector<std::string>& args) {
    CliOptions options;

    if (args.size() <= 1 || is_help_flag(args[1])) {
        options.command = CommandKind::Help;
        return Result<CliOptions>::success(options);
    }

    if (args[1] == "--inspect") {
        if (args.size() < 3) {
            return Result<CliOptions>::failure("--inspect requires a model path");
        }
        options.command = CommandKind::Inspect;
        options.model_path = args[2];
        return Result<CliOptions>::success(options);
    }

    if (args[1] == "inspect") {
        options.command = CommandKind::Inspect;
        for (std::size_t i = 2; i < args.size(); ++i) {
            if (args[i] == "--model") {
                auto value = require_value(args, i, "--model");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                options.model_path = value.value();
                ++i;
            } else {
                return Result<CliOptions>::failure("unknown inspect option: " + args[i]);
            }
        }
        if (options.model_path.empty()) {
            return Result<CliOptions>::failure("inspect requires --model <file.gguf>");
        }
        return Result<CliOptions>::success(options);
    }

    if (args[1] == "tensor") {
        options.command = CommandKind::Tensor;
        for (std::size_t i = 2; i < args.size(); ++i) {
            const auto& arg = args[i];
            if (arg == "--model") {
                auto value = require_value(args, i, "--model");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                options.model_path = value.value();
                ++i;
            } else if (arg == "--name") {
                auto value = require_value(args, i, "--name");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                options.tensor_name = value.value();
                ++i;
            } else if (arg == "--bytes") {
                auto value = require_value(args, i, "--bytes");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.max_bytes = static_cast<std::uint64_t>(std::stoull(value.value()));
                } catch (...) {
                    return Result<CliOptions>::failure("--bytes must be an unsigned integer");
                }
                ++i;
            } else {
                return Result<CliOptions>::failure("unknown tensor option: " + arg);
            }
        }
        if (options.model_path.empty()) {
            return Result<CliOptions>::failure("tensor requires --model <file.gguf>");
        }
        if (options.tensor_name.empty()) {
            return Result<CliOptions>::failure("tensor requires --name <tensor>");
        }
        return Result<CliOptions>::success(options);
    }

    if (args[1] == "dit-block") {
        options.command = CommandKind::DitBlock;
        for (std::size_t i = 2; i < args.size(); ++i) {
            const auto& arg = args[i];
            if (arg == "--model") {
                auto value = require_value(args, i, "--model");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                options.model_path = value.value();
                ++i;
            } else if (arg == "--block") {
                auto value = require_value(args, i, "--block");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.block_index = std::stoi(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--block must be an integer");
                }
                ++i;
            } else if (arg == "--block-count") {
                auto value = require_value(args, i, "--block-count");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.block_count = std::stoi(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--block-count must be an integer");
                }
                ++i;
            } else if (arg == "--tokens") {
                auto value = require_value(args, i, "--tokens");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.tokens = std::stoi(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--tokens must be an integer");
                }
                ++i;
            } else if (arg == "--context-tokens") {
                auto value = require_value(args, i, "--context-tokens");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.context_tokens = std::stoi(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--context-tokens must be an integer");
                }
                ++i;
            } else if (arg == "--context-dim") {
                auto value = require_value(args, i, "--context-dim");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.context_dim = std::stoi(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--context-dim must be an integer");
                }
                ++i;
            } else if (arg == "--heads") {
                auto value = require_value(args, i, "--heads");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.heads = std::stoi(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--heads must be an integer");
                }
                ++i;
            } else if (arg == "--head-dim") {
                auto value = require_value(args, i, "--head-dim");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.head_dim = std::stoi(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--head-dim must be an integer");
                }
                ++i;
            } else if (arg == "--timestep") {
                auto value = require_value(args, i, "--timestep");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.timestep = std::stof(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--timestep must be a number");
                }
                ++i;
            } else if (arg == "--dry-run") {
                options.dry_run = true;
            } else if (arg == "--no-mlp") {
                options.no_mlp = true;
            } else if (arg == "--no-cross-attn") {
                options.no_cross_attn = true;
            } else if (arg == "--no-timestep") {
                options.no_timestep = true;
            } else {
                return Result<CliOptions>::failure("unknown dit-block option: " + arg);
            }
        }
        if (options.model_path.empty()) {
            return Result<CliOptions>::failure("dit-block requires --model <file.gguf>");
        }
        if (options.block_index < 0) {
            return Result<CliOptions>::failure("--block must be non-negative");
        }
        if (options.block_count <= 0) {
            return Result<CliOptions>::failure("--block-count must be positive");
        }
        if (options.tokens <= 0 || options.context_tokens <= 0 || options.context_dim <= 0 ||
            options.heads <= 0 || options.head_dim <= 0) {
            return Result<CliOptions>::failure("--tokens, --context-tokens, --context-dim, --heads, and --head-dim must be positive");
        }
        return Result<CliOptions>::success(options);
    }

    if (args[1] == "dit-forward") {
        options.command = CommandKind::DitForward;
        options.block_index = 0;
        options.block_count = 1;
        options.tokens = 1;
        options.latent_dim = 64;
        options.context_tokens = 1;
        options.context_dim = 1024;
        for (std::size_t i = 2; i < args.size(); ++i) {
            const auto& arg = args[i];
            if (arg == "--model") {
                auto value = require_value(args, i, "--model");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                options.model_path = value.value();
                ++i;
            } else if (arg == "--block") {
                auto value = require_value(args, i, "--block");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.block_index = std::stoi(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--block must be an integer");
                }
                ++i;
            } else if (arg == "--block-count") {
                auto value = require_value(args, i, "--block-count");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.block_count = std::stoi(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--block-count must be an integer");
                }
                ++i;
            } else if (arg == "--latent-tokens") {
                auto value = require_value(args, i, "--latent-tokens");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.tokens = std::stoi(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--latent-tokens must be an integer");
                }
                ++i;
            } else if (arg == "--latent-dim") {
                auto value = require_value(args, i, "--latent-dim");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.latent_dim = std::stoi(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--latent-dim must be an integer");
                }
                ++i;
            } else if (arg == "--context-tokens") {
                auto value = require_value(args, i, "--context-tokens");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.context_tokens = std::stoi(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--context-tokens must be an integer");
                }
                ++i;
            } else if (arg == "--context-dim") {
                auto value = require_value(args, i, "--context-dim");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.context_dim = std::stoi(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--context-dim must be an integer");
                }
                ++i;
            } else if (arg == "--latent-bin") {
                auto value = require_value(args, i, "--latent-bin");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                options.latent_path = value.value();
                ++i;
            } else if (arg == "--context-bin") {
                auto value = require_value(args, i, "--context-bin");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                options.context_path = value.value();
                ++i;
            } else if (arg == "--heads") {
                auto value = require_value(args, i, "--heads");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.heads = std::stoi(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--heads must be an integer");
                }
                ++i;
            } else if (arg == "--head-dim") {
                auto value = require_value(args, i, "--head-dim");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.head_dim = std::stoi(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--head-dim must be an integer");
                }
                ++i;
            } else if (arg == "--timestep") {
                auto value = require_value(args, i, "--timestep");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.timestep = std::stof(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--timestep must be a number");
                }
                ++i;
            } else if (arg == "--dry-run") {
                options.dry_run = true;
            } else {
                return Result<CliOptions>::failure("unknown dit-forward option: " + arg);
            }
        }
        if (options.model_path.empty()) {
            return Result<CliOptions>::failure("dit-forward requires --model <file.gguf>");
        }
        if (options.block_index < 0 || options.block_count < 0) {
            return Result<CliOptions>::failure("--block and --block-count must be non-negative");
        }
        if (options.tokens <= 0 || options.latent_dim <= 0 || options.context_tokens <= 0 ||
            options.context_dim <= 0 || options.heads <= 0 || options.head_dim <= 0) {
            return Result<CliOptions>::failure("dit-forward dimensions must be positive");
        }
        return Result<CliOptions>::success(options);
    }

    if (args[1] == "generate") {
        options.command = CommandKind::Generate;
        bool steps_set = false;
        for (std::size_t i = 2; i < args.size(); ++i) {
            const auto& arg = args[i];
            if (arg == "--backend") {
                auto value = require_value(args, i, "--backend");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                options.backend = value.value();
                ++i;
            } else if (arg == "--image") {
                auto value = require_value(args, i, "--image");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                options.image_path = value.value();
                ++i;
            } else if (arg == "--model" || arg == "--model-path") {
                auto value = require_value(args, i, arg);
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                options.model_path = value.value();
                ++i;
            } else if (arg == "--out") {
                auto value = require_value(args, i, "--out");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                options.output_path = value.value();
                ++i;
            } else if (arg == "--device") {
                auto value = require_value(args, i, "--device");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                options.device = value.value();
                ++i;
            } else if (arg == "--quality") {
                auto value = require_value(args, i, "--quality");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                options.quality = value.value();
                ++i;
            } else if (arg == "--steps") {
                auto value = require_value(args, i, "--steps");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.steps = std::stoi(value.value());
                    steps_set = true;
                } catch (...) {
                    return Result<CliOptions>::failure("--steps must be an integer");
                }
                ++i;
            } else if (arg == "--seed") {
                auto value = require_value(args, i, "--seed");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.seed = std::stoi(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--seed must be an integer");
                }
                ++i;
            } else if (arg == "--low-vram") {
                options.low_vram = true;
            } else if (arg == "--dry-run") {
                options.dry_run = true;
            } else if (arg == "--no-rembg") {
                options.no_rembg = true;
            } else {
                return Result<CliOptions>::failure("unknown generate option: " + arg);
            }
        }
        if (options.image_path.empty()) {
            return Result<CliOptions>::failure("generate requires --image <input.png>");
        }
        if (options.output_path.empty()) {
            return Result<CliOptions>::failure("generate requires --out <output.glb>");
        }
        if (!is_quality(options.quality)) {
            return Result<CliOptions>::failure("--quality must be smoke, draft, normal, character-normal, or final");
        }
        if (!steps_set) {
            options.steps = steps_for_quality(options.quality);
        }
        if (options.steps <= 0) {
            return Result<CliOptions>::failure("--steps must be positive");
        }
        if (options.device != "cuda" && options.device != "cpu") {
            return Result<CliOptions>::failure("--device must be cuda or cpu");
        }
        return Result<CliOptions>::success(options);
    }

    if (args[1] == "texture") {
        options.command = CommandKind::Texture;
        for (std::size_t i = 2; i < args.size(); ++i) {
            const auto& arg = args[i];
            if (arg == "--backend") {
                auto value = require_value(args, i, "--backend");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                options.backend = value.value();
                ++i;
            } else if (arg == "--mesh") {
                auto value = require_value(args, i, "--mesh");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                options.mesh_path = value.value();
                ++i;
            } else if (arg == "--image") {
                auto value = require_value(args, i, "--image");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                options.image_path = value.value();
                ++i;
            } else if (arg == "--model" || arg == "--model-path") {
                auto value = require_value(args, i, arg);
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                options.model_path = value.value();
                ++i;
            } else if (arg == "--out") {
                auto value = require_value(args, i, "--out");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                options.output_path = value.value();
                ++i;
            } else if (arg == "--device") {
                auto value = require_value(args, i, "--device");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                options.device = value.value();
                ++i;
            } else if (arg == "--max-views") {
                auto value = require_value(args, i, "--max-views");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.max_views = std::stoi(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--max-views must be an integer");
                }
                ++i;
            } else if (arg == "--resolution") {
                auto value = require_value(args, i, "--resolution");
                if (!value.ok()) {
                    return Result<CliOptions>::failure(value.error());
                }
                try {
                    options.resolution = std::stoi(value.value());
                } catch (...) {
                    return Result<CliOptions>::failure("--resolution must be an integer");
                }
                ++i;
            } else if (arg == "--dry-run") {
                options.dry_run = true;
            } else if (arg == "--no-remesh") {
                options.no_remesh = true;
            } else {
                return Result<CliOptions>::failure("unknown texture option: " + arg);
            }
        }
        if (options.mesh_path.empty()) {
            return Result<CliOptions>::failure("texture requires --mesh <input.glb>");
        }
        if (options.image_path.empty()) {
            return Result<CliOptions>::failure("texture requires --image <input.png>");
        }
        if (options.output_path.empty()) {
            return Result<CliOptions>::failure("texture requires --out <output.glb>");
        }
        if (options.device != "cuda" && options.device != "cpu") {
            return Result<CliOptions>::failure("--device must be cuda or cpu");
        }
        if (options.max_views < 6 || options.max_views > 12) {
            return Result<CliOptions>::failure("--max-views must be between 6 and 12");
        }
        if (options.resolution != 512 && options.resolution != 768) {
            return Result<CliOptions>::failure("--resolution must be 512 or 768");
        }
        return Result<CliOptions>::success(options);
    }

    return Result<CliOptions>::failure("unknown command: " + args[1]);
}

} // namespace hy3d
