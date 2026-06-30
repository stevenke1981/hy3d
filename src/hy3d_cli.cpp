#include "hy3d_cli.h"

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <sstream>

namespace hy3d {
namespace {

constexpr int kMaxBlockCount = 4096;
constexpr int kMaxTokens = 1 << 20;
constexpr int kMaxDimension = 1 << 20;
constexpr int kMaxHeads = 1024;
constexpr int kMaxHeadDimension = 1 << 16;
constexpr int kMaxInferenceSteps = 10000;
constexpr std::uint64_t kMaxRuntimeValues = 100000000ULL;

bool is_help_flag(const std::string& arg) {
    return arg == "--help" || arg == "-h" || arg == "help";
}

Result<std::string> consume_value(
    const std::vector<std::string>& args,
    std::size_t& index,
    const std::string& option) {
    if (index + 1 >= args.size()) {
        return Result<std::string>::failure("missing value for " + option);
    }
    ++index;
    return Result<std::string>::success(args[index]);
}

Result<int> parse_integer(const std::string& text, const std::string& option) {
    int value = 0;
    const auto* first = text.data();
    const auto* last = first + text.size();
    const auto parsed = std::from_chars(first, last, value);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != last) {
        return Result<int>::failure(option + " must be an integer");
    }
    return Result<int>::success(value);
}

Result<int> parse_bounded_integer(
    const std::string& text,
    const std::string& option,
    int minimum,
    int maximum) {
    auto parsed = parse_integer(text, option);
    if (!parsed.ok()) {
        return parsed;
    }
    if (parsed.value() < minimum || parsed.value() > maximum) {
        return Result<int>::failure(
            option + " must be between " + std::to_string(minimum) + " and " + std::to_string(maximum));
    }
    return parsed;
}

Result<std::uint64_t> parse_unsigned_integer(const std::string& text, const std::string& option) {
    std::uint64_t value = 0;
    const auto* first = text.data();
    const auto* last = first + text.size();
    const auto parsed = std::from_chars(first, last, value);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != last) {
        return Result<std::uint64_t>::failure(option + " must be an unsigned integer");
    }
    return Result<std::uint64_t>::success(value);
}

Result<float> parse_number(const std::string& text, const std::string& option) {
    if (text.empty()) {
        return Result<float>::failure(option + " must be a finite number");
    }
    char* end = nullptr;
    errno = 0;
    const auto value = std::strtof(text.c_str(), &end);
    if (errno == ERANGE || end != text.c_str() + text.size() || !std::isfinite(value)) {
        return Result<float>::failure(option + " must be a finite number");
    }
    return Result<float>::success(value);
}

bool fits_runtime_allocation(std::initializer_list<int> dimensions) {
    std::uint64_t values = 1;
    for (const auto dimension : dimensions) {
        if (dimension <= 0 ||
            values > kMaxRuntimeValues / static_cast<std::uint64_t>(dimension)) {
            return false;
        }
        values *= static_cast<std::uint64_t>(dimension);
    }
    return values <= kMaxRuntimeValues;
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

Result<CliOptions> option_error(const Result<std::string>& value) {
    return Result<CliOptions>::failure(value.error());
}

Result<CliOptions> integer_error(const Result<int>& value) {
    return Result<CliOptions>::failure(value.error());
}

Result<CliOptions> unsigned_error(const Result<std::uint64_t>& value) {
    return Result<CliOptions>::failure(value.error());
}

Result<CliOptions> number_error(const Result<float>& value) {
    return Result<CliOptions>::failure(value.error());
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

Result<CliOptions> parse_inspect_args(const std::vector<std::string>& args, std::size_t start) {
    CliOptions options;
    options.command = CommandKind::Inspect;
    for (std::size_t i = start; i < args.size(); ++i) {
        if (args[i] != "--model") {
            return Result<CliOptions>::failure("unknown inspect option: " + args[i]);
        }
        auto value = consume_value(args, i, "--model");
        if (!value.ok()) {
            return option_error(value);
        }
        options.model_path = value.take_value();
    }
    if (options.model_path.empty()) {
        return Result<CliOptions>::failure("inspect requires --model <file.gguf>");
    }
    return Result<CliOptions>::success(std::move(options));
}

Result<CliOptions> parse_tensor_args(const std::vector<std::string>& args, std::size_t start) {
    CliOptions options;
    options.command = CommandKind::Tensor;
    for (std::size_t i = start; i < args.size(); ++i) {
        const auto& arg = args[i];
        if (arg == "--model" || arg == "--name") {
            auto value = consume_value(args, i, arg);
            if (!value.ok()) {
                return option_error(value);
            }
            if (arg == "--model") {
                options.model_path = value.take_value();
            } else {
                options.tensor_name = value.take_value();
            }
        } else if (arg == "--bytes") {
            auto text = consume_value(args, i, arg);
            if (!text.ok()) {
                return option_error(text);
            }
            auto value = parse_unsigned_integer(text.value(), arg);
            if (!value.ok()) {
                return unsigned_error(value);
            }
            options.max_bytes = value.value();
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
    return Result<CliOptions>::success(std::move(options));
}

Result<CliOptions> parse_dit_block_args(const std::vector<std::string>& args, std::size_t start) {
    CliOptions options;
    options.command = CommandKind::DitBlock;
    for (std::size_t i = start; i < args.size(); ++i) {
        const auto& arg = args[i];
        if (arg == "--model") {
            auto value = consume_value(args, i, arg);
            if (!value.ok()) {
                return option_error(value);
            }
            options.model_path = value.take_value();
        } else if (
            arg == "--block" || arg == "--block-count" || arg == "--tokens" ||
            arg == "--context-tokens" || arg == "--context-dim" || arg == "--heads" ||
            arg == "--head-dim") {
            auto text = consume_value(args, i, arg);
            if (!text.ok()) {
                return option_error(text);
            }
            int minimum = 1;
            int maximum = kMaxDimension;
            if (arg == "--block") {
                minimum = 0;
                maximum = kMaxBlockCount;
            } else if (arg == "--block-count") {
                maximum = kMaxBlockCount;
            } else if (arg == "--tokens" || arg == "--context-tokens") {
                maximum = kMaxTokens;
            } else if (arg == "--heads") {
                maximum = kMaxHeads;
            } else if (arg == "--head-dim") {
                maximum = kMaxHeadDimension;
            }
            auto value = parse_bounded_integer(text.value(), arg, minimum, maximum);
            if (!value.ok()) {
                return integer_error(value);
            }
            if (arg == "--block") {
                options.block_index = value.value();
            } else if (arg == "--block-count") {
                options.block_count = value.value();
            } else if (arg == "--tokens") {
                options.tokens = value.value();
            } else if (arg == "--context-tokens") {
                options.context_tokens = value.value();
            } else if (arg == "--context-dim") {
                options.context_dim = value.value();
            } else if (arg == "--heads") {
                options.heads = value.value();
            } else {
                options.head_dim = value.value();
            }
        } else if (arg == "--timestep") {
            auto text = consume_value(args, i, arg);
            if (!text.ok()) {
                return option_error(text);
            }
            auto value = parse_number(text.value(), arg);
            if (!value.ok()) {
                return number_error(value);
            }
            options.timestep = value.value();
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
        return Result<CliOptions>::failure(
            "--tokens, --context-tokens, --context-dim, --heads, and --head-dim must be positive");
    }
    if (!fits_runtime_allocation({options.tokens, options.heads, options.head_dim}) ||
        !fits_runtime_allocation({options.context_tokens, options.context_dim})) {
        return Result<CliOptions>::failure("dit-block tensor dimensions exceed the runtime allocation limit");
    }
    return Result<CliOptions>::success(std::move(options));
}

Result<CliOptions> parse_dit_forward_args(const std::vector<std::string>& args, std::size_t start) {
    CliOptions options;
    options.command = CommandKind::DitForward;
    options.latent_dim = 64;
    options.context_dim = 1024;
    for (std::size_t i = start; i < args.size(); ++i) {
        const auto& arg = args[i];
        if (arg == "--model" || arg == "--latent-bin" || arg == "--context-bin") {
            auto value = consume_value(args, i, arg);
            if (!value.ok()) {
                return option_error(value);
            }
            if (arg == "--model") {
                options.model_path = value.take_value();
            } else if (arg == "--latent-bin") {
                options.latent_path = value.take_value();
            } else {
                options.context_path = value.take_value();
            }
        } else if (
            arg == "--block" || arg == "--block-count" || arg == "--latent-tokens" ||
            arg == "--latent-dim" || arg == "--context-tokens" || arg == "--context-dim" ||
            arg == "--heads" || arg == "--head-dim") {
            auto text = consume_value(args, i, arg);
            if (!text.ok()) {
                return option_error(text);
            }
            int minimum = 1;
            int maximum = kMaxDimension;
            if (arg == "--block" || arg == "--block-count") {
                minimum = 0;
                maximum = kMaxBlockCount;
            } else if (arg == "--latent-tokens" || arg == "--context-tokens") {
                maximum = kMaxTokens;
            } else if (arg == "--heads") {
                maximum = kMaxHeads;
            } else if (arg == "--head-dim") {
                maximum = kMaxHeadDimension;
            }
            auto value = parse_bounded_integer(text.value(), arg, minimum, maximum);
            if (!value.ok()) {
                return integer_error(value);
            }
            if (arg == "--block") {
                options.block_index = value.value();
            } else if (arg == "--block-count") {
                options.block_count = value.value();
            } else if (arg == "--latent-tokens") {
                options.tokens = value.value();
            } else if (arg == "--latent-dim") {
                options.latent_dim = value.value();
            } else if (arg == "--context-tokens") {
                options.context_tokens = value.value();
            } else if (arg == "--context-dim") {
                options.context_dim = value.value();
            } else if (arg == "--heads") {
                options.heads = value.value();
            } else {
                options.head_dim = value.value();
            }
        } else if (arg == "--timestep") {
            auto text = consume_value(args, i, arg);
            if (!text.ok()) {
                return option_error(text);
            }
            auto value = parse_number(text.value(), arg);
            if (!value.ok()) {
                return number_error(value);
            }
            options.timestep = value.value();
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
    if (!fits_runtime_allocation({options.tokens, options.latent_dim}) ||
        !fits_runtime_allocation({options.context_tokens, options.context_dim}) ||
        !fits_runtime_allocation({options.tokens + 1, options.heads, options.head_dim})) {
        return Result<CliOptions>::failure("dit-forward tensor dimensions exceed the runtime allocation limit");
    }
    return Result<CliOptions>::success(std::move(options));
}

Result<CliOptions> parse_generate_args(const std::vector<std::string>& args, std::size_t start) {
    CliOptions options;
    options.command = CommandKind::Generate;
    bool steps_set = false;
    for (std::size_t i = start; i < args.size(); ++i) {
        const auto& arg = args[i];
        if (
            arg == "--backend" || arg == "--image" || arg == "--model" || arg == "--model-path" ||
            arg == "--out" || arg == "--device" || arg == "--quality") {
            auto value = consume_value(args, i, arg);
            if (!value.ok()) {
                return option_error(value);
            }
            if (arg == "--backend") {
                options.backend = value.take_value();
            } else if (arg == "--image") {
                options.image_path = value.take_value();
            } else if (arg == "--model" || arg == "--model-path") {
                options.model_path = value.take_value();
            } else if (arg == "--out") {
                options.output_path = value.take_value();
            } else if (arg == "--device") {
                options.device = value.take_value();
            } else {
                options.quality = value.take_value();
            }
        } else if (arg == "--steps" || arg == "--seed") {
            auto text = consume_value(args, i, arg);
            if (!text.ok()) {
                return option_error(text);
            }
            auto value = arg == "--steps"
                ? parse_bounded_integer(text.value(), arg, 1, kMaxInferenceSteps)
                : parse_integer(text.value(), arg);
            if (!value.ok()) {
                return integer_error(value);
            }
            if (arg == "--steps") {
                options.steps = value.value();
                steps_set = true;
            } else {
                options.seed = value.value();
            }
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
        return Result<CliOptions>::failure(
            "--quality must be smoke, draft, normal, character-normal, or final");
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
    return Result<CliOptions>::success(std::move(options));
}

Result<CliOptions> parse_texture_args(const std::vector<std::string>& args, std::size_t start) {
    CliOptions options;
    options.command = CommandKind::Texture;
    for (std::size_t i = start; i < args.size(); ++i) {
        const auto& arg = args[i];
        if (
            arg == "--backend" || arg == "--mesh" || arg == "--image" || arg == "--model" ||
            arg == "--model-path" || arg == "--out" || arg == "--device") {
            auto value = consume_value(args, i, arg);
            if (!value.ok()) {
                return option_error(value);
            }
            if (arg == "--backend") {
                options.backend = value.take_value();
            } else if (arg == "--mesh") {
                options.mesh_path = value.take_value();
            } else if (arg == "--image") {
                options.image_path = value.take_value();
            } else if (arg == "--model" || arg == "--model-path") {
                options.model_path = value.take_value();
            } else if (arg == "--out") {
                options.output_path = value.take_value();
            } else {
                options.device = value.take_value();
            }
        } else if (arg == "--max-views" || arg == "--resolution") {
            auto text = consume_value(args, i, arg);
            if (!text.ok()) {
                return option_error(text);
            }
            auto value = arg == "--max-views"
                ? parse_bounded_integer(text.value(), arg, 6, 12)
                : parse_bounded_integer(text.value(), arg, 512, 768);
            if (!value.ok()) {
                return integer_error(value);
            }
            if (arg == "--max-views") {
                options.max_views = value.value();
            } else {
                options.resolution = value.value();
            }
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
    return Result<CliOptions>::success(std::move(options));
}

Result<CliOptions> parse_args(const std::vector<std::string>& args) {
    CliOptions options;
    if (args.size() <= 1 || is_help_flag(args[1])) {
        options.command = CommandKind::Help;
        return Result<CliOptions>::success(std::move(options));
    }
    if (args[1] == "--inspect") {
        if (args.size() != 3) {
            return Result<CliOptions>::failure("--inspect requires exactly one model path");
        }
        options.command = CommandKind::Inspect;
        options.model_path = args[2];
        return Result<CliOptions>::success(std::move(options));
    }
    if (args[1] == "inspect") {
        return parse_inspect_args(args, 2);
    }
    if (args[1] == "tensor") {
        return parse_tensor_args(args, 2);
    }
    if (args[1] == "dit-block") {
        return parse_dit_block_args(args, 2);
    }
    if (args[1] == "dit-forward") {
        return parse_dit_forward_args(args, 2);
    }
    if (args[1] == "generate") {
        return parse_generate_args(args, 2);
    }
    if (args[1] == "texture") {
        return parse_texture_args(args, 2);
    }
    return Result<CliOptions>::failure("unknown command: " + args[1]);
}

} // namespace hy3d
