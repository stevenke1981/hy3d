#pragma once

#include "hy3d_result.h"

#include <string>
#include <vector>

namespace hy3d {

enum class CommandKind {
    Help,
    Inspect,
    Tensor,
    Generate,
    Texture,
    DitBlock,
    DitForward,
};

struct CliOptions {
    CommandKind command = CommandKind::Help;
    std::string backend = "python";
    std::string device = "cuda";
    std::string quality = "normal";
    std::string image_path;
    std::string mesh_path;
    std::string model_path;
    std::string tensor_name;
    std::string output_path;
    std::string latent_path;
    std::string context_path;
    std::uint64_t max_bytes = 64;
    int steps = 30;
    int seed = 42;
    int block_index = 0;
    int block_count = 1;
    int max_views = 6;
    int tokens = 1;
    int latent_dim = 64;
    int context_tokens = 1;
    int context_dim = 1024;
    int heads = 16;
    int head_dim = 128;
    int resolution = 512;
    float timestep = 0.0f;
    bool low_vram = false;
    bool dry_run = false;
    bool no_rembg = false;
    bool no_remesh = false;
    bool no_mlp = false;
    bool no_cross_attn = false;
    bool no_timestep = false;
};

Result<CliOptions> parse_args(const std::vector<std::string>& args);
std::string help_text();

} // namespace hy3d
