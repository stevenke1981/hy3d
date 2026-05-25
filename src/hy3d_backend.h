#pragma once

#include "hy3d_result.h"

#include <string>

namespace hy3d {

struct GenerateRequest {
    std::string backend = "python";
    std::string device = "cuda";
    std::string quality = "normal";
    std::string image_path;
    std::string output_path;
    std::string model_path;
    int steps = 30;
    int seed = 42;
    bool low_vram = false;
    bool dry_run = false;
    bool no_rembg = false;
};

struct TextureRequest {
    std::string backend = "python";
    std::string device = "cuda";
    std::string mesh_path;
    std::string image_path;
    std::string output_path;
    std::string model_path;
    int max_views = 6;
    int resolution = 512;
    bool dry_run = false;
    bool no_remesh = false;
};

Result<int> run_generate(const GenerateRequest& request);
Result<int> run_texture(const TextureRequest& request);

} // namespace hy3d
