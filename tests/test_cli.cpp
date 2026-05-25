#include "hy3d_cli.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << "\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    {
        const std::vector<std::string> args = {"hy3d", "--help"};
        const auto parsed = hy3d::parse_args(args);
        require(parsed.ok(), "help should parse");
        require(parsed.value().command == hy3d::CommandKind::Help, "help command kind");
    }

    {
        const std::vector<std::string> args = {"hy3d", "inspect", "--model", "model.gguf"};
        const auto parsed = hy3d::parse_args(args);
        require(parsed.ok(), "inspect should parse");
        require(parsed.value().command == hy3d::CommandKind::Inspect, "inspect command kind");
        require(parsed.value().model_path == "model.gguf", "inspect model path");
    }

    {
        const std::vector<std::string> args = {"hy3d", "--inspect", "model.gguf"};
        const auto parsed = hy3d::parse_args(args);
        require(parsed.ok(), "inspect alias should parse");
        require(parsed.value().command == hy3d::CommandKind::Inspect, "inspect alias command kind");
        require(parsed.value().model_path == "model.gguf", "inspect alias model path");
    }

    {
        const std::vector<std::string> args = {
            "hy3d", "generate", "--backend", "python", "--image", "input.png",
            "--out", "out.glb", "--model-path", "D:/models/Hunyuan3D-2.1", "--steps", "5",
            "--seed", "7", "--device", "cuda", "--low-vram", "--no-rembg"};
        const auto parsed = hy3d::parse_args(args);
        require(parsed.ok(), "generate should parse");
        require(parsed.value().command == hy3d::CommandKind::Generate, "generate command kind");
        require(parsed.value().backend == "python", "generate backend");
        require(parsed.value().image_path == "input.png", "generate image path");
        require(parsed.value().output_path == "out.glb", "generate output path");
        require(parsed.value().model_path == "D:/models/Hunyuan3D-2.1", "generate model path");
        require(parsed.value().steps == 5, "generate steps");
        require(parsed.value().seed == 7, "generate seed");
        require(parsed.value().device == "cuda", "generate device");
        require(parsed.value().low_vram, "generate low vram");
        require(parsed.value().no_rembg, "generate no rembg");
    }

    {
        const std::vector<std::string> args = {
            "hy3d", "generate", "--backend", "python", "--image", "input.png",
            "--out", "out.glb", "--quality", "smoke"};
        const auto parsed = hy3d::parse_args(args);
        require(parsed.ok(), "generate quality should parse");
        require(parsed.value().quality == "smoke", "generate quality");
        require(parsed.value().steps == 5, "generate smoke steps");
    }

    {
        const std::vector<std::string> args = {
            "hy3d", "generate", "--image", "input.png", "--out", "out.glb",
            "--quality", "smoke", "--steps", "8"};
        const auto parsed = hy3d::parse_args(args);
        require(parsed.ok(), "generate quality plus steps should parse");
        require(parsed.value().steps == 8, "explicit steps override quality");
    }

    {
        const std::vector<std::string> args = {
            "hy3d", "texture", "--backend", "python", "--mesh", "shape.glb",
            "--image", "input.png", "--out", "textured.glb", "--model-path",
            "D:/models/Hunyuan3D-2.1", "--resolution", "512", "--max-views", "6",
            "--no-remesh"};
        const auto parsed = hy3d::parse_args(args);
        require(parsed.ok(), "texture should parse");
        require(parsed.value().command == hy3d::CommandKind::Texture, "texture command kind");
        require(parsed.value().backend == "python", "texture backend");
        require(parsed.value().mesh_path == "shape.glb", "texture mesh path");
        require(parsed.value().image_path == "input.png", "texture image path");
        require(parsed.value().output_path == "textured.glb", "texture output path");
        require(parsed.value().model_path == "D:/models/Hunyuan3D-2.1", "texture model path");
        require(parsed.value().resolution == 512, "texture resolution");
        require(parsed.value().max_views == 6, "texture max views");
        require(parsed.value().no_remesh, "texture no remesh");
    }

    {
        const std::vector<std::string> args = {
            "hy3d", "tensor", "--model", "model.gguf", "--name", "blocks.0.weight", "--bytes", "16"};
        const auto parsed = hy3d::parse_args(args);
        require(parsed.ok(), "tensor should parse");
        require(parsed.value().command == hy3d::CommandKind::Tensor, "tensor command kind");
        require(parsed.value().model_path == "model.gguf", "tensor model path");
        require(parsed.value().tensor_name == "blocks.0.weight", "tensor name");
        require(parsed.value().max_bytes == 16, "tensor max bytes");
    }

    {
        const std::vector<std::string> args = {
            "hy3d", "dit-block", "--model", "model.gguf", "--block", "0",
            "--tokens", "1", "--block-count", "2", "--context-tokens", "2", "--context-dim", "1024",
            "--heads", "16", "--head-dim", "128", "--timestep", "12.5",
            "--no-cross-attn", "--no-timestep", "--no-mlp", "--dry-run"};
        const auto parsed = hy3d::parse_args(args);
        require(parsed.ok(), "dit-block should parse");
        require(parsed.value().command == hy3d::CommandKind::DitBlock, "dit-block command kind");
        require(parsed.value().model_path == "model.gguf", "dit-block model path");
        require(parsed.value().block_index == 0, "dit-block index");
        require(parsed.value().block_count == 2, "dit-block count");
        require(parsed.value().tokens == 1, "dit-block tokens");
        require(parsed.value().context_tokens == 2, "dit-block context tokens");
        require(parsed.value().context_dim == 1024, "dit-block context dim");
        require(parsed.value().heads == 16, "dit-block heads");
        require(parsed.value().head_dim == 128, "dit-block head dim");
        require(parsed.value().timestep > 12.4f && parsed.value().timestep < 12.6f, "dit-block timestep");
        require(parsed.value().no_cross_attn, "dit-block no cross-attn");
        require(parsed.value().no_timestep, "dit-block no timestep");
        require(parsed.value().no_mlp, "dit-block no mlp");
        require(parsed.value().dry_run, "dit-block dry run");
    }


    {
        const std::vector<std::string> args = {"hy3d", "inspect"};
        const auto parsed = hy3d::parse_args(args);
        require(!parsed.ok(), "incomplete inspect should fail");
        require(parsed.error().find("--model") != std::string::npos, "incomplete inspect error");
    }

    return 0;
}
