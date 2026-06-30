#include "hy3d_backend.h"

#include "backend_paths.h"
#include "process_runner.h"

#include <filesystem>

namespace hy3d {
namespace {

Result<int> validate_generate_request(const GenerateRequest& request) {
    if (request.image_path.empty()) {
        return Result<int>::failure("generate requires an image path");
    }
    if (!std::filesystem::exists(request.image_path)) {
        return Result<int>::failure("image not found: " + request.image_path);
    }
    if (request.output_path.empty()) {
        return Result<int>::failure("generate requires an output path");
    }

    const auto parent = std::filesystem::path(request.output_path).parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            return Result<int>::failure("failed to create output directory: " + parent.string());
        }
    }

    return Result<int>::success(0);
}

Result<int> validate_texture_request(const TextureRequest& request) {
    if (request.mesh_path.empty()) {
        return Result<int>::failure("texture requires a mesh path");
    }
    if (!std::filesystem::exists(request.mesh_path)) {
        return Result<int>::failure("mesh not found: " + request.mesh_path);
    }
    if (request.image_path.empty()) {
        return Result<int>::failure("texture requires an image path");
    }
    if (!std::filesystem::exists(request.image_path)) {
        return Result<int>::failure("image not found: " + request.image_path);
    }
    if (request.output_path.empty()) {
        return Result<int>::failure("texture requires an output path");
    }
    if (request.max_views < 6 || request.max_views > 12) {
        return Result<int>::failure("texture max views must be between 6 and 12");
    }
    if (request.resolution != 512 && request.resolution != 768) {
        return Result<int>::failure("texture resolution must be 512 or 768");
    }

    const auto parent = std::filesystem::path(request.output_path).parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            return Result<int>::failure("failed to create output directory: " + parent.string());
        }
    }

    return Result<int>::success(0);
}

} // namespace

Result<int> run_generate(const GenerateRequest& request) {
    if (request.backend == "native") {
        return Result<int>::failure("native Hunyuan3D inference is not implemented yet");
    }

    if (request.backend != "python") {
        return Result<int>::failure("unknown backend: " + request.backend);
    }

    auto validation = validate_generate_request(request);
    if (!validation.ok()) {
        return validation;
    }

    const auto script = resolve_backend_script("run_python_backend.ps1");
    if (!script.ok()) {
        return Result<int>::failure(script.error());
    }

    if (request.dry_run) {
        return Result<int>::success(0);
    }

    ProcessCommand command;
    command.executable = "powershell.exe";
    command.arguments = {
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        script.value().string(),
        "-ImagePath",
        request.image_path,
        "-OutputPath",
        request.output_path,
        "-Device",
        request.device,
        "-Quality",
        request.quality,
        "-Steps",
        std::to_string(request.steps),
        "-Seed",
        std::to_string(request.seed),
    };

    if (!request.model_path.empty()) {
        command.arguments.push_back("-ModelPath");
        command.arguments.push_back(request.model_path);
    }
    if (request.low_vram) {
        command.arguments.push_back("-LowVram");
    }
    if (request.no_rembg) {
        command.arguments.push_back("-NoRembg");
    }

    return run_process(command);
}

Result<int> run_texture(const TextureRequest& request) {
    if (request.backend == "native") {
        return Result<int>::failure("native Hunyuan3D texture inference is not implemented yet");
    }

    if (request.backend != "python") {
        return Result<int>::failure("unknown backend: " + request.backend);
    }

    auto validation = validate_texture_request(request);
    if (!validation.ok()) {
        return validation;
    }

    const auto script = resolve_backend_script("run_texture_backend.ps1");
    if (!script.ok()) {
        return Result<int>::failure(script.error());
    }

    if (request.dry_run) {
        return Result<int>::success(0);
    }

    ProcessCommand command;
    command.executable = "powershell.exe";
    command.arguments = {
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        script.value().string(),
        "-MeshPath",
        request.mesh_path,
        "-ImagePath",
        request.image_path,
        "-OutputPath",
        request.output_path,
        "-Device",
        request.device,
        "-MaxViews",
        std::to_string(request.max_views),
        "-Resolution",
        std::to_string(request.resolution),
    };

    if (!request.model_path.empty()) {
        command.arguments.push_back("-ModelPath");
        command.arguments.push_back(request.model_path);
    }
    if (request.no_remesh) {
        command.arguments.push_back("-NoRemesh");
    }

    return run_process(command);
}

} // namespace hy3d
