#include "hy3d_backend.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#undef assert
#define assert(expr)                                                                                                   \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            std::cerr << "assertion failed: " #expr << "\n";                                                          \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

namespace {

std::filesystem::path write_temp_image() {
    const auto path = std::filesystem::temp_directory_path() / "hy3d-test-input.png";
    std::ofstream out(path, std::ios::binary);
    out.write("\x89PNG\r\n\x1a\n", 8);
    return path;
}

std::filesystem::path write_temp_mesh() {
    const auto path = std::filesystem::temp_directory_path() / "hy3d-test-input.glb";
    std::ofstream out(path, std::ios::binary);
    out.write("glTF", 4);
    return path;
}

} // namespace

int main() {
    {
        hy3d::GenerateRequest request;
        request.backend = "native";
        request.image_path = "missing.png";
        request.output_path = "out.obj";
        const auto result = hy3d::run_generate(request);
        assert(!result.ok());
        assert(result.error().find("native Hunyuan3D inference is not implemented yet") != std::string::npos);
    }

    {
        hy3d::GenerateRequest request;
        request.backend = "python";
        request.image_path = "missing.png";
        request.output_path = "out.glb";
        const auto result = hy3d::run_generate(request);
        assert(!result.ok());
        assert(result.error().find("image not found") != std::string::npos);
    }

    {
        const auto image = write_temp_image();
        hy3d::GenerateRequest request;
        request.backend = "python";
        request.quality = "character-normal";
        request.image_path = image.string();
        request.output_path = (std::filesystem::temp_directory_path() / "hy3d-test-output.glb").string();
        request.dry_run = true;

        const auto result = hy3d::run_generate(request);
        assert(result.ok());
        assert(result.value() == 0);
        std::filesystem::remove(image);
    }

    {
        hy3d::TextureRequest request;
        request.backend = "native";
        request.mesh_path = "missing.glb";
        request.image_path = "missing.png";
        request.output_path = "out.glb";
        const auto result = hy3d::run_texture(request);
        assert(!result.ok());
        assert(result.error().find("native Hunyuan3D texture inference is not implemented yet") != std::string::npos);
    }

    {
        hy3d::TextureRequest request;
        request.backend = "python";
        request.mesh_path = "missing.glb";
        request.image_path = "missing.png";
        request.output_path = "out.glb";
        const auto result = hy3d::run_texture(request);
        assert(!result.ok());
        assert(result.error().find("mesh not found") != std::string::npos);
    }

    {
        const auto image = write_temp_image();
        const auto mesh = write_temp_mesh();
        hy3d::TextureRequest request;
        request.backend = "python";
        request.mesh_path = mesh.string();
        request.image_path = image.string();
        request.output_path = (std::filesystem::temp_directory_path() / "hy3d-test-textured.glb").string();
        request.dry_run = true;

        const auto result = hy3d::run_texture(request);
        assert(result.ok());
        assert(result.value() == 0);
        std::filesystem::remove(image);
        std::filesystem::remove(mesh);
    }

    return 0;
}
