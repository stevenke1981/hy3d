#include "backend_paths.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

void touch(const std::filesystem::path& path) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    output << "test";
}

} // namespace

int main() {
    const auto root =
        std::filesystem::temp_directory_path() / ("hy3d-backend-paths-" + std::to_string(std::rand()));
    const auto working_directory = root / "unrelated-working-directory";
    const auto script = root / "scripts" / "run_python_backend.ps1";
    touch(script);
    std::filesystem::create_directories(working_directory);

    const auto packaged = hy3d::find_backend_script(
        root / "bin" / "hy3d.exe",
        working_directory,
        "run_python_backend.ps1");
    require(packaged.ok(), "packaged executable did not find sibling scripts directory");
    require(
        std::filesystem::equivalent(packaged.value(), script),
        "packaged executable resolved the wrong backend script");

    const auto multi_config_build = hy3d::find_backend_script(
        root / "build" / "Debug" / "hy3d.exe",
        working_directory,
        "run_python_backend.ps1");
    require(multi_config_build.ok(), "multi-config build executable did not find repository scripts directory");
    require(
        std::filesystem::equivalent(multi_config_build.value(), script),
        "multi-config build executable resolved the wrong backend script");

    const auto missing =
        hy3d::find_backend_script(root / "bin" / "hy3d.exe", working_directory, "missing.ps1");
    require(!missing.ok(), "missing backend script must return a failure");

    std::filesystem::remove_all(root);
    return 0;
}
