#include "backend_paths.h"

#include <array>
#include <cstdlib>
#include <system_error>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace hy3d {
namespace {

Result<std::filesystem::path> current_executable_path() {
#ifdef _WIN32
    std::vector<wchar_t> buffer(32768);
    const auto size = GetModuleFileNameW(nullptr, buffer.data(), static_cast<unsigned long>(buffer.size()));
    if (size == 0 || size >= buffer.size()) {
        return Result<std::filesystem::path>::failure(
            "GetModuleFileNameW failed with Windows error " + std::to_string(GetLastError()));
    }
    return Result<std::filesystem::path>::success(std::filesystem::path(buffer.data(), buffer.data() + size));
#else
    std::array<char, 4096> buffer{};
    const auto size = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (size <= 0) {
        return Result<std::filesystem::path>::failure("failed to resolve /proc/self/exe");
    }
    return Result<std::filesystem::path>::success(
        std::filesystem::path(std::string(buffer.data(), static_cast<std::size_t>(size))));
#endif
}

Result<std::filesystem::path> existing_script(
    const std::filesystem::path& root,
    const std::string& filename) {
    const auto candidate = root / "scripts" / filename;
    std::error_code error;
    if (!std::filesystem::is_regular_file(candidate, error) || error) {
        return Result<std::filesystem::path>::failure("not found");
    }
    const auto canonical = std::filesystem::weakly_canonical(candidate, error);
    return Result<std::filesystem::path>::success(error ? candidate : canonical);
}

std::string script_root_override() {
#ifdef _WIN32
    char* value = nullptr;
    std::size_t size = 0;
    if (_dupenv_s(&value, &size, "HY3D_SCRIPT_ROOT") != 0 || value == nullptr) {
        return {};
    }
    std::string result(value);
    std::free(value);
    return result;
#else
    const char* value = std::getenv("HY3D_SCRIPT_ROOT");
    return value == nullptr ? std::string{} : std::string(value);
#endif
}

} // namespace

Result<std::filesystem::path> find_backend_script(
    const std::filesystem::path& executable_path,
    const std::filesystem::path& working_directory,
    const std::string& filename) {
    auto directory = executable_path.parent_path();
    for (int depth = 0; depth < 3 && !directory.empty(); ++depth) {
        auto script = existing_script(directory, filename);
        if (script.ok()) {
            return script;
        }
        const auto parent = directory.parent_path();
        if (parent == directory) {
            break;
        }
        directory = parent;
    }

    auto working_script = existing_script(working_directory, filename);
    if (working_script.ok()) {
        return working_script;
    }
    return Result<std::filesystem::path>::failure("backend script not found: " + filename);
}

Result<std::filesystem::path> resolve_backend_script(const std::string& filename) {
    const auto override = script_root_override();
    if (!override.empty()) {
        for (const auto& candidate : {
                 std::filesystem::path(override) / filename,
                 std::filesystem::path(override) / "scripts" / filename,
             }) {
            std::error_code error;
            if (std::filesystem::is_regular_file(candidate, error) && !error) {
                return Result<std::filesystem::path>::success(std::filesystem::weakly_canonical(candidate, error));
            }
        }
        return Result<std::filesystem::path>::failure(
            "backend script not found under HY3D_SCRIPT_ROOT: " + override);
    }

    auto executable = current_executable_path();
    if (!executable.ok()) {
        return Result<std::filesystem::path>::failure(executable.error());
    }
    return find_backend_script(executable.value(), std::filesystem::current_path(), filename);
}

} // namespace hy3d
