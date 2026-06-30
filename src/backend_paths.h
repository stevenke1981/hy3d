#pragma once

#include "hy3d_result.h"

#include <filesystem>
#include <string>

namespace hy3d {

Result<std::filesystem::path> find_backend_script(
    const std::filesystem::path& executable_path,
    const std::filesystem::path& working_directory,
    const std::string& filename);

Result<std::filesystem::path> resolve_backend_script(const std::string& filename);

} // namespace hy3d
