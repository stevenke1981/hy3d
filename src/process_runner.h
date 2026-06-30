#pragma once

#include "hy3d_result.h"

#include <string>
#include <vector>

namespace hy3d {

struct ProcessCommand {
    std::string executable;
    std::vector<std::string> arguments;
};

Result<int> run_process(const ProcessCommand& command);

} // namespace hy3d
