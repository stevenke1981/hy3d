#include "process_runner.h"

#include <cstdlib>
#include <sstream>

namespace hy3d {

std::string quote_arg(const std::string& value) {
    if (value.find_first_of(" \t\"") == std::string::npos) {
        return value;
    }

    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char ch : value) {
        if (ch == '"') {
            escaped += "\\\"";
        } else {
            escaped.push_back(ch);
        }
    }
    escaped.push_back('"');
    return escaped;
}

Result<int> run_process(const ProcessCommand& command) {
    if (command.executable.empty()) {
        return Result<int>::failure("process executable is empty");
    }

    std::ostringstream line;
    line << quote_arg(command.executable);
    for (const auto& argument : command.arguments) {
        line << ' ' << quote_arg(argument);
    }

    const int exit_code = std::system(line.str().c_str());
    return Result<int>::success(exit_code);
}

} // namespace hy3d
