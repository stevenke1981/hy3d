#include "hy3d_commands.h"

#include <iostream>
#include <sstream>

int main() {
    hy3d::CliOptions options;
    options.command = hy3d::CommandKind::Help;

    std::ostringstream captured;
    auto* previous = std::cout.rdbuf(captured.rdbuf());
    const auto result = hy3d::run_help_command(options);
    std::cout.rdbuf(previous);

    if (result != 0 || captured.str().find("Usage:") == std::string::npos) {
        std::cerr << "help command handler failed\n";
        return 1;
    }
    return 0;
}
