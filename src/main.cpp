#include "hy3d_cli.h"
#include "hy3d_commands.h"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::vector<std::string> args;
    args.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }

    const auto parsed = hy3d::parse_args(args);
    if (!parsed.ok()) {
        std::cerr << "error: " << parsed.error() << "\n\n" << hy3d::help_text();
        return 2;
    }
    return hy3d::run_command(parsed.value());
}
