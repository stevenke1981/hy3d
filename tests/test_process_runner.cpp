#include "process_runner.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

int write_child_arguments(int argc, char** argv) {
    if (argc < 3) {
        return 64;
    }

    std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
    if (!output) {
        return 65;
    }

    for (int index = 3; index < argc; ++index) {
        const std::string value = argv[index];
        output << value.size() << '\n';
        output.write(value.data(), static_cast<std::streamsize>(value.size()));
        output << '\n';
    }
    return output ? 0 : 66;
}

std::vector<std::string> read_child_arguments(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "child argument output was not created");

    std::vector<std::string> values;
    std::size_t size = 0;
    while (input >> size) {
        require(input.get() == '\n', "invalid child argument length separator");
        std::string value(size, '\0');
        input.read(value.data(), static_cast<std::streamsize>(size));
        require(static_cast<std::size_t>(input.gcount()) == size, "child argument was truncated");
        require(input.get() == '\n', "invalid child argument value separator");
        values.push_back(std::move(value));
    }
    return values;
}

} // namespace

int main(int argc, char** argv) {
    if (std::filesystem::path(argv[0]).filename() == "powershell.exe") {
        return 99;
    }
    if (argc >= 2 && std::string(argv[1]) == "--child") {
        return write_child_arguments(argc, argv);
    }

    const auto executable = std::filesystem::absolute(argv[0]);
    const auto output_path = std::filesystem::temp_directory_path() / "hy3d-process-runner-arguments.txt";
    std::filesystem::remove(output_path);

    const std::vector<std::string> expected = {
        "plain",
        "with space",
        "",
        "left&definitely_not_a_command",
        "quote\"inside",
        "trailing\\",
    };

    hy3d::ProcessCommand command;
    command.executable = executable.string();
    command.arguments = {"--child", output_path.string()};
    command.arguments.insert(command.arguments.end(), expected.begin(), expected.end());

    const auto result = hy3d::run_process(command);
    require(result.ok(), "valid child process failed to launch");
    require(result.value() == 0, "valid child process returned a non-zero exit code");
    require(read_child_arguments(output_path) == expected, "child process arguments did not round-trip exactly");
    std::filesystem::remove(output_path);

    hy3d::ProcessCommand missing;
    missing.executable = (std::filesystem::temp_directory_path() / "hy3d-executable-that-does-not-exist.exe").string();
    const auto missing_result = hy3d::run_process(missing);
    require(!missing_result.ok(), "missing executable must be reported as a launch failure");

#ifdef _WIN32
    const auto original_directory = std::filesystem::current_path();
    const auto hijack_directory =
        std::filesystem::temp_directory_path() / "hy3d-process-runner-path-hijack";
    std::filesystem::remove_all(hijack_directory);
    std::filesystem::create_directories(hijack_directory);
    std::filesystem::copy_file(
        executable,
        hijack_directory / "powershell.exe",
        std::filesystem::copy_options::overwrite_existing);
    std::filesystem::current_path(hijack_directory);

    hy3d::ProcessCommand path_lookup;
    path_lookup.executable = "powershell.exe";
    path_lookup.arguments = {"-NoProfile", "-Command", "exit 7"};
    const auto path_result = hy3d::run_process(path_lookup);
    std::filesystem::current_path(original_directory);
    std::filesystem::remove_all(hijack_directory);
    require(path_result.ok(), "executable name was not resolved through the Windows search path");
    require(path_result.value() == 7, "PATH-resolved child returned the wrong exit code");
#endif

    return 0;
}
