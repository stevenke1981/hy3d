#include "process_runner.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace hy3d {
namespace {

#ifdef _WIN32
Result<std::wstring> to_wide(const std::string& value) {
    if (value.empty()) {
        return Result<std::wstring>::success({});
    }

    auto convert = [&](unsigned int code_page, unsigned long flags) -> std::wstring {
        const int size = MultiByteToWideChar(
            code_page,
            flags,
            value.data(),
            static_cast<int>(value.size()),
            nullptr,
            0);
        if (size <= 0) {
            return {};
        }
        std::wstring result(static_cast<std::size_t>(size), L'\0');
        if (MultiByteToWideChar(
                code_page,
                flags,
                value.data(),
                static_cast<int>(value.size()),
                result.data(),
                size) <= 0) {
            return {};
        }
        return result;
    };

    auto result = convert(CP_UTF8, MB_ERR_INVALID_CHARS);
    if (result.empty()) {
        result = convert(CP_ACP, 0);
    }
    if (result.empty()) {
        return Result<std::wstring>::failure("failed to convert process argument to UTF-16");
    }
    return Result<std::wstring>::success(std::move(result));
}

std::wstring quote_windows_argument(const std::wstring& value) {
    if (!value.empty() && value.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        return value;
    }

    std::wstring escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back(L'"');

    std::size_t backslashes = 0;
    for (const wchar_t ch : value) {
        if (ch == L'\\') {
            ++backslashes;
            continue;
        }
        if (ch == L'"') {
            escaped.append(backslashes * 2 + 1, L'\\');
            escaped.push_back(L'"');
            backslashes = 0;
            continue;
        }
        escaped.append(backslashes, L'\\');
        backslashes = 0;
        escaped.push_back(ch);
    }

    escaped.append(backslashes * 2, L'\\');
    escaped.push_back(L'"');
    return escaped;
}

Result<std::wstring> resolve_windows_executable(const std::wstring& executable) {
    const std::filesystem::path path(executable);
    if (path.has_parent_path() || path.has_root_name()) {
        return Result<std::wstring>::success(executable);
    }

    const unsigned long path_size = GetEnvironmentVariableW(L"PATH", nullptr, 0);
    if (path_size == 0) {
        return Result<std::wstring>::failure("PATH is empty while resolving process executable");
    }
    std::wstring search_path(static_cast<std::size_t>(path_size), L'\0');
    if (GetEnvironmentVariableW(L"PATH", search_path.data(), path_size) == 0) {
        return Result<std::wstring>::failure("failed to read PATH while resolving process executable");
    }
    search_path.resize(std::char_traits<wchar_t>::length(search_path.c_str()));

    const wchar_t* extension = path.has_extension() ? nullptr : L".exe";
    const unsigned long required =
        SearchPathW(search_path.c_str(), executable.c_str(), extension, 0, nullptr, nullptr);
    if (required == 0) {
        return Result<std::wstring>::failure("process executable not found on PATH");
    }

    std::vector<wchar_t> resolved(static_cast<std::size_t>(required) + 1, L'\0');
    const unsigned long written = SearchPathW(
        search_path.c_str(),
        executable.c_str(),
        extension,
        static_cast<unsigned long>(resolved.size()),
        resolved.data(),
        nullptr);
    if (written == 0 || written >= resolved.size()) {
        return Result<std::wstring>::failure("failed to resolve process executable on PATH");
    }
    return Result<std::wstring>::success(std::wstring(resolved.data(), written));
}

std::string windows_error(const char* operation, unsigned long code) {
    return std::string(operation) + " failed with Windows error " + std::to_string(code);
}
#endif

} // namespace

Result<int> run_process(const ProcessCommand& command) {
    if (command.executable.empty()) {
        return Result<int>::failure("process executable is empty");
    }

#ifdef _WIN32
    auto executable = to_wide(command.executable);
    if (!executable.ok()) {
        return Result<int>::failure(executable.error());
    }
    auto resolved_executable = resolve_windows_executable(executable.value());
    if (!resolved_executable.ok()) {
        return Result<int>::failure(resolved_executable.error() + ": " + command.executable);
    }

    std::wstring command_line = quote_windows_argument(resolved_executable.value());
    for (const auto& argument : command.arguments) {
        auto wide_argument = to_wide(argument);
        if (!wide_argument.ok()) {
            return Result<int>::failure(wide_argument.error());
        }
        command_line.push_back(L' ');
        command_line += quote_windows_argument(wide_argument.value());
    }

    std::vector<wchar_t> mutable_command_line(command_line.begin(), command_line.end());
    mutable_command_line.push_back(L'\0');

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(
            resolved_executable.value().c_str(),
            mutable_command_line.data(),
            nullptr,
            nullptr,
            FALSE,
            CREATE_UNICODE_ENVIRONMENT,
            nullptr,
            nullptr,
            &startup,
            &process)) {
        return Result<int>::failure(windows_error("CreateProcessW", GetLastError()));
    }

    const unsigned long wait_result = WaitForSingleObject(process.hProcess, INFINITE);
    if (wait_result != WAIT_OBJECT_0) {
        const auto error = GetLastError();
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        return Result<int>::failure(windows_error("WaitForSingleObject", error));
    }

    unsigned long exit_code = 0;
    if (!GetExitCodeProcess(process.hProcess, &exit_code)) {
        const auto error = GetLastError();
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        return Result<int>::failure(windows_error("GetExitCodeProcess", error));
    }

    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return Result<int>::success(static_cast<int>(exit_code));
#else
    int error_pipe[2]{};
    if (pipe(error_pipe) != 0) {
        return Result<int>::failure(std::string("pipe failed: ") + std::strerror(errno));
    }
    if (fcntl(error_pipe[1], F_SETFD, FD_CLOEXEC) == -1) {
        const auto error = errno;
        close(error_pipe[0]);
        close(error_pipe[1]);
        return Result<int>::failure(std::string("fcntl failed: ") + std::strerror(error));
    }

    std::vector<char*> arguments;
    arguments.reserve(command.arguments.size() + 2);
    arguments.push_back(const_cast<char*>(command.executable.c_str()));
    for (const auto& argument : command.arguments) {
        arguments.push_back(const_cast<char*>(argument.c_str()));
    }
    arguments.push_back(nullptr);

    const pid_t child = fork();
    if (child == -1) {
        const auto error = errno;
        close(error_pipe[0]);
        close(error_pipe[1]);
        return Result<int>::failure(std::string("fork failed: ") + std::strerror(error));
    }
    if (child == 0) {
        close(error_pipe[0]);
        execvp(command.executable.c_str(), arguments.data());
        const int error = errno;
        (void)write(error_pipe[1], &error, sizeof(error));
        _exit(127);
    }

    close(error_pipe[1]);
    int launch_error = 0;
    const auto error_bytes = read(error_pipe[0], &launch_error, sizeof(launch_error));
    close(error_pipe[0]);

    int status = 0;
    while (waitpid(child, &status, 0) == -1) {
        if (errno != EINTR) {
            return Result<int>::failure(std::string("waitpid failed: ") + std::strerror(errno));
        }
    }
    if (error_bytes > 0) {
        return Result<int>::failure(std::string("execvp failed: ") + std::strerror(launch_error));
    }
    if (WIFEXITED(status)) {
        return Result<int>::success(WEXITSTATUS(status));
    }
    if (WIFSIGNALED(status)) {
        return Result<int>::success(128 + WTERMSIG(status));
    }
    return Result<int>::failure("child process ended without an exit status");
#endif
}

} // namespace hy3d
