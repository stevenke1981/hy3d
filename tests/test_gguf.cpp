#include "hy3d_gguf.h"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#undef assert
#define assert(expr)                                                                                                   \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            std::cerr << "assertion failed: " #expr << "\n";                                                          \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

namespace {

void write_u32(std::ofstream& out, std::uint32_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_u64(std::ofstream& out, std::uint64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_string(std::ofstream& out, const std::string& value) {
    write_u64(out, static_cast<std::uint64_t>(value.size()));
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void pad_to(std::ofstream& out, std::uint64_t alignment) {
    const auto position = static_cast<std::uint64_t>(out.tellp());
    const auto remainder = position % alignment;
    if (remainder != 0) {
        std::vector<char> padding(static_cast<std::size_t>(alignment - remainder), 0);
        out.write(padding.data(), static_cast<std::streamsize>(padding.size()));
    }
}

void write_header(
    std::ofstream& out,
    std::uint32_t version,
    std::uint64_t tensor_count,
    std::uint64_t metadata_count) {
    out.write("GGUF", 4);
    write_u32(out, version);
    write_u64(out, tensor_count);
    write_u64(out, metadata_count);
}

std::filesystem::path write_minimal_gguf() {
    const auto path = std::filesystem::temp_directory_path() / "hy3d-test-minimal.gguf";
    std::ofstream out(path, std::ios::binary);
    write_header(out, 3, 0, 0);
    return path;
}

} // namespace

int main() {
    {
        const auto path = write_minimal_gguf();
        const auto info = hy3d::inspect_gguf(path.string());
        assert(info.ok());
        assert(info.value().version == 3);
        assert(info.value().tensor_count == 0);
        assert(info.value().metadata_count == 0);
        assert(info.value().tensor_names.empty());
        std::filesystem::remove(path);
    }

    {
        const auto path = std::filesystem::temp_directory_path() / "hy3d-test-invalid.gguf";
        std::ofstream out(path, std::ios::binary);
        out.write("NOPE", 4);
        out.close();

        const auto info = hy3d::inspect_gguf(path.string());
        assert(!info.ok());
        assert(info.error().find("GGUF") != std::string::npos);
        std::filesystem::remove(path);
    }

    {
        const auto path = std::filesystem::temp_directory_path() / "hy3d-test-version.gguf";
        std::ofstream out(path, std::ios::binary);
        write_header(out, 99, 0, 0);
        out.close();

        const auto info = hy3d::inspect_gguf(path.string());
        assert(!info.ok());
        assert(info.error().find("version") != std::string::npos);
        std::filesystem::remove(path);
    }

    {
        const auto path = std::filesystem::temp_directory_path() / "hy3d-test-metadata-count.gguf";
        std::ofstream out(path, std::ios::binary);
        write_header(out, 3, 0, std::numeric_limits<std::uint64_t>::max());
        out.close();

        const auto info = hy3d::inspect_gguf(path.string());
        assert(!info.ok());
        assert(info.error().find("metadata count") != std::string::npos);
        std::filesystem::remove(path);
    }

    {
        const auto path = std::filesystem::temp_directory_path() / "hy3d-test-tensor-count.gguf";
        std::ofstream out(path, std::ios::binary);
        write_header(out, 3, std::numeric_limits<std::uint64_t>::max(), 0);
        out.close();

        const auto info = hy3d::inspect_gguf(path.string());
        assert(!info.ok());
        assert(info.error().find("tensor count") != std::string::npos);
        std::filesystem::remove(path);
    }

    {
        const auto path = std::filesystem::temp_directory_path() / "hy3d-test-array-overflow.gguf";
        std::ofstream out(path, std::ios::binary);
        write_header(out, 3, 0, 1);
        write_string(out, "array");
        write_u32(out, 9);
        write_u32(out, 10);
        write_u64(out, std::numeric_limits<std::uint64_t>::max());
        out.close();

        const auto info = hy3d::inspect_gguf(path.string());
        assert(!info.ok());
        assert(info.error().find("array") != std::string::npos);
        std::filesystem::remove(path);
    }

    {
        const auto path = std::filesystem::temp_directory_path() / "hy3d-test-duplicate-tensor.gguf";
        std::ofstream out(path, std::ios::binary);
        write_header(out, 3, 2, 0);
        for (int index = 0; index < 2; ++index) {
            write_string(out, "duplicate.weight");
            write_u32(out, 1);
            write_u64(out, 0);
            write_u32(out, static_cast<std::uint32_t>(hy3d::GgmlType::F32));
            write_u64(out, 0);
        }
        pad_to(out, 32);
        out.close();

        const auto info = hy3d::inspect_gguf(path.string());
        assert(!info.ok());
        assert(info.error().find("duplicate") != std::string::npos);
        std::filesystem::remove(path);
    }

    {
        const auto path = std::filesystem::temp_directory_path() / "hy3d-test-truncated-tensor.gguf";
        std::ofstream out(path, std::ios::binary);
        write_header(out, 3, 1, 0);
        write_string(out, "truncated.weight");
        write_u32(out, 1);
        write_u64(out, 4);
        write_u32(out, static_cast<std::uint32_t>(hy3d::GgmlType::F32));
        write_u64(out, 0);
        pad_to(out, 32);
        out.close();

        const auto info = hy3d::inspect_gguf(path.string());
        assert(!info.ok());
        assert(info.error().find("range") != std::string::npos);
        std::filesystem::remove(path);
    }

    {
        const auto path = std::filesystem::temp_directory_path() / "hy3d-test-offset-overflow.gguf";
        std::ofstream out(path, std::ios::binary);
        write_header(out, 3, 1, 0);
        write_string(out, "overflow.weight");
        write_u32(out, 1);
        write_u64(out, 0);
        write_u32(out, static_cast<std::uint32_t>(hy3d::GgmlType::F32));
        write_u64(out, std::numeric_limits<std::uint64_t>::max());
        pad_to(out, 32);
        out.close();

        const auto info = hy3d::inspect_gguf(path.string());
        assert(!info.ok());
        assert(info.error().find("offset") != std::string::npos);
        std::filesystem::remove(path);
    }

    return 0;
}
