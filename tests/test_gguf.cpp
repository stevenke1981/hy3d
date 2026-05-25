#include "hy3d_gguf.h"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

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

std::filesystem::path write_minimal_gguf() {
    const auto path = std::filesystem::temp_directory_path() / "hy3d-test-minimal.gguf";
    std::ofstream out(path, std::ios::binary);
    out.write("GGUF", 4);
    write_u32(out, 3);
    write_u64(out, 0);
    write_u64(out, 0);
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

    return 0;
}
