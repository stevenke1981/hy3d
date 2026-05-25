#include "hy3d_gguf.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
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
    const auto pos = static_cast<std::uint64_t>(out.tellp());
    const auto remainder = pos % alignment;
    if (remainder != 0) {
        std::vector<char> padding(static_cast<std::size_t>(alignment - remainder), 0);
        out.write(padding.data(), static_cast<std::streamsize>(padding.size()));
    }
}

std::filesystem::path write_tensor_gguf() {
    const auto path = std::filesystem::temp_directory_path() / "hy3d-test-tensor.gguf";
    std::ofstream out(path, std::ios::binary);
    out.write("GGUF", 4);
    write_u32(out, 3);
    write_u64(out, 1);
    write_u64(out, 0);

    write_string(out, "linear.weight");
    write_u32(out, 2);
    write_u64(out, 2);
    write_u64(out, 2);
    write_u32(out, static_cast<std::uint32_t>(hy3d::GgmlType::F32));
    write_u64(out, 0);

    pad_to(out, 32);
    const float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    out.write(reinterpret_cast<const char*>(values), sizeof(values));
    return path;
}

} // namespace

int main() {
    const auto path = write_tensor_gguf();
    const auto info = hy3d::inspect_gguf(path.string());
    assert(info.ok());
    assert(info.value().tensor_count == 1);
    assert(info.value().tensor_infos.size() == 1);
    assert(info.value().tensor_infos[0].name == "linear.weight");
    assert(info.value().tensor_infos[0].dimensions == std::vector<std::uint64_t>({2, 2}));
    assert(info.value().tensor_infos[0].type == hy3d::GgmlType::F32);
    assert(info.value().tensor_infos[0].byte_size == 16);
    assert(info.value().data_start_offset % 32 == 0);

    const auto bytes = hy3d::read_gguf_tensor_data(path.string(), info.value(), "linear.weight");
    assert(bytes.ok());
    assert(bytes.value().size() == 16);

    float decoded[4] = {};
    std::memcpy(decoded, bytes.value().data(), sizeof(decoded));
    assert(decoded[0] == 1.0f);
    assert(decoded[1] == 2.0f);
    assert(decoded[2] == 3.0f);
    assert(decoded[3] == 4.0f);

    const auto missing = hy3d::read_gguf_tensor_data(path.string(), info.value(), "missing.weight");
    assert(!missing.ok());

    std::filesystem::remove(path);
    return 0;
}
