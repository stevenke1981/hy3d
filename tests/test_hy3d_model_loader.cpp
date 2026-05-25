#include "hy3d_model_loader.h"

#include <cassert>
#include <algorithm>
#include <cstdint>
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

struct TensorRecord {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::vector<float> values;
};

std::filesystem::path write_block_gguf() {
    const auto path = std::filesystem::temp_directory_path() / "hy3d-test-block.gguf";
    const std::vector<TensorRecord> tensors = {
        {"blocks.0.attn1.to_q.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}},
        {"blocks.0.attn1.to_k.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}},
        {"blocks.0.attn1.to_v.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}},
        {"blocks.0.attn1.out_proj.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}},
    };

    std::ofstream out(path, std::ios::binary);
    out.write("GGUF", 4);
    write_u32(out, 3);
    write_u64(out, tensors.size());
    write_u64(out, 0);

    std::uint64_t offset = 0;
    for (const auto& tensor : tensors) {
        write_string(out, tensor.name);
        write_u32(out, static_cast<std::uint32_t>(tensor.dims.size()));
        for (const auto dim : tensor.dims) {
            write_u64(out, dim);
        }
        write_u32(out, static_cast<std::uint32_t>(hy3d::GgmlType::F32));
        write_u64(out, offset);
        offset += tensor.values.size() * sizeof(float);
        const auto remainder = offset % 32;
        if (remainder != 0) {
            offset += 32 - remainder;
        }
    }

    pad_to(out, 32);
    for (const auto& tensor : tensors) {
        pad_to(out, 32);
        out.write(reinterpret_cast<const char*>(tensor.values.data()), static_cast<std::streamsize>(tensor.values.size() * sizeof(float)));
    }
    return path;
}

} // namespace

int main() {
    const auto names = hy3d::hunyuan_dit_block_tensor_names(0, false);
    assert(names[0] == "blocks.0.norm1.weight");
    assert(names[6] == "blocks.0.attn1.to_q.weight");
    const auto full_names = hy3d::hunyuan_dit_block_tensor_names(0, true, true, true);
    assert(std::find(full_names.begin(), full_names.end(), "blocks.0.attn2.to_k.weight") != full_names.end());
    assert(std::find(full_names.begin(), full_names.end(), "blocks.0.norm3.weight") != full_names.end());
    assert(std::find(full_names.begin(), full_names.end(), "t_embedder.mlp.0.weight") != full_names.end());

    const auto path = write_block_gguf();
    const auto loaded = hy3d::load_hunyuan_dit_block_from_gguf(path.string(), 0, false);
    assert(loaded.ok());
    assert(loaded.value().tensor_count() == 4);
    assert(loaded.value().find_mapped_tensor("blocks.0", "attn_q.weight") != nullptr);
    const auto output = loaded.value().run_dit_block("blocks.0", {1.0f, 0.0f, 0.0f, 1.0f}, 2, 1, 2);
    assert(output.ok());
    assert(output.value().size() == 4);

    std::filesystem::remove(path);
    return 0;
}
