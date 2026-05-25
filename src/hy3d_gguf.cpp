#include "hy3d_gguf.h"

#include <array>
#include <algorithm>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>

namespace hy3d {
namespace {

enum class GgufValueType : std::uint32_t {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
};

template <typename T>
bool read_pod(std::ifstream& input, T& value) {
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    return static_cast<bool>(input);
}

std::uint64_t align_offset(std::uint64_t offset, std::uint64_t alignment) {
    const auto remainder = offset % alignment;
    if (remainder == 0) {
        return offset;
    }
    return offset + (alignment - remainder);
}

Result<std::uint64_t> ggml_type_size(GgmlType type) {
    switch (type) {
    case GgmlType::F32:
        return Result<std::uint64_t>::success(4);
    case GgmlType::F16:
        return Result<std::uint64_t>::success(2);
    }
    return Result<std::uint64_t>::failure("unsupported GGML tensor type");
}

Result<std::uint64_t> tensor_byte_size(GgmlType type, const std::vector<std::uint64_t>& dimensions) {
    auto element_size = ggml_type_size(type);
    if (!element_size.ok()) {
        return element_size;
    }

    std::uint64_t elements = 1;
    for (const auto dim : dimensions) {
        if (dim != 0 && elements > std::numeric_limits<std::uint64_t>::max() / dim) {
            return Result<std::uint64_t>::failure("GGUF tensor size overflow");
        }
        elements *= dim;
    }
    if (elements > std::numeric_limits<std::uint64_t>::max() / element_size.value()) {
        return Result<std::uint64_t>::failure("GGUF tensor byte size overflow");
    }
    return Result<std::uint64_t>::success(elements * element_size.value());
}

Result<std::string> read_string(std::ifstream& input) {
    std::uint64_t size = 0;
    if (!read_pod(input, size)) {
        return Result<std::string>::failure("unexpected end of file while reading GGUF string length");
    }
    if (size > 1024ull * 1024ull * 64ull) {
        return Result<std::string>::failure("GGUF string is unreasonably large");
    }
    std::string value(static_cast<std::size_t>(size), '\0');
    if (size > 0) {
        input.read(value.data(), static_cast<std::streamsize>(size));
        if (!input) {
            return Result<std::string>::failure("unexpected end of file while reading GGUF string");
        }
    }
    return Result<std::string>::success(value);
}

std::uint64_t scalar_size(GgufValueType type) {
    switch (type) {
    case GgufValueType::Uint8:
    case GgufValueType::Int8:
    case GgufValueType::Bool:
        return 1;
    case GgufValueType::Uint16:
    case GgufValueType::Int16:
        return 2;
    case GgufValueType::Uint32:
    case GgufValueType::Int32:
    case GgufValueType::Float32:
        return 4;
    case GgufValueType::Uint64:
    case GgufValueType::Int64:
    case GgufValueType::Float64:
        return 8;
    case GgufValueType::String:
    case GgufValueType::Array:
        return 0;
    }
    return 0;
}

Result<int> skip_value(std::ifstream& input, GgufValueType type);

Result<int> skip_array(std::ifstream& input) {
    std::uint32_t raw_type = 0;
    std::uint64_t count = 0;
    if (!read_pod(input, raw_type) || !read_pod(input, count)) {
        return Result<int>::failure("unexpected end of file while reading GGUF array metadata");
    }

    const auto type = static_cast<GgufValueType>(raw_type);
    if (type == GgufValueType::String || type == GgufValueType::Array) {
        for (std::uint64_t i = 0; i < count; ++i) {
            auto skipped = skip_value(input, type);
            if (!skipped.ok()) {
                return skipped;
            }
        }
        return Result<int>::success(0);
    }

    const auto bytes = scalar_size(type);
    if (bytes == 0) {
        return Result<int>::failure("unknown GGUF array value type");
    }

    input.seekg(static_cast<std::streamoff>(bytes * count), std::ios::cur);
    if (!input) {
        return Result<int>::failure("unexpected end of file while skipping GGUF array");
    }
    return Result<int>::success(0);
}

Result<int> skip_value(std::ifstream& input, GgufValueType type) {
    if (type == GgufValueType::String) {
        auto value = read_string(input);
        if (!value.ok()) {
            return Result<int>::failure(value.error());
        }
        return Result<int>::success(0);
    }

    if (type == GgufValueType::Array) {
        return skip_array(input);
    }

    const auto bytes = scalar_size(type);
    if (bytes == 0) {
        return Result<int>::failure("unknown GGUF metadata value type");
    }
    input.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
    if (!input) {
        return Result<int>::failure("unexpected end of file while skipping GGUF metadata value");
    }
    return Result<int>::success(0);
}

} // namespace

Result<GgufInfo> inspect_gguf(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return Result<GgufInfo>::failure("model not found: " + path);
    }

    std::array<char, 4> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!input || magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') {
        return Result<GgufInfo>::failure("file is not a GGUF model: " + path);
    }

    GgufInfo info;
    if (!read_pod(input, info.version) || !read_pod(input, info.tensor_count) || !read_pod(input, info.metadata_count)) {
        return Result<GgufInfo>::failure("file ended before GGUF header was complete");
    }

    for (std::uint64_t i = 0; i < info.metadata_count; ++i) {
        auto key = read_string(input);
        if (!key.ok()) {
            return Result<GgufInfo>::failure(key.error());
        }
        std::uint32_t raw_type = 0;
        if (!read_pod(input, raw_type)) {
            return Result<GgufInfo>::failure("unexpected end of file while reading GGUF metadata type");
        }
        auto skipped = skip_value(input, static_cast<GgufValueType>(raw_type));
        if (!skipped.ok()) {
            return Result<GgufInfo>::failure(skipped.error());
        }
    }

    for (std::uint64_t i = 0; i < info.tensor_count; ++i) {
        auto name = read_string(input);
        if (!name.ok()) {
            return Result<GgufInfo>::failure(name.error());
        }
        GgufTensorInfo tensor;
        tensor.name = name.value();
        info.tensor_names.push_back(tensor.name);

        std::uint32_t n_dims = 0;
        if (!read_pod(input, n_dims)) {
            return Result<GgufInfo>::failure("unexpected end of file while reading GGUF tensor dimensions");
        }
        if (n_dims > 16) {
            return Result<GgufInfo>::failure("GGUF tensor has too many dimensions");
        }
        for (std::uint32_t dim = 0; dim < n_dims; ++dim) {
            std::uint64_t value = 0;
            if (!read_pod(input, value)) {
                return Result<GgufInfo>::failure("unexpected end of file while reading GGUF tensor shape");
            }
            tensor.dimensions.push_back(value);
        }
        std::uint32_t type = 0;
        std::uint64_t offset = 0;
        if (!read_pod(input, type) || !read_pod(input, offset)) {
            return Result<GgufInfo>::failure("unexpected end of file while reading GGUF tensor info");
        }
        tensor.type = static_cast<GgmlType>(type);
        tensor.data_offset = offset;
        auto byte_size = tensor_byte_size(tensor.type, tensor.dimensions);
        if (!byte_size.ok()) {
            return Result<GgufInfo>::failure(byte_size.error());
        }
        tensor.byte_size = byte_size.value();
        info.tensor_infos.push_back(tensor);
    }

    const auto current = static_cast<std::uint64_t>(input.tellg());
    info.data_start_offset = align_offset(current, 32);

    return Result<GgufInfo>::success(info);
}

Result<std::vector<std::uint8_t>> read_gguf_tensor_data(
    const std::string& path,
    const GgufInfo& info,
    const std::string& tensor_name) {
    const auto it = std::find_if(
        info.tensor_infos.begin(),
        info.tensor_infos.end(),
        [&](const GgufTensorInfo& tensor) { return tensor.name == tensor_name; });
    if (it == info.tensor_infos.end()) {
        return Result<std::vector<std::uint8_t>>::failure("tensor not found: " + tensor_name);
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return Result<std::vector<std::uint8_t>>::failure("model not found: " + path);
    }

    const auto absolute_offset = info.data_start_offset + it->data_offset;
    input.seekg(static_cast<std::streamoff>(absolute_offset), std::ios::beg);
    if (!input) {
        return Result<std::vector<std::uint8_t>>::failure("failed to seek to tensor data: " + tensor_name);
    }

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(it->byte_size));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!input) {
            return Result<std::vector<std::uint8_t>>::failure("unexpected end of file while reading tensor: " + tensor_name);
        }
    }
    return Result<std::vector<std::uint8_t>>::success(bytes);
}

std::string format_gguf_info(const GgufInfo& info) {
    std::ostringstream out;
    out << "format: GGUF\n"
        << "version: " << info.version << "\n"
        << "metadata_count: " << info.metadata_count << "\n"
        << "tensor_count: " << info.tensor_count << "\n";
    if (!info.tensor_names.empty()) {
        out << "tensors:\n";
        for (const auto& name : info.tensor_names) {
            out << "  - " << name << "\n";
        }
    }
    return out.str();
}

} // namespace hy3d
