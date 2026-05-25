#pragma once

#include "hy3d_result.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hy3d {

enum class GgmlType : std::uint32_t {
    F32 = 0,
    F16 = 1,
};

struct GgufTensorInfo {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    GgmlType type = GgmlType::F32;
    std::uint64_t data_offset = 0;
    std::uint64_t byte_size = 0;
};

struct GgufInfo {
    std::uint32_t version = 0;
    std::uint64_t tensor_count = 0;
    std::uint64_t metadata_count = 0;
    std::uint64_t data_start_offset = 0;
    std::vector<std::string> tensor_names;
    std::vector<GgufTensorInfo> tensor_infos;
};

Result<GgufInfo> inspect_gguf(const std::string& path);
Result<std::vector<std::uint8_t>> read_gguf_tensor_data(
    const std::string& path,
    const GgufInfo& info,
    const std::string& tensor_name);
std::string format_gguf_info(const GgufInfo& info);

} // namespace hy3d
