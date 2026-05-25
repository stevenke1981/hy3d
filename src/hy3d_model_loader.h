#pragma once

#include "hy3d_runtime.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hy3d {

std::vector<std::string> hunyuan_dit_block_tensor_names(
    std::uint32_t block_index,
    bool include_mlp,
    bool include_cross_attention = false,
    bool include_timestep = false);
std::vector<std::string> hunyuan_dit_block_range_tensor_names(
    std::uint32_t first_block,
    std::uint32_t block_count,
    bool include_mlp,
    bool include_cross_attention = false,
    bool include_timestep = false);
Result<HunyuanDitModel> load_hunyuan_dit_block_from_gguf(
    const std::string& path,
    std::uint32_t block_index,
    bool include_mlp,
    bool include_cross_attention = false,
    bool include_timestep = false);
Result<HunyuanDitModel> load_hunyuan_dit_blocks_from_gguf(
    const std::string& path,
    std::uint32_t first_block,
    std::uint32_t block_count,
    bool include_mlp,
    bool include_cross_attention = false,
    bool include_timestep = false);

} // namespace hy3d
