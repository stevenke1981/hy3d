#pragma once

#include "hy3d_cli.h"

namespace hy3d {

int run_help_command(const CliOptions& options);
int run_inspect_command(const CliOptions& options);
int run_tensor_command(const CliOptions& options);
int run_dit_block_command(const CliOptions& options);
int run_dit_forward_command(const CliOptions& options);
int run_generate_command(const CliOptions& options);
int run_texture_command(const CliOptions& options);
int run_command(const CliOptions& options);

} // namespace hy3d
