# Hunyuan3D GGUF C++ CLI Spec

## Goal

Build a Windows-native local C++ command line tool for Hunyuan3D workflows. The first implementation provides a usable CLI shell, GGUF inspection, and a Python backend bridge. Native GGUF inference for Hunyuan3D shape generation is a later phase because GGUF is only the model container; Hunyuan3D still requires its image encoder, DiT diffusion model, scheduler, and mesh decoder runtime.

## Target User

The primary user runs on Windows and wants local command line usage without WSL. The tool should be friendly to PowerShell, UTF-8 paths, CUDA-capable machines, and future model backends.

## References

- llama.cpp: GGUF conventions, ggml backend style, CLI option patterns.
- whisper.cpp: compact C/C++ CLI shape, model loading UX, progress and error style.
- CrisperASR: local Windows ggml/GGUF/CUDA reference already validated on this machine.

## MVP Scope

The first implementation must provide:

- `hy3d.exe --help`
- `hy3d.exe inspect --model <file.gguf>`
- `hy3d.exe --inspect <file.gguf>` as a convenience alias
- `hy3d.exe dit-block --model <file.gguf> [--block N] [--block-count N] [--tokens N] [--context-tokens N] [--context-dim N] [--timestep N] [--heads N] [--head-dim N] [--no-cross-attn] [--no-timestep] [--no-mlp] [--dry-run]`
- `hy3d.exe dit-forward --model <file.gguf> [--block N] [--block-count N] [--latent-tokens N] [--latent-dim N] [--context-tokens N] [--context-dim N] [--latent-bin f32.bin] [--context-bin f32.bin] [--timestep N] [--dry-run]`
- `hy3d.exe generate --backend python --image <input.png> --out <output.glb> [--model-path <path>] [--low-vram]`
- Clear Windows-friendly error messages.
- CMake build and CTest smoke tests.

The first implementation does not provide native Hunyuan3D inference. If `--backend native` is requested, the CLI must fail explicitly with a message that native inference is not implemented yet.

## Architecture

The executable is split into small C++ modules:

- `hy3d_cli`: parse command line arguments into a typed command.
- `hy3d_gguf`: read GGUF header, metadata, and tensor count for inspection.
- `process_runner`: run backend scripts and capture exit codes.
- `hy3d_backend`: dispatch generate requests to Python or future native backends.
- `main`: thin entrypoint that prints errors and returns exit codes.

The Python backend bridge delegates to `scripts/run_python_backend.ps1`. The script is intentionally thin and validates inputs before calling a future official Hunyuan3D Python checkout.

## CLI Contract

### Help

```powershell
hy3d.exe --help
```

Expected behavior:

- exit code `0`
- output includes `Hunyuan3D GGUF C++ CLI`
- output lists `inspect` and `generate`

### GGUF Inspect

```powershell
hy3d.exe inspect --model models\hy3d-shape-q8.gguf
hy3d.exe --inspect models\hy3d-shape-q8.gguf
```

Expected behavior:

- exit code `0` for valid GGUF
- output includes `format: GGUF`, `version`, `metadata_count`, and `tensor_count`
- invalid or missing model returns a non-zero exit code with a readable error

### Python Generate Bridge

```powershell
hy3d.exe generate --backend python --image examples\input.png --out outputs\demo.glb --model-path D:\models\Hunyuan3D-2.1 --low-vram
```

Expected behavior:

- validates the input image path before invoking the backend
- creates the output directory when possible
- calls `scripts/run_python_backend.ps1`
- propagates the backend exit code

### Native Backend Placeholder

```powershell
hy3d.exe generate --backend native --image examples\input.png --model models\hy3d-shape-q8.gguf --out outputs\shape.obj
```

Expected behavior:

- returns non-zero
- prints `native Hunyuan3D inference is not implemented yet`

### Native DiT Block Smoke

```powershell
hy3d.exe dit-block --model models\hy3d-shape-f16.gguf --block 0 --tokens 1 --heads 16 --head-dim 128
```

Expected behavior:

- loads official Hunyuan3D GGUF tensor names for the requested block
- maps `attn1.to_q/to_k/to_v`, `attn1.q_norm/k_norm`, `attn2.to_q/to_k/to_v`, `attn2.q_norm/k_norm`, `attn1/attn2.out_proj`, `norm1/norm2/norm3`, `mlp.fc1/fc2`, and `t_embedder.mlp.*` into the current runtime
- maps MoE tensors including `moe.gate`, `moe.experts.*.net.0.proj`, `moe.experts.*.net.2`, and `moe.shared_experts.*`
- runs the current CPU `run_dit_blocks_conditioned()` primitive over `--block-count`
- prints loaded tensor count, output value count, and an L1 checksum

### Native DiT Forward Scaffold

```powershell
hy3d.exe dit-forward --model models\hy3d-shape-f16.gguf --block-count 1 --latent-tokens 1 --latent-dim 64 --context-tokens 1 --context-dim 1024
```

Expected behavior:

- loads `x_embedder`, selected `blocks.N.*`, optional `pooler/extra_embedder`, and `final_layer`
- accepts raw little-endian F32 latent/context files when `--latent-bin` or `--context-bin` is provided
- projects latents into hidden size, prepends timestep/context conditioning, runs selected blocks, applies `final_layer`, and prints output size/checksum
- treats missing pooler tensors as optional for shape configs with `use_attention_pooling: false`

## Future Native GGUF Inference Phases

1. Feed real image encoder context and scheduler latents into `dit-forward`.
2. Complete all-block DiT execution with production-sized token counts and final projection.
3. Implement native image preprocessing and optional ONNX image encoder bridge.
4. Run the diffusion denoising loop with the scheduler.
5. Decode VAE latents into density fields and meshes.
6. Add CPU Q8 correctness tests.
7. Add CUDA backend validation.
8. Add quantized Q4_K/Q5_K model support.

## Non-Goals

- No texture/PBR generation in the first C++ native phase.
- No custom GUI.
- No WSL-only setup.
- No silent fallback from native to Python; backend choice must be explicit.

## Release Packaging

The Windows CUDA release is produced by `scripts/make_release.ps1`. The default package does not bundle model weights or the Python virtual environment; `hy3d-setup.cmd` installs the Python backend, builds the Hunyuan3D-Paint Windows extensions, and downloads model files from Hugging Face. `-IncludeModels` and `-IncludeVenv` create a larger mostly offline package.

## Native Runtime Direction

The native C++ runtime now includes tensor loading, CPU linear/self-attention/cross-attention math, head-wise RMS q/k normalization, timestep embedding projection, optional attention pooling, layer norm, GELU, real GGUF block tensor-name mapping, a selective block/range/forward loader, a sequential multi-block DiT primitive, simplified MoE inference routing, x-embedding, final-layer projection, an Euler scheduler step, and a density-grid mesh decoder primitive. Remaining native work is to feed real image encoder context and scheduler latents through production-sized all-block execution, run the denoising loop, decode VAE latents into density grids, and write final GLB output.
