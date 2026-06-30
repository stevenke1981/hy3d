# Hunyuan3D Windows CLI

Windows-native C++17 command-line tooling for local Hunyuan3D shape and PBR
texture workflows.

The production path is the C++ CLI calling the pinned official Python/CUDA
backend. The native GGUF runtime can inspect/load tensors and run verified DiT
primitive/block smoke paths, but it is not yet an end-to-end image-to-3D
backend.

## Choose Your Path

| Goal | Start here |
|---|---|
| Generate a GLB from an image | [Quick Start](#quick-start-for-humans) |
| Build or test the C++ project | [Developer Build](#developer-build) |
| Create or verify a Windows release | [Windows Release](#windows-release) |
| Continue implementation as an agent | [`AGENTS.md`](AGENTS.md), then [`todos.md`](todos.md) |
| Review acceptance evidence | [`test.md`](test.md) and [`final.md`](final.md) |

## Current Status

- Verified on Windows with an RTX 3070 Ti: CUDA shape generation and
  Hunyuan3D-Paint texture generation.
- Core Debug and Release suites: 24/24 non-slow tests passing.
- Release packaging: clean configure/build, zip extraction to a Unicode path,
  package-wide SHA-256 verification, and executable smoke are automated.
- Remaining release gate: run online source/model download, fresh virtual
  environment setup, shape smoke, and texture smoke starting only from a newly
  extracted release zip.
- Source and model downloads are pinned. The resolved Windows/Python
  3.10/CUDA 12.4 lock contains 136 packages.

## Prerequisites

- Windows 10 or newer.
- CMake and Visual Studio C++ Build Tools.
- NVIDIA driver and a supported CUDA toolkit for GPU generation.
- Python 3.10 for the official backend. The setup script creates
  `.venv-hy3d`.
- Git and network access for first-time source/model setup.

Large models, generated outputs, build trees, and virtual environments are
local artifacts and must not be committed.

Clean online setup, shape, and texture smoke are verified. Extract to a path
containing only ASCII characters before running setup: PyTorch/Ninja's Windows
native-extension toolchain still corrupts non-ASCII source paths. Unicode
paths remain covered for zip extraction, hash verification, and CLI startup.

## Quick Start for Humans

From PowerShell:

```powershell
.\scripts\download_hy3d_models.ps1
.\scripts\setup_hy3d_python.ps1

.\scripts\generate_3d_model.ps1 `
  -ImagePath "C:\path\to\input.png" `
  -OutputPath .\outputs\character.glb `
  -Quality character-normal `
  -Seed 42
```

Add `-Texture` to run Hunyuan3D-Paint after shape generation. Every backend
run writes a `.log.txt` and `.json` sidecar next to the GLB. Start with
`-Quality smoke` when checking a new machine.

Do not use `--low-vram` for the official Python backend on this checkout. The
upstream offload path currently mixes CPU and CUDA scheduler tensors.

## Developer Build

```powershell
cmake -S . -B build
cmake --build build --config Debug --parallel
ctest --test-dir build -C Debug -LE slow --output-on-failure
cmake --build build --config Release --parallel
ctest --test-dir build -C Release -LE slow --output-on-failure
```

Run the slower release-package gate separately:

```powershell
ctest --test-dir build -C Release -R '^make_release$' --output-on-failure
```

## Windows Release

Create both a redistributable Windows CUDA folder and zip:

```powershell
.\scripts\make_release.ps1 -Zip
```

The output is `dist\hy3d-win-cuda`. It contains `hy3d-setup.cmd`, `hy3d-generate-smoke.cmd`, `hy3d-texture-smoke.cmd`, and `hy3d.cmd`. Setup downloads the pinned Hunyuan3D source/model revisions, installs the Python backend, verifies the RealESRGAN checkpoint hash, and builds the Hunyuan3D-Paint Windows extensions. The release script configures its build directory automatically.

After extracting the zip, verify its complete contents before setup:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\verify_release.ps1
.\hy3d-setup.cmd
.\hy3d-generate-smoke.cmd
.\hy3d-texture-smoke.cmd
```

`verify_release.ps1` rejects missing, additional, modified, duplicate, absolute,
or package-escaping manifest paths. It also runs `bin\hy3d.exe --help` from a
temporary working directory outside the package. This structural check does
not download models or perform CUDA inference.

For the setup and CUDA commands, move the verified package to an ASCII-only
path such as `D:\hy3d-release` if the extraction path contains non-ASCII text.

For a mostly offline package from an already prepared checkout:

```powershell
.\scripts\make_release.ps1 -IncludeSource -IncludeModels -IncludeVenv -Zip
```

## Image-to-3D Options

Use `scripts\generate_3d_model.ps1` to turn any local image into a GLB through the verified Windows CUDA Python backend:

```powershell
.\scripts\generate_3d_model.ps1 `
  -ImagePath "C:\Users\steven\Downloads\ChatGPT Image 2026年5月25日 下午11_20_07.png" `
  -OutputPath .\outputs\frieren-character-normal.glb `
  -Quality character-normal `
  -Seed 42
```

Useful options:

- `-Quality smoke|draft|normal|character-normal|final`: maps to the backend generation presets.
- `-Steps N`: overrides the quality preset step count.
- `-Texture`: runs Hunyuan3D-Paint after shape generation and writes `*-textured.glb`.
- `-NoRembg`: keeps the original image background.
- `-LowVram`: forwards the upstream low-VRAM flag, but see the note below before using it on this checkout.
- `-DryRun`: validates paths and command wiring without producing the GLB.
- `-NoBuild`: fails if `build\Release\hy3d.exe` is missing instead of building it.

Each run writes sidecars next to the output model:

- `output.glb.log.txt`
- `output.glb.json`

Preset steps:

- `smoke`: 5 steps, CUDA/backend checks only.
- `draft`: 10 steps, quick preview.
- `normal`: 30 steps, general shape generation.
- `character-normal`: 40 steps, the default wrapper preset for single-image character references.
- `final`: 50 steps, slower final pass.

## Download Hunyuan3D Models

The reproducible path is:

```powershell
.\scripts\download_hy3d_models.ps1
```

It checks out source revision `82920d643c0dc2f7bfd7255f45f62d386edfe60c`, downloads model revision `0b94677654c57bb9a6b6845cd7b704ccf551d327`, and verifies the RealESRGAN SHA-256. To download manually, use the current Hugging Face CLI, `hf`, and pass the same revision. The older `huggingface-cli` command is deprecated.

Shape generation checkpoints:

```powershell
hf download tencent/Hunyuan3D-2.1 `
  --revision 0b94677654c57bb9a6b6845cd7b704ccf551d327 `
  README.md LICENSE Notice.txt demo.py `
  "hunyuan3d-dit-v2-1/config.yaml" `
  "hunyuan3d-dit-v2-1/model.fp16.ckpt" `
  "hunyuan3d-vae-v2-1/config.yaml" `
  "hunyuan3d-vae-v2-1/model.fp16.ckpt" `
  --local-dir .\models\Hunyuan3D-2.1
```

The downloaded `.ckpt` files are official PyTorch checkpoints, not GGUF files. Use them directly with the Python backend or convert the shape checkpoint with the GGUF converter below.

PBR texture checkpoints:

```powershell
hf download tencent/Hunyuan3D-2.1 `
  --revision 0b94677654c57bb9a6b6845cd7b704ccf551d327 `
  --local-dir .\models\Hunyuan3D-2.1 `
  --include "hunyuan3d-paintpbr-v2-1/*"
```

Hunyuan3D-Paint also needs RealESRGAN:

```powershell
New-Item -ItemType Directory -Force -Path .\third_party\Hunyuan3D-2.1\hy3dpaint\ckpt | Out-Null
Invoke-WebRequest `
  -Uri https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth `
  -OutFile .\third_party\Hunyuan3D-2.1\hy3dpaint\ckpt\RealESRGAN_x4plus.pth
```

## Convert Shape Checkpoint to GGUF

Create a local Python environment and install CUDA PyTorch:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .\.venv\Scripts\python.exe numpy
uv pip install --python .\.venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu130
```

Verify CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Convert the Hunyuan3D shape DiT checkpoint:

```powershell
.\.venv\Scripts\python.exe tools\convert-hy3d-shape-to-gguf.py `
  --input .\models\Hunyuan3D-2.1\hunyuan3d-dit-v2-1\model.fp16.ckpt `
  --config .\models\Hunyuan3D-2.1\hunyuan3d-dit-v2-1\config.yaml `
  --output .\models\hy3d-shape-f16.gguf `
  --outtype f16
```

Checkpoint loading is safe by default (`torch.load(weights_only=True)`). Only for a trusted legacy checkpoint that cannot be loaded safely, add `--allow-unsafe-pickle`; this explicitly enables arbitrary pickle deserialization.

Inspect the result:

```powershell
.\build\Release\hy3d.exe inspect --model .\models\hy3d-shape-f16.gguf
```

Read a tensor's metadata and leading bytes from C++:

```powershell
.\build\Release\hy3d.exe tensor `
  --model .\models\hy3d-shape-f16.gguf `
  --name blocks.0.attn1.k_norm.weight `
  --bytes 32
```

This verifies that the native executable can seek into the GGUF tensor data section and read actual tensor bytes.

Run a real block-0 DiT smoke pass from the converted GGUF:

```powershell
.\build\Release\hy3d.exe dit-block `
  --model .\models\hy3d-shape-f16.gguf `
  --block 0 `
  --block-count 2 `
  --tokens 1 `
  --heads 16 `
  --head-dim 128
```

By default this smoke path now loads one or more DiT blocks, attn1 q/k RMSNorm, attn2 cross-attention, MLP or MoE feed-forward tensors, skip projection tensors, and the top-level timestep embedder. It prepends one timestep conditioning token to the synthetic latent token buffer, matching the official Hunyuan3D block input shape more closely. Use `--block-count N`, `--no-cross-attn`, `--no-timestep`, `--no-mlp`, or `--dry-run` to narrow the smoke path.

On the local converted `hy3d-shape-f16.gguf`, blocks 0-1 load 52 tensors and run through the native multi-block path. Block 18 loads and runs the MoE path with `--no-cross-attn --no-timestep`.

Run the fuller native DiT scaffold:

```powershell
.\build\Release\hy3d.exe dit-forward `
  --model .\models\hy3d-shape-f16.gguf `
  --block-count 1 `
  --latent-tokens 1 `
  --latent-dim 64 `
  --context-tokens 1 `
  --context-dim 1024 `
  --heads 16 `
  --head-dim 128
```

`dit-forward` loads `x_embedder`, optional `pooler/extra_embedder`, the selected DiT blocks, and `final_layer`. It accepts raw little-endian F32 input files through `--latent-bin` and `--context-bin`; when omitted, it uses zero-filled smoke tensors.

## Native Runtime Status

Implemented:

- GGUF tensor table parsing.
- GGUF tensor data offset calculation.
- Named tensor byte loading through `hy3d::read_gguf_tensor_data`.
- `hy3d tensor` CLI command for native tensor readback.
- Initial C++ interfaces for `HunyuanDitModel`, `DiffusionScheduler`, and `MeshDecoder`.
- CPU F16-to-F32 tensor decode.
- CPU linear projection primitive for `x @ W^T + b`.
- `HunyuanDitModel::project_linear()` for named runtime tensors.
- Numerically stable CPU softmax.
- CPU scaled dot-product attention over `{tokens, heads, head_dim}` buffers.
- CPU cross-attention over separate query and context token counts.
- Head-wise RMS q/k normalization for official `attn1.q_norm/k_norm` and `attn2.q_norm/k_norm` tensors.
- `HunyuanDitModel::project_attention_qkv()` for named `attn_q/attn_k/attn_v` tensors.
- Batched CPU linear projection for flattened token sequences.
- `HunyuanDitModel::run_attention_block()` for `Q/K/V projection -> attention -> output projection`.
- CPU layer norm and GELU.
- `HunyuanDitModel::run_dit_block()` for `norm1 -> attn1 -> residual -> optional norm2/attn2 -> residual -> norm3/MLP -> residual`.
- `HunyuanDitModel::run_dit_blocks_conditioned()` for sequential multi-block smoke forward with skip projection support.
- `HunyuanDitModel::run_moe_block()` for simplified inference-time top-k MoE routing using `moe.gate`, routed experts, and shared experts.
- Top-level `t_embedder` projection for timestep conditioning token smoke tests.
- `x_embedder` latent projection for raw latent tokens.
- Optional `pooler`/`extra_embedder` context pooling path. The local Hunyuan3D-2.1 shape config has `use_attention_pooling: false`, so missing pooler tensors are treated as optional.
- `final_layer` projection that drops the conditioning token and emits latent-channel outputs.
- `hy3d dit-forward` scaffold for `x_embedder -> timestep/context conditioning -> blocks -> final_layer`.
- Real GGUF tensor-name mapping for official Hunyuan3D names such as `blocks.0.attn1.to_q.weight`, `blocks.0.attn1.out_proj.weight`, and `blocks.0.mlp.fc1.weight`.
- Selective GGUF block tensor loading through `load_hunyuan_dit_block_from_gguf()`.
- `hy3d dit-block` CLI command that loads a real block from GGUF and runs the current native DiT block primitive.
- Euler scheduler step primitive.
- Density-grid mesh decoder primitive that emits OBJ-style surface triangles.

Not implemented yet:

- Full Hunyuan3D DiT forward graph across every block, attention pooling, final projection, and native latent I/O.
- Image encoder conditioning.
- Diffusion denoising loop.
- Full VAE latent-to-density decode.
- GLB mesh writing from native inference.

## Usage

```powershell
.\build\Release\hy3d.exe --help
.\build\Release\hy3d.exe inspect --model .\models\hy3d-shape-q8.gguf
.\build\Release\hy3d.exe --inspect .\models\hy3d-shape-q8.gguf
.\build\Release\hy3d.exe dit-block --model .\models\hy3d-shape-f16.gguf --block 0 --block-count 2 --tokens 1 --heads 16 --head-dim 128
.\build\Release\hy3d.exe dit-block --model .\models\hy3d-shape-f16.gguf --block 0 --tokens 1 --heads 16 --head-dim 128 --no-cross-attn --no-timestep --no-mlp
.\build\Release\hy3d.exe dit-block --model .\models\hy3d-shape-f16.gguf --block 18 --tokens 1 --heads 16 --head-dim 128 --no-cross-attn --no-timestep
.\build\Release\hy3d.exe dit-forward --model .\models\hy3d-shape-f16.gguf --block-count 1 --latent-tokens 1 --latent-dim 64 --context-tokens 1 --context-dim 1024 --heads 16 --head-dim 128
```

Python backend bridge, verified on this machine with RTX 3070 Ti:

```powershell
.\build\Release\hy3d.exe generate `
  --backend python `
  --quality smoke `
  --image .\third_party\Hunyuan3D-2.1\hy3dshape\demos\demo.png `
  --out .\outputs\demo.glb `
  --model-path .\models\Hunyuan3D-2.1 `
  --device cuda `
  --seed 42
```

Quality presets:

- `smoke`: 5 steps, useful for CUDA checks.
- `draft`: 10 steps.
- `normal`: 30 steps, default.
- Explicit `--steps N` overrides `--quality`.

Each Python backend run writes sidecars next to the GLB:

- `output.glb.log.txt`
- `output.glb.json`

PBR texture generation, verified on this machine:

```powershell
.\build\Release\hy3d.exe texture `
  --backend python `
  --mesh .\outputs\hy3d-cli-smoke-5.glb `
  --image .\third_party\Hunyuan3D-2.1\hy3dshape\demos\demo.png `
  --out .\outputs\hy3d-texture-smoke.glb `
  --model-path .\models\Hunyuan3D-2.1 `
  --resolution 512 `
  --max-views 6 `
  --no-remesh `
  --device cuda
```

First-time setup:

```powershell
.\scripts\download_hy3d_models.ps1
.\scripts\setup_hy3d_python.ps1
```

The direct requirement files are compile inputs. Setup installs the complete
136-package Windows/Python 3.10/CUDA 12.4 resolution from
`requirements-win-cu124.lock.txt`, then writes `hy3d-dependencies.json` with
the actual Python/package/torch/CUDA and pinned source/model revision details.

If the Hunyuan3D-Paint native extensions need to be rebuilt:

```powershell
.\scripts\build_hy3dpaint_windows.ps1
```

The build script detects CUDA, Visual Studio C++ tools, the Windows SDK, and
the Python import library. It also applies a compatibility ceiling: CUDA 12.1
selects MSVC 14.29 on the verified machine because MSVC 14.51 crashes
`cudafe++` with status `0xC0000409`. Override a detected value when needed:

```powershell
.\scripts\build_hy3dpaint_windows.ps1 -CudaRoot "C:\CUDA\v12.4" -CudaArchitecture "8.6"
```

The latest verified shape output is `outputs\acceptance-next-shape.glb`
(11,250,604 bytes). A fresh 5-step/seed-42 run took 249.656 seconds on the RTX
3070 Ti and independently reopened as one geometry with 312,722 vertices and
624,760 faces.

The latest verified texture output is `outputs\acceptance-next-textured.glb`
(17,695,540 bytes). It completed in 1284.569 seconds with resolution 512,
6 views and remeshing disabled, and independently reopened with PBR material,
texture visuals and UVs. Hunyuan3D-Paint officially recommends much more VRAM
for this setting, so expect long runtimes on 8 GB cards.

Real-model loader benchmark:

```powershell
.\build\Release\benchmark_gguf_loader.exe .\models\hy3d-shape-f16.gguf 0 1
```

On the 6,101,566,528-byte local GGUF this measured 0.090 seconds to inspect 752
tensors and 2.532 seconds to load 28 block-0 tensors, with peak RSS of
197,054,464 bytes.

The end-to-end native backend is intentionally not implemented yet:

```powershell
.\build\Release\hy3d.exe generate --backend native --image .\examples\input.png --out .\outputs\shape.obj
```
