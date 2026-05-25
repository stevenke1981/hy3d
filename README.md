# Hunyuan3D GGUF C++ CLI

Windows-native C++ CLI shell for local Hunyuan3D workflows.

This first version provides:

- CLI parsing and help
- GGUF inspection
- Python backend bridge
- Shape generation through official CUDA PyTorch backend
- PBR texture generation through official Hunyuan3D-Paint backend
- Initial native GGUF tensor loading and DiT block forward smoke path

## Build

```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

## One-Click Windows Release

Create a redistributable Windows CUDA folder:

```powershell
.\scripts\make_release.ps1
```

The output is `dist\hy3d-win-cuda`. It contains `hy3d-setup.cmd`, `hy3d-generate-smoke.cmd`, `hy3d-texture-smoke.cmd`, and `hy3d.cmd`. Setup installs the Python backend, builds the Hunyuan3D-Paint Windows extensions, and downloads official shape/PBR model files.

For a mostly offline package from an already prepared checkout:

```powershell
.\scripts\make_release.ps1 -IncludeModels -IncludeVenv -Zip
```

## Download Hunyuan3D Models

Use the current Hugging Face CLI, `hf`. The older `huggingface-cli` command is deprecated.

Shape generation checkpoints:

```powershell
hf download tencent/Hunyuan3D-2.1 `
  README.md LICENSE Notice.txt demo.py `
  "hunyuan3d-dit-v2-1/config.yaml" `
  "hunyuan3d-dit-v2-1/model.fp16.ckpt" `
  "hunyuan3d-vae-v2-1/config.yaml" `
  "hunyuan3d-vae-v2-1/model.fp16.ckpt" `
  --local-dir .\models\Hunyuan3D-2.1
```

The downloaded `.ckpt` files are official PyTorch checkpoints, not GGUF files. Use them with the Python backend first, then convert them to GGUF when the converter is implemented.

PBR texture checkpoints:

```powershell
hf download tencent/Hunyuan3D-2.1 `
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
.\scripts\setup_hy3d_python.ps1
```

If the Hunyuan3D-Paint native extensions need to be rebuilt:

```powershell
.\scripts\build_hy3dpaint_windows.ps1
```

The verified smoke output is `outputs\hy3d-cli-smoke-5.glb`, about 11 MB, and loads as a GLB scene with one geometry. A 5-step run took about 210 seconds on the RTX 3070 Ti.

The verified texture smoke output is `outputs\hy3d-texture-smoke.glb`, about 17.7 MB. It completed on the RTX 3070 Ti in about 1083 seconds with `--resolution 512 --max-views 6 --no-remesh`. Hunyuan3D-Paint officially recommends much more VRAM for this setting, so expect long runtimes on 8 GB cards.

Do not use `--low-vram` yet for the official Python backend on this checkout. The upstream CPU offload path currently mixes CPU and CUDA scheduler tensors. The non-offload CUDA path works for the downloaded shape model on this machine.

The end-to-end native backend is intentionally not implemented yet:

```powershell
.\build\Release\hy3d.exe generate --backend native --image .\examples\input.png --out .\outputs\shape.obj
```
