# Hunyuan3D GGUF C++ CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first Windows-native Hunyuan3D C++ CLI with help output, GGUF inspection, and a Python backend bridge.

**Architecture:** The CLI is a small C++17 executable split into argument parsing, GGUF inspection, process execution, and backend dispatch modules. Native Hunyuan3D GGUF inference is documented as a later phase because it requires model architecture implementation beyond GGUF file loading.

**Tech Stack:** C++17, CMake, CTest, PowerShell backend bridge, Windows-first filesystem/process behavior.

---

### Task 1: Documentation and Scope

**Files:**
- Create: `spec.md`
- Create: `plan.md`

- [x] **Step 1: Write the MVP spec**

Define supported commands, backend behavior, non-goals, and future native GGUF phases in `spec.md`.

- [x] **Step 2: Write the implementation plan**

Define this task list in `plan.md` so implementation can proceed without guessing.

### Task 2: Test Harness

**Files:**
- Create: `CMakeLists.txt`
- Create: `tests/test_cli.cpp`
- Create: `tests/test_gguf.cpp`
- Create: `tests/test_backend.cpp`

- [x] **Step 1: Add failing tests for CLI parsing**

Expected behavior:

```cpp
auto parsed = hy3d::parse_args({"hy3d", "inspect", "--model", "model.gguf"});
assert(parsed.ok());
assert(parsed.value().command == hy3d::CommandKind::Inspect);
assert(parsed.value().model_path == "model.gguf");
```

- [x] **Step 2: Add failing tests for GGUF inspection**

Expected behavior:

```cpp
auto info = hy3d::inspect_gguf("tmp-valid.gguf");
assert(info.ok());
assert(info.value().version == 3);
assert(info.value().metadata_count == 0);
assert(info.value().tensor_count == 0);
```

- [x] **Step 3: Add failing tests for backend dispatch**

Expected behavior:

```cpp
hy3d::GenerateRequest request;
request.backend = "native";
auto result = hy3d::run_generate(request);
assert(!result.ok());
assert(result.error().find("native Hunyuan3D inference is not implemented yet") != std::string::npos);
```

- [x] **Step 4: Run tests and verify they fail before implementation**

Run:

```powershell
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Expected: compilation or tests fail because production headers/functions are missing.

### Task 3: CLI Parser and Main Entrypoint

**Files:**
- Create: `src/hy3d_result.h`
- Create: `src/hy3d_cli.h`
- Create: `src/hy3d_cli.cpp`
- Create: `src/main.cpp`

- [x] **Step 1: Implement a small result type**

`hy3d_result.h` provides `Result<T>` and `Status` for readable errors without exceptions at module boundaries.

- [x] **Step 2: Implement argument parsing**

Support `--help`, `inspect --model`, `--inspect`, and `generate` command forms.

- [x] **Step 3: Implement main dispatch**

`main.cpp` calls the parser, then dispatches to inspect or generate.

- [x] **Step 4: Run CLI tests**

Run:

```powershell
cmake --build build
ctest --test-dir build -R cli --output-on-failure
```

Expected: CLI tests pass.

### Task 4: GGUF Inspect

**Files:**
- Create: `src/hy3d_gguf.h`
- Create: `src/hy3d_gguf.cpp`

- [x] **Step 1: Implement GGUF header reading**

Read magic `GGUF`, version, tensor count, and metadata count.

- [x] **Step 2: Implement metadata skipping**

Support the scalar and array value types needed to skip valid GGUF metadata without decoding all values.

- [x] **Step 3: Print inspect output**

Include format, version, tensor count, metadata count, and tensor names when present.

- [x] **Step 4: Run GGUF tests**

Run:

```powershell
cmake --build build
ctest --test-dir build -R gguf --output-on-failure
```

Expected: GGUF tests pass.

### Task 5: Backend Bridge

**Files:**
- Create: `src/process_runner.h`
- Create: `src/process_runner.cpp`
- Create: `src/hy3d_backend.h`
- Create: `src/hy3d_backend.cpp`
- Create: `scripts/run_python_backend.ps1`

- [x] **Step 1: Implement backend request validation**

Validate image path, output path, backend name, and native placeholder behavior.

- [x] **Step 2: Implement PowerShell script command construction**

Call `scripts/run_python_backend.ps1` for `--backend python`.

- [x] **Step 3: Implement the PowerShell backend script**

The script validates input and prints the exact official Hunyuan3D command it would run unless `HY3D_PYTHON_BACKEND` is set to an existing script.

- [x] **Step 4: Run backend tests**

Run:

```powershell
cmake --build build
ctest --test-dir build -R backend --output-on-failure
```

Expected: backend tests pass.

### Task 6: Full Verification

**Files:**
- Modify: `README.md`

- [x] **Step 1: Add README usage**

Document build commands, inspect examples, and Python backend bridge usage.

- [x] **Step 2: Build release**

Run:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Expected: `hy3d.exe` builds.

- [x] **Step 3: Run all tests**

Run:

```powershell
ctest --test-dir build --output-on-failure
```

Expected: all tests pass.

### Task 7: Windows CUDA Release Package

**Files:**
- Create: `scripts/make_release.ps1`
- Create: `scripts/download_hy3d_models.ps1`
- Modify: `README.md`

- [x] **Step 1: Add release packager**

Build `hy3d.exe`, copy backend scripts, examples, converter, docs, and required Hunyuan3D source files into `dist\hy3d-win-cuda`.

- [x] **Step 2: Add one-click launchers**

Add `hy3d-setup.cmd`, `hy3d.cmd`, `hy3d-generate-smoke.cmd`, and `hy3d-texture-smoke.cmd`.

- [x] **Step 3: Add model downloader**

Use Hugging Face `hf` or `uvx --from huggingface-hub hf` to download shape/PBR models and RealESRGAN.

### Task 8: Native Runtime Push

**Files:**
- Modify: `src/hy3d_math.*`
- Modify: `src/hy3d_runtime.*`
- Modify: `tests/test_hy3d_math.cpp`
- Modify: `tests/test_hy3d_runtime.cpp`

- [x] **Step 1: Add CPU normalization and activation primitives**

Implement batched layer norm and GELU.

- [x] **Step 2: Add DiT block composition**

Implement `norm -> attention -> residual -> norm -> MLP -> residual` for already-loaded tensors.

- [x] **Step 3: Add scheduler and mesh decoder primitives**

Implement Euler stepping and a density-grid-to-OBJ surface primitive.

- [x] **Step 4: Verify**

Run CMake release build and CTest.

### Task 9: Real GGUF DiT Block Wiring

**Files:**
- Modify: `src/hy3d_runtime.*`
- Create: `src/hy3d_model_loader.*`
- Modify: `src/hy3d_cli.*`
- Modify: `src/main.cpp`
- Modify: `tests/test_hy3d_runtime.cpp`
- Create: `tests/test_hy3d_model_loader.cpp`
- Modify: `README.md`
- Modify: `spec.md`

- [x] **Step 1: Map official Hunyuan3D tensor names**

Wire canonical runtime names such as `attn_q.weight`, `attn_output.weight`, and `mlp_fc1.weight` to real GGUF names such as `attn1.to_q.weight`, `attn1.out_proj.weight`, and `mlp.fc1.weight`.

- [x] **Step 2: Load a DiT block from GGUF**

Add a selective block loader that reads only the requested block's attention, norm, and optional MLP tensors from GGUF into `HunyuanDitModel`.

- [x] **Step 3: Add `dit-block` CLI smoke path**

Add a command that loads `blocks.N` from GGUF and runs `run_dit_block()` on a small token buffer.

- [x] **Step 4: Verify with tests and real model**

Run release build, CTest, and a real `hy3d-shape-f16.gguf` block-0 smoke pass with and without MLP.

### Task 10: Official Block Structure Push

**Files:**
- Modify: `src/hy3d_math.*`
- Modify: `src/hy3d_runtime.*`
- Modify: `src/hy3d_model_loader.*`
- Modify: `src/hy3d_cli.*`
- Modify: `src/main.cpp`
- Modify: `tests/test_hy3d_runtime.cpp`
- Modify: `tests/test_hy3d_model_loader.cpp`
- Modify: `tests/test_cli.cpp`
- Modify: `README.md`
- Modify: `spec.md`

- [x] **Step 1: Add q/k head normalization**

Implement head-wise RMSNorm for official `attn1.q_norm/k_norm` and `attn2.q_norm/k_norm` tensors.

- [x] **Step 2: Add attn2 cross-attention**

Support separate query/context token counts and context input width for `attn2.to_k/to_v` tensors.

- [x] **Step 3: Add timestep conditioning smoke path**

Load `t_embedder.mlp.*`, project a sinusoidal timestep embedding, and prepend it as the official conditioning token for block smoke tests.

- [x] **Step 4: Verify and publish**

Run CTest, run real GGUF smoke with default conditioned block path, then push to GitHub.

### Task 11: Multi-Block and MoE Forward

**Files:**
- Modify: `src/hy3d_runtime.*`
- Modify: `src/hy3d_model_loader.*`
- Modify: `src/hy3d_cli.*`
- Modify: `src/main.cpp`
- Modify: `tests/test_hy3d_runtime.cpp`
- Modify: `tests/test_hy3d_model_loader.cpp`
- Modify: `tests/test_cli.cpp`
- Modify: `README.md`
- Modify: `spec.md`

- [x] **Step 1: Add multi-block CLI control**

Add `--block-count N` so `dit-block` can load and run a contiguous block range.

- [x] **Step 2: Load range tensors**

Add block-range tensor name generation and GGUF loading for repeated `blocks.N.*` tensors.

- [x] **Step 3: Add sequential block runner**

Run `blocks.first..first+count-1` with optional skip projection support when `skip_linear/skip_norm` tensors exist.

- [x] **Step 4: Add MoE inference primitive**

Implement simplified inference-time top-k MoE routing over `moe.gate`, routed experts, and `shared_experts`.

- [x] **Step 5: Verify**

Run CTest, real GGUF block-count smoke, and real block 18 MoE smoke.

### Task 12: DiT Forward Scaffold

**Files:**
- Modify: `src/hy3d_runtime.*`
- Modify: `src/hy3d_model_loader.*`
- Modify: `src/hy3d_cli.*`
- Modify: `src/main.cpp`
- Modify: `tests/test_hy3d_runtime.cpp`
- Modify: `tests/test_hy3d_model_loader.cpp`
- Modify: `tests/test_cli.cpp`
- Modify: `README.md`
- Modify: `spec.md`

- [x] **Step 1: Add forward-level tensor loading**

Load `x_embedder`, `final_layer`, optional `pooler/extra_embedder`, and selected DiT block tensors.

- [x] **Step 2: Add attention pooling scaffold**

Implement optional context pooling compatible with official `AttentionPool` tensor names. Treat missing pooler tensors as optional because the local shape config has `use_attention_pooling: false`.

- [x] **Step 3: Add final layer projection**

Apply `final_layer.norm_final`, drop the prepended conditioning token, and project back to latent channels through `final_layer.linear`.

- [x] **Step 4: Add raw latent/context input CLI**

Add `hy3d dit-forward` with `--latent-bin` and `--context-bin` raw little-endian F32 inputs, falling back to zero smoke tensors when omitted.

- [x] **Step 5: Verify**

Run CTest, real GGUF dry-run, real block-count 0 final-layer smoke, real block-count 1 scaffold smoke, and raw F32 input smoke.

### Task 13: Reusable Image-to-3D Script

**Files:**
- Create: `scripts/generate_3d_model.ps1`
- Modify: `README.md`

- [x] **Step 1: Add a reusable Windows PowerShell wrapper**

Wrap the verified `hy3d.exe generate --backend python` path with image/output/model/device/quality/seed parameters, automatic release build, UTF-8 environment setup, and optional Hunyuan3D-Paint texture generation.

- [x] **Step 2: Validate with a real local image**

Generate `outputs\frieren-character-smoke.glb` from `C:\Users\steven\Downloads\ChatGPT Image 2026年5月25日 下午11_20_07.png` through CUDA PyTorch.

- [x] **Step 3: Document reuse**

Add a README section with the reusable command, options, and output sidecars.
