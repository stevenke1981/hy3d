# Hunyuan 測試與驗證計畫

> 本文件同時記錄 2026-06-30 實際基線，以及改善項目的驗收測試。
>
> 原則：核心測試不依賴 CUDA／大型模型；真實模型測試獨立標記並保存可重現版本資訊。

## 1. 已執行基線與改善後驗收

執行環境：Windows PowerShell，CMake Visual Studio generator。

```powershell
cmake -S . -B build
cmake --build build --config Debug --parallel
ctest --test-dir build -C Debug --output-on-failure
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

初始檢視結果：

| 組態 | Build | CTest |
|---|---:|---:|
| Debug | 通過 | 8/8 通過，0.90 秒 |
| Release | 通過 | 8/8 通過，0.78 秒 |

通過的 CTest：`cli`、`gguf`、`gguf_loader`、`hy3d_runtime`、`hy3d_model_loader`、`hy3d_math`、`backend`、`python_converter`。

這個基線只證明目前小型 fixture 與 dry-run 行為通過；未執行真實 CUDA shape/texture 推論，也未驗證 release zip 的 clean-machine setup。

### 2026-06-30 改善後結果

```powershell
cmake --build build --config Debug --parallel
ctest --test-dir build -C Debug -LE slow --output-on-failure --timeout 60
cmake --build build --config Release --parallel
ctest --test-dir build -C Release -LE slow --output-on-failure --timeout 60
ctest --test-dir build -C Release -R '^make_release$' --output-on-failure
```

| 組態／Gate | 結果 |
|---|---:|
| Debug build | 通過，所有 first-party targets 使用 `/W4 /permissive-` |
| Debug 非 slow CTest | 15/15 通過，3.37 秒 |
| Release build | 通過 |
| Release 非 slow CTest | 15/15 通過，3.69 秒 |
| Clean release test | 1/1 通過，16.50 秒 |

新增覆蓋：child process argv round-trip/launch failure、backend path、wrapper missing-backend、惡意 GGUF corpus、`Result::take_value()`、pinned local source checkout、safe checkpoint load、generate/texture export failure與原子輸出、clean release configure/package。

仍未執行：真實 CUDA shape/texture、完整線上 release setup、真實模型 native parity、loader peak RSS/時間 benchmark。

## 2. 現有覆蓋與缺口

| 區域 | 現有覆蓋 | 主要缺口 |
|---|---|---|
| CLI | 常見 parse 成功/失敗 | 整數邊界、極大維度、各 subcommand property-based cases |
| Backend | request validation、dry-run、argv round-trip、launch failure、path resolver、wrapper 非零失敗 | 真實 Python backend 非零/缺產物整合測試 |
| GGUF | 最小合法檔、錯誤 magic、單 tensor、版本/count/array/offset/range/duplicate 負向 fixture | 持續 fuzzing 與更完整 GGML type corpus |
| Runtime | 小型線性層、attention、block、scheduler、mesh fixture | 官方模型數值 parity、NaN/Inf、效能與記憶體 |
| Converter | writer metadata/tensor、name mapping、safe default/unsafe opt-in | 重複名稱、部分輸出清理、大模型 peak RSS |
| Python pipelines | generate/texture export failure、error sidecar、舊輸出保留、partial cleanup | import/init/inference 各階段、metadata write failure、缺失 CUDA/model |
| Release/setup | clean configure/build/package、pinned local source checkout | 真實線上下載、zip 解壓後 CUDA shape/texture smoke |

## 3. P0/P1 自動測試

### 3.1 Process runner round-trip

新增一個只回顯 argv 到 UTF-8 JSON 的測試 helper，測試 `run_process()`：

1. 一般 ASCII。
2. 含空白與空字串。
3. 中文與 emoji。
4. 內嵌雙引號、尾端反斜線。
5. `& | < > ^ % !`。
6. 不存在 executable。
7. child 分別回傳 0、1、42。

驗收：helper 實收 argv 與輸入逐項完全相同；測試不得啟動 shell command。

### 3.2 Malformed GGUF corpus

每個 fixture 都限制在數 KB，測試需在短時間、低記憶體下失敗：

1. header 在每一欄位截斷。
2. 不支援的 GGUF version。
3. `metadata_count`／`tensor_count = UINT64_MAX`。
4. string length 超過上限。
5. array `bytes * count` overflow。
6. nested array 超過深度。
7. tensor dimension product overflow。
8. `data_start + tensor_offset` overflow。
9. tensor range超過實際 file size。
10. 重複 tensor name 與未知 GGML type。

額外加入 libFuzzer/AFL++ target（Linux 可跑即可）；seed corpus 使用現有最小 GGUF fixture。

### 3.3 Checkpoint loader safety

1. 純 state dict 在 `weights_only=True` 成功。
2. 需要 pickle object 的 checkpoint 在預設模式被拒絕。
3. `--allow-unsafe-pickle` 未指定時不得 fallback。
4. 重複 mapped name 回傳非零並刪除／不提交部分 GGUF。
5. safetensors 路徑可在不啟用 pickle 下完成。

### 3.4 Python sidecar failure matrix

以 mock/fake pipeline 注入下列故障：

- dependency import failure
- CUDA requested but unavailable
- pipeline constructor failure
- inference failure
- export failure
- metadata write failure
- 使用者中斷（至少保證 log flush；是否寫 `cancelled` 由設計決定）

驗收：

- generate 與 texture 均回傳穩定非零 code。
- sidecar 最終狀態不是 `started`。
- error 摘要存在，且不包含 secret/env dump。
- 舊輸出不會被誤報為本次成功輸出；半成品不留在 final path。

### 3.5 Backend path 與 wrapper

1. 在 repo root、`build\Debug`、隨機 temp cwd 執行 dry-run。
2. executable/package path 含空白與中文。
3. backend env override 指向不存在檔案。
4. Python executable 不存在。
5. backend script 回傳非零。
6. backend 回傳 0 但未產生輸出。

驗收：成功/失敗與檔案結果一致，不能出現 “Would run” 後 exit 0。

## 4. 效能與數值測試

### 4.1 GGUF loader benchmark

固定同一檔案與冷/暖 cache 條件，記錄：

- inspect 時間
- 載入 N 個 tensors 的 wall time
- file open 次數
- peak RSS
- tensor lookup 每秒次數

比較基線與「單次開檔 + hash index + move」版本。主要目標不是任意百分比，而是：

- file open 次數由 O(tensors) 降至 O(1)。
- tensor info/runtime lookup 由線性掃描降至平均 O(1)。
- 移除已知的 tensor byte vector 額外複製。

### 4.2 Native parity

為每一個已宣稱可用的 primitive 建立 Python/reference fixture：

1. tensor 名稱與 shape。
2. 輸入輸出皆為 finite。
3. FP32 使用明確 `atol/rtol`。
4. 至少比較一個 attention、conditioned block、timestep、final layer。
5. 固定 seed 與模型 revision。

不得只比 L1 checksum；checksum 可作 smoke signal，但不能定位維度錯置或正負抵銷。

### 4.3 真實模型 smoke

標記為 `model`／`cuda`，不放進每次 PR 的快速 job：

```powershell
.\build\Release\hy3d.exe generate --backend python --quality smoke `
  --image .\third_party\Hunyuan3D-2.1\hy3dshape\demos\demo.png `
  --out .\outputs\review-shape.glb `
  --model-path .\models\Hunyuan3D-2.1 --device cuda --seed 42

.\build\Release\hy3d.exe texture --backend python `
  --mesh .\outputs\review-shape.glb `
  --image .\third_party\Hunyuan3D-2.1\hy3dshape\demos\demo.png `
  --out .\outputs\review-textured.glb `
  --model-path .\models\Hunyuan3D-2.1 `
  --resolution 512 --max-views 6 --no-remesh --device cuda
```

除 exit code 外，驗證：

- GLB 存在、非零且可由獨立 parser 重新開啟。
- scene 至少有一個 mesh、合理 vertex/face 數。
- texture 流程輸出材質與 image references。
- JSON sidecar 為 `ok`，記錄 model revision、torch/CUDA、seed、elapsed。

## 5. Release 驗收

在乾淨暫存目錄：

1. checkout 後刪除/確認不存在 `build`、`.venv-hy3d`、`third_party`、`models`。
2. 執行 release build，確認會自行 configure。
3. 解壓 release zip 到含空白與中文的路徑。
4. 僅依 `README_RELEASE.md` 執行 setup。
5. 執行 generate/texture smoke。
6. 逐項驗證 `SHA256SUMS.txt` 使用 package-relative path 且 hash 全部正確。
7. 從 release root 以外 cwd 執行 `bin\hy3d.exe`。

任何未宣告的本機全域依賴、硬編碼 VS/CUDA/Python 路徑或缺失 source checkout，都視為 release gate 失敗。

## 6. CI 建議

快速 PR jobs：

```powershell
cmake -S . -B build
cmake --build build --config Debug --parallel
ctest --test-dir build -C Debug --output-on-failure
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
python -m unittest tests/test_convert_hy3d_shape_to_gguf.py
```

另設：

- Windows process/path job。
- malformed GGUF + sanitizer/fuzzer job。
- nightly clean-release job。
- 手動或 nightly CUDA/model parity job。

## 7. 完成門檻

- P0 測試全部進 CI 且可重現。
- 核心測試不需要下載模型或使用 CUDA。
- 真實模型測試有固定 revision 與結果驗證，不只檢查檔案存在。
- 所有「成功」路徑同時符合 exit code、輸出檔與 sidecar 狀態。
- 效能改善附 before/after 數據；正確性改善附 regression test。
