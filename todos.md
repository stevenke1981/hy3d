# Hunyuan 專案改善 TODO

> 檢視日期：2026-06-30
>
> 方法：CBM 專案索引 `cbm+hunyuan`（63 files / 353 symbols / 843 edges）、關鍵符號與呼叫路徑檢視、Debug/Release 實際建置、CTest 與真實 CUDA/model smoke。
>
> 原則：先處理安全與錯誤成功，再做大型模型的 I/O／查找優化，最後拆分結構與補齊工程化。

## 2026-06-30 實作狀態

已完成 14/15 項。CLI parser/handler、Python pipeline orchestration、136-package transitive lock、真實 GGUF benchmark 與 NumPy parity 均已完成；真實 RTX 3070 Ti shape/texture smoke 也通過。release zip 已能在含空白/中文的路徑解壓、逐檔驗證 SHA-256，並從套件外 cwd 啟動 executable。唯一保留未勾選的是「從全新 release zip 執行線上 setup 與 CUDA smoke」的 clean-machine 閉環。

## P0：安全阻斷

- [x] **移除 `std::system()` 字串式程序啟動**
  - 位置：`src/process_runner.cpp:8-40`
  - 證據：`quote_arg()` 只在空白、Tab 或雙引號存在時加引號；`a&whoami`、`a|...`、`<`、`>`、`^`、`%` 等 Windows shell meta character 可保持未引用，之後整條命令交給 `std::system()`。
  - 改善：Windows 使用 `CreateProcessW`／`ShellExecuteExW` 的參數陣列與正確 Windows quoting；若保留跨平台層，將平台實作封裝在 `process_runner_win.cpp`。不要經過 `cmd.exe`。
  - 驗收：
    - 含空白、Unicode、引號、尾端反斜線及 `&|<>^%!` 的參數，子程序收到的值須逐字一致。
    - 啟動失敗回傳 `Result::failure`，正常非零結束碼與「無法啟動」可區分。
    - 新增 `test_process_runner`，不可用只測 `quote_arg()` 取代真實子程序 round-trip。

- [x] **禁止轉換器以不安全 pickle 模式載入任意 checkpoint**
  - 位置：`tools/convert-hy3d-shape-to-gguf.py:303-311`
  - 證據：`torch.load(..., weights_only=False)` 允許 checkpoint 反序列化時執行 Python pickle payload。
  - 改善：預設 `weights_only=True`；不相容舊檔必須以明確的 `--allow-unsafe-pickle` opt-in 才能載入，並在執行前輸出強警告。優先支援 safetensors。
  - 驗收：
    - 預設模式拒絕需要任意 pickle 物件的 checkpoint。
    - 一般 state dict 仍可轉換。
    - CLI help 與 README 清楚說明信任邊界。

## P1：正確性、健壯性與主要效能

- [x] **為 GGUF parser 加入檔案邊界與資源上限**
  - 位置：`src/hy3d_gguf.cpp:116-279`
  - 證據：
    - `metadata_count`、`tensor_count`、array `count` 沒有合理上限。
    - `bytes * count`、`data_start_offset + data_offset` 缺少完整溢位檢查。
    - `read_gguf_tensor_data()` 依檔案宣告尺寸先配置 vector，再確認資料是否真的存在。
    - nested array 可遞迴進入 `skip_value()`，沒有深度限制。
  - 改善：開檔時取得實際 file size；所有 offset／length 使用 checked arithmetic；配置前驗證範圍；限制 metadata、tensor、array、string、rank 與遞迴深度；拒絕重複 tensor 名稱與不支援的 GGUF version。
  - 驗收：截斷、超大 count、溢位 offset、未知 type、重複名稱、深層 array 均快速回傳錯誤，且不做大型配置。

- [x] **將 GGUF 模型載入改為單次開檔、索引查找與 move**
  - 位置：`src/hy3d_model_loader.cpp:19-52`、`src/hy3d_runtime.cpp:180-209`
  - 證據：每個要求 tensor 都線性搜尋 `tensor_infos`，再由 `read_gguf_tensor_data()` 重新開啟同一檔案；`runtime_tensor.bytes = bytes.value()` 又複製一次資料。runtime 的 `find_tensor()` 也每次線性掃描。
  - 改善：
    - 建立 `name -> tensor info/index` hash index。
    - loader 共用一個 file handle，依 offset 批次讀取；後續可評估 read-only memory mapping。
    - 提供 `Result<T>::take_value()` 或等價 move API。
    - `HunyuanDitModel` 維護名稱索引，避免 inference hot path 反覆 O(N) 搜尋。
  - 驗收：以實際模型或可重現大型 fixture 比較載入時間、開檔次數與 peak RSS；輸出數值須與修改前一致。
  - 本輪驗收：既有 GGUF/model/runtime 數值 fixture 全部通過；真實模型時間與 peak RSS 統一留在「效能基準與 native parity gate」項目。

- [x] **讓 backend script 定位不依賴目前工作目錄**
  - 位置：`src/hy3d_backend.cpp:10-16`
  - 證據：script path 使用 `std::filesystem::current_path()/scripts/...`；從非 repo／release 根目錄執行 `hy3d.exe` 會找不到 backend。
  - 改善：以 executable 位置推導安裝根目錄，並保留明確的環境變數／CLI override；錯誤訊息列出實際探測路徑。
  - 驗收：從任意 cwd 執行 `bin\hy3d.exe` 的 generate/texture dry-run 均能定位 package 內 scripts。

- [x] **禁止 wrapper 在 backend 缺失時假成功**
  - 位置：`scripts/run_python_backend.ps1`、`scripts/run_texture_backend.ps1`
  - 證據：找不到 backend 時只輸出 “Would run...” 並 `exit 0`，上層會把未產生輸出的情況視為成功。
  - 改善：缺失或無法啟動 backend 時使用非零 exit code；成功後驗證預期輸出與 sidecar 狀態。
  - 驗收：錯誤的 `HY3D_PYTHON_BACKEND`／`HY3D_TEXTURE_BACKEND` 必須失敗，且不得留下 `status=ok`。

- [x] **統一 generate/texture 的非預期例外與 sidecar 狀態**
  - 位置：`scripts/hy3d_generate.py:86-230`、`scripts/hy3d_texture.py:118-284`
  - 證據：texture 有 top-level `except Exception` 並寫入 `status=error`；generate 只有 `finally`。generate 在 inference/export 例外時會保留 `status=started` 或沒有完整錯誤 metadata。
  - 改善：抽出共用 run context（tee、metadata、elapsed、cleanup）；所有 exit path 原子寫入 final metadata；輸出先寫暫存檔再 rename，避免把舊檔或半成品當成功。
  - 驗收：模擬 import、pipeline init、inference、export、metadata write 失敗，log 與 JSON 均有穩定非零狀態與錯誤摘要。

- [ ] **修正 clean-machine release/setup 流程**
  - 位置：`scripts/make_release.ps1`、`scripts/setup_hy3d_python.ps1`、`scripts/download_hy3d_models.ps1`
  - 證據：
    - `make_release.ps1` 直接 `cmake --build build`，全新 checkout 尚未 configure 時會失敗。
    - release setup 先 build `third_party\Hunyuan3D-2.1` extensions；下載腳本只抓 model repo 內容，沒有建立 source checkout。
    - 產出的 `README_RELEASE.md` 宣稱 setup 會下載 official source/model files，與腳本行為不一致。
  - 改善：release 腳本先 configure；明確下載／要求 pinned source revision；依序取得 source → 建 venv → 安裝依賴 → build extensions → 下載 models → smoke test。
  - 驗收：在乾淨暫存目錄由 release zip 開始，依 README 的四個命令可完成 package verification、setup、shape smoke、texture smoke。
  - 進度：獨立空 build 目錄 configure/build/package 已自動驗收；source/model revision、setup 順序已修正。線上下載後的 CUDA shape/texture smoke 尚未執行。
  - 本輪進度：release 測試新增共用 lifecycle、toolchain、resolved lock、manifest writer 的 package closure gate；另新增 `verify_release.ps1`，自動建立 zip、解壓到含空白/中文的路徑、拒絕不安全 manifest path、驗證 26 個 package files 的 SHA-256，並從套件外 cwd 執行 `hy3d.exe --help`。repo 內 pinned source/model 已完成真實 CUDA shape/texture smoke，custom rasterizer 與 mesh inpaint extension 亦實際重建成功。尚未從全新 release zip 重新下載約數 GB 模型並建立全新 venv，因此維持未勾選。
  - 最新實跑：修正 `%USERPROFILE%\.local\bin\uvx.exe` 探測與 `huggingface-hub==0.30.2` 必須呼叫 `huggingface-cli` 的相容性後，全新 zip 已完成 pinned source/model 下載與全新 136-package venv。下一個阻斷是 freshly cloned upstream source 未自動套用 prepared checkout 的 Windows `int64_t`/CUDA custom-rasterizer patch；Unicode 路徑另會使 PyTorch/Ninja 路徑亂碼。shape/texture 尚未由該 clean zip 執行，因此維持未勾選。

- [x] **鎖定依賴與下載來源**
  - 位置：`scripts/setup_hy3d_python.ps1`、`scripts/download_hy3d_models.ps1`
  - 證據：`pillow`、`pythreejs`、`uvx --from huggingface-hub` 未鎖版；Hugging Face download 未指定 revision；RealESRGAN 下載未驗證 checksum。
  - 改善：建立可審查 lock/constraints；model/source 固定 revision；小型二進位資產驗 SHA-256；metadata 記錄實際 revision 與依賴版本。
  - 驗收：同一 release manifest 可重建相同依賴集合，下載內容不符 hash 時立即失敗。
  - 進度：直接依賴作為 compile inputs，`requirements-win-cu124.lock.txt` 固定 Windows/Python 3.10/CUDA 12.4 的 136 個 transitive packages；`uv pip install --dry-run` 在空 venv 完整解析。setup 使用 resolved lock 並以 `write_dependency_manifest.py` 原子記錄 Python、平台、實際 packages、torch/CUDA、source/model revisions；model/source revision 與 RealESRGAN SHA-256 皆固定。

- [x] **建立 Windows CI 與負向安全測試**
  - 位置：新增 `.github/workflows/ci.yml`（或專案既有 CI 平台）
  - 改善：Windows Debug/Release build + CTest + Python converter test；加入 malformed GGUF corpus、程序參數 round-trip、PowerShell wrapper failure tests。
  - 驗收：PR 自動執行；任一測試失敗阻止合併；不依賴本機 models/CUDA 的核心測試可完整跑完。

## P2：可維護性與可攜性

- [x] **拆分大型 command dispatcher**
  - 位置：`src/hy3d_cli.cpp:78-632`、`src/main.cpp:38-310`
  - 改善：每個 subcommand 有獨立 parser/validator/handler；共用 bounded numeric parser；`main()` 只負責 dispatch 與 exit-code mapping。
  - 驗收：現有 CLI 行為與 help 相容；每個 subcommand 可獨立單元測試；移除重複 `try/stoi` 區塊。
  - 進度：每個 subcommand 都有獨立 parser 與 handler；`run_command()` 只做 switch dispatch。共用 parser 使用 full-consumption、型別 overflow、語意上限及 100M runtime-value allocation gate，拒絕 `5junk`、整數 overflow 與維度乘積爆量。

- [x] **拆分 Python pipeline orchestration**
  - 位置：`scripts/hy3d_generate.py`、`scripts/hy3d_texture.py`
  - 改善：把參數解析、環境探測、pipeline 建立、推論、輸出提交、metadata 分成小函式；共用 logging/sidecar 模組。
  - 驗收：不載入 torch/CUDA 即可測輸入驗證與錯誤 metadata；generate/texture 錯誤格式一致。
  - 進度：generate/texture 都拆成 preflight、dependency load、pipeline create、inference/output commit；preflight 與 import failure 不需載入 torch/CUDA 即可測。共用 `RunContext` 處理 tee、elapsed、cleanup、原子 metadata，metadata write failure 固定回傳 98。

- [x] **自動探測 Windows toolchain**
  - 位置：`scripts/build_hy3dpaint_windows.ps1`
  - 證據：CUDA 12.1、VS 18 路徑、MSVC 14.29、Windows SDK 10.0.26100.0、`python310.lib`、GPU arch 8.6 均硬編碼。
  - 改善：使用 `vswhere`、Python `sysconfig`、CUDA env/CMake 探測；GPU arch 允許參數化；輸出探測摘要。
  - 驗收：至少兩個已支援 VS/CUDA/Python 組合可重現建置；不支援組合給出具體診斷。
  - 本輪驗收：resolver fixture 驗證多版本選擇、明確 override、無效路徑診斷及 `CUDA_PATH` 指向 `bin` 的正規化。實機發現 CUDA 12.1 + MSVC 14.51 會使 `cudafe++` 以 `0xC0000409` 崩潰；resolver 改選最高相容的 14.29 後 custom rasterizer 與 `mesh_inpaint_processor.cp310-win_amd64.pyd` 均建置成功。SDK 10.0.26100.0、Python import library 由 `sysconfig` 取得，GPU arch 可參數化。

- [x] **補齊效能基準與 native parity gate**
  - 位置：新增 `benchmarks/`、測試 fixture 與文件
  - 改善：量測 GGUF inspect/load、tensor lookup、單 block forward；以 Python/reference fixture 比對 shape、有限值、誤差、checksum。
  - 驗收：優化 PR 附 before/after；native 功能不可只以「程式有跑完」作為完成標準。
  - 進度：20,000 tensors / 1,000,000 lookups Release benchmark 為 0.16 秒；NumPy fixture 覆蓋 attention、conditioned block、timestep、final layer（finite + `1e-5`）。6.10 GB / 752-tensor 真實 GGUF inspect 0.090 秒、block-0 28-tensor load 2.532 秒、peak RSS 197,054,464 bytes；真實 block forward 3.385 秒、4096 finite outputs、L1 17409.8。

- [x] **將警告／靜態分析套用到所有 first-party targets**
  - 位置：`CMakeLists.txt`
  - 證據：`/W4` 或 `-Wall -Wextra -Wpedantic` 目前只套用 `hy3d_core`，未涵蓋 `hy3d` 與 tests。
  - 改善：建立共用 interface target；CI 增加 clang-tidy（先採選定規則）與可選 ASan/UBSan job。
  - 驗收：main/tests 同樣啟用警告；新增程式碼不引入警告。
  - 進度：`/W4 /permissive-` 或 `-Wall -Wextra -Wpedantic` 已套用所有 targets；CMake 新增 opt-in clang-tidy、ASan/UBSan，CI 加入獨立 Linux jobs。

## 建議執行順序

1. P0 程序啟動與 checkpoint 安全。
2. GGUF 邊界檢查及其負向測試。
3. backend 路徑、wrapper exit code、sidecar 正確性。
4. loader 單次開檔、索引與 move；建立 benchmark 後再做 mmap。
5. clean-machine release、依賴鎖定、CI。
6. CLI/Python 結構拆分、toolchain 自動探測與 native parity。
