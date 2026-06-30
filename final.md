# Hunyuan 專案檢視總結

> 檢視日期：2026-06-30
>
> CBM project：`cbm+hunyuan`
>
> 索引規模：51 files、263 symbols、592 edges（245 call edges）

## 結論

專案目前是一個可建置、核心小型測試全綠的 Windows/C++17 Hunyuan3D CLI，主要可用路徑仍是：

```text
hy3d.exe
  -> C++ CLI parse / request validation
  -> PowerShell wrapper
  -> Python official Hunyuan3D shape / paint pipeline
  -> GLB + log + JSON sidecar
```

同時存在一套逐步成形的 native GGUF/CPU runtime（GGUF inspect/load、tensor mapping、attention/DiT primitives、scheduler、mesh fixture），但 end-to-end native inference 仍明確未完成。現階段適合定位為「Python backend 的 Windows orchestration CLI + native runtime 原型」，不宜把 native 路徑描述成完整推論引擎。

兩輪改善已把非 slow 測試由 8 個擴充到 21 個，並完成原 P0、GGUF 邊界、backend 成功語意、原子輸出、單次開檔/hash index、toolchain 探測、品質分析 gates 與 clean release build。主要剩餘風險是線上 clean-machine/CUDA 驗收、transitive Python lock、每個 CLI handler 的細拆，以及真實模型 native parity/peak RSS。

## 高優先級問題處理狀態

| 等級 | 問題 | 狀態 | 本輪處理 |
|---|---|---|---|
| P0 | shell 字串式程序啟動 | 已修復 | Windows 使用 `CreateProcessW`，POSIX 使用 `fork/execvp`；真實 child argv round-trip 與 launch failure 測試通過 |
| P0 | checkpoint 不安全反序列化 | 已修復 | 預設 `weights_only=True`；只有 `--allow-unsafe-pickle` 可明確 opt-in |
| P1 | GGUF 資源與邊界驗證不足 | 已修復 | version/count/string/array/rank/offset/file-range/duplicate 檢查及小型惡意 fixture |
| P1 | tensor 載入與查找不適合大型模型 | 核心實作完成 | loader 單次開檔、metadata/runtime hash index、移除 byte copy；真實模型 benchmark 待補 |
| P1 | backend script 綁定 cwd | 已修復 | executable-relative ancestor search、cwd fallback、`HY3D_SCRIPT_ROOT` override |
| P1 | backend 缺失時假成功 | 已修復 | wrapper 固定非零退出並有 PowerShell regression test |
| P1 | Python 例外 sidecar/輸出不完整 | 已修復 | generate/texture error sidecar 對齊，final output 使用 `.partial` + atomic replace，失敗保留舊輸出 |
| P1 | release/setup 非 clean-machine 閉環 | 部分完成 | release 自行 configure/build/package，pinned source checkout 與 setup 順序已測；線上 CUDA smoke 待執行 |
| P1 | 供應鏈未完全鎖定 | 部分完成 | source/model/uvx revision 與資產 hash 已固定；新增兩份精確直接依賴 lock，transitive lock/installed manifest 待補 |

## 主要可維護性與工程缺口

- `src/hy3d_cli.cpp::parse_args()` 仍約 555 行且重複 numeric parse；`main()` 已縮為 parse + `run_command()`，下一步是每個 subcommand parser/handler。
- generate/texture 已共用 `RunContext` 的 tee、metadata、timing 與 cleanup；重型 pipeline 建立／推論仍在各自 `main()`。
- toolchain 已由 `vswhere`/versioned dirs、CUDA env、Windows SDK 與 Python `sysconfig` 探測，仍需在第二組實體工具鏈執行 extension build。
- CI 已有 Windows Debug/Release、Linux ASan/UBSan 與 clang-tidy；仍缺 fuzzer 與真實 CUDA nightly job。
- native runtime 新增 NumPy attention fixture與 tensor lookup benchmark，仍缺官方模型 fixture、load/peak RSS 與 block-forward benchmark。
- CMake warning flags 已套用所有 first-party targets；clang-tidy/sanitizer 尚未加入。

## 正向觀察

- C++ 核心、CLI、backend、GGUF、math、runtime、model loader 已分成獨立檔案與 target。
- request validation 在啟動重型 backend 前執行，dry-run 適合快速檢查基本參數。
- 測試同時涵蓋 Debug/Release 可建置，converter 也已納入 CTest。
- GGUF tensor dimension product 已有 `uint64_t` overflow 防護，這是可延伸的正確方向。
- texture Python path 已有 top-level exception-to-sidecar 處理，可作為 generate 共用化的基準。
- release manifest 使用 package-relative path 與內建 SHA-256；clean build/package 已自動測試。

## 已執行與後續路線

### 第一階段：先讓信任邊界可靠

1. `[完成]` `CreateProcessW`/`execvp` 取代 `std::system()`。
2. `[完成]` converter 預設 `weights_only=True`。
3. `[完成]` GGUF count/offset/length checked validation。
4. `[完成]` process round-trip 與 malformed GGUF regression tests。

### 第二階段：消除錯誤成功與部署脆弱性

1. `[完成]` script path 改以 executable/package root 定位。
2. `[完成]` wrapper 缺失 backend 非零退出。
3. `[完成]` generate/texture error lifecycle 與 temp + atomic rename。
4. `[部分]` 乾淨目錄 release configure/build/package 已測；線上 setup/download/CUDA smoke 待測。

### 第三階段：大型模型效能

1. `[完成]` tensor metadata hash index。
2. `[完成]` 模型載入共用單一 file handle，直接填入 runtime bytes。
3. `[完成]` runtime tensor name index。
4. `[待辦]` 真實模型時間/peak RSS benchmark 後再決定是否 memory-map。

### 第四階段：可維護性與 native 完成度

1. 拆分 CLI handlers 與 Python orchestration。
2. 自動探測 toolchain，鎖定依賴/model/source revision。
3. 建 Windows CI、nightly release、CUDA/model parity jobs。
4. 以 reference parity 和輸出模型有效性作為 native 功能的 Definition of Done。

## 本次實際驗證

```powershell
cmake -S . -B build
cmake --build build --config Debug --parallel
ctest --test-dir build -C Debug -LE slow --output-on-failure --timeout 60
cmake --build build --config Release --parallel
ctest --test-dir build -C Release -LE slow --output-on-failure --timeout 60
ctest --test-dir build -C Release -R '^make_release$' --output-on-failure
```

結果：

- Debug build：通過。
- Debug 非 slow CTest：21/21 通過（5.28 秒）。
- Release build：通過。
- Release 非 slow CTest：21/21 通過（3.27 秒）。
- Clean release configure/build/package：1/1 通過（18.10 秒）。
- Release tensor lookup benchmark：1,000,000 次查找 0.15 秒。
- Python `py_compile`、toolchain 實機探測與 `git diff --check` 通過。

未執行：

- 真實 CUDA shape/texture 生成。
- 真實模型 native parity。
- release zip 的線上 source/model 下載後 CUDA smoke。
- 惡意 pickle payload 的實際執行（預設安全模式與明確 opt-in 已由參數測試驗證）。

詳細執行清單見 `todos.md`，測試矩陣與驗收 gate 見 `test.md`。

## 工作區注意事項

`.opencode/memory.db*` 是本機執行期資料，刻意排除於本次提交。其餘本輪原始碼、測試、CI、腳本與三份審查文件將在驗收後提交。
