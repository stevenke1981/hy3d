param(
    [string] $PythonVersion = "3.10",
    [string] $VenvPath = ".\.venv-hy3d",
    [string] $SourceRevision = "82920d643c0dc2f7bfd7255f45f62d386edfe60c",
    [string] $ModelRevision = "0b94677654c57bb9a6b6845cd7b704ccf551d327",
    [string] $ManifestPath = ".\hy3d-dependencies.json"
)

$ErrorActionPreference = "Stop"

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCommand) {
    $uvExecutable = $uvCommand.Source
} else {
    $uvExecutable = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (-not (Test-Path -LiteralPath $uvExecutable)) {
        throw "uv was not found on PATH or at '$uvExecutable'. Install uv before running setup."
    }
}

& $uvExecutable venv --python $PythonVersion $VenvPath
if ($LASTEXITCODE -ne 0) {
    throw "failed to create Python virtual environment"
}
$python = Join-Path $VenvPath "Scripts\python.exe"
$repoRoot = Split-Path -Parent $PSScriptRoot

& $uvExecutable pip install --python $python `
  --index-url https://pypi.org/simple `
  --extra-index-url https://download.pytorch.org/whl/cu124 `
  --index-strategy unsafe-best-match `
  --requirement (Join-Path $repoRoot "requirements-win-cu124.lock.txt")
if ($LASTEXITCODE -ne 0) {
    throw "failed to install resolved dependencies"
}

$sourceRoot = Join-Path $repoRoot "third_party\Hunyuan3D-2.1"
$modelRoot = Join-Path $repoRoot "models\Hunyuan3D-2.1"
& (Join-Path $PSScriptRoot "build_hy3dpaint_windows.ps1") `
  -PythonPath $python `
  -SourceRoot $sourceRoot `
  -SourceRevision $SourceRevision `
  -UvPath $uvExecutable

& $python (Join-Path $PSScriptRoot "hy3d_generate.py") `
  --image (Join-Path $sourceRoot "hy3dshape\demos\demo.png") `
  --out (Join-Path $repoRoot "outputs\setup-dry-run.glb") `
  --model-path $modelRoot `
  --low-vram `
  --dry-run
if ($LASTEXITCODE -ne 0) {
    throw "setup dry-run failed"
}

& $python (Join-Path $PSScriptRoot "write_dependency_manifest.py") `
  --out $ManifestPath `
  --source-revision $SourceRevision `
  --model-revision $ModelRevision
if ($LASTEXITCODE -ne 0) {
    throw "failed to write dependency manifest"
}
