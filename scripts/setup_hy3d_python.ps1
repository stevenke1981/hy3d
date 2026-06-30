param(
    [string] $PythonVersion = "3.10",
    [string] $VenvPath = ".\.venv-hy3d"
)

$ErrorActionPreference = "Stop"

uv venv --python $PythonVersion $VenvPath
$python = Join-Path $VenvPath "Scripts\python.exe"
$repoRoot = Split-Path -Parent $PSScriptRoot

uv pip install --python $python `
  --index-url https://download.pytorch.org/whl/cu124 `
  --requirement (Join-Path $repoRoot "requirements-torch-cu124.lock.txt")
uv pip install --python $python `
  --requirement (Join-Path $repoRoot "requirements-hy3d.lock.txt")

& .\scripts\build_hy3dpaint_windows.ps1 -PythonPath $python

& $python scripts\hy3d_generate.py `
  --image third_party\Hunyuan3D-2.1\hy3dshape\demos\demo.png `
  --out outputs\setup-dry-run.glb `
  --model-path models\Hunyuan3D-2.1 `
  --low-vram `
  --dry-run
