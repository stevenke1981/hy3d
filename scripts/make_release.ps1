param(
    [string] $ReleaseDir = ".\dist\hy3d-win-cuda",
    [switch] $IncludeModels,
    [switch] $IncludeVenv,
    [switch] $Zip
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$release = Join-Path $root $ReleaseDir

cmake --build (Join-Path $root "build") --config Release
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (Test-Path -LiteralPath $release) {
    Remove-Item -LiteralPath $release -Recurse -Force
}

New-Item -ItemType Directory -Force -Path `
    (Join-Path $release "bin"), `
    (Join-Path $release "scripts"), `
    (Join-Path $release "examples"), `
    (Join-Path $release "tools"), `
    (Join-Path $release "third_party") | Out-Null

Copy-Item -LiteralPath (Join-Path $root "build\Release\hy3d.exe") -Destination (Join-Path $release "bin\hy3d.exe")
Copy-Item -LiteralPath (Join-Path $root "scripts\hy3d_generate.py") -Destination (Join-Path $release "scripts\hy3d_generate.py")
Copy-Item -LiteralPath (Join-Path $root "scripts\hy3d_texture.py") -Destination (Join-Path $release "scripts\hy3d_texture.py")
Copy-Item -LiteralPath (Join-Path $root "scripts\run_python_backend.ps1") -Destination (Join-Path $release "scripts\run_python_backend.ps1")
Copy-Item -LiteralPath (Join-Path $root "scripts\run_texture_backend.ps1") -Destination (Join-Path $release "scripts\run_texture_backend.ps1")
Copy-Item -LiteralPath (Join-Path $root "scripts\setup_hy3d_python.ps1") -Destination (Join-Path $release "scripts\setup_hy3d_python.ps1")
Copy-Item -LiteralPath (Join-Path $root "scripts\build_hy3dpaint_windows.ps1") -Destination (Join-Path $release "scripts\build_hy3dpaint_windows.ps1")
Copy-Item -LiteralPath (Join-Path $root "scripts\download_hy3d_models.ps1") -Destination (Join-Path $release "scripts\download_hy3d_models.ps1")
Copy-Item -LiteralPath (Join-Path $root "tools\convert-hy3d-shape-to-gguf.py") -Destination (Join-Path $release "tools\convert-hy3d-shape-to-gguf.py")
Copy-Item -LiteralPath (Join-Path $root "examples\generate.ps1") -Destination (Join-Path $release "examples\generate.ps1")
Copy-Item -LiteralPath (Join-Path $root "examples\texture.ps1") -Destination (Join-Path $release "examples\texture.ps1")
Copy-Item -LiteralPath (Join-Path $root "README.md") -Destination (Join-Path $release "README.md")
Copy-Item -LiteralPath (Join-Path $root "spec.md") -Destination (Join-Path $release "spec.md")
Copy-Item -LiteralPath (Join-Path $root "plan.md") -Destination (Join-Path $release "plan.md")

$sourceRoot = Join-Path $root "third_party\Hunyuan3D-2.1"
if (Test-Path -LiteralPath $sourceRoot) {
    $targetSource = Join-Path $release "third_party\Hunyuan3D-2.1"
    New-Item -ItemType Directory -Force -Path $targetSource | Out-Null
    foreach ($name in @("hy3dshape", "hy3dpaint", "torchvision_fix.py", "README.md", "LICENSE", "Notice.txt", "demo.py")) {
        $src = Join-Path $sourceRoot $name
        if (Test-Path -LiteralPath $src) {
            Copy-Item -LiteralPath $src -Destination $targetSource -Recurse -Force
        }
    }
}

if ($IncludeModels) {
    $modelRoot = Join-Path $root "models"
    if (Test-Path -LiteralPath $modelRoot) {
        Copy-Item -LiteralPath $modelRoot -Destination (Join-Path $release "models") -Recurse -Force
    }
}

if ($IncludeVenv) {
    $venvRoot = Join-Path $root ".venv-hy3d"
    if (Test-Path -LiteralPath $venvRoot) {
        Copy-Item -LiteralPath $venvRoot -Destination (Join-Path $release ".venv-hy3d") -Recurse -Force
    }
}

@'
@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\setup_hy3d_python.ps1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\download_hy3d_models.ps1"
endlocal
'@ | Set-Content -Path (Join-Path $release "hy3d-setup.cmd") -Encoding ASCII

@'
@echo off
setlocal
cd /d "%~dp0"
".\bin\hy3d.exe" %*
endlocal
'@ | Set-Content -Path (Join-Path $release "hy3d.cmd") -Encoding ASCII

@'
@echo off
setlocal
cd /d "%~dp0"
".\bin\hy3d.exe" generate --backend python --quality smoke --image ".\third_party\Hunyuan3D-2.1\hy3dshape\demos\demo.png" --out ".\outputs\hy3d-release-smoke.glb" --model-path ".\models\Hunyuan3D-2.1" --device cuda --seed 42
endlocal
'@ | Set-Content -Path (Join-Path $release "hy3d-generate-smoke.cmd") -Encoding ASCII

@'
@echo off
setlocal
cd /d "%~dp0"
".\bin\hy3d.exe" texture --backend python --mesh ".\outputs\hy3d-release-smoke.glb" --image ".\third_party\Hunyuan3D-2.1\hy3dshape\demos\demo.png" --out ".\outputs\hy3d-release-textured.glb" --model-path ".\models\Hunyuan3D-2.1" --resolution 512 --max-views 6 --no-remesh --device cuda
endlocal
'@ | Set-Content -Path (Join-Path $release "hy3d-texture-smoke.cmd") -Encoding ASCII

@'
# Hunyuan3D Windows CUDA Release

Run in order:

1. `hy3d-setup.cmd`
2. `hy3d-generate-smoke.cmd`
3. `hy3d-texture-smoke.cmd`

The generated files and sidecar logs are written under `outputs\`.

Advanced usage:

```powershell
.\hy3d.cmd --help
.\hy3d.cmd generate --backend python --quality smoke --image .\input.png --out .\outputs\shape.glb --model-path .\models\Hunyuan3D-2.1 --device cuda
.\hy3d.cmd texture --backend python --mesh .\outputs\shape.glb --image .\input.png --out .\outputs\textured.glb --model-path .\models\Hunyuan3D-2.1 --resolution 512 --max-views 6 --no-remesh --device cuda
```

Models are not bundled by default. `hy3d-setup.cmd` installs Python dependencies, builds the Windows paint extensions, and downloads the official shape/PBR model files. To build a mostly offline package from a prepared checkout, run `scripts\make_release.ps1 -IncludeModels -IncludeVenv`.
'@ | Set-Content -Path (Join-Path $release "README_RELEASE.md") -Encoding UTF8

$hashes = Get-ChildItem -LiteralPath $release -Recurse -File |
    Where-Object { $_.FullName -notlike "*.zip" } |
    ForEach-Object {
        $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
        $relative = Resolve-Path -LiteralPath $_.FullName -Relative
        "$($hash.Hash)  $relative"
    }
$hashes | Set-Content -Path (Join-Path $release "SHA256SUMS.txt") -Encoding ASCII

if ($Zip) {
    $zipPath = "$release.zip"
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Compress-Archive -Path (Join-Path $release "*") -DestinationPath $zipPath -Force
    Write-Host "Release zip: $zipPath"
}

Write-Host "Release dir: $release"
