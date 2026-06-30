param(
    [string] $ReleaseDir = ".\dist\hy3d-win-cuda",
    [string] $BuildDir = ".\build",
    [switch] $IncludeSource,
    [switch] $IncludeModels,
    [switch] $IncludeVenv,
    [switch] $Zip
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$root = [System.IO.Path]::GetFullPath($root)

function Resolve-RepoPath {
    param([string] $Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $root $Path))
}

function Get-Sha256 {
    param([string] $Path)

    $stream = [System.IO.File]::OpenRead($Path)
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = $sha256.ComputeHash($stream)
        return ([System.BitConverter]::ToString($bytes)).Replace("-", "")
    } finally {
        $sha256.Dispose()
        $stream.Dispose()
    }
}

function Get-RelativePath {
    param(
        [string] $BasePath,
        [string] $Path
    )

    $base = [System.IO.Path]::GetFullPath($BasePath).TrimEnd("\") + "\"
    $target = [System.IO.Path]::GetFullPath($Path)
    $baseUri = New-Object System.Uri($base)
    $targetUri = New-Object System.Uri($target)
    return [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString()).Replace("/", "\")
}

$release = Resolve-RepoPath $ReleaseDir
$build = Resolve-RepoPath $BuildDir

if ($release -eq $root -or $release -eq [System.IO.Path]::GetPathRoot($release)) {
    throw "unsafe release directory: $release"
}

cmake -S $root -B $build
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

cmake --build $build --config Release --parallel
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

$executable = @(
    (Join-Path $build "Release\hy3d.exe"),
    (Join-Path $build "hy3d.exe")
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $executable) {
    throw "release executable not found under build directory: $build"
}

Copy-Item -LiteralPath $executable -Destination (Join-Path $release "bin\hy3d.exe")
Copy-Item -LiteralPath (Join-Path $root "scripts\hy3d_generate.py") -Destination (Join-Path $release "scripts\hy3d_generate.py")
Copy-Item -LiteralPath (Join-Path $root "scripts\hy3d_texture.py") -Destination (Join-Path $release "scripts\hy3d_texture.py")
Copy-Item -LiteralPath (Join-Path $root "scripts\hy3d_run_context.py") -Destination (Join-Path $release "scripts\hy3d_run_context.py")
Copy-Item -LiteralPath (Join-Path $root "scripts\hy3d_toolchain.ps1") -Destination (Join-Path $release "scripts\hy3d_toolchain.ps1")
Copy-Item -LiteralPath (Join-Path $root "scripts\write_dependency_manifest.py") -Destination (Join-Path $release "scripts\write_dependency_manifest.py")
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
Copy-Item -LiteralPath (Join-Path $root "requirements-hy3d.lock.txt") -Destination (Join-Path $release "requirements-hy3d.lock.txt")
Copy-Item -LiteralPath (Join-Path $root "requirements-torch-cu124.lock.txt") -Destination (Join-Path $release "requirements-torch-cu124.lock.txt")
Copy-Item -LiteralPath (Join-Path $root "requirements-win-cu124.lock.txt") -Destination (Join-Path $release "requirements-win-cu124.lock.txt")

$sourceRoot = Join-Path $root "third_party\Hunyuan3D-2.1"
if ($IncludeSource) {
    if (-not (Test-Path -LiteralPath $sourceRoot)) {
        throw "source checkout not found for -IncludeSource: $sourceRoot"
    }
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
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\download_hy3d_models.ps1"
if errorlevel 1 exit /b %errorlevel%
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\setup_hy3d_python.ps1"
if errorlevel 1 exit /b %errorlevel%
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

Source and models are not bundled by default. `hy3d-setup.cmd` downloads the pinned official source/model files, installs Python dependencies, and builds the Windows paint extensions. To build a mostly offline package from a prepared checkout, run `scripts\make_release.ps1 -IncludeSource -IncludeModels -IncludeVenv`.
'@ | Set-Content -Path (Join-Path $release "README_RELEASE.md") -Encoding UTF8

$hashes = Get-ChildItem -LiteralPath $release -Recurse -File |
    Where-Object { $_.FullName -notlike "*.zip" } |
    ForEach-Object {
        $hash = Get-Sha256 -Path $_.FullName
        $relative = Get-RelativePath -BasePath $release -Path $_.FullName
        "$hash  $relative"
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
