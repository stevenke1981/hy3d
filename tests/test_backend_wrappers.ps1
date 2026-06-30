param(
    [Parameter(Mandatory = $true)]
    [string] $RepoRoot
)

$ErrorActionPreference = "Stop"

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("hy3d-wrapper-test-" + [Guid]::NewGuid().ToString("N"))
$imagePath = Join-Path $tempRoot "input.png"
$meshPath = Join-Path $tempRoot "input.glb"
$outputPath = Join-Path $tempRoot "output.glb"

New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
Set-Content -LiteralPath $imagePath -Value "image"
Set-Content -LiteralPath $meshPath -Value "mesh"

try {
    $env:HY3D_PYTHON_BACKEND = Join-Path $tempRoot "missing-generate-backend.py"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $RepoRoot "scripts\run_python_backend.ps1") `
        -ImagePath $imagePath `
        -OutputPath $outputPath `
        -DryRun
    if ($LASTEXITCODE -ne 10) {
        throw "generate wrapper returned $LASTEXITCODE for a missing backend; expected 10"
    }

    $env:HY3D_TEXTURE_BACKEND = Join-Path $tempRoot "missing-texture-backend.py"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $RepoRoot "scripts\run_texture_backend.ps1") `
        -MeshPath $meshPath `
        -ImagePath $imagePath `
        -OutputPath $outputPath `
        -DryRun
    if ($LASTEXITCODE -ne 10) {
        throw "texture wrapper returned $LASTEXITCODE for a missing backend; expected 10"
    }
} finally {
    Remove-Item Env:\HY3D_PYTHON_BACKEND -ErrorAction SilentlyContinue
    Remove-Item Env:\HY3D_TEXTURE_BACKEND -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
