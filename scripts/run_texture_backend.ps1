param(
    [Parameter(Mandatory = $true)]
    [string] $MeshPath,

    [Parameter(Mandatory = $true)]
    [string] $ImagePath,

    [Parameter(Mandatory = $true)]
    [string] $OutputPath,

    [string] $ModelPath = "",

    [string] $Device = "cuda",

    [int] $MaxViews = 6,

    [int] $Resolution = 512,

    [switch] $DryRun,

    [switch] $NoRemesh
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $MeshPath)) {
    Write-Error "mesh not found: $MeshPath"
    exit 2
}

if (-not (Test-Path -LiteralPath $ImagePath)) {
    Write-Error "image not found: $ImagePath"
    exit 3
}

$outputDir = Split-Path -Parent $OutputPath
if ($outputDir -and -not (Test-Path -LiteralPath $outputDir)) {
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
}

$backend = $env:HY3D_TEXTURE_BACKEND
if (-not $backend) {
    $backend = Join-Path (Split-Path -Parent $PSScriptRoot) "scripts\hy3d_texture.py"
}

$python = $env:HY3D_PYTHON
if (-not $python) {
    $localPython = Join-Path (Split-Path -Parent $PSScriptRoot) ".venv-hy3d\Scripts\python.exe"
    if (Test-Path -LiteralPath $localPython) {
        $python = $localPython
    } else {
        $python = "python"
    }
}

if ($backend -and (Test-Path -LiteralPath $backend)) {
    $argsList = @("--mesh", $MeshPath, "--image", $ImagePath, "--out", $OutputPath, "--device", $Device, "--max-views", "$MaxViews", "--resolution", "$Resolution")
    if ($ModelPath) {
        $argsList += @("--model-path", $ModelPath)
    }
    if ($DryRun) {
        $argsList += "--dry-run"
    }
    if ($NoRemesh) {
        $argsList += "--no-remesh"
    }
    & $python $backend @argsList
    exit $LASTEXITCODE
}

Write-Host "No usable texture backend script was found."
Write-Host "Would run official Hunyuan3D-Paint Python pipeline with:"
Write-Host "  mesh: $MeshPath"
Write-Host "  image: $ImagePath"
Write-Host "  out: $OutputPath"
if ($ModelPath) {
    Write-Host "  model-path: $ModelPath"
}
Write-Host "Set HY3D_TEXTURE_BACKEND to a local texture script or keep scripts/hy3d_texture.py in place."
exit 0
