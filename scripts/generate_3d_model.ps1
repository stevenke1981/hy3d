param(
    [Parameter(Mandatory = $true)]
    [string] $ImagePath,

    [string] $OutputPath = "",

    [string] $ModelPath = "",

    [ValidateSet("smoke", "draft", "normal")]
    [string] $Quality = "smoke",

    [int] $Steps = 0,

    [int] $Seed = 42,

    [ValidateSet("cuda", "cpu")]
    [string] $Device = "cuda",

    [switch] $NoRembg,

    [switch] $LowVram,

    [switch] $Texture,

    [string] $TextureOutputPath = "",

    [ValidateSet(512, 768)]
    [int] $TextureResolution = 512,

    [ValidateRange(6, 12)]
    [int] $TextureMaxViews = 6,

    [switch] $NoRemesh,

    [switch] $DryRun,

    [switch] $NoBuild
)

$ErrorActionPreference = "Stop"

function Resolve-FullPath([string] $Path) {
    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
}

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path -LiteralPath $ImagePath)) {
    throw "image not found: $ImagePath"
}

$resolvedImage = Resolve-FullPath $ImagePath

if (-not $ModelPath) {
    $ModelPath = Join-Path $root "models\Hunyuan3D-2.1"
}
$resolvedModel = Resolve-FullPath $ModelPath
if (-not (Test-Path -LiteralPath $resolvedModel)) {
    throw "model path not found: $resolvedModel"
}

if (-not $OutputPath) {
    $imageStem = [System.IO.Path]::GetFileNameWithoutExtension($resolvedImage)
    $safeStem = ($imageStem -replace '[^\p{L}\p{Nd}_-]+', '-').Trim('-')
    if (-not $safeStem) {
        $safeStem = "hy3d-model"
    }
    $OutputPath = Join-Path $root ("outputs\{0}.glb" -f $safeStem)
}
$resolvedOutput = Resolve-FullPath $OutputPath

$exe = Join-Path $root "build\Release\hy3d.exe"
if (-not (Test-Path -LiteralPath $exe)) {
    if ($NoBuild) {
        throw "hy3d.exe not found: $exe"
    }
    cmake -S $root -B (Join-Path $root "build") | Out-Host
    cmake --build (Join-Path $root "build") --config Release | Out-Host
}

$outputDir = Split-Path -Parent $resolvedOutput
if ($outputDir -and -not (Test-Path -LiteralPath $outputDir)) {
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$generateArgs = @(
    "generate",
    "--backend", "python",
    "--quality", $Quality,
    "--image", $resolvedImage,
    "--out", $resolvedOutput,
    "--model-path", $resolvedModel,
    "--device", $Device,
    "--seed", "$Seed"
)

if ($Steps -gt 0) {
    $generateArgs += @("--steps", "$Steps")
}
if ($NoRembg) {
    $generateArgs += "--no-rembg"
}
if ($LowVram) {
    $generateArgs += "--low-vram"
}
if ($DryRun) {
    $generateArgs += "--dry-run"
}

Write-Host "hy3d generate"
Write-Host "  image: $resolvedImage"
Write-Host "  output: $resolvedOutput"
Write-Host "  model: $resolvedModel"
Write-Host "  quality: $Quality"
if ($Steps -gt 0) {
    Write-Host "  steps: $Steps"
}
Write-Host "  device: $Device"

& $exe @generateArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($Texture -and -not $DryRun) {
    if (-not $TextureOutputPath) {
        $base = [System.IO.Path]::GetFileNameWithoutExtension($resolvedOutput)
        $TextureOutputPath = Join-Path (Split-Path -Parent $resolvedOutput) ("{0}-textured.glb" -f $base)
    }
    $resolvedTextureOutput = Resolve-FullPath $TextureOutputPath

    $textureArgs = @(
        "texture",
        "--backend", "python",
        "--mesh", $resolvedOutput,
        "--image", $resolvedImage,
        "--out", $resolvedTextureOutput,
        "--model-path", $resolvedModel,
        "--resolution", "$TextureResolution",
        "--max-views", "$TextureMaxViews",
        "--device", $Device
    )
    if ($NoRemesh) {
        $textureArgs += "--no-remesh"
    }

    Write-Host "hy3d texture"
    Write-Host "  mesh: $resolvedOutput"
    Write-Host "  output: $resolvedTextureOutput"
    & $exe @textureArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "done"
Write-Host "  model: $resolvedOutput"
Write-Host "  log:   $resolvedOutput.log.txt"
Write-Host "  meta:  $resolvedOutput.json"
if ($Texture -and -not $DryRun) {
    Write-Host "  textured: $resolvedTextureOutput"
    Write-Host "  texture log:  $resolvedTextureOutput.log.txt"
    Write-Host "  texture meta: $resolvedTextureOutput.json"
}
