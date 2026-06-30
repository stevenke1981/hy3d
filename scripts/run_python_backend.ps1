param(
    [Parameter(Mandatory = $true)]
    [string] $ImagePath,

    [Parameter(Mandatory = $true)]
    [string] $OutputPath,

    [string] $ModelPath = "",

    [string] $Device = "cuda",

    [ValidateSet("smoke", "draft", "normal", "character-normal", "final")]
    [string] $Quality = "normal",

    [int] $Steps = 30,

    [int] $Seed = 42,

    [switch] $LowVram,

    [switch] $DryRun,

    [switch] $NoRembg
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ImagePath)) {
    Write-Error "image not found: $ImagePath"
    exit 2
}

$outputDir = Split-Path -Parent $OutputPath
if ($outputDir -and -not (Test-Path -LiteralPath $outputDir)) {
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
}

$backend = $env:HY3D_PYTHON_BACKEND
if (-not $backend) {
    $backend = Join-Path (Split-Path -Parent $PSScriptRoot) "scripts\hy3d_generate.py"
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
    $argsList = @("--image", $ImagePath, "--out", $OutputPath, "--device", $Device, "--quality", $Quality, "--steps", "$Steps", "--seed", "$Seed")
    if ($ModelPath) {
        $argsList += @("--model-path", $ModelPath)
    }
    if ($LowVram) {
        $argsList += "--low-vram"
    }
    if ($DryRun) {
        $argsList += "--dry-run"
    }
    if ($NoRembg) {
        $argsList += "--no-rembg"
    }
    & $python $backend @argsList
    exit $LASTEXITCODE
}

Write-Error "No usable Python backend script was found: $backend" -ErrorAction Continue
Write-Error "Set HY3D_PYTHON_BACKEND to an existing generate.py or keep scripts/hy3d_generate.py in place." -ErrorAction Continue
exit 10
