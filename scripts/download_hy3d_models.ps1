param(
    [string] $ModelDir = ".\models\Hunyuan3D-2.1",
    [string] $SourceRoot = ".\third_party\Hunyuan3D-2.1",
    [switch] $ShapeOnly,
    [switch] $SkipPaint
)

$ErrorActionPreference = "Stop"

function Invoke-HfDownload {
    param([string[]] $Arguments)

    $hf = Get-Command hf -ErrorAction SilentlyContinue
    if ($hf) {
        & hf @Arguments
        return
    }

    $uvx = Get-Command uvx -ErrorAction SilentlyContinue
    if ($uvx) {
        & uvx --from huggingface-hub hf @Arguments
        return
    }

    throw "Neither hf nor uvx is available. Install Hugging Face CLI or uv first."
}

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

Invoke-HfDownload @(
    "download",
    "tencent/Hunyuan3D-2.1",
    "--local-dir",
    $ModelDir,
    "--include",
    "README.md",
    "--include",
    "LICENSE",
    "--include",
    "Notice.txt",
    "--include",
    "demo.py",
    "--include",
    "hunyuan3d-dit-v2-1/*",
    "--include",
    "hunyuan3d-vae-v2-1/*"
)

if (-not $ShapeOnly -and -not $SkipPaint) {
    Invoke-HfDownload @(
        "download",
        "tencent/Hunyuan3D-2.1",
        "--local-dir",
        $ModelDir,
        "--include",
        "hunyuan3d-paintpbr-v2-1/*"
    )

    $ckptDir = Join-Path $SourceRoot "hy3dpaint\ckpt"
    $ckptPath = Join-Path $ckptDir "RealESRGAN_x4plus.pth"
    New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null
    if (-not (Test-Path -LiteralPath $ckptPath)) {
        Invoke-WebRequest `
            -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" `
            -OutFile $ckptPath
    }
}

Write-Host "Model dir: $ModelDir"
