param(
    [string] $ModelDir = ".\models\Hunyuan3D-2.1",
    [string] $SourceRoot = ".\third_party\Hunyuan3D-2.1",
    [string] $SourceRepo = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git",
    [string] $SourceRevision = "82920d643c0dc2f7bfd7255f45f62d386edfe60c",
    [string] $ModelRevision = "0b94677654c57bb9a6b6845cd7b704ccf551d327",
    [string] $UvPath,
    [switch] $SkipSource,
    [switch] $SkipModels,
    [switch] $ShapeOnly,
    [switch] $SkipPaint
)

$ErrorActionPreference = "Stop"

function Invoke-HfDownload {
    param([string[]] $Arguments)

    $uvxExecutable = $UvPath
    if ($uvxExecutable) {
        if (-not (Test-Path -LiteralPath $uvxExecutable -PathType Leaf)) {
            throw "specified uvx executable does not exist: $uvxExecutable"
        }
    } else {
        $uvxCommand = Get-Command uvx -ErrorAction SilentlyContinue
        if ($uvxCommand) {
            $uvxExecutable = $uvxCommand.Source
        } else {
            $userLocalUvx = Join-Path $env:USERPROFILE ".local\bin\uvx.exe"
            if (Test-Path -LiteralPath $userLocalUvx -PathType Leaf) {
                $uvxExecutable = $userLocalUvx
            }
        }
    }

    if ($uvxExecutable) {
        & $uvxExecutable --from "huggingface-hub==0.30.2" huggingface-cli @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Hugging Face download failed with exit code $LASTEXITCODE"
        }
        return
    }

    $hf = Get-Command hf -ErrorAction SilentlyContinue
    if ($hf) {
        & hf @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Hugging Face download failed with exit code $LASTEXITCODE"
        }
        return
    }

    throw "Neither hf nor uvx is available. Install Hugging Face CLI or uv first."
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

if (-not $SkipSource) {
    $sourceParent = Split-Path -Parent $SourceRoot
    if ($sourceParent) {
        New-Item -ItemType Directory -Force -Path $sourceParent | Out-Null
    }

    if (-not (Test-Path -LiteralPath $SourceRoot)) {
        git clone --filter=blob:none --no-checkout $SourceRepo $SourceRoot
        if ($LASTEXITCODE -ne 0) {
            throw "failed to clone source repository: $SourceRepo"
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $SourceRoot ".git"))) {
        throw "source path exists but is not a git checkout: $SourceRoot"
    }

    git -C $SourceRoot cat-file -e "$SourceRevision^{commit}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        git -C $SourceRoot fetch --depth 1 origin $SourceRevision
        if ($LASTEXITCODE -ne 0) {
            throw "failed to fetch source revision: $SourceRevision"
        }
    }
    git -C $SourceRoot checkout --detach $SourceRevision
    if ($LASTEXITCODE -ne 0) {
        throw "failed to checkout source revision: $SourceRevision"
    }
    Write-Host "Source revision: $SourceRevision"
}

if ($SkipModels) {
    Write-Host "Source dir: $SourceRoot"
    return
}

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

Invoke-HfDownload @(
    "download",
    "tencent/Hunyuan3D-2.1",
    "--revision",
    $ModelRevision,
    "--local-dir",
    $ModelDir,
    "--include",
    "README.md",
    "LICENSE",
    "Notice.txt",
    "demo.py",
    "hunyuan3d-dit-v2-1/*",
    "hunyuan3d-vae-v2-1/*"
)

if (-not $ShapeOnly -and -not $SkipPaint) {
    Invoke-HfDownload @(
        "download",
        "tencent/Hunyuan3D-2.1",
        "--revision",
        $ModelRevision,
        "--local-dir",
        $ModelDir,
        "--include",
        "hunyuan3d-paintpbr-v2-1/*"
    )

    $ckptDir = Join-Path $SourceRoot "hy3dpaint\ckpt"
    $ckptPath = Join-Path $ckptDir "RealESRGAN_x4plus.pth"
    $expectedHash = "4FA0D38905F75AC06EB49A7951B426670021BE3018265FD191D2125DF9D682F1"
    New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null
    if (-not (Test-Path -LiteralPath $ckptPath)) {
        $tempPath = "$ckptPath.download"
        Invoke-WebRequest `
            -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" `
            -OutFile $tempPath
        $actualHash = Get-Sha256 -Path $tempPath
        if ($actualHash -ne $expectedHash) {
            Remove-Item -LiteralPath $tempPath -Force
            throw "RealESRGAN checkpoint SHA-256 mismatch: $actualHash"
        }
        Move-Item -LiteralPath $tempPath -Destination $ckptPath
    } else {
        $actualHash = Get-Sha256 -Path $ckptPath
        if ($actualHash -ne $expectedHash) {
            throw "existing RealESRGAN checkpoint SHA-256 mismatch: $actualHash"
        }
    }
}

Write-Host "Model revision: $ModelRevision"
Write-Host "Model dir: $ModelDir"
