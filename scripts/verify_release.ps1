param(
    [string] $ReleaseRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string] $Path)

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

$release = (Resolve-Path -LiteralPath $ReleaseRoot).Path.TrimEnd("\")
$releasePrefix = $release + "\"

$requiredFiles = @(
    "bin\hy3d.exe",
    "scripts\hy3d_generate.py",
    "scripts\hy3d_texture.py",
    "scripts\hy3d_run_context.py",
    "scripts\hy3d_toolchain.ps1",
    "scripts\verify_release.ps1",
    "scripts\write_dependency_manifest.py",
    "scripts\run_python_backend.ps1",
    "scripts\run_texture_backend.ps1",
    "scripts\setup_hy3d_python.ps1",
    "scripts\build_hy3dpaint_windows.ps1",
    "scripts\download_hy3d_models.ps1",
    "requirements-win-cu124.lock.txt",
    "README_RELEASE.md",
    "SHA256SUMS.txt",
    "hy3d.cmd",
    "hy3d-setup.cmd",
    "hy3d-generate-smoke.cmd",
    "hy3d-texture-smoke.cmd"
)

foreach ($relative in $requiredFiles) {
    $path = Join-Path $release $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "release package is missing required file: $relative"
    }
}

$manifestPath = Join-Path $release "SHA256SUMS.txt"
$manifestEntries = @{}
foreach ($line in Get-Content -LiteralPath $manifestPath) {
    if (-not $line) {
        continue
    }
    if ($line -notmatch "^([0-9A-Fa-f]{64})  (.+)$") {
        throw "invalid SHA256SUMS.txt line: $line"
    }

    $expected = $Matches[1].ToUpperInvariant()
    $relative = $Matches[2]
    if ([System.IO.Path]::IsPathRooted($relative)) {
        throw "manifest path must be package-relative: $relative"
    }

    $path = [System.IO.Path]::GetFullPath((Join-Path $release $relative))
    if (-not $path.StartsWith($releasePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "manifest path escapes release root: $relative"
    }
    if ($manifestEntries.ContainsKey($path)) {
        throw "duplicate manifest path: $relative"
    }
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "manifest file is missing: $relative"
    }

    $actual = Get-Sha256 -Path $path
    if ($actual -ne $expected) {
        throw "SHA-256 mismatch for ${relative}: expected $expected, got $actual"
    }
    $manifestEntries[$path] = $relative
}

$packageFiles = Get-ChildItem -LiteralPath $release -Recurse -File |
    Where-Object { $_.FullName -ne $manifestPath }
foreach ($file in $packageFiles) {
    if (-not $manifestEntries.ContainsKey($file.FullName)) {
        throw "release file is not covered by SHA256SUMS.txt: $($file.FullName)"
    }
}
if ($manifestEntries.Count -ne $packageFiles.Count) {
    throw "manifest entry count does not match package file count"
}

$probeDir = Join-Path ([System.IO.Path]::GetTempPath()) ("hy3d-release-verify-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $probeDir | Out-Null
try {
    Push-Location $probeDir
    try {
        $helpOutput = & (Join-Path $release "bin\hy3d.exe") --help 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "bin\hy3d.exe --help failed with exit code $LASTEXITCODE"
        }
        if (-not ($helpOutput -join "`n").Trim()) {
            throw "bin\hy3d.exe --help returned no output"
        }
    } finally {
        Pop-Location
    }
} finally {
    if (Test-Path -LiteralPath $probeDir) {
        Remove-Item -LiteralPath $probeDir -Recurse -Force
    }
}

Write-Host "Release verification passed: $($manifestEntries.Count) hashed files and executable smoke."
