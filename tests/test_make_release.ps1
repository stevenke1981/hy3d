param(
    [Parameter(Mandatory = $true)]
    [string] $RepoRoot
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path -LiteralPath $RepoRoot).Path
$buildDir = Join-Path $repo "build\release-script-test-build"
$releaseDir = Join-Path $repo "build\release-script-test-output"

foreach ($path in @($buildDir, $releaseDir)) {
    $full = [System.IO.Path]::GetFullPath($path)
    if (-not $full.StartsWith($repo, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "refusing to use test path outside repository: $full"
    }
    if (Test-Path -LiteralPath $full) {
        Remove-Item -LiteralPath $full -Recurse -Force
    }
}

try {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $repo "scripts\make_release.ps1") `
        -BuildDir $buildDir `
        -ReleaseDir $releaseDir
    if ($LASTEXITCODE -ne 0) {
        throw "make_release.ps1 failed with exit code $LASTEXITCODE"
    }

    foreach ($relative in @(
        "bin\hy3d.exe",
        "scripts\hy3d_generate.py",
        "scripts\hy3d_texture.py",
        "scripts\hy3d_run_context.py",
        "scripts\hy3d_toolchain.ps1",
        "scripts\write_dependency_manifest.py",
        "scripts\run_python_backend.ps1",
        "requirements-win-cu124.lock.txt",
        "README_RELEASE.md",
        "SHA256SUMS.txt"
    )) {
        if (-not (Test-Path -LiteralPath (Join-Path $releaseDir $relative))) {
            throw "release output is missing: $relative"
        }
    }
} finally {
    foreach ($path in @($buildDir, $releaseDir)) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Recurse -Force
        }
    }
}
