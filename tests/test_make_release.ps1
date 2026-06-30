param(
    [Parameter(Mandatory = $true)]
    [string] $RepoRoot
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path -LiteralPath $RepoRoot).Path
$buildDir = Join-Path $repo "build\release-script-test-build"
$releaseDir = Join-Path $repo "build\release-script-test-output"
$zipPath = "$releaseDir.zip"
$extractDir = Join-Path $repo "build\release 驗收 path"
$outsideCwd = Join-Path $repo "build\release-verifier-cwd"

function Assert-ReleaseVerificationFails {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Verifier,
        [Parameter(Mandatory = $true)]
        [string] $ReleaseRoot,
        [Parameter(Mandatory = $true)]
        [string] $Reason
    )

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
            $Verifier -ReleaseRoot $ReleaseRoot *> $null
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -eq 0) {
        throw "release verification unexpectedly accepted $Reason"
    }
}

foreach ($path in @($buildDir, $releaseDir, $zipPath, $extractDir, $outsideCwd)) {
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
        -ReleaseDir $releaseDir `
        -Zip
    if ($LASTEXITCODE -ne 0) {
        throw "make_release.ps1 failed with exit code $LASTEXITCODE"
    }

    foreach ($relative in @(
        "bin\hy3d.exe",
        "scripts\hy3d_generate.py",
        "scripts\hy3d_texture.py",
        "scripts\hy3d_run_context.py",
        "scripts\hy3d_toolchain.ps1",
        "scripts\hy3d_setup_helpers.ps1",
        "scripts\patch_hy3dpaint_windows.py",
        "scripts\verify_release.ps1",
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

    if (-not (Test-Path -LiteralPath $zipPath -PathType Leaf)) {
        throw "release zip is missing: $zipPath"
    }

    New-Item -ItemType Directory -Force -Path $extractDir, $outsideCwd | Out-Null
    Expand-Archive -LiteralPath $zipPath -DestinationPath $extractDir -Force

    Push-Location $outsideCwd
    try {
        $verifier = Join-Path $extractDir "scripts\verify_release.ps1"
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
            $verifier `
            -ReleaseRoot $extractDir
        if ($LASTEXITCODE -ne 0) {
            throw "verify_release.ps1 failed with exit code $LASTEXITCODE"
        }

        $unexpectedFile = Join-Path $extractDir "unexpected.txt"
        Set-Content -LiteralPath $unexpectedFile -Value "not in manifest" -Encoding UTF8
        Assert-ReleaseVerificationFails `
            -Verifier $verifier `
            -ReleaseRoot $extractDir `
            -Reason "an unlisted package file"
        Remove-Item -LiteralPath $unexpectedFile -Force

        Add-Content -LiteralPath (Join-Path $extractDir "README_RELEASE.md") -Value "tampered"
        Assert-ReleaseVerificationFails `
            -Verifier $verifier `
            -ReleaseRoot $extractDir `
            -Reason "a modified package file"
    } finally {
        Pop-Location
    }
} finally {
    foreach ($path in @($buildDir, $releaseDir, $zipPath, $extractDir, $outsideCwd)) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Recurse -Force
        }
    }
}
