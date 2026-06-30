param(
    [Parameter(Mandatory = $true)]
    [string] $RepoRoot
)

$ErrorActionPreference = "Stop"

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("hy3d-source-download-test-" + [Guid]::NewGuid().ToString("N"))
$origin = Join-Path $tempRoot "origin"
$checkout = Join-Path $tempRoot "checkout"

New-Item -ItemType Directory -Force -Path $origin | Out-Null
try {
    git -C $origin init --quiet
    Set-Content -LiteralPath (Join-Path $origin "source.txt") -Value "pinned source"
    git -C $origin add source.txt
    git -C $origin -c user.name=hy3d-test -c user.email=hy3d@example.invalid commit --quiet -m "fixture"
    if ($LASTEXITCODE -ne 0) {
        throw "failed to create source fixture"
    }
    $revision = (git -C $origin rev-parse HEAD).Trim()

    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $RepoRoot "scripts\download_hy3d_models.ps1") `
        -SourceRepo $origin `
        -SourceRevision $revision `
        -SourceRoot $checkout `
        -SkipModels
    if ($LASTEXITCODE -ne 0) {
        throw "source downloader failed with exit code $LASTEXITCODE"
    }

    $actual = (git -C $checkout rev-parse HEAD).Trim()
    if ($actual -ne $revision) {
        throw "source checkout revision mismatch: expected $revision, got $actual"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $checkout "source.txt"))) {
        throw "source checkout is missing fixture content"
    }
} finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}
