param(
    [Parameter(Mandatory = $true)]
    [string] $RepoRoot
)

$ErrorActionPreference = "Stop"

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("hy3d-source-download-test-" + [Guid]::NewGuid().ToString("N"))
$origin = Join-Path $tempRoot "origin"
$checkout = Join-Path $tempRoot "checkout"
$modelDir = Join-Path $tempRoot "models"
$fakeUvx = Join-Path $tempRoot "fake-uvx.cmd"
$uvxMarker = Join-Path $tempRoot "uvx-arguments.txt"

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

    @'
@echo off
echo %* > "%HY3D_TEST_UVX_MARKER%"
exit /b 0
'@ | Set-Content -LiteralPath $fakeUvx -Encoding ASCII

    $previousMarker = $env:HY3D_TEST_UVX_MARKER
    $env:HY3D_TEST_UVX_MARKER = $uvxMarker
    try {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
            (Join-Path $RepoRoot "scripts\download_hy3d_models.ps1") `
            -SkipSource `
            -ShapeOnly `
            -ModelDir $modelDir `
            -UvPath $fakeUvx
        if ($LASTEXITCODE -ne 0) {
            throw "model downloader failed with explicit uvx path: $LASTEXITCODE"
        }
    } finally {
        $env:HY3D_TEST_UVX_MARKER = $previousMarker
    }

    if (-not (Test-Path -LiteralPath $uvxMarker -PathType Leaf)) {
        throw "explicit uvx path was not invoked"
    }
    $uvxArguments = Get-Content -Raw -LiteralPath $uvxMarker
    if ($uvxArguments -notmatch [regex]::Escape("huggingface-hub==0.30.2") -or
        $uvxArguments -notmatch [regex]::Escape("huggingface-cli") -or
        $uvxArguments -notmatch [regex]::Escape("0b94677654c57bb9a6b6845cd7b704ccf551d327")) {
        throw "uvx invocation is missing the pinned tool or model revision: $uvxArguments"
    }
    if ([regex]::Matches($uvxArguments, "(?:^|\s)--include(?:\s|$)").Count -ne 1) {
        throw "legacy huggingface-cli requires one --include followed by all patterns: $uvxArguments"
    }
} finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}
