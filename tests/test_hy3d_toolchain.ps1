param([Parameter(Mandatory = $true)][string] $RepoRoot)

$ErrorActionPreference = "Stop"
. (Join-Path $RepoRoot "scripts\hy3d_toolchain.ps1")

$buildScript = Get-Content -Raw -LiteralPath (Join-Path $RepoRoot "scripts\build_hy3dpaint_windows.ps1")
if ($buildScript -notmatch '\[string\]\s*\$UvPath' -or
    $buildScript -notmatch '&\s+\$UvPath\s+pip') {
    throw "extension build does not accept the resolved uv executable"
}
if ($buildScript -notmatch [regex]::Escape("patch_hy3dpaint_windows.py") -or
    $buildScript -notmatch '\[string\]\s*\$SourceRevision') {
    throw "extension build does not apply the revision-guarded Windows source patch"
}
if (-not (Test-Hy3dAsciiPath "D:\hy3d release") -or
    (Test-Hy3dAsciiPath "D:\hy3d 驗收")) {
    throw "ASCII path detection is incorrect"
}
$common = Get-Hy3dCommonPath `
    "D:\hy3d 驗收\.venv-hy3d\Scripts\python.exe" `
    "D:\hy3d 驗收\third_party\Hunyuan3D-2.1"
if ($common -ne "D:\hy3d 驗收") {
    throw "common Unicode release root is incorrect: $common"
}
$mapped = ConvertTo-Hy3dMappedPath `
    -Path "D:\hy3d 驗收\.venv-hy3d\Scripts\python.exe" `
    -SourceRoot $common `
    -MappedRoot "Z:\"
if ($mapped -ne "Z:\.venv-hy3d\Scripts\python.exe") {
    throw "mapped Python path is incorrect: $mapped"
}
if ($buildScript -notmatch [regex]::Escape("subst.exe") -or
    $buildScript -notmatch '\[switch\]\s*\$SkipUnicodeRemap') {
    throw "extension build does not remap Unicode roots to an ASCII drive"
}
if ($buildScript -notmatch [regex]::Escape("--reinstall") -or
    $buildScript -match '--no-build-isolation\s+-e\s+') {
    throw "custom rasterizer must be installed non-editably for temporary drive mappings"
}

$root = Join-Path ([System.IO.Path]::GetTempPath()) ("hy3d-toolchain-" + [guid]::NewGuid())
try {
    $cudaBase = Join-Path $root "CUDA"
    $msvcBase = Join-Path $root "MSVC"
    $sdkBase = Join-Path $root "SDK"

    foreach ($path in @(
        (Join-Path $cudaBase "v12.1\bin"),
        (Join-Path $cudaBase "v13.2\bin"),
        (Join-Path $msvcBase "14.29.30133\bin\HostX64\x64"),
        (Join-Path $msvcBase "14.44.35207\bin\HostX64\x64"),
        (Join-Path $sdkBase "Include\10.0.19041.0"),
        (Join-Path $sdkBase "Include\10.0.26100.0")
    )) {
        New-Item -ItemType Directory -Force -Path $path | Out-Null
    }
    New-Item -ItemType File -Force -Path (Join-Path $cudaBase "v12.1\bin\nvcc.exe") | Out-Null
    New-Item -ItemType File -Force -Path (Join-Path $cudaBase "v13.2\bin\nvcc.exe") | Out-Null
    New-Item -ItemType File -Force -Path (Join-Path $msvcBase "14.29.30133\bin\HostX64\x64\cl.exe") | Out-Null
    New-Item -ItemType File -Force -Path (Join-Path $msvcBase "14.44.35207\bin\HostX64\x64\cl.exe") | Out-Null

    $cuda = Resolve-Hy3dVersionedRoot -BasePath $cudaBase -RequiredRelativePath "bin\nvcc.exe" -Kind "CUDA"
    $msvc = Resolve-Hy3dVersionedRoot -BasePath $msvcBase -RequiredRelativePath "bin\HostX64\x64\cl.exe" -Kind "MSVC"
    $compatibleMsvc = Resolve-Hy3dVersionedRoot `
        -BasePath $msvcBase `
        -RequiredRelativePath "bin\HostX64\x64\cl.exe" `
        -Kind "MSVC" `
        -MaximumVersion ([version]"14.39.99999")
    $sdk = Resolve-Hy3dWindowsSdk -SdkRoot $sdkBase

    if ((Split-Path -Leaf $cuda) -ne "v13.2") { throw "did not choose latest CUDA: $cuda" }
    if ((Split-Path -Leaf $msvc) -ne "14.44.35207") { throw "did not choose latest MSVC: $msvc" }
    if ((Split-Path -Leaf $compatibleMsvc) -ne "14.29.30133") {
        throw "did not choose CUDA-compatible MSVC: $compatibleMsvc"
    }
    if ($sdk.Version -ne "10.0.26100.0") { throw "did not choose latest SDK: $($sdk.Version)" }

    $explicit = Resolve-Hy3dVersionedRoot -ExplicitPath (Join-Path $cudaBase "v12.1") -RequiredRelativePath "bin\nvcc.exe" -Kind "CUDA"
    if ((Split-Path -Leaf $explicit) -ne "v12.1") { throw "explicit CUDA path was ignored" }
    $fromBin = Resolve-Hy3dCudaRoot -ExplicitPath (Join-Path $cudaBase "v12.1\bin") -BasePath $cudaBase
    if ((Split-Path -Leaf $fromBin) -ne "v12.1") { throw "CUDA_PATH ending in bin was not normalized" }

    $failed = $false
    try {
        Resolve-Hy3dVersionedRoot -ExplicitPath (Join-Path $root "missing") -RequiredRelativePath "bin\nvcc.exe" -Kind "CUDA"
    } catch {
        $failed = $_.Exception.Message -match "CUDA"
    }
    if (-not $failed) { throw "invalid explicit path did not produce a specific diagnostic" }
} finally {
    Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "toolchain resolver tests passed"
