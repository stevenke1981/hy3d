param(
    [string] $PythonPath = ".\.venv-hy3d\Scripts\python.exe",
    [string] $SourceRoot = ".\third_party\Hunyuan3D-2.1",
    [string] $SourceRevision = "82920d643c0dc2f7bfd7255f45f62d386edfe60c",
    [string] $CudaRoot,
    [string] $MsvcRoot,
    [string] $WindowsSdkVersion,
    [string] $CudaArchitecture = "8.6",
    [string] $UvPath
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "hy3d_toolchain.ps1")

$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
$SourceRoot = (Resolve-Path -LiteralPath $SourceRoot).Path
if (-not $UvPath) {
    $uvCommand = Get-Command uv -ErrorAction SilentlyContinue
    $UvPath = if ($uvCommand) { $uvCommand.Source } else { Join-Path $env:USERPROFILE ".local\bin\uv.exe" }
}
if (-not (Test-Path -LiteralPath $UvPath)) {
    throw "uv executable not found: $UvPath"
}
$cudaBase = "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA"
$cudaExplicit = if ($CudaRoot) { $CudaRoot } elseif ($env:CUDA_PATH) { $env:CUDA_PATH } else { $null }
$CudaRoot = Resolve-Hy3dCudaRoot -ExplicitPath $cudaExplicit -BasePath $cudaBase
$cudaVersion = Get-Hy3dVersion (Split-Path -Leaf $CudaRoot)
$maximumMsvcVersion = $null
if ($cudaVersion.Major -eq 12 -and $cudaVersion.Minor -le 1) {
    $maximumMsvcVersion = [version]"14.39.99999"
}
$msvcBase = if ($MsvcRoot) { $null } else { Find-Hy3dMsvcBase }
$MsvcRoot = Resolve-Hy3dVersionedRoot `
    -ExplicitPath $MsvcRoot `
    -BasePath $msvcBase `
    -RequiredRelativePath "bin\HostX64\x64\cl.exe" `
    -Kind "MSVC" `
    -MaximumVersion $maximumMsvcVersion
$sdk = Resolve-Hy3dWindowsSdk -ExplicitVersion $WindowsSdkVersion
$sdkRoot = $sdk.Root
$WindowsSdkVersion = $sdk.Version

$env:DISTUTILS_USE_SDK = "1"
$env:CUDA_HOME = $CudaRoot
$env:CUDA_PATH = $CudaRoot
$env:TORCH_CUDA_ARCH_LIST = $CudaArchitecture
$env:MAX_JOBS = "1"
$env:PATH = "$CudaRoot\bin;$MsvcRoot\bin\HostX64\x64;$sdkRoot\bin\$WindowsSdkVersion\x64;$env:PATH"
$env:INCLUDE = "$MsvcRoot\include;$sdkRoot\Include\$WindowsSdkVersion\ucrt;$sdkRoot\Include\$WindowsSdkVersion\um;$sdkRoot\Include\$WindowsSdkVersion\shared;$sdkRoot\Include\$WindowsSdkVersion\winrt;$sdkRoot\Include\$WindowsSdkVersion\cppwinrt"
$env:LIB = "$MsvcRoot\lib\x64;$sdkRoot\Lib\$WindowsSdkVersion\ucrt\x64;$sdkRoot\Lib\$WindowsSdkVersion\um\x64"

& $PythonPath (Join-Path $PSScriptRoot "patch_hy3dpaint_windows.py") `
    --source-root $SourceRoot `
    --expected-revision $SourceRevision
if ($LASTEXITCODE -ne 0) {
    throw "failed to apply the Windows custom-rasterizer source patch"
}

$customRasterizer = Join-Path $SourceRoot "hy3dpaint\custom_rasterizer"
& $UvPath pip install --python $PythonPath --no-build-isolation -e $customRasterizer
if ($LASTEXITCODE -ne 0) {
    throw "failed to build custom_rasterizer"
}

$renderer = Join-Path $SourceRoot "hy3dpaint\DifferentiableRenderer"
$configJson = & $PythonPath -c "import json, pathlib, pybind11, sys, sysconfig; print(json.dumps({'ext': sysconfig.get_config_var('EXT_SUFFIX'), 'py_include': sysconfig.get_paths()['include'], 'py_lib': str(pathlib.Path(sys.base_prefix) / 'libs'), 'pybind11_include': pybind11.get_include()}))"
$config = $configJson | ConvertFrom-Json
$pythonLibrary = & $PythonPath -c "import sysconfig; print(sysconfig.get_config_var('LDLIBRARY') or '')"
if (-not $pythonLibrary) {
    $pythonLibrary = & $PythonPath -c "import sys; print(f'python{sys.version_info.major}{sys.version_info.minor}.lib')"
}

Write-Host "Hunyuan3D toolchain:"
Write-Host "  CUDA:  $CudaRoot (arch $CudaArchitecture)"
Write-Host "  MSVC:  $MsvcRoot"
Write-Host "  SDK:   $sdkRoot ($WindowsSdkVersion)"
Write-Host "  Python: $PythonPath ($pythonLibrary)"

Push-Location $renderer
try {
    cl /utf-8 /O2 /LD /EHsc /std:c++17 mesh_inpaint_processor.cpp /I"$($config.py_include)" /I"$($config.pybind11_include)" /link /LIBPATH:"$($config.py_lib)" $pythonLibrary /OUT:"mesh_inpaint_processor$($config.ext)"
    if ($LASTEXITCODE -ne 0) {
        throw "failed to build mesh_inpaint_processor"
    }
} finally {
    Pop-Location
}
