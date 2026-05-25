param(
    [string] $PythonPath = ".\.venv-hy3d\Scripts\python.exe",
    [string] $SourceRoot = ".\third_party\Hunyuan3D-2.1",
    [string] $CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
    [string] $MsvcRoot = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.29.30133",
    [string] $WindowsSdkVersion = "10.0.26100.0"
)

$ErrorActionPreference = "Stop"

$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
$SourceRoot = (Resolve-Path -LiteralPath $SourceRoot).Path
$sdkRoot = "C:\Program Files (x86)\Windows Kits\10"

if (-not (Test-Path -LiteralPath $CudaRoot)) {
    throw "CUDA toolkit not found: $CudaRoot"
}
if (-not (Test-Path -LiteralPath $MsvcRoot)) {
    throw "MSVC toolset not found: $MsvcRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path $sdkRoot "Include\$WindowsSdkVersion"))) {
    throw "Windows SDK not found: $sdkRoot\Include\$WindowsSdkVersion"
}

$env:DISTUTILS_USE_SDK = "1"
$env:CUDA_HOME = $CudaRoot
$env:CUDA_PATH = $CudaRoot
$env:TORCH_CUDA_ARCH_LIST = "8.6"
$env:MAX_JOBS = "1"
$env:PATH = "$CudaRoot\bin;$MsvcRoot\bin\HostX64\x64;$sdkRoot\bin\$WindowsSdkVersion\x64;$env:PATH"
$env:INCLUDE = "$MsvcRoot\include;$sdkRoot\Include\$WindowsSdkVersion\ucrt;$sdkRoot\Include\$WindowsSdkVersion\um;$sdkRoot\Include\$WindowsSdkVersion\shared;$sdkRoot\Include\$WindowsSdkVersion\winrt;$sdkRoot\Include\$WindowsSdkVersion\cppwinrt"
$env:LIB = "$MsvcRoot\lib\x64;$sdkRoot\Lib\$WindowsSdkVersion\ucrt\x64;$sdkRoot\Lib\$WindowsSdkVersion\um\x64"

$customRasterizer = Join-Path $SourceRoot "hy3dpaint\custom_rasterizer"
uv pip install --python $PythonPath --no-build-isolation -e $customRasterizer

$renderer = Join-Path $SourceRoot "hy3dpaint\DifferentiableRenderer"
$configJson = & $PythonPath -c "import json, pathlib, pybind11, sys, sysconfig; print(json.dumps({'ext': sysconfig.get_config_var('EXT_SUFFIX'), 'py_include': sysconfig.get_paths()['include'], 'py_lib': str(pathlib.Path(sys.base_prefix) / 'libs'), 'pybind11_include': pybind11.get_include()}))"
$config = $configJson | ConvertFrom-Json

Push-Location $renderer
try {
    cl /utf-8 /O2 /LD /EHsc /std:c++17 mesh_inpaint_processor.cpp /I"$($config.py_include)" /I"$($config.pybind11_include)" /link /LIBPATH:"$($config.py_lib)" python310.lib /OUT:"mesh_inpaint_processor$($config.ext)"
    if ($LASTEXITCODE -ne 0) {
        throw "failed to build mesh_inpaint_processor"
    }
} finally {
    Pop-Location
}
