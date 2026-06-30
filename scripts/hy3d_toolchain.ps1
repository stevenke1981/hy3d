function Get-Hy3dVersion {
    param([Parameter(Mandatory = $true)][string] $Name)

    $normalized = $Name -replace '^[^0-9]*', ''
    $parsed = $null
    if ([version]::TryParse($normalized, [ref] $parsed)) {
        return $parsed
    }
    return [version]"0.0"
}

function Resolve-Hy3dVersionedRoot {
    param(
        [string] $ExplicitPath,
        [Parameter(Mandatory = $true)][string] $RequiredRelativePath,
        [Parameter(Mandatory = $true)][string] $Kind,
        [string] $BasePath
    )

    if ($ExplicitPath) {
        $required = Join-Path $ExplicitPath $RequiredRelativePath
        if (-not (Test-Path -LiteralPath $required)) {
            throw "$Kind toolchain is invalid: expected '$required'."
        }
        return (Resolve-Path -LiteralPath $ExplicitPath).Path
    }

    if (-not $BasePath -or -not (Test-Path -LiteralPath $BasePath)) {
        throw "$Kind toolchain base directory was not found: '$BasePath'."
    }

    $candidate = Get-ChildItem -LiteralPath $BasePath -Directory |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName $RequiredRelativePath) } |
        Sort-Object { Get-Hy3dVersion $_.Name } -Descending |
        Select-Object -First 1
    if (-not $candidate) {
        throw "No valid $Kind toolchain under '$BasePath' (expected '$RequiredRelativePath')."
    }
    return $candidate.FullName
}

function Resolve-Hy3dCudaRoot {
    param(
        [string] $ExplicitPath,
        [string] $BasePath = "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA"
    )

    if ($ExplicitPath) {
        $trimmed = $ExplicitPath.TrimEnd('\', '/')
        if ((Split-Path -Leaf $trimmed) -ieq "bin" -and (Test-Path -LiteralPath (Join-Path $trimmed "nvcc.exe"))) {
            $ExplicitPath = Split-Path -Parent $trimmed
        }
    }
    return Resolve-Hy3dVersionedRoot `
        -ExplicitPath $ExplicitPath `
        -BasePath $BasePath `
        -RequiredRelativePath "bin\nvcc.exe" `
        -Kind "CUDA"
}

function Resolve-Hy3dWindowsSdk {
    param(
        [string] $SdkRoot = "${env:ProgramFiles(x86)}\Windows Kits\10",
        [string] $ExplicitVersion
    )

    $includeRoot = Join-Path $SdkRoot "Include"
    if ($ExplicitVersion) {
        $include = Join-Path $includeRoot $ExplicitVersion
        if (-not (Test-Path -LiteralPath $include)) {
            throw "Windows SDK $ExplicitVersion was not found under '$includeRoot'."
        }
        return [pscustomobject]@{ Root = (Resolve-Path -LiteralPath $SdkRoot).Path; Version = $ExplicitVersion }
    }
    if (-not (Test-Path -LiteralPath $includeRoot)) {
        throw "Windows SDK include directory was not found: '$includeRoot'."
    }

    $candidate = Get-ChildItem -LiteralPath $includeRoot -Directory |
        Sort-Object { Get-Hy3dVersion $_.Name } -Descending |
        Select-Object -First 1
    if (-not $candidate) {
        throw "No Windows SDK versions were found under '$includeRoot'."
    }
    return [pscustomobject]@{ Root = (Resolve-Path -LiteralPath $SdkRoot).Path; Version = $candidate.Name }
}

function Find-Hy3dMsvcBase {
    $vswhere = Join-Path "${env:ProgramFiles(x86)}" "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $installationPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $installationPath) {
            $base = Join-Path ($installationPath | Select-Object -First 1) "VC\Tools\MSVC"
            if (Test-Path -LiteralPath $base) {
                return $base
            }
        }
    }

    $fallbacks = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"
    )
    foreach ($candidate in $fallbacks) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    throw "Visual Studio C++ toolsets were not found. Install Desktop development with C++ or pass -MsvcRoot."
}
