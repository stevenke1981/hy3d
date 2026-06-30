function Test-Hy3dAsciiPath {
    param([Parameter(Mandatory = $true)][string] $Path)

    foreach ($character in $Path.ToCharArray()) {
        if ([int] $character -gt 127) {
            return $false
        }
    }
    return $true
}

function Get-Hy3dCommonPath {
    param(
        [Parameter(Mandatory = $true, Position = 0)][string] $PathA,
        [Parameter(Mandatory = $true, Position = 1)][string] $PathB
    )

    $fullA = [System.IO.Path]::GetFullPath($PathA)
    $fullB = [System.IO.Path]::GetFullPath($PathB)
    $rootA = [System.IO.Path]::GetPathRoot($fullA)
    $rootB = [System.IO.Path]::GetPathRoot($fullB)
    if ($rootA -ine $rootB) {
        throw "paths do not share a drive root: '$fullA' and '$fullB'"
    }
    $segmentsA = $fullA.Substring($rootA.Length).Split("\", [System.StringSplitOptions]::RemoveEmptyEntries)
    $segmentsB = $fullB.Substring($rootB.Length).Split("\", [System.StringSplitOptions]::RemoveEmptyEntries)
    $shared = New-Object System.Collections.Generic.List[string]
    for ($index = 0; $index -lt [Math]::Min($segmentsA.Count, $segmentsB.Count); ++$index) {
        if ($segmentsA[$index] -ine $segmentsB[$index]) {
            break
        }
        $shared.Add($segmentsA[$index])
    }
    if ($shared.Count -eq 0) {
        return $rootA.TrimEnd("\")
    }
    return Join-Path $rootA ($shared -join "\")
}

function ConvertTo-Hy3dMappedPath {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][string] $SourceRoot,
        [Parameter(Mandatory = $true)][string] $MappedRoot
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $fullSource = [System.IO.Path]::GetFullPath($SourceRoot).TrimEnd("\")
    if ($fullPath -ine $fullSource -and
        -not $fullPath.StartsWith($fullSource + "\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "path is outside mapped source root: $fullPath"
    }
    $relative = $fullPath.Substring($fullSource.Length).TrimStart("\")
    if (-not $relative) {
        return $MappedRoot.TrimEnd("\")
    }
    return $MappedRoot.TrimEnd("\") + "\" + $relative
}

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
        [string] $BasePath,
        [version] $MaximumVersion
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
        Where-Object {
            (Test-Path -LiteralPath (Join-Path $_.FullName $RequiredRelativePath)) -and
            (-not $MaximumVersion -or (Get-Hy3dVersion $_.Name) -le $MaximumVersion)
        } |
        Sort-Object { Get-Hy3dVersion $_.Name } -Descending |
        Select-Object -First 1
    if (-not $candidate) {
        $versionHint = if ($MaximumVersion) { " at or below $MaximumVersion" } else { "" }
        throw "No valid $Kind toolchain$versionHint under '$BasePath' (expected '$RequiredRelativePath')."
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
