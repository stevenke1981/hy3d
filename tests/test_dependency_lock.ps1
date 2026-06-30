param([Parameter(Mandatory = $true)][string] $RepoRoot)

$ErrorActionPreference = "Stop"

$setup = Get-Content -Raw -LiteralPath (Join-Path $RepoRoot "scripts\setup_hy3d_python.ps1")
$locks = @(
    "requirements-hy3d.lock.txt",
    "requirements-torch-cu124.lock.txt",
    "requirements-win-cu124.lock.txt"
)

foreach ($lock in $locks) {
    $path = Join-Path $RepoRoot $lock
    if (-not (Test-Path -LiteralPath $path)) {
        throw "missing dependency lock: $lock"
    }
    foreach ($line in Get-Content -LiteralPath $path) {
        $requirement = $line.Trim()
        if (-not $requirement -or $requirement.StartsWith("#")) {
            continue
        }
        if ($requirement -notmatch '^[A-Za-z0-9_.-]+==[^;\s]+(?:\s*;.*)?$') {
            throw "dependency is not exactly pinned in ${lock}: $requirement"
        }
    }
}

if ($setup -notmatch [regex]::Escape("requirements-win-cu124.lock.txt")) {
    throw "setup script does not consume the resolved dependency lock"
}
if ($setup -notmatch [regex]::Escape("write_dependency_manifest.py")) {
    throw "setup script does not write an installed dependency manifest"
}
if ($setup -notmatch [regex]::Escape("Get-Command uv") -or
    $setup -notmatch [regex]::Escape(".local\bin\uv.exe")) {
    throw "setup script does not resolve the common user-local uv installation"
}
if ($setup -notmatch "failed to install resolved dependencies" -or
    $setup -notmatch "setup dry-run failed") {
    throw "setup script does not enforce native-command exit codes"
}
if ($setup -notmatch [regex]::Escape("hy3d_setup_helpers.ps1") -or
    $setup -notmatch '\[switch\]\s*\$RecreateVenv' -or
    $setup -notmatch "Get-Hy3dVenvPlan") {
    throw "setup script does not support resumable venv lifecycle"
}

Write-Host "dependency lock tests passed"
