param([Parameter(Mandatory = $true)][string] $RepoRoot)

$ErrorActionPreference = "Stop"

$setup = Get-Content -Raw -LiteralPath (Join-Path $RepoRoot "scripts\setup_hy3d_python.ps1")
$locks = @(
    "requirements-hy3d.lock.txt",
    "requirements-torch-cu124.lock.txt"
)

foreach ($lock in $locks) {
    $path = Join-Path $RepoRoot $lock
    if (-not (Test-Path -LiteralPath $path)) {
        throw "missing dependency lock: $lock"
    }
    if ($setup -notmatch [regex]::Escape($lock)) {
        throw "setup script does not consume dependency lock: $lock"
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

Write-Host "dependency lock tests passed"
