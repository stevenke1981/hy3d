param([Parameter(Mandatory = $true)][string] $RepoRoot)

$ErrorActionPreference = "Stop"
. (Join-Path $RepoRoot "scripts\hy3d_setup_helpers.ps1")

$root = Join-Path ([System.IO.Path]::GetTempPath()) ("hy3d-setup-helper-" + [guid]::NewGuid())
try {
    $venv = Join-Path $root ".venv-hy3d"

    $create = Get-Hy3dVenvPlan -VenvPath $venv
    if ($create.Action -ne "create") {
        throw "missing venv should be created, got: $($create.Action)"
    }

    New-Item -ItemType Directory -Force -Path (Join-Path $venv "Scripts") | Out-Null
    New-Item -ItemType File -Force -Path (Join-Path $venv "Scripts\python.exe") | Out-Null
    $reuse = Get-Hy3dVenvPlan -VenvPath $venv
    if ($reuse.Action -ne "reuse") {
        throw "complete venv should be reused, got: $($reuse.Action)"
    }

    $recreate = Get-Hy3dVenvPlan -VenvPath $venv -RecreateVenv
    if ($recreate.Action -ne "recreate") {
        throw "explicit recreation should clear the venv, got: $($recreate.Action)"
    }

    Remove-Item -LiteralPath (Join-Path $venv "Scripts\python.exe") -Force
    $failed = $false
    try {
        Get-Hy3dVenvPlan -VenvPath $venv | Out-Null
    } catch {
        $failed = $_.Exception.Message -match "RecreateVenv"
    }
    if (-not $failed) {
        throw "partial venv was not rejected with recovery guidance"
    }
} finally {
    Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "setup helper tests passed"
