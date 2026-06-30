function Get-Hy3dVenvPlan {
    param(
        [Parameter(Mandatory = $true)][string] $VenvPath,
        [switch] $RecreateVenv
    )

    $fullVenv = [System.IO.Path]::GetFullPath($VenvPath)
    $python = Join-Path $fullVenv "Scripts\python.exe"
    if ($RecreateVenv) {
        return [pscustomobject]@{
            Action = "recreate"
            VenvPath = $fullVenv
            PythonPath = $python
        }
    }
    if (-not (Test-Path -LiteralPath $fullVenv)) {
        return [pscustomobject]@{
            Action = "create"
            VenvPath = $fullVenv
            PythonPath = $python
        }
    }
    if (Test-Path -LiteralPath $python -PathType Leaf) {
        return [pscustomobject]@{
            Action = "reuse"
            VenvPath = $fullVenv
            PythonPath = $python
        }
    }
    throw "virtual environment is incomplete at '$fullVenv'; rerun with -RecreateVenv to replace it"
}
