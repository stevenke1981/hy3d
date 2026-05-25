param(
    [string] $ImagePath = "",
    [string] $OutputPath = "",
    [string] $TextureOutputPath = "",
    [int] $Seed = 42,
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $ImagePath) {
    $ImagePath = Join-Path $root "outputs\frieren-reference.png"
}
if (-not $OutputPath) {
    $OutputPath = Join-Path $root "outputs\character-final-80step.glb"
}
if (-not $TextureOutputPath) {
    $TextureOutputPath = Join-Path $root "outputs\character-final-80step-textured.glb"
}

if ($DryRun) {
    & (Join-Path $PSScriptRoot "generate_3d_model.ps1") `
        -ImagePath $ImagePath `
        -OutputPath $OutputPath `
        -Quality final `
        -Steps 80 `
        -Seed $Seed `
        -Texture `
        -TextureOutputPath $TextureOutputPath `
        -TextureResolution 512 `
        -TextureMaxViews 6 `
        -NoRemesh `
        -DryRun
    exit $LASTEXITCODE
}

& (Join-Path $PSScriptRoot "generate_3d_model.ps1") `
    -ImagePath $ImagePath `
    -OutputPath $OutputPath `
    -Quality final `
    -Steps 80 `
    -Seed $Seed `
    -Texture `
    -TextureOutputPath $TextureOutputPath `
    -TextureResolution 512 `
    -TextureMaxViews 6 `
    -NoRemesh
exit $LASTEXITCODE
