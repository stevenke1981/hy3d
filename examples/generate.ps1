$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

.\build\Release\hy3d.exe generate `
  --backend python `
  --quality smoke `
  --image .\third_party\Hunyuan3D-2.1\hy3dshape\demos\demo.png `
  --out .\outputs\hy3d-example-smoke.glb `
  --model-path .\models\Hunyuan3D-2.1 `
  --seed 42 `
  --device cuda

Write-Host "Output: $root\outputs\hy3d-example-smoke.glb"
Write-Host "Log:    $root\outputs\hy3d-example-smoke.glb.log.txt"
Write-Host "Meta:   $root\outputs\hy3d-example-smoke.glb.json"
