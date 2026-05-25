$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

.\build\Release\hy3d.exe texture `
  --backend python `
  --mesh .\outputs\hy3d-example-smoke.glb `
  --image .\third_party\Hunyuan3D-2.1\hy3dshape\demos\demo.png `
  --out .\outputs\hy3d-example-textured.glb `
  --model-path .\models\Hunyuan3D-2.1 `
  --resolution 512 `
  --max-views 6 `
  --device cuda

Write-Host "Output: $root\outputs\hy3d-example-textured.glb"
Write-Host "Log:    $root\outputs\hy3d-example-textured.glb.log.txt"
Write-Host "Meta:   $root\outputs\hy3d-example-textured.glb.json"
