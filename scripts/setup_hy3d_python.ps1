param(
    [string] $PythonVersion = "3.10",
    [string] $VenvPath = ".\.venv-hy3d"
)

$ErrorActionPreference = "Stop"

uv venv --python $PythonVersion $VenvPath
$python = Join-Path $VenvPath "Scripts\python.exe"

uv pip install --python $python torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

uv pip install --python $python `
  diffusers==0.30.0 `
  transformers==4.46.0 `
  accelerate==1.1.1 `
  huggingface-hub==0.30.2 `
  safetensors==0.4.4 `
  numpy==1.24.4 `
  scipy==1.14.1 `
  einops==0.8.0 `
  opencv-python==4.10.0.84 `
  pillow `
  trimesh==4.4.7 `
  pymeshlab==2022.2.post3 `
  pygltflib==1.16.3 `
  xatlas==0.0.9 `
  omegaconf==2.3.0 `
  pyyaml==6.0.2 `
  tqdm==4.66.5 `
  rembg==2.0.65 `
  onnxruntime==1.16.3 `
  setuptools==70.3.0 `
  timm==1.0.22 `
  torchdiffeq==0.2.5 `
  basicsr==1.4.2 `
  realesrgan==0.3.0 `
  pytorch-lightning==1.9.5 `
  torchmetrics==1.6.0 `
  pandas==2.2.2 `
  imageio==2.36.0 `
  scikit-image==0.24.0 `
  configargparse==1.7 `
  psutil==6.0.0 `
  cupy-cuda12x==13.4.1 `
  pythreejs `
  pybind11==2.13.4 `
  ninja==1.11.1.1

& .\scripts\build_hy3dpaint_windows.ps1 -PythonPath $python

& $python scripts\hy3d_generate.py `
  --image third_party\Hunyuan3D-2.1\hy3dshape\demos\demo.png `
  --out outputs\setup-dry-run.glb `
  --model-path models\Hunyuan3D-2.1 `
  --low-vram `
  --dry-run
