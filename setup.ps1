# Sets up the hackathon environment on a Windows laptop.
# Needs Python 3.12 installed (winget install Python.Python.3.12).
# Run from this folder in PowerShell:  powershell -ExecutionPolicy Bypass -File setup.ps1

py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip

# PyTorch: GPU build if an NVIDIA card is found, otherwise the CPU build
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "NVIDIA GPU found, installing PyTorch with CUDA"
    .\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
} else {
    Write-Host "No NVIDIA GPU, installing PyTorch for CPU"
    .\.venv\Scripts\python -m pip install torch torchvision
}

.\.venv\Scripts\python -m pip install transformers accelerate datasets pandas scikit-learn gradio opencv-python pillow pillow-heif ultralytics einops timm sentencepiece matplotlib jupyter

.\.venv\Scripts\python -c "import torch; print('PyTorch', torch.__version__, '| GPU:', torch.cuda.is_available())"

.\.venv\Scripts\python download_models.py

Write-Host "Done. Start working with: .venv\Scripts\activate  then  set HF_HUB_OFFLINE=1 (cmd) or `$env:HF_HUB_OFFLINE=1 (PowerShell)"
