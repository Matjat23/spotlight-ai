# Setup guide

How the hackathon laptop was set up, so anyone on the team can do the same.
Takes about 20-30 minutes, mostly downloads (around 6 GB). Do it on good Wi-Fi.

## Quick way (one script)
1. Install Python 3.12 if you don't have it:
   `winget install Python.Python.3.12`
2. Get this folder (copy it, or clone the repo once it is on GitHub).
3. Open PowerShell in the folder and run:
   `powershell -ExecutionPolicy Bypass -File setup.ps1`

The script does every step below by itself.

## Step by step (what the script does)

| Step | Command | What it does |
|---|---|---|
| 1 | `py -3.12 -m venv .venv` | Creates a private Python just for this project in the `.venv` folder, so its libraries don't mix with anything else on the computer. We use 3.12 because the newest Python (3.14) is too new for some AI libraries. |
| 2 | `.venv\Scripts\activate` | Turns that private Python on in this terminal. You see `(.venv)` at the start of the line. |
| 3 | `python -m pip install --upgrade pip` | Updates the installer. |
| 4a | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` | PyTorch, the AI engine, built for NVIDIA graphics cards (CUDA). Use this if `nvidia-smi` works on your laptop. |
| 4b | `pip install torch torchvision` | The same for laptops without an NVIDIA card (runs on the CPU, slower). |
| 5 | `pip install transformers accelerate datasets pandas scikit-learn gradio opencv-python pillow ultralytics einops timm sentencepiece matplotlib jupyter` | The tools: Hugging Face models (transformers), data tables (pandas), metrics (scikit-learn), demo app (gradio), images and video (opencv, pillow), YOLO tracking (ultralytics), notebooks (jupyter). |
| 6 | `python -c "import torch; print(torch.cuda.is_available())"` | Checks the GPU. `True` = the graphics card is used. |
| 7 | `python download_models.py` | Downloads the models once, so they work without internet at the venue. |

## Every time you start working
```
cd C:\Users\<you>\Hackathons\ai4good-2026
.venv\Scripts\activate
set HF_HUB_OFFLINE=1
```
- `activate` turns on the project's Python.
- `set HF_HUB_OFFLINE=1` tells the Hugging Face library to use the downloaded models and not try the internet. In PowerShell write `$env:HF_HUB_OFFLINE=1` instead.

## Models we have
| Model | Used for |
|---|---|
| openai/clip-vit-base-patch32 | Search photos with text, zero-shot tags |
| Salesforce/blip-image-captioning-large | Photo captions |
| microsoft/Florence-2-base | Captions and object tags |
| MCG-NJU/videomae-base-finetuned-kinetics | Actions in video |
| trpakov/vit-face-expression | Emotions from faces |
| sentence-transformers/all-MiniLM-L6-v2 | Text embeddings |
| YOLO11 n/s and pose (weights/) | Detect and track people, body pose |

Models are stored in `C:\Users\<you>\.cache\huggingface`, YOLO in `weights\`.

## Laptops
- Matty: RTX 4070 (8 GB), 32 GB RAM: runs everything on the GPU.
- Without an NVIDIA GPU everything still works, just slower. Then use the small models and fewer frames.
