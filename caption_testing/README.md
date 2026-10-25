# Caption testing

Copy the `caption_testing` folder directly into your existing `spotlight-ai` project.
Do not replace Matty's downloader, setup scripts, frontend, or existing tests.
This package contains code only: no photos, captions, names, tokens, or model weights.
Your old `compare_captions.py`, `test_caption.py`, and `test_qwen.py` can stay as backups.

## Install (Windows PowerShell, from the project root)

Use your existing project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\caption_testing\requirements.txt
```

PyTorch must already be installed. If it isn't, install the CPU version with
` .\.venv\Scripts\python.exe -m pip install torch torchvision `.
For NVIDIA use the matching CUDA PyTorch build; this package does not change your GPU installation.
Native Florence support needs a recent Transformers release. If imports fail, check
` .\.venv\Scripts\python.exe -m pip show transformers ` and update the caption requirements.
Do not run pip updates during an active model test.

## First run: one photo, smaller models

```powershell
.\.venv\Scripts\python.exe -m caption_testing.run --folder 'C:\Hackaton\Test batches\Batch 1' --models blip florence-base --limit 1
```

## Test your new models

```powershell
.\.venv\Scripts\python.exe -m caption_testing.run --folder 'C:\Hackaton\Test batches\Batch 1' --models smolvlm-500m florence-large --limit 5
```

```powershell
.\.venv\Scripts\python.exe -m caption_testing.run --folder 'C:\Hackaton\Test batches\Batch 1' --models smolvlm2-2.2b --limit 1
```

```powershell
.\.venv\Scripts\python.exe -m caption_testing.run --folder 'C:\Hackaton\Test batches\Batch 1' --models moondream2 --limit 1
```

Qwen is deliberately opt-in because it was slow and memory-heavy on your CPU:

```powershell
.\.venv\Scripts\python.exe -m caption_testing.run --folder 'C:\Hackaton\Test batches\Batch 1' --models qwen-vl --limit 1
```

List keys without downloading/loading models:

```powershell
.\.venv\Scripts\python.exe -m caption_testing.run --list-models
```

| Key | Model |
| --- | --- |
| blip | Salesforce/blip-image-captioning-large |
| florence-base | florence-community/Florence-2-base |
| florence-large | florence-community/Florence-2-large |
| qwen-vl | Qwen/Qwen2.5-VL-3B-Instruct |
| smolvlm-500m | HuggingFaceTB/SmolVLM-500M-Instruct |
| smolvlm2-2.2b | HuggingFaceTB/SmolVLM2-2.2B-Instruct |
| moondream2 | vikhyatk/moondream2 (2025-06-21 release) |

## Controls

- `--limit 5`: first five supported photos in filename order, not a random sample.
  Omit it to run every supported image in the folder.
- `--recursive`: include subfolders; CSV photo names retain relative paths.
- `--threads 1`: lower CPU parallelism. Default 2. Does not guarantee a temperature.
- `--device cpu`: force CPU. `--device cuda`: require NVIDIA CUDA. Default auto.
- `--max-new-tokens 80`: shorter output limit for every selected model. Can truncate answers.
- `--prompt "Describe only the visible setting."`: Qwen, SmolVLM and Moondream only.
  BLIP's normal captioning does not follow this instruction.
- `--florence-task '<CAPTION>'`: shorter Florence captions.
  Default is `<MORE_DETAILED_CAPTION>`; `<DETAILED_CAPTION>` is also supported.
- `--qwen-min-pixels 100352 --qwen-max-pixels 200704`: reduced Qwen image detail.
  Defaults preserve your original Qwen script: 200704 and 401408 respectively.
- `--offline`: use only cached files and disable Hugging Face network access.
- `--output-dir 'C:\Hackaton\results'`: choose a results directory.

## Results

Each run creates new timestamped files under `outputs/caption_tests`:

- CSV: one row per photo/model, original caption/scoring columns first.
- JSON: the run's photo list, settings, model keys, and library versions.

Every completed row is flushed immediately. Ctrl+C stops the run and preserves
completed results; it cannot save a caption still being generated. There is no resume
mode: rerunning creates a fresh CSV and recomputes the selected photos.

Rows have status `ok`, `image_error`, or `load_error`. A load error creates one row
with a blank photo and the error message. Review error rows separately from scores.
A hard process crash appears in the terminal and may have no CSV error row.

Processing time includes preprocessing, image encoding, generation, and decoding;
it excludes opening the image and loading/downloading the model. GPU work is synchronized.
Loading time is recorded separately. The first image can include warm-up costs.
Models run in separate processes, sequentially, to release their memory between models.
If a single model exhausts system memory, subprocess isolation cannot prevent Windows
from becoming slow; stop it or use a smaller model / different hardware.

CPU uses float32; Qwen/SmolVLM/Moondream use float16 on CUDA. BLIP and Florence
retain float32, matching your previous comparison script. Moondream may manage
some internal precision itself. Record the library/model versions and hardware when
comparing runs. Do not interpret token counts as identical text lengths across models.

Scores are left blank for your team to review. No automated accuracy judgments are made.
The same photos are read independently by each model, not passed from model to model.
HEIC/HEIF are decoded locally with pillow-heif. No intermediate JPEG is written.
For multi-image HEIC containers, only the primary image opened by Pillow is used.

## Model downloads and offline operation

Use Matty's existing `download_models.py`, or let this runner fetch selected model files.
Weights use the normal Hugging Face cache; they are not stored in this code folder.
The community Florence checkpoints are distinct cache entries from `microsoft/Florence-2-*`.
Moondream uses custom code and the pinned `2025-06-21` release; its upstream code also
loads `moondream/starmie-v1`. Run once online with a non-sensitive test image to cache
all required components, then verify the same run with `--offline` before relying on it.
The runner makes no hosted inference calls; inference happens on your machine.
Moondream's custom loader compatibility has not been inference-tested in this environment.

Keep actual photos and derived captions/metadata local or in your event-approved private
environment. The existing project `.gitignore` already excludes `outputs/`.
Do not commit outputs if your ignore rules have changed.

## Validation of this package

Verified Python syntax, help/model listing, photo discovery, CSV writing, error handling,
and adapter input/output routing with mocks. No model weights were loaded and no real
caption inference was run here. Start with `--limit 1` on your laptop.

## Sources for adapters

- https://huggingface.co/Salesforce/blip-image-captioning-large
- https://huggingface.co/docs/transformers/model_doc/florence2
- https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl
- https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct
- https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
- https://huggingface.co/vikhyatk/moondream2
