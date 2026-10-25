import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

register_heif_opener()

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
parser.add_argument("--limit", type=int, default=1)
args = parser.parse_args()

extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
photos = sorted(
    p for p in args.folder.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
)
photos = photos[:args.limit]
if not photos:
    raise SystemExit("No supported photos found.")

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

prompt = (
    "Describe this photo using only clearly visible information. "
    "Include the setting, objects, activities, and clothing/accessories "
    "where visible. Keep descriptions attached to the correct person. "
    "Do not guess identities, relationships, medical conditions, emotions, "
    "or specific venues. Omit uncertain details."
)

print(f"Loading {model_id} on {device}...", flush=True)

processor = AutoProcessor.from_pretrained(
    model_id,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28,
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    dtype=dtype,
    attn_implementation="sdpa",
).to(device)
model.eval()

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
output_path = output_dir / f"qwen_test_{stamp}.csv"

fields = [
    "batch", "photo", "model", "task", "caption", "seconds",
    "accuracy_1_to_5", "detail_1_to_5", "hallucinations",
    "error", "prompt",
]

with output_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for number, photo in enumerate(photos, 1):
        print(f"\n[{number}/{len(photos)}] {photo.name}", flush=True)
        row = {
            "batch": args.folder.name,
            "photo": photo.name,
            "model": model_id,
            "task": "prompted caption",
            "prompt": prompt,
        }

        try:
            with Image.open(photo) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")

            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=[image], return_tensors="pt"
            ).to(device)

            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=256, do_sample=False
                )

            # Remove the input prompt; decode only the generated answer.
            answer = generated[:, inputs["input_ids"].shape[1]:]
            caption = processor.batch_decode(
                answer, skip_special_tokens=True
            )[0]

            if device == "cuda":
                torch.cuda.synchronize()
            row["seconds"] = round(time.perf_counter() - start, 2)
            row["caption"] = caption

            print(caption, flush=True)
            print(f"Processing time: {row['seconds']:.2f} seconds", flush=True)

        except Exception as error:
            row["error"] = str(error)
            print(f"ERROR: {error}", flush=True)

        writer.writerow(row)
        file.flush()

print(f"\nResults saved to: {output_path.resolve()}", flush=True)