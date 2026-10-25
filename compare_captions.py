import argparse
import csv
import gc
import time
from datetime import datetime
from pathlib import Path
from pillow_heif import register_heif_opener

register_heif_opener()


import torch
from PIL import Image, ImageOps
from transformers import (
    AutoProcessor,
    BlipProcessor,
    BlipForConditionalGeneration,
    Florence2ForConditionalGeneration,
)

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
args = parser.parse_args()

extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
photos = sorted(
    p for p in args.folder.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
)
if not photos:
    raise SystemExit("No supported images found in that folder.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Found {len(photos)} photos. Using {device}.", flush=True)

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
output_path = output_dir / f"caption_test_{stamp}.csv"

models = [
    (
        "Salesforce/blip-image-captioning-large",
        BlipProcessor,
        BlipForConditionalGeneration,
        "",
    ),
    (
        "florence-community/Florence-2-base",
        AutoProcessor,
        Florence2ForConditionalGeneration,
        "<MORE_DETAILED_CAPTION>",
    ),
]

fields = [
    "batch", "photo", "model", "task", "caption", "seconds",
    "accuracy_1_to_5", "detail_1_to_5", "hallucinations", "error",
]

with output_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for model_id, processor_class, model_class, task in models:
        print(f"\nLoading {model_id}...", flush=True)
        processor = processor_class.from_pretrained(model_id)
        model = model_class.from_pretrained(model_id).to(device)
        model.eval()

        for number, photo in enumerate(photos, start=1):
            row = {
                "batch": args.folder.name,
                "photo": photo.name,
                "model": model_id,
                "task": task or "caption",
            }
            print(f"[{number}/{len(photos)}] {photo.name}", flush=True)

            try:
                with Image.open(photo) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")

                start = time.perf_counter()
                options = {"text": task} if task else {}
                inputs = processor(
                    images=image, return_tensors="pt", **options
                ).to(device)

                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=256 if task else 80,
                        do_sample=False,
                        num_beams=3,
                    )

                if task:
                    raw = processor.batch_decode(
                        generated, skip_special_tokens=False
                    )[0]
                    caption = processor.post_process_generation(
                        raw, task=task, image_size=image.size
                    )[task]
                else:
                    caption = processor.decode(
                        generated[0], skip_special_tokens=True
                    )

                row["caption"] = caption
                row["seconds"] = round(time.perf_counter() - start, 2)
                print(caption, flush=True)
                print(f"Processing time: {row['seconds']:.2f} seconds\n", flush=True)

            except Exception as error:
                row["error"] = str(error)
                print(f"ERROR: {error}", flush=True)

            writer.writerow(row)
            file.flush()

        # Release this model before loading the next one.
        del model, processor
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

print(f"\nResults saved to: {output_path.resolve()}", flush=True)