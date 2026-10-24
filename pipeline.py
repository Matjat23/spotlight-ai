"""Read every photo once and save captions, tags and CLIP embeddings to outputs/.

The search app reads outputs/ so searching is instant. Run again when new photos are added.

    python pipeline.py                  # all photos in data/raw
    python pipeline.py --limit 20       # quick test on 20 photos
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from transformers import BlipForConditionalGeneration, BlipProcessor, CLIPModel, CLIPProcessor

register_heif_opener()

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
EXTS = {".jpg", ".jpeg", ".png", ".heic"}

CLIP_NAME = "openai/clip-vit-base-patch32"
BLIP_NAME = "Salesforce/blip-image-captioning-large"

# Tag list for Dreams Come True photos. Each tag is scored with CLIP against the photo.
# Change these after talking to the partner, they should be the words staff search with.
TAGS = {
    "headshot": "a close-up headshot portrait of one person",
    "family": "a family posing together",
    "group": "a large group of people",
    "posed portrait": "a posed studio portrait in front of a backdrop",
    "cutout": "a person cut out on a plain white background",
    "shopping": "children shopping in a store",
    "theme park": "a family at a theme park",
    "hospital": "a child in a hospital",
    "staff and volunteers": "volunteers or staff in matching shirts",
    "smiling": "people smiling happily",
    "outdoors": "a photo taken outdoors",
    "indoors": "a photo taken indoors",
    "celebration": "a party or celebration",
    "child": "a young child",
    "teenager": "a teenager",
}
TAG_THRESHOLD = 0.22  # cosine similarity, tune on the test set
TOP_TAGS = 3


def features(out):
    """Newer transformers return an output object instead of a tensor."""
    if not torch.is_tensor(out):
        out = out.pooler_output
    return out / out.norm(dim=-1, keepdim=True)


def clean_caption(cap):
    cap = cap.strip()
    for start in ("there is ", "there are ", "arafed ", "araffe "):
        if cap.startswith(start):
            cap = cap[len(start):]
    return cap[:1].upper() + cap[1:]


def load_image(path, max_side=1024):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def photo_year(path):
    try:
        exif = Image.open(path).getexif()
        dt = exif.get(306) or exif.get_ifd(0x8769).get(36867)
    except Exception:
        return ""
    year = (dt or "")[:4]
    return year if year.isdigit() and 2000 <= int(year) <= 2030 else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="only the first N photos")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    files = sorted(p for p in DATA.iterdir() if p.suffix.lower() in EXTS)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No photos in {DATA}")
    OUT.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"{len(files)} photos, device {device}")

    clip = CLIPModel.from_pretrained(CLIP_NAME, torch_dtype=dtype).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(CLIP_NAME)
    blip = BlipForConditionalGeneration.from_pretrained(BLIP_NAME, torch_dtype=dtype).to(device).eval()
    blip_proc = BlipProcessor.from_pretrained(BLIP_NAME)

    # Text embeddings for the tags, computed once
    with torch.no_grad():
        t = clip_proc(text=list(TAGS.values()), return_tensors="pt", padding=True).to(device)
        tag_emb = features(clip.get_text_features(**t))

    rows, embeddings = [], []
    start = time.time()
    for i in range(0, len(files), args.batch):
        batch = files[i : i + args.batch]
        images = [load_image(p) for p in batch]

        with torch.no_grad():
            c = clip_proc(images=images, return_tensors="pt").to(device)
            img_emb = features(clip.get_image_features(pixel_values=c["pixel_values"].to(dtype)))
            sims = (img_emb @ tag_emb.T).float().cpu().numpy()

            b = blip_proc(images=images, return_tensors="pt").to(device)
            ids = blip.generate(pixel_values=b["pixel_values"].to(dtype), max_new_tokens=30, num_beams=3)
            captions = blip_proc.batch_decode(ids, skip_special_tokens=True)

        for path, img, cap, sim in zip(batch, images, captions, sims):
            # Best 3 tags above the threshold, otherwise every photo gets half the list
            best = sorted(zip(sim, TAGS), reverse=True)[:TOP_TAGS]
            tags = [name for s, name in best if s >= TAG_THRESHOLD]
            rows.append({
                "id": len(rows),
                "file": path.name,
                "caption": clean_caption(cap),
                "tags": ";".join(tags),
                "year": photo_year(path),
                "orientation": "portrait" if img.height > img.width else "landscape",
            })
        embeddings.append(img_emb.float().cpu().numpy())
        print(f"  {min(i + args.batch, len(files))}/{len(files)}")

    np.save(OUT / "clip_embeddings.npy", np.concatenate(embeddings))
    with open(OUT / "photos.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    took = time.time() - start
    print(f"Done in {took:.0f} s ({took / len(files):.2f} s per photo). Saved to {OUT}")


if __name__ == "__main__":
    main()
