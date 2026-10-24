"""Read every photo once and save a structured record per photo to outputs/.

For each photo: a caption, the event/activity, the location, non-biometric person
attributes (clothing and gear), an event group, plus embeddings for search.
The search app reads outputs/ so searching is instant. Run again when new photos are added.

    python pipeline.py                  # all photos in data/raw
    python pipeline.py --limit 20       # quick test on 20 photos

No face detection, face crops or face embeddings are made anywhere. "Same person"
in the app is matched on the attribute text only.
"""
import argparse
import csv
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from sklearn.cluster import AgglomerativeClustering
from transformers import (AutoModel, AutoTokenizer, BlipForConditionalGeneration, BlipProcessor,
                          CLIPModel, CLIPProcessor)

register_heif_opener()

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
EXTS = {".jpg", ".jpeg", ".png", ".heic"}

CLIP_NAME = "openai/clip-vit-base-patch32"
BLIP_NAME = "Salesforce/blip-image-captioning-large"
TEXT_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Label lists scored with CLIP. Change them after talking to DCT, they should match
# the words staff use. Key = label shown in the app, value = prompt given to CLIP.
ACTIVITIES = {
    "birthday party": "a birthday party with a cake",
    "shopping spree": "children shopping in a store with bags",
    "theme park visit": "a family visiting a theme park",
    "meeting a character": "a child meeting a costumed character or mascot",
    "photo session": "a posed photo session in front of a backdrop",
    "gift giving": "a child receiving a gift",
    "fundraiser or gala": "people at a fundraising gala event",
    "run or sports event": "people at a charity run or sports event",
    "holiday party": "a holiday party with decorations",
    "family time": "a family relaxing together",
    "staff and volunteers": "volunteers or staff working together in matching shirts",
}
LOCATIONS = {
    "theme park": "a theme park",
    "store": "inside a store",
    "restaurant": "inside a restaurant",
    "hotel or resort": "a hotel or resort",
    "hospital": "a hospital room",
    "home": "inside a home",
    "park or outdoors": "an outdoor park with grass and trees",
    "beach": "a beach",
    "event hall": "an event hall or ballroom",
    "office": "an office",
    "studio backdrop": "a photo studio with a plain backdrop",
}
# Only clothing and gear, never faces, skin, age or gender
ATTRIBUTES = {
    "mouse ears hat": "a person wearing mouse ears",
    "cap or hat": "a person wearing a cap or hat",
    "glasses": "a person wearing glasses",
    "princess dress": "a child wearing a princess dress",
    "costume": "a person wearing a costume",
    "red top": "a person wearing a red shirt",
    "blue top": "a person wearing a blue shirt",
    "pink top": "a person wearing a pink shirt",
    "white top": "a person wearing a white shirt",
    "black top": "a person wearing a black shirt",
    "yellow top": "a person wearing a yellow shirt",
    "green top": "a person wearing a green shirt",
    "matching shirts": "a group wearing matching t-shirts",
    "backpack": "a person with a backpack",
    "wheelchair": "a person in a wheelchair",
    "stroller": "a child in a stroller",
    "shopping bags": "a person holding shopping bags",
    "balloons": "a person holding balloons",
    "stuffed animal": "a child holding a stuffed animal",
}
ATTR_THRESHOLD = 0.23  # cosine similarity, tune on the test set
TOP_ATTRS = 3
EVENT_DISTANCE = 0.18  # how different two undated photos can be and still count as one event


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


def taken_at(path):
    """Date and time from EXIF, or None. Ignores clearly wrong camera clocks."""
    try:
        exif = Image.open(path).getexif()
        dt = exif.get_ifd(0x8769).get(36867) or exif.get(306)
        when = datetime.strptime(dt.strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None
    return when if 2000 <= when.year <= datetime.now().year else None


def clip_text(clip, proc, prompts, device):
    with torch.no_grad():
        t = proc(text=prompts, return_tensors="pt", padding=True).to(device)
        return features(clip.get_text_features(**t))


def best_label(sim, labels):
    """Single label with a confidence from a softmax over all labels."""
    probs = torch.softmax(torch.tensor(sim) * 100, dim=0).numpy()
    i = int(probs.argmax())
    return labels[i], float(probs[i])


def text_embed(texts, device):
    tok = AutoTokenizer.from_pretrained(TEXT_NAME)
    model = AutoModel.from_pretrained(TEXT_NAME).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tok(texts[i : i + 64], padding=True, truncation=True, return_tensors="pt").to(device)
            h = model(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1)
            e = (h * mask).sum(1) / mask.sum(1)
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    return np.concatenate(out)


def group_events(rows, img_emb):
    """Same day = same event. Undated photos are grouped by how similar they look."""
    by_day = {}
    for r in rows:
        if r["taken"]:
            by_day.setdefault(r["taken"][:10], []).append(r["id"])
    undated = [r["id"] for r in rows if not r["taken"]]
    groups = list(by_day.values())
    if len(undated) > 1:
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=EVENT_DISTANCE,
                                         metric="cosine", linkage="average").fit_predict(img_emb[undated])
        for lab in sorted(set(labels)):
            groups.append([undated[j] for j in range(len(undated)) if labels[j] == lab])
    elif undated:
        groups.append(undated)

    for n, ids in enumerate(sorted(groups, key=len, reverse=True), 1):
        name = Counter(rows[i]["activity"] for i in ids).most_common(1)[0][0]
        for i in ids:
            rows[i]["event_id"] = f"E{n:02d}"
            rows[i]["event_name"] = name


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

    act_emb = clip_text(clip, clip_proc, list(ACTIVITIES.values()), device)
    loc_emb = clip_text(clip, clip_proc, list(LOCATIONS.values()), device)
    attr_emb = clip_text(clip, clip_proc, list(ATTRIBUTES.values()), device)

    rows, embeddings = [], []
    start = time.time()
    for i in range(0, len(files), args.batch):
        batch = files[i : i + args.batch]
        images = [load_image(p) for p in batch]

        with torch.no_grad():
            c = clip_proc(images=images, return_tensors="pt").to(device)
            img_emb = features(clip.get_image_features(pixel_values=c["pixel_values"].to(dtype)))
            act_sim = (img_emb @ act_emb.T).float().cpu().numpy()
            loc_sim = (img_emb @ loc_emb.T).float().cpu().numpy()
            attr_sim = (img_emb @ attr_emb.T).float().cpu().numpy()

            b = blip_proc(images=images, return_tensors="pt").to(device)
            ids = blip.generate(pixel_values=b["pixel_values"].to(dtype), max_new_tokens=30, num_beams=3)
            captions = blip_proc.batch_decode(ids, skip_special_tokens=True)

        for k, (path, img, cap) in enumerate(zip(batch, images, captions)):
            activity, act_conf = best_label(act_sim[k], list(ACTIVITIES))
            location, loc_conf = best_label(loc_sim[k], list(LOCATIONS))
            best = sorted(zip(attr_sim[k], ATTRIBUTES), reverse=True)[:TOP_ATTRS]
            attrs = [name for s, name in best if s >= ATTR_THRESHOLD]
            when = taken_at(path)
            rows.append({
                "id": len(rows),
                "file": path.name,
                "caption": clean_caption(cap),
                "activity": activity,
                "activity_conf": round(act_conf, 2),
                "location": location,
                "location_conf": round(loc_conf, 2),
                "person_attributes": ";".join(attrs),
                "taken": when.isoformat(sep=" ") if when else "",
                "orientation": "portrait" if img.height > img.width else "landscape",
                "status": "ai",  # ai, accepted or edited, changed by staff in the app
            })
        embeddings.append(img_emb.float().cpu().numpy())
        print(f"  {min(i + args.batch, len(files))}/{len(files)}")

    img_all = np.concatenate(embeddings)
    group_events(rows, img_all)

    # Text embeddings: one for full search, one for the attributes only (same-person pivot)
    search_text = [f"{r['caption']}. {r['activity']} at {r['location']}. {r['person_attributes'].replace(';', ', ')}"
                   for r in rows]
    attr_text = [r["person_attributes"].replace(";", ", ") or "no visible attributes" for r in rows]
    np.save(OUT / "clip_embeddings.npy", img_all)
    np.save(OUT / "text_embeddings.npy", text_embed(search_text, device))
    np.save(OUT / "attr_embeddings.npy", text_embed(attr_text, device))

    with open(OUT / "photos.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT / "photos.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)

    took = time.time() - start
    events = len({r["event_id"] for r in rows})
    print(f"Done in {took:.0f} s ({took / len(files):.2f} s per photo), {events} events. Saved to {OUT}")


if __name__ == "__main__":
    main()
