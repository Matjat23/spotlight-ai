"""Read every photo once and save a structured record per photo to outputs/.

For each photo: a caption, the event/activity, the location, non-biometric person
attributes (clothing and gear), visible objects, an event group, plus embeddings for search.
The search app reads outputs/ so searching is instant. Run again when new photos are added.

    python pipeline.py                      # all photos, captions by Qwen2.5-VL
    python pipeline.py --limit 20           # quick test on 20 photos
    python pipeline.py --captioner blip     # older, faster BLIP captions

Face matching (approved by the organizers): faces are grouped into anonymous person
groups (P01, P02...) so staff can find the same person in other photos. No names, no
identity lookup, no face crops saved, and the face numbers are thrown away after grouping.
Clothing and gear are the backup when no face is visible. Skip faces with --no-faces.
"""
import argparse
import csv
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from sklearn.cluster import AgglomerativeClustering
from transformers import (AutoModel, AutoProcessor, AutoTokenizer, BitsAndBytesConfig,
                          BlipForConditionalGeneration, BlipProcessor, CLIPModel, CLIPProcessor,
                          Qwen2_5_VLForConditionalGeneration)

register_heif_opener()

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
EXTS = {".jpg", ".jpeg", ".png", ".heic"}

CLIP_NAME = "openai/clip-vit-base-patch32"
BLIP_NAME = "Salesforce/blip-image-captioning-large"
QWEN_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
TEXT_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Label lists. Key = label shown in the app, value = sentence CLIP compares with the photo.
# Qwen picks from the same keys, so both models use one vocabulary. Change them with DCT.
ACTIVITIES = {
    "birthday party": "a photo of a birthday party with a cake",
    "shopping spree": "a photo of children shopping in a store with bags",
    "theme park visit": "a photo of a family visiting a theme park",
    "meeting a character": "a photo of a child meeting a costumed character or mascot",
    "photo session": "a posed group photo in front of a backdrop",
    "gift giving": "a photo of a child receiving a gift",
    "fundraiser or gala": "a photo of people at a fundraising gala event",
    "run or sports event": "a photo of people at a charity run or sports event",
    "holiday party": "a photo of a holiday party with decorations",
    "meal together": "a photo of people eating a meal together at a table",
    "staff and volunteers": "a photo of volunteers or staff working together in matching shirts",
}
LOCATIONS = {
    "theme park": "a photo taken at a theme park",
    "store": "a photo taken inside a store",
    "restaurant": "a photo taken inside a restaurant",
    "hotel or resort": "a photo taken at a hotel or resort",
    "hospital": "a photo taken in a hospital room",
    "meeting room or office": "a photo taken in a meeting room or office with a table and chairs",
    "living room": "a photo taken in a living room with a sofa",
    "playroom or arcade": "a photo taken in a playroom or game arcade",
    "park or outdoors": "a photo taken outdoors with grass and trees",
    "beach": "a photo taken at a beach",
    "event hall or gym": "a photo taken in an event hall, ballroom or gym",
    "stage": "a photo of people on a stage at an event",
    "photo backdrop": "a posed photo in front of a plain photo backdrop",
}
# What a person wears or uses. Only these are used for "Same person", never faces,
# skin, age or gender. Objects like toys or bags are kept separately in "objects".
CLOTHING = {
    "mouse ears hat": "a person wearing mouse ears",
    "cap or hat": "a person wearing a cap or hat",
    "glasses": "a person wearing glasses",
    "princess dress": "a child wearing a princess dress",
    "costume": "a person wearing a costume",
    "red top": "a person wearing a red shirt",
    "blue top": "a person wearing a blue shirt",
    "pink top": "a person wearing a pink shirt",
    "purple top": "a person wearing a purple shirt",
    "white top": "a person wearing a white shirt",
    "black top": "a person wearing a black shirt",
    "gray top": "a person wearing a gray shirt",
    "yellow top": "a person wearing a yellow shirt",
    "green top": "a person wearing a green shirt",
    "orange top": "a person wearing an orange shirt",
    "striped top": "a person wearing a striped shirt",
    "jersey": "a person wearing a sports jersey",
    "matching shirts": "a group wearing matching t-shirts",
    "backpack": "a person with a backpack",
    "wheelchair": "a person in a wheelchair",
    "stroller": "a child in a stroller",
}
# Visible moments, not feelings. We never label what a child feels inside (sad, scared, tired),
# only what anyone can see in the photo. Staff can correct them in the app.
MOMENTS = ["smiling", "laughing", "hugging", "cheering", "celebrating", "dancing", "posing", "playing"]
ATTR_THRESHOLD = 0.23  # cosine similarity for the CLIP fallback, tune on the test set
TOP_ATTRS = 3
EVENT_DISTANCE = 0.18  # how different two undated photos can be and still count as one event
SAME_SCENE = 0.90      # photos this similar in one event are the same scene and share labels

QWEN_PROMPT = f"""Describe this event photo for a searchable photo archive of a children's charity.
Answer with JSON only, in this form:
{{"caption": "...", "activity": "...", "location": "...", "clothing": [...], "objects": [...], "moments": [...]}}
Rules:
- caption: one plain sentence about what is clearly visible. Say "a child", "an adult" or "a group".
  Never guess names, family relationships (mother, father, son, family), illness or feelings.
- activity: exactly one of {list(ACTIVITIES)} or "other".
- location: exactly one of {list(LOCATIONS)} or "other".
- clothing: every item from this list that a person in the photo clearly wears or uses: {list(CLOTHING)}.
  Use an empty list if you are not sure.
- objects: up to 5 clearly visible objects. Leave out anything you are not sure about.
- moments: every item from this list that is clearly visible: {MOMENTS}. Empty list if none.
  Never describe feelings like sad, scared or tired.
  A person in a full animal costume is a "mascot", not a stuffed animal."""

# BLIP and Qwen sometimes guess family roles. We do not know who people are, so captions stay neutral.
RELATION_WORDS = {
    "mother": "an adult", "mom": "an adult", "father": "an adult", "dad": "an adult",
    "grandmother": "an adult", "grandfather": "an adult", "parents": "adults", "parent": "an adult",
    "husband": "an adult", "wife": "an adult",
    "son": "a child", "daughter": "a child", "brother": "a child", "sister": "a child",
    "family": "a group",
}


def features(out):
    """Newer transformers return an output object instead of a tensor."""
    if not torch.is_tensor(out):
        out = out.pooler_output
    return out / out.norm(dim=-1, keepdim=True)


def clean_caption(cap):
    cap = cap.strip().rstrip(".")
    for start in ("there is ", "there are ", "arafed ", "araffe "):
        if cap.lower().startswith(start):
            cap = cap[len(start):]
    # "family" used as an adjective ("a family dinner") is just dropped
    cap = re.sub(r"\bfamily (?=party|photo|picture|dinner|meal|time|gathering|trip|vacation)", "", cap, flags=re.IGNORECASE)
    for word, neutral in RELATION_WORDS.items():
        cap = re.sub(rf"\b(?:(?:a|an|the|her|his|their) )?{word}\b", neutral, cap, flags=re.IGNORECASE)
    return cap[:1].upper() + cap[1:]


ROTATIONS = OUT / "rotations.json"  # file name -> degrees clockwise, for photos saved sideways


def rotations():
    return json.loads(ROTATIONS.read_text(encoding="utf-8")) if ROTATIONS.exists() else {}


def set_rotation(file_name, degrees):
    known = rotations()
    known[file_name] = degrees % 360
    if not known[file_name]:
        del known[file_name]
    ROTATIONS.write_text(json.dumps(known, indent=1), encoding="utf-8")


def open_upright(path):
    """The photo the right way up: the camera's own rotation tag first, then our fix for photos
    whose tag was lost (found by fix_rotation.py or the Rotate button in the app)."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    degrees = rotations().get(Path(path).name, 0)
    return img.rotate(-degrees, expand=True) if degrees else img


def speech_for(files):
    """For each file: the words said in a video from this moment until the next moment (from videos.py),
    or "" for photos. Searching "happy birthday" then finds the moment it was said."""
    info_file = OUT / "videos.json"
    info = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}
    moments = {}
    for n, path in enumerate(files):
        m = re.match(r"^zz_video_(V[0-9]+)_([0-9]+)[.]jpg$", path.name)
        if m and m.group(1) in info:
            moments.setdefault(m.group(1), []).append((int(m.group(2)), n))
    said = [""] * len(files)
    for vid, times in moments.items():
        times.sort()
        for k, (t, n) in enumerate(times):
            start = 0 if k == 0 else t
            end = times[k + 1][0] if k + 1 < len(times) else 1e9
            said[n] = " ".join(text for a, b, text in info[vid].get("speech", []) if start <= a < end)
    return said


def load_image(path, max_side=1024):
    img = open_upright(path)
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


def label_probs(sim):
    """Softmax over all labels, so the scores add up to 1 and read as a confidence."""
    return torch.softmax(torch.tensor(sim) * 100, dim=0).numpy()


def text_embed(texts, tok, model, device):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tok(texts[i : i + 64], padding=True, truncation=True, return_tensors="pt").to(device)
            h = model(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1)
            e = (h * mask).sum(1) / mask.sum(1)
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    return np.concatenate(out)


# Captions

def parse_json(answer):
    match = re.search(r"\{.*\}", answer, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {}


def describe_with_qwen(files, times=None):
    """One structured description per photo. Runs alone on the GPU, then frees it.
    If a list is given as times, the seconds per photo are added to it."""
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_NAME, quantization_config=quant, device_map="cuda")
    proc = AutoProcessor.from_pretrained(QWEN_NAME, min_pixels=128 * 28 * 28, max_pixels=400 * 28 * 28)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QWEN_PROMPT}]}]
    prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    results = []
    for n, path in enumerate(files, 1):
        started = time.time()
        inputs = proc(text=[prompt], images=[load_image(path, 896)], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        answer = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        results.append(parse_json(answer))
        if times is not None:
            times.append(time.time() - started)
        print(f"  Qwen {n}/{len(files)}", flush=True)

    del model
    torch.cuda.empty_cache()
    return results


def caption_with_blip(images, blip, blip_proc, dtype):
    b = blip_proc(images=images, return_tensors="pt").to(blip.device)
    ids = blip.generate(pixel_values=b["pixel_values"].to(dtype), max_new_tokens=30, num_beams=3)
    return blip_proc.batch_decode(ids, skip_special_tokens=True)


def pick(qwen_value, choices, probs):
    """Qwen's label if it is one of ours, else CLIP's best. Confidence is CLIP's score for that label."""
    labels = list(choices)
    if qwen_value in choices:
        return qwen_value, float(probs[labels.index(qwen_value)])
    i = int(probs.argmax())
    return labels[i], float(probs[i])


# Events

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


def harmonize_scenes(rows, img_emb):
    """Photos of the same scene (same event, almost the same picture) get the same location
    and activity, decided by a majority vote. Stops one room getting two different labels."""
    before = [(r["location"], r["activity"]) for r in rows]
    for r in rows:
        same = [x["id"] for x in rows if x["event_id"] == r["event_id"]
                and float(img_emb[x["id"]] @ img_emb[r["id"]]) >= SAME_SCENE]
        if len(same) < 2:
            continue
        for k, field in enumerate(("location", "activity")):
            votes = Counter(before[i][k] for i in same).most_common()
            if votes[0][1] > len(same) / 2:
                r[field] = votes[0][0]


# Faces

FACE_MIN_SCORE = 0.6   # detector confidence, lower picks up mascot heads and blurry faces
FACE_MIN_SIZE = 40     # pixels, smaller faces give unreliable numbers
FACE_DISTANCE = 0.5    # cosine distance, average linkage. Looser values chain different people together


def find_faces(files):
    """For each face: (photo index, 512 numbers from ArcFace, box, detector score).
    The box is where the face is, as fractions of the photo size, used for the People page cover."""
    import onnxruntime  # noqa: F401  (loads before insightface so it finds the CUDA libraries from torch)
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    faces = []
    for n, path in enumerate(files):
        img = np.array(load_image(path, 1280))[:, :, ::-1]  # RGB to BGR, what insightface expects
        h, w = img.shape[:2]
        for f in app.get(img):
            x1, y1, x2, y2 = f.bbox
            if f.det_score >= FACE_MIN_SCORE and min(x2 - x1, y2 - y1) >= FACE_MIN_SIZE:
                box = [round(float(x1) / w, 4), round(float(y1) / h, 4), round(float(x2) / w, 4), round(float(y2) / h, 4)]
                faces.append((n, f.normed_embedding, box, round(float(f.det_score), 3)))
        if (n + 1) % 20 == 0 or n + 1 == len(files):
            print(f"  Faces {n + 1}/{len(files)}", flush=True)
    return faces


def group_people(faces, count):
    """Groups faces that look alike into P01, P02... Only people seen in 2+ photos get a group.
    Returns the groups per photo and a list of grouped faces (photo, person, box, score)."""
    people = [set() for _ in range(count)]
    if len(faces) < 2:
        return people, []
    emb = np.stack([f[1] for f in faces])
    # Average linkage: a face joins a group only if it is close to the group on average. DBSCAN
    # chained different people together (A looks like B, B like C) in our tests.
    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=FACE_DISTANCE,
                                     metric="cosine", linkage="average").fit_predict(emb)
    by_group = {}
    for f, lab in zip(faces, labels):
        by_group.setdefault(lab, set()).add(f[0])
    groups = [(lab, photos) for lab, photos in by_group.items() if len(photos) >= 2]
    names = {}
    for n, (lab, photos) in enumerate(sorted(groups, key=lambda g: len(g[1]), reverse=True), 1):
        names[lab] = f"P{n:02d}"
        for photo in photos:
            people[photo].add(names[lab])
    grouped = [{"photo": f[0], "person": names[lab], "box": f[2], "score": f[3]}
               for f, lab in zip(faces, labels) if lab in names]
    return people, grouped


def process_photos(files, start_id):
    """Describes photos added in the app, without running the whole archive again.
    Returns the new records and their CLIP, text and clothing embeddings. Faces are not
    grouped here, new photos join person groups at the next full run."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    described = describe_with_qwen(files) if device == "cuda" else [{} for _ in files]

    clip = CLIPModel.from_pretrained(CLIP_NAME, torch_dtype=dtype).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(CLIP_NAME)
    act_emb = clip_text(clip, clip_proc, list(ACTIVITIES.values()), device)
    loc_emb = clip_text(clip, clip_proc, list(LOCATIONS.values()), device)
    images = [load_image(f) for f in files]
    with torch.no_grad():
        c = clip_proc(images=images, return_tensors="pt").to(device)
        img_emb = features(clip.get_image_features(pixel_values=c["pixel_values"].to(dtype)))
        act_sim = (img_emb @ act_emb.T).float().cpu().numpy()
        loc_sim = (img_emb @ loc_emb.T).float().cpu().numpy()

    rows = []
    for k, (path, img, d) in enumerate(zip(files, images, described)):
        activity, act_conf = pick(d.get("activity"), ACTIVITIES, label_probs(act_sim[k]))
        location, loc_conf = pick(d.get("location"), LOCATIONS, label_probs(loc_sim[k]))
        when = taken_at(path)
        rows.append({
            "id": start_id + k,
            "file": path.name,
            "caption": clean_caption(d.get("caption") or "No description yet"),
            "activity": activity,
            "activity_conf": round(act_conf, 2),
            "location": location,
            "location_conf": round(loc_conf, 2),
            "person_attributes": ";".join(dict.fromkeys(c for c in d.get("clothing", []) if c in CLOTHING)),
            "objects": ";".join(str(o).strip().lower() for o in d.get("objects", [])[:5]),
            "moments": ";".join(dict.fromkeys(m for m in d.get("moments", []) if m in MOMENTS)),
            "person_ids": "",
            "taken": when.isoformat(sep=" ") if when else "",
            "orientation": "portrait" if img.height > img.width else "landscape",
            "status": "ai",
        })

    text_tok = AutoTokenizer.from_pretrained(TEXT_NAME)
    text_model = AutoModel.from_pretrained(TEXT_NAME).to(device).eval()
    search_text = [f"{r['caption']}. {r['activity']} at {r['location']}. "
                   f"{r['person_attributes'].replace(';', ', ')}. {r['objects'].replace(';', ', ')}. "
                   f"{r['moments'].replace(';', ', ')}. {r.get('speech', '')}" for r in rows]
    attr_text = [r["person_attributes"].replace(";", ", ") or "no visible attributes" for r in rows]
    result = (rows, img_emb.float().cpu().numpy(), text_embed(search_text, text_tok, text_model, device),
              text_embed(attr_text, text_tok, text_model, device))
    del clip, text_model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="only the first N photos")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--captioner", choices=["qwen", "blip"], default="qwen")
    parser.add_argument("--no-faces", action="store_true", help="skip face grouping")
    args = parser.parse_args()

    files = sorted(p for p in DATA.iterdir() if p.suffix.lower() in EXTS)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No photos in {DATA}")
    OUT.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"{len(files)} photos, device {device}, captions by {args.captioner}")
    start = time.time()

    # Pass 1: Qwen describes every photo (it needs most of the GPU, so it runs first and alone)
    use_qwen = args.captioner == "qwen" and device == "cuda"
    described = describe_with_qwen(files) if use_qwen else [{} for _ in files]

    # Pass 2: anonymous person groups from faces
    if args.no_faces:
        people, face_boxes = [set() for _ in files], []
    else:
        people, face_boxes = group_people(find_faces(files), len(files))

    # Pass 3: CLIP embeddings and label scores (and BLIP captions when Qwen is not used)
    clip = CLIPModel.from_pretrained(CLIP_NAME, torch_dtype=dtype).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(CLIP_NAME)
    if not use_qwen:
        blip = BlipForConditionalGeneration.from_pretrained(BLIP_NAME, torch_dtype=dtype).to(device).eval()
        blip_proc = BlipProcessor.from_pretrained(BLIP_NAME)
    text_tok = AutoTokenizer.from_pretrained(TEXT_NAME)
    text_model = AutoModel.from_pretrained(TEXT_NAME).to(device).eval()

    act_emb = clip_text(clip, clip_proc, list(ACTIVITIES.values()), device)
    loc_emb = clip_text(clip, clip_proc, list(LOCATIONS.values()), device)
    cloth_emb = clip_text(clip, clip_proc, list(CLOTHING.values()), device)

    said = speech_for(files)
    rows, embeddings = [], []
    for i in range(0, len(files), args.batch):
        batch = files[i : i + args.batch]
        images = [load_image(p) for p in batch]

        with torch.no_grad():
            c = clip_proc(images=images, return_tensors="pt").to(device)
            img_emb = features(clip.get_image_features(pixel_values=c["pixel_values"].to(dtype)))
            act_sim = (img_emb @ act_emb.T).float().cpu().numpy()
            loc_sim = (img_emb @ loc_emb.T).float().cpu().numpy()
            cloth_sim = (img_emb @ cloth_emb.T).float().cpu().numpy()
            blip_caps = caption_with_blip(images, blip, blip_proc, dtype) if not use_qwen else [""] * len(batch)

        for k, (path, img) in enumerate(zip(batch, images)):
            d = described[len(rows)]
            activity, act_conf = pick(d.get("activity"), ACTIVITIES, label_probs(act_sim[k]))
            location, loc_conf = pick(d.get("location"), LOCATIONS, label_probs(loc_sim[k]))
            if use_qwen:
                clothing = list(dict.fromkeys(c for c in d.get("clothing", []) if c in CLOTHING))
            else:
                best = sorted(zip(cloth_sim[k], CLOTHING), reverse=True)[:TOP_ATTRS]
                clothing = [name for s, name in best if s >= ATTR_THRESHOLD]
            objects = [str(o).strip().lower() for o in d.get("objects", [])][:5]
            when = taken_at(path)
            rows.append({
                "id": len(rows),
                "file": path.name,
                "caption": clean_caption(d.get("caption") or blip_caps[k] or "No description"),
                "activity": activity,
                "activity_conf": round(act_conf, 2),
                "location": location,
                "location_conf": round(loc_conf, 2),
                "person_attributes": ";".join(clothing),
                "objects": ";".join(objects),
                "moments": ";".join(dict.fromkeys(m for m in d.get("moments", []) if m in MOMENTS)),
                "taken": when.isoformat(sep=" ") if when else "",
                "orientation": "portrait" if img.height > img.width else "landscape",
                "person_ids": ";".join(sorted(people[len(rows)])),
                "speech": said[len(rows)],
                "status": "ai",  # ai, accepted or edited, changed by staff in the app
            })
        embeddings.append(img_emb.float().cpu().numpy())
        print(f"  CLIP {min(i + args.batch, len(files))}/{len(files)}")

    img_all = np.concatenate(embeddings)
    group_events(rows, img_all)
    harmonize_scenes(rows, img_all)

    # Text embeddings: one for full search, one for the clothing only (same-person pivot)
    search_text = [f"{r['caption']}. {r['activity']} at {r['location']}. "
                   f"{r['person_attributes'].replace(';', ', ')}. {r['objects'].replace(';', ', ')}. "
                   f"{r['moments'].replace(';', ', ')}. {r['speech']}" for r in rows]
    attr_text = [r["person_attributes"].replace(";", ", ") or "no visible attributes" for r in rows]
    np.save(OUT / "clip_embeddings.npy", img_all)
    np.save(OUT / "text_embeddings.npy", text_embed(search_text, text_tok, text_model, device))
    np.save(OUT / "attr_embeddings.npy", text_embed(attr_text, text_tok, text_model, device))

    with open(OUT / "photos.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT / "photos.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    # Only where each grouped face is (no face numbers, no crops), for the People page covers
    with open(OUT / "faces.json", "w", encoding="utf-8") as f:
        json.dump(face_boxes, f)

    took = time.time() - start
    events = len({r["event_id"] for r in rows})
    print(f"Done in {took:.0f} s ({took / len(files):.2f} s per photo), {events} events. Saved to {OUT}")


if __name__ == "__main__":
    main()
