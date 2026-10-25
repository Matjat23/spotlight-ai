"""Print what each captioning model says about the same photos, with the time it took.

For the model comparison sheet. Does not touch outputs/, so the app keeps working.

    python compare_models.py 2 5 17          # photo numbers from the app
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from transformers import AutoProcessor, BlipForConditionalGeneration, BlipProcessor, Florence2ForConditionalGeneration

import pipeline as p

# Microsoft's Florence-2 weights converted for the built-in transformers class (the original
# custom code no longer runs on transformers 5)
FLORENCE_NAME = "florence-community/Florence-2-base"


def timed(fn):
    start = time.time()
    out = fn()
    return out, time.time() - start


def run_blip(images):
    proc = BlipProcessor.from_pretrained(p.BLIP_NAME)
    model = BlipForConditionalGeneration.from_pretrained(p.BLIP_NAME, torch_dtype=torch.float16).to("cuda").eval()
    results = []
    for img in images:
        def go():
            b = proc(images=[img], return_tensors="pt").to("cuda")
            ids = model.generate(pixel_values=b["pixel_values"].half(), max_new_tokens=30, num_beams=3)
            return proc.batch_decode(ids, skip_special_tokens=True)[0]
        results.append(timed(go))
    del model
    torch.cuda.empty_cache()
    return results


def run_florence(images):
    proc = AutoProcessor.from_pretrained(FLORENCE_NAME)
    model = Florence2ForConditionalGeneration.from_pretrained(FLORENCE_NAME, dtype=torch.float16).to("cuda").eval()
    task = "<MORE_DETAILED_CAPTION>"
    results = []
    for img in images:
        def go():
            inputs = proc(text=task, images=img, return_tensors="pt").to("cuda", torch.float16)
            ids = model.generate(**inputs, max_new_tokens=120, num_beams=3)
            return proc.batch_decode(ids, skip_special_tokens=True)[0]
        results.append(timed(go))
    del model
    torch.cuda.empty_cache()
    return results


def run_qwen(files):
    times = []
    described = p.describe_with_qwen(files, times)
    out = []
    for d, per_photo in zip(described, times):
        text = d.get("caption", "")
        extra = [f"activity: {d.get('activity')}", f"location: {d.get('location')}",
                 f"clothing: {', '.join(d.get('clothing', []))}", f"objects: {', '.join(d.get('objects', []))}"]
        out.append((text + " | " + " | ".join(extra), per_photo))
    return out


def main():
    ids = [int(x) for x in sys.argv[1:]]
    if not ids:
        raise SystemExit("Give photo numbers, for example: python compare_models.py 2 5 17")
    all_files = sorted(f for f in p.DATA.iterdir() if f.suffix.lower() in p.EXTS)
    files = [all_files[i - 1] for i in ids]  # app numbers start at 1
    images = [p.load_image(f) for f in files]

    answers = {"Salesforce BLIP": run_blip(images)}
    try:
        answers["Microsoft Florence-2"] = run_florence(images)
    except Exception as e:
        print(f"Florence-2 could not run: {e}")
    answers["Qwen2.5-VL-3B (4-bit)"] = run_qwen(files)

    for k, photo in enumerate(ids):
        print(f"\n=== Photo {photo} ===")
        for model, results in answers.items():
            text, seconds = results[k]
            print(f"- {model} ({seconds:.1f} s):\n  {p.clean_caption(text) if model != 'Qwen2.5-VL-3B (4-bit)' else text}")
    chart(answers, ids)


def chart(answers, ids):
    """Bar charts of seconds and caption length per model, saved as an image to open in VS Code."""
    names = list(answers)
    seconds = [sum(t for _, t in answers[m]) / len(ids) for m in names]
    words = [sum(len(text.split()) for text, _ in answers[m]) / len(ids) for m in names]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["#D9822B", "#7B4FB8", "#2E8B57"][: len(names)]
    a.bar(names, seconds, color=colors)
    a.set_title("Seconds per photo (lower is faster)")
    b.bar(names, words, color=colors)
    b.set_title("Words per answer (more detail, more to check)")
    for ax in (a, b):
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"Model comparison on photos {', '.join(map(str, ids))}")
    fig.tight_layout()
    out = Path(__file__).parent / "outputs" / "model_comparison.png"
    fig.savefig(out, dpi=120)
    print(f"\nChart saved: {out} (open it in VS Code)")


if __name__ == "__main__":
    main()
