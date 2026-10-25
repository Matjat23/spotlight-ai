"""Draws docs/architecture.png: how a raw photo or video becomes searchable in spotlight.io."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY, NAVY2, GOLD, LIGHT, INK, GREY = "#24214f", "#3d3878", "#b9770e", "#f7f5fb", "#1c1a3d", "#6f6a92"

fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor(LIGHT)


def box(x, y, w, h, title, lines, color=NAVY, text="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=color, ec="none"))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=12.5, weight="bold", color=text)
    ax.text(x + w / 2, y + h - 0.8, "\n".join(lines), ha="center", va="top", fontsize=9.5, color=text,
            linespacing=1.45)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, lw=2, color=GOLD))


ax.text(0.4, 8.55, "spotlight.io: from a raw photo to a searchable record", fontsize=18, weight="bold", color=INK)
ax.text(0.4, 8.1, "Everything runs on one laptop (RTX 4070). No photo, video or text leaves it.", fontsize=11,
        color=GREY)

# 1. Input
box(0.4, 4.2, 2.6, 3.2, "1. Input", ["Event photos", "(jpg, png, iPhone heic)", "", "Phone videos (.mov)", "",
                                      "Upload in the app", "or a folder"], color=NAVY2)
# 2. Prepare
box(3.5, 4.2, 2.6, 3.2, "2. Prepare", ["Turn photos upright", "(EXIF + face check)", "", "videos.py:",
                                        "browser copy, frames", "at scene changes,", "sound for Whisper"],
    color=NAVY2)
# 3. Describe and index
box(6.6, 1.5, 4.4, 5.9, "3. Describe  (pipeline.py)", [
    "Qwen2.5-VL-3B, 4-bit", "caption, activity, location, clothing,", "objects, moments as JSON", "",
    "CLIP ViT-B/32", "image vector + zero-shot labels", "",
    "MiniLM-L6", "text vector of the description", "",
    "InsightFace buffalo_l", "anonymous person groups P01, P02 (no names)", "",
    "Whisper small", "what is said in videos, with times", "",
    "Events: same day or agglomerative clustering"])
# 4. Index
box(11.5, 4.2, 4.1, 3.2, "4. Index (outputs/)", ["photos.json: one record per photo", "or video moment",
                                                  "clip / text vectors (.npy)", "faces.json (boxes only)",
                                                  "videos.json (speech + times)", "audit_log.jsonl"], color=GOLD)
# 5. Search
box(11.5, 0.4, 4.1, 3.3, "5. Search and review  (app.py)", [
    "Hybrid search: CLIP + MiniLM + BM25", "merged with Reciprocal Rank Fusion", "",
    "Pivot: same event / location / person", "Accept or edit, photo release, bin", "Staff accounts, review log",
    "NiceGUI + FastAPI on 127.0.0.1"])

arrow(3.0, 5.8, 3.5, 5.8)
arrow(6.1, 5.8, 6.6, 5.8)
arrow(11.0, 5.8, 11.5, 5.8)
arrow(13.55, 4.2, 13.55, 3.7)
# staff edits flow back into the index
ax.text(8.8, 0.8, "Staff edits, answers and photo releases\nare saved to the index and kept\nwhen pipeline.py runs again",
        ha="center", va="center", fontsize=9.5, color=INK, style="italic")

ax.text(0.4, 3.6, "Search query", fontsize=11, weight="bold", color=INK)
ax.text(0.4, 3.2, "\"child hugging a mascot\"\nbecomes a CLIP and a MiniLM vector\nand BM25 keywords, compared\n"
        "with every record in milliseconds", fontsize=9.5, color=GREY, va="top", linespacing=1.45)

out = Path(__file__).parent / "docs" / "architecture.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, bbox_inches="tight", facecolor=LIGHT)
print(f"Saved {out}")
