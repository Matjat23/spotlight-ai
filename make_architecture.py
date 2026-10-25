"""Draws docs/architecture.png: how a raw photo or video becomes searchable in spotlight.io.

Two lanes: offline indexing (pipeline.py, runs once per batch) and online search (app.py).
Boxes are processing steps, cylinders are stored data, arrow labels say what flows.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

INK, MUTED, LINE = "#1f2330", "#6b7280", "#c9ccd3"
MODEL, STORE, UI, ZONE = "#eef0fb", "#fdf6e9", "#eaf5f1", "#fafafa"
ACCENT_M, ACCENT_S, ACCENT_U = "#4b4fa8", "#b9770e", "#2f7d68"
FONT = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(16, 8.6), dpi=160)
ax.set_xlim(0, 160)
ax.set_ylim(0, 86)
ax.axis("off")
fig.patch.set_facecolor("white")
plt.rcParams["font.family"] = FONT


def zone(x, y, w, h, title):
    ax.add_patch(Rectangle((x, y), w, h, fc=ZONE, ec=LINE, lw=1, ls=(0, (4, 3))))
    ax.text(x + 1.5, y + h - 2.2, title.upper(), fontsize=8.5, color=MUTED, weight="bold", va="top")


def comp(x, y, w, h, name, sub, fill, accent):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fill, ec=accent, lw=1.2))
    ax.text(x + w / 2, y + h / 2 + (1.6 if sub else 0), name, ha="center", va="center", fontsize=10.5,
            color=INK, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.2, sub, ha="center", va="center", fontsize=8.3, color=MUTED)


def store(x, y, w, h, name, sub):
    e = 2.2
    ax.add_patch(Rectangle((x, y + e / 2), w, h - e, fc=STORE, ec="none"))
    ax.add_patch(Ellipse((x + w / 2, y + e / 2), w, e, fc=STORE, ec=ACCENT_S, lw=1.2))
    ax.plot([x, x], [y + e / 2, y + h - e / 2], color=ACCENT_S, lw=1.2)
    ax.plot([x + w, x + w], [y + e / 2, y + h - e / 2], color=ACCENT_S, lw=1.2)
    ax.add_patch(Ellipse((x + w / 2, y + h - e / 2), w, e, fc=STORE, ec=ACCENT_S, lw=1.2))
    ax.text(x + w / 2, y + h / 2 + 0.6, name, ha="center", va="center", fontsize=10, color=INK, weight="bold")
    ax.text(x + w / 2, y + h / 2 - 2.6, sub, ha="center", va="center", fontsize=8, color=MUTED)


def flow(p1, p2, label="", dx=0, dy=1.2, rad=0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11, lw=1.1, color="#555b6b",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))
    if label:
        ax.text((p1[0] + p2[0]) / 2 + dx, (p1[1] + p2[1]) / 2 + dy, label, ha="center", va="bottom",
                fontsize=7.8, color=MUTED, style="italic",
                bbox=dict(fc="white", ec="none", pad=0.6))


# Title
ax.text(2, 83, "spotlight.io  ·  system architecture", fontsize=15, weight="bold", color=INK, va="top")
ax.text(2, 79, "Problem 2, Track B. All components run locally on one laptop (RTX 4070); no data leaves it.",
        fontsize=9.5, color=MUTED, va="top")

# Lanes
zone(1, 40, 158, 34, "Offline indexing  ·  pipeline.py, videos.py")
zone(1, 3, 158, 33, "Online search and review  ·  app.py (NiceGUI + FastAPI, 127.0.0.1)")

# Offline lane
comp(4, 51, 19, 13, "Raw media", "photos, HEIC, .mov", "white", LINE)
comp(29, 51, 19, 13, "Preprocess", "auto-rotate, frames", MODEL, ACCENT_M)
# the models sit in one group: each reads the prepared media, none feeds another
ax.add_patch(FancyBboxPatch((54, 42), 46, 28, boxstyle="round,pad=0,rounding_size=1.5", fc="white", ec=ACCENT_M,
                            lw=1, ls=(0, (3, 2))))
ax.text(56, 68.4, "Describe and embed", fontsize=8.5, color=ACCENT_M, weight="bold", va="top")
for i, (name, sub) in enumerate([("Qwen2.5-VL-3B", "caption + labels"), ("CLIP ViT-B/32", "image vector"),
                                 ("InsightFace", "anon. person groups")]):
    comp(56, 57.5 - i * 7.5, 20, 6.5, name, sub, MODEL, ACCENT_M)
for i, (name, sub) in enumerate([("Whisper small", "speech + times"), ("MiniLM-L6", "text vector"),
                                 ("Event grouping", "date, clustering")]):
    comp(78, 57.5 - i * 7.5, 20, 6.5, name, sub, MODEL, ACCENT_M)

store(108, 56, 22, 13, "Record store", "photos.json")
store(108, 42, 22, 12, "Vector index", "clip / text .npy")
store(136, 56, 21, 13, "Staff edits", "audit_log.jsonl")

flow((23, 57.5), (29, 57.5))
flow((48, 57.5), (54, 57.5), "frames, audio", dy=1.4)
flow((100, 62), (108, 62), "records", dy=1.2)
flow((100, 48), (108, 48), "vectors", dy=1.2)
flow((136, 62.5), (130, 62.5), "kept on re-run", dy=1.6)

# Online lane
comp(4, 14, 17, 11, "DCT staff", "browser · login", "white", LINE)
comp(28, 14, 20, 11, "Query encoder", "CLIP + MiniLM + BM25", UI, ACCENT_U)
comp(56, 14, 20, 11, "Hybrid ranking", "Reciprocal Rank Fusion", UI, ACCENT_U)
comp(84, 21, 22, 9, "Gallery + filters", "event · place · attribute", UI, ACCENT_U)
comp(84, 8, 22, 9, "3-way pivot", "same event · place · person", UI, ACCENT_U)
comp(114, 14, 20, 11, "Review panel", "accept · edit · release", UI, ACCENT_U)
comp(140, 14, 17, 11, "Export", "JPEG · ZIP · CSV", UI, ACCENT_U)

flow((21, 19.5), (28, 19.5), "text query")
flow((48, 19.5), (56, 19.5), "3 rankings")
flow((76, 21), (84, 25), "")
flow((76, 18), (84, 12.5), "")
flow((106, 25), (114, 21), "")
flow((106, 12.5), (114, 17), "")
flow((134, 19.5), (140, 19.5))
# lanes connect through the stores
flow((119, 42), (68, 25), "vectors", dx=8, dy=-2, rad=0.1)
flow((126, 25), (146, 56), "edits, answers", dx=7, dy=0, rad=0.15)

# Legend
lx, ly = 104, 80
for i, (fill, ec, label) in enumerate([(MODEL, ACCENT_M, "ML model / processing"),
                                       (STORE, ACCENT_S, "stored data"),
                                       (UI, ACCENT_U, "app component")]):
    ax.add_patch(FancyBboxPatch((lx + i * 18, ly - 2.4), 3, 2.2, boxstyle="round,pad=0,rounding_size=0.5",
                                fc=fill, ec=ec, lw=1))
    ax.text(lx + i * 18 + 4, ly - 1.3, label, fontsize=7.8, color=MUTED, va="center")

out = Path(__file__).parent / "docs" / "architecture.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
