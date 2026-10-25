"""Writes docs/architecture.drawio, an editable version of the architecture diagram.

Open it at app.diagrams.net or in VS Code with the Draw.io Integration extension.
"""
from pathlib import Path
from xml.sax.saxutils import escape

MODEL = "rounded=1;whiteSpace=wrap;html=1;fillColor=#eef0fb;strokeColor=#4b4fa8;fontColor=#1f2330;arcSize=12;"
APP = "rounded=1;whiteSpace=wrap;html=1;fillColor=#eaf5f1;strokeColor=#2f7d68;fontColor=#1f2330;arcSize=12;"
PLAIN = "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#c9ccd3;fontColor=#1f2330;arcSize=12;"
STORE = ("shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=10;fillColor=#fdf6e9;"
         "strokeColor=#b9770e;fontColor=#1f2330;")
ZONE = ("rounded=0;whiteSpace=wrap;html=1;fillColor=#fafafa;strokeColor=#c9ccd3;dashed=1;verticalAlign=top;"
        "align=left;spacingLeft=10;spacingTop=6;fontColor=#6b7280;fontStyle=1;fontSize=11;")
GROUP = ("rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#4b4fa8;dashed=1;verticalAlign=top;"
         "align=left;spacingLeft=8;spacingTop=4;fontColor=#4b4fa8;fontStyle=1;fontSize=11;arcSize=4;")
TEXT = "text;html=1;align=left;verticalAlign=top;whiteSpace=wrap;"
EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;endFill=1;strokeColor=#555b6b;"
        "fontColor=#6b7280;fontSize=10;fontStyle=2;labelBackgroundColor=#ffffff;")

cells = []


def label(name, sub=""):
    text = f"<b>{escape(name)}</b>"
    if sub:
        text += f"<br><font style='font-size:10px' color='#6b7280'>{escape(sub)}</font>"
    return escape(text)


def node(id_, x, y, w, h, style, value):
    cells.append(f'<mxCell id="{id_}" value="{value}" style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def edge(id_, src, dst, text=""):
    cells.append(f'<mxCell id="{id_}" value="{escape(text)}" style="{EDGE}" edge="1" parent="1" source="{src}" '
                 f'target="{dst}"><mxGeometry relative="1" as="geometry"/></mxCell>')


node("title", 20, 10, 900, 50, TEXT, escape(
    "<b style='font-size:22px'>spotlight.io · system architecture</b><br><font color='#6b7280'>"
    "Problem 2, Track B. All components run locally on one laptop (RTX 4070); no data leaves it.</font>"))

node("zone1", 10, 80, 1580, 340, ZONE, "OFFLINE INDEXING · pipeline.py, videos.py")
node("zone2", 10, 450, 1580, 330, ZONE, "ONLINE SEARCH AND REVIEW · app.py (NiceGUI + FastAPI, 127.0.0.1)")

# Offline lane
node("raw", 40, 200, 180, 90, PLAIN, label("Raw media", "photos, HEIC, .mov"))
node("prep", 280, 200, 180, 90, MODEL, label("Preprocess", "auto-rotate, frames"))
node("group", 520, 110, 460, 290, GROUP, "Describe and embed")
models = [("qwen", "Qwen2.5-VL-3B", "4-bit, caption + labels"), ("clip", "CLIP ViT-B/32", "image vector"),
          ("face", "InsightFace", "anonymous person groups"), ("whisper", "Whisper small", "speech + times"),
          ("minilm", "MiniLM-L6", "text vector"), ("events", "Event grouping", "date, clustering")]
for i, (id_, name, sub) in enumerate(models):
    node(id_, 540 + (i // 3) * 220, 145 + (i % 3) * 82, 200, 68, MODEL, label(name, sub))
node("records", 1060, 130, 200, 120, STORE, label("Record store", "photos.json"))
node("vectors", 1060, 280, 200, 110, STORE, label("Vector index", "clip / text .npy"))
node("edits", 1340, 130, 200, 120, STORE, label("Staff edits", "audit_log.jsonl"))

edge("e1", "raw", "prep")
edge("e2", "prep", "group", "frames, audio")
edge("e3", "group", "records", "records")
edge("e4", "group", "vectors", "vectors")
edge("e5", "edits", "records", "kept on re-run")

# Online lane
node("staff", 40, 590, 170, 90, PLAIN, label("DCT staff", "browser, login"))
node("encoder", 270, 590, 200, 90, APP, label("Query encoder", "CLIP + MiniLM + BM25"))
node("rank", 530, 590, 200, 90, APP, label("Hybrid ranking", "Reciprocal Rank Fusion"))
node("gallery", 800, 520, 220, 80, APP, label("Gallery + filters", "event, place, attribute"))
node("pivot", 800, 660, 220, 80, APP, label("3-way pivot", "same event, place, person"))
node("review", 1090, 590, 200, 90, APP, label("Review panel", "accept, edit, release"))
node("export", 1360, 590, 180, 90, APP, label("Export", "JPEG, ZIP, CSV"))

edge("e6", "staff", "encoder", "text query")
edge("e7", "encoder", "rank", "3 rankings")
edge("e8", "rank", "gallery")
edge("e9", "rank", "pivot")
edge("e10", "gallery", "review")
edge("e11", "pivot", "review")
edge("e12", "review", "export")
edge("e13", "vectors", "rank", "vectors")
edge("e14", "review", "edits", "edits, answers")

# Legend
for i, (style, text) in enumerate([(MODEL, "ML model / processing"), (STORE, "stored data"), (APP, "app component")]):
    node(f"lg{i}", 1080 + i * 170, 30, 22, 16, style, "")
    node(f"lt{i}", 1108 + i * 170, 26, 140, 24, TEXT, escape(f"<font color='#6b7280'>{text}</font>"))

xml = ('<mxfile host="app.diagrams.net"><diagram name="spotlight.io architecture">'
       '<mxGraphModel dx="1600" dy="820" grid="1" gridSize="10" page="1" pageWidth="1600" pageHeight="800" '
       'background="#ffffff"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
       + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
out = Path(__file__).parent / "docs" / "architecture.drawio"
out.parent.mkdir(exist_ok=True)
out.write_text(xml, encoding="utf-8")
print(f"Saved {out}")
