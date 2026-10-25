"""Writes the sample structured output for the submission: docs/sample_index.json and .csv.

One record per photo or video moment with its path, event, location, person attributes and the
other fields search uses. Many original file names contain people's names, so the path is replaced
with an anonymous one (data/raw/photo_001.jpg). The real names stay in outputs/photos.json on
this laptop.

    python make_sample_output.py          (all photos)
    python make_sample_output.py 25       (only the first 25)
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"
FIELDS = ["path", "photo", "type", "video_second", "caption", "event_id", "event_name", "activity", "location",
          "person_attributes", "objects", "moments", "person_groups", "taken", "consent", "status"]


def anonymous_path(r):
    """data/raw/photo_007.jpg, or data/videos/video_02.mp4#t=12 for a moment in a video."""
    name = r["file"]
    if name.startswith("zz_video_"):
        _, _, vid, sec = Path(name).stem.split("_")
        return f"data/videos/video_{vid[1:]}.mp4#t={int(sec)}", "video moment", int(sec)
    return f"data/raw/photo_{r['id'] + 1:03d}{Path(name).suffix.lower()}", "photo", ""


def main():
    rows = json.loads((ROOT / "outputs" / "photos.json").read_text(encoding="utf-8"))
    rows = [r for r in rows if not r.get("removed")]
    if len(sys.argv) > 1:
        rows = rows[: int(sys.argv[1])]
    records = []
    for r in rows:
        path, kind, second = anonymous_path(r)
        records.append({
            "path": path, "photo": r["id"] + 1, "type": kind, "video_second": second,
            "caption": r["caption"], "event_id": r.get("event_id", ""), "event_name": r.get("event_name", ""),
            "activity": r["activity"], "location": r["location"],
            "person_attributes": [a for a in r["person_attributes"].split(";") if a],
            "objects": [o for o in r.get("objects", "").split(";") if o],
            "moments": [m for m in r.get("moments", "").split(";") if m],
            "person_groups": [p for p in r.get("person_ids", "").split(";") if p],
            "taken": r.get("taken", ""), "consent": r.get("consent", "unknown"), "status": r.get("status", "ai"),
        })
    DOCS.mkdir(exist_ok=True)
    (DOCS / "sample_index.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    with open(DOCS / "sample_index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: ";".join(v) if isinstance(v, list) else v for k, v in rec.items()})
    print(f"Wrote {len(records)} records to docs/sample_index.json and docs/sample_index.csv")


if __name__ == "__main__":
    main()
