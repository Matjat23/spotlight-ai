"""Search engine over the records that pipeline.py writes to outputs/.

Hybrid search: CLIP (query text vs photo), MiniLM (query text vs caption text) and
BM25 keywords, merged with Reciprocal Rank Fusion. Pivots find photos from the same
event, the same location, or with the same person attributes (clothing and gear only).
Staff changes are saved to outputs/photos.json and logged to outputs/audit_log.jsonl.

    python search.py "child meeting a mascot at the theme park"
"""
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
CLIP_NAME = "openai/clip-vit-base-patch32"
TEXT_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60  # standard constant for Reciprocal Rank Fusion
EDITABLE = {"caption", "activity", "location", "person_attributes", "objects"}


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Small BM25 keyword index, enough for a few thousand photos."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query):
        words = tokenize(query)
        out = np.zeros(len(self.docs))
        for i, (d, length) in enumerate(zip(self.docs, self.lens)):
            for w in words:
                if w in d:
                    tf = d[w]
                    out[i] += self.idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg))
        return out


def search_text(r):
    attrs = r["person_attributes"].replace(";", ", ")
    objects = r.get("objects", "").replace(";", ", ")
    return f"{r['caption']}. {r['activity']} at {r['location']}. {attrs}. {objects}"


class SearchEngine:
    def __init__(self):
        self.rows = json.loads((OUT / "photos.json").read_text(encoding="utf-8"))
        self.clip_emb = np.load(OUT / "clip_embeddings.npy")
        self.text_emb = np.load(OUT / "text_embeddings.npy")
        self.attr_emb = np.load(OUT / "attr_embeddings.npy")
        self.bm25 = BM25([search_text(r) for r in self.rows])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip = CLIPModel.from_pretrained(CLIP_NAME).to(self.device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(CLIP_NAME)
        self.tok = AutoTokenizer.from_pretrained(TEXT_NAME)
        self.text_model = AutoModel.from_pretrained(TEXT_NAME).to(self.device).eval()

    # Query embeddings

    def _clip_query(self, q):
        with torch.no_grad():
            t = self.clip_proc(text=[q], return_tensors="pt", padding=True).to(self.device)
            e = self.clip.get_text_features(**t)
            if not torch.is_tensor(e):
                e = e.pooler_output
            return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()[0]

    def _text_query(self, q):
        with torch.no_grad():
            b = self.tok([q], return_tensors="pt", truncation=True).to(self.device)
            h = self.text_model(**b).last_hidden_state
            e = h.mean(dim=1)
            return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()[0]

    # Search

    def search(self, query, filters=None, top=40, use=("image", "caption", "keyword")):
        """Returns a list of (row, score, why) best first. Empty query lists everything.

        use picks which searches are combined, evaluate.py uses it to compare them."""
        start = time.time()
        ids = self._filtered(filters or {})
        if not query.strip():
            hits = [(self.rows[i], 0.0, {}) for i in ids]
            return hits[:top], time.time() - start

        lists = {
            "image": self.clip_emb @ self._clip_query(query),
            "caption": self.text_emb @ self._text_query(query),
            "keyword": self.bm25.scores(query),
        }
        fused = {}
        why = {}
        for name, scores in lists.items():
            if name not in use:
                continue
            order = [i for i in np.argsort(-scores) if i in ids]
            if name == "keyword":
                order = [i for i in order if scores[i] > 0]
            for rank, i in enumerate(order):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
                why.setdefault(i, {})[name] = rank + 1
        best = sorted(fused, key=fused.get, reverse=True)[:top]
        hits = [(self.rows[i], fused[i], why[i]) for i in best]
        return hits, time.time() - start

    def _filtered(self, filters):
        ids = set(range(len(self.rows)))
        for field in ("activity", "location", "event_id"):
            if filters.get(field):
                ids = {i for i in ids if self.rows[i][field] == filters[field]}
        if filters.get("attribute"):
            ids = {i for i in ids if filters["attribute"] in self.rows[i]["person_attributes"].split(";")}
        if filters.get("year"):
            ids = {i for i in ids if self.rows[i]["taken"].startswith(str(filters["year"]))}
        return ids

    # Pivots

    def pivot(self, photo_id, kind, top=40):
        """kind: event, location or person. Never uses faces, only stored fields and text."""
        r = self.rows[photo_id]
        if kind == "event":
            same = [x for x in self.rows if x["event_id"] == r["event_id"] and x["id"] != photo_id]
            return same[:top]
        if kind == "location":
            same = [x for x in self.rows if x["location"] == r["location"] and x["id"] != photo_id]
            # Most similar looking first, so the same room comes before other stores
            same.sort(key=lambda x: -float(self.clip_emb[x["id"]] @ self.clip_emb[photo_id]))
            return same[:top]
        if kind == "person":
            # 1. Face groups: photos that share an anonymous person group (P01...) come first
            groups = set(r.get("person_ids", "").split(";")) - {""}
            by_face = [x for x in self.rows if x["id"] != photo_id
                       and groups & set(x.get("person_ids", "").split(";"))]
            # 2. Clothing backup: at least 2 shared items, ranked by Jaccard (shared / all).
            # One shared item like "blue top" matches half the archive, so it is not enough.
            mine = set(r["person_attributes"].split(";")) - {""}
            if len(mine) < 2:
                return by_face[:top]
            seen = {x["id"] for x in by_face}
            scored = []
            for x in self.rows:
                theirs = set(x["person_attributes"].split(";")) - {""}
                shared = mine & theirs
                if x["id"] != photo_id and x["id"] not in seen and len(shared) >= 2:
                    scored.append((len(shared) / len(mine | theirs), x))
            scored.sort(key=lambda t: -t[0])
            return (by_face + [x for _, x in scored])[:top]
        raise ValueError(kind)

    # Staff review

    def facets(self):
        return {
            "activity": sorted({r["activity"] for r in self.rows}),
            "location": sorted({r["location"] for r in self.rows}),
            "attribute": sorted({a for r in self.rows for a in r["person_attributes"].split(";") if a}),
            "year": sorted({r["taken"][:4] for r in self.rows if r["taken"]}, reverse=True),
        }

    def review(self, photo_id, decision, changes=None, reviewer="staff", note=""):
        """decision: accepted or edited. Saves the record and appends to the audit log."""
        r = self.rows[photo_id]
        changes = {k: v for k, v in (changes or {}).items() if k in EDITABLE and v != r.get(k, "")}
        entry = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "photo_id": photo_id,
            "decision": "edited" if changes else decision,
            "reviewer": reviewer,
            "before": {k: r.get(k, "") for k in changes},
            "after": changes,
            "note": note,
        }
        r.update(changes)
        r["status"] = entry["decision"]
        if changes:
            self.bm25 = BM25([search_text(x) for x in self.rows])
        (OUT / "photos.json").write_text(json.dumps(self.rows, indent=1), encoding="utf-8")
        with open(OUT / "audit_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def audit_log(self, last=50):
        path = OUT / "audit_log.jsonl"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(x) for x in lines[-last:]][::-1]


if __name__ == "__main__":
    engine = SearchEngine()
    query = " ".join(sys.argv[1:]) or "child meeting a mascot at the theme park"
    hits, took = engine.search(query, top=8)
    print(f"{query!r}: {took * 1000:.0f} ms")
    for r, score, why in hits:
        print(f"  {score:.4f}  {r['caption']}  [{r['activity']} | {r['location']}]  {why}")
