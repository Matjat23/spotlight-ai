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
DATA = ROOT / "data" / "raw"
REMOVED = ROOT / "data" / "removed"
CLIP_NAME = "openai/clip-vit-base-patch32"
TEXT_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60  # standard constant for Reciprocal Rank Fusion
# When is a photo a match at all (tuned on a few searches, check with the test set)
MIN_TEXT = 0.30       # caption meaning (MiniLM cosine)
TEXT_SHARE = 0.6      # and at least this share of the best caption match
KEYWORD_SHARE = 0.5   # keyword score at least this share of the best keyword match
MIN_IMAGE = 0.21      # photo content (CLIP cosine)
IMAGE_MARGIN = 0.03   # and at most this far below the best photo match
EDITABLE = {"caption", "activity", "location", "person_attributes", "objects", "moments", "consent"}


# Filler words appear in almost every caption, so they would match every photo
STOPWORDS = {"a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "with", "by", "is", "are",
             "it", "its", "this", "that", "some", "their", "his", "her", "photo", "picture", "image"}


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


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
    moments = r.get("moments", "").replace(";", ", ")
    return f"{r['caption']}. {r['activity']} at {r['location']}. {attrs}. {objects}. {moments}. {r.get('speech', '')}"


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

    def search(self, query, filters=None, top=40, use=("image", "caption", "keyword"), only_matches=True):
        """Returns a list of (row, score, why) best first. Empty query lists everything.

        use picks which searches are combined, evaluate.py uses it to compare them.
        only_matches drops photos that are not close to the search at all, otherwise every
        search returns the whole archive in a different order."""
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
        if only_matches:
            fused = {i: s for i, s in fused.items() if self._matches(i, lists, use)}
        best = sorted(fused, key=fused.get, reverse=True)[:top]
        hits = [(self.rows[i], fused[i], why[i]) for i in best]
        return hits, time.time() - start

    def _matches(self, i, lists, use):
        """A photo counts as a match if the words appear in its description, its caption means
        nearly the same as the search, or it looks almost as close as the best image match."""
        words = lists["keyword"]
        if "keyword" in use and words[i] > 0 and words[i] >= KEYWORD_SHARE * words.max():
            return True
        text = lists["caption"]
        if "caption" in use and text[i] >= max(MIN_TEXT, TEXT_SHARE * text.max()):
            return True
        image = lists["image"]
        return "image" in use and image[i] >= max(MIN_IMAGE, image.max() - IMAGE_MARGIN)

    def _filtered(self, filters):
        ids = {i for i, r in enumerate(self.rows) if not r.get("removed")}
        for field in ("activity", "location", "event_id"):
            if filters.get(field):
                ids = {i for i in ids if self.rows[i][field] == filters[field]}
        if filters.get("attribute"):
            ids = {i for i in ids if filters["attribute"] in self.rows[i]["person_attributes"].split(";")}
        if filters.get("person"):
            ids = {i for i in ids if filters["person"] in self.rows[i].get("person_ids", "").split(";")}
        if filters.get("moment"):
            ids = {i for i in ids if filters["moment"] in self.rows[i].get("moments", "").split(";")}
        if filters.get("consent"):
            ids = {i for i in ids if self.rows[i].get("consent", "unknown") == filters["consent"]}
        if filters.get("ids") is not None:
            ids &= set(filters["ids"])
        if filters.get("year"):
            ids = {i for i in ids if self.rows[i]["taken"].startswith(str(filters["year"]))}
        return ids

    # Pivots

    def pivot(self, photo_id, kind, top=40):
        """kind: event, location or person. Never uses faces, only stored fields and text."""
        r = self.rows[photo_id]
        rows = [x for x in self.rows if not x.get("removed")]
        if kind == "event":
            same = [x for x in rows if x["event_id"] == r["event_id"] and x["id"] != photo_id]
            return same[:top]
        if kind == "location":
            same = [x for x in rows if x["location"] == r["location"] and x["id"] != photo_id]
            # Most similar looking first, so the same room comes before other stores
            same.sort(key=lambda x: -float(self.clip_emb[x["id"]] @ self.clip_emb[photo_id]))
            return same[:top]
        if kind == "person":
            # 1. Face groups: photos that share an anonymous person group (P01...) come first
            groups = set(r.get("person_ids", "").split(";")) - {""}
            by_face = [x for x in rows if x["id"] != photo_id
                       and groups & set(x.get("person_ids", "").split(";"))]
            # 2. Clothing backup: at least 2 shared items, ranked by Jaccard (shared / all).
            # One shared item like "blue top" matches half the archive, so it is not enough.
            """mine = set(r["person_attributes"].split(";")) - {""}
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
            return (by_face + [x for _, x in scored])[:top]"""
            return by_face[:top]
        raise ValueError(kind)

    # Staff review

    def facets(self):
        rows = [r for r in self.rows if not r.get("removed")]
        return {
            "activity": sorted({r["activity"] for r in rows}),
            "location": sorted({r["location"] for r in rows}),
            "attribute": sorted({a for r in rows for a in r["person_attributes"].split(";") if a}),
            "moment": sorted({m for r in rows for m in r.get("moments", "").split(";") if m}),
            "year": sorted({r["taken"][:4] for r in rows if r["taken"]}, reverse=True),
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
        self._save_rows()
        self._log(entry)
        return entry

    # Adding and removing photos

    def add(self, rows, clip_emb, text_emb, attr_emb, reviewer="staff"):
        """Appends newly processed photos. Same day as a known event = that event, else a new one."""
        days = {x["taken"][:10]: x for x in self.rows if x["taken"]}
        next_event = max(int(x["event_id"][1:]) for x in self.rows) + 1 if self.rows else 1
        for r in rows:
            known = days.get(r["taken"][:10]) if r["taken"] else None
            if known:
                r["event_id"], r["event_name"] = known["event_id"], known["event_name"]
            else:
                r["event_id"], r["event_name"] = f"E{next_event:02d}", r["activity"]
                next_event += 1
                if r["taken"]:
                    days[r["taken"][:10]] = r
            r.setdefault("person_ids", "")
            self.rows.append(r)
            self._log({"time": datetime.now().isoformat(timespec="seconds"), "photo_id": r["id"],
                       "decision": "added", "reviewer": reviewer, "before": {}, "after": {}, "note": ""})
        self.clip_emb = np.vstack([self.clip_emb, clip_emb])
        self.text_emb = np.vstack([self.text_emb, text_emb])
        self.attr_emb = np.vstack([self.attr_emb, attr_emb])
        np.save(OUT / "clip_embeddings.npy", self.clip_emb)
        np.save(OUT / "text_embeddings.npy", self.text_emb)
        np.save(OUT / "attr_embeddings.npy", self.attr_emb)
        self.bm25 = BM25([search_text(x) for x in self.rows])
        self._save_rows()

    def remove(self, photo_id, reviewer="staff", note=""):
        """Hides a photo everywhere and moves the file to data/removed. The record keeps its
        number, so every other photo number (and the test set) stays the same."""
        r = self.rows[photo_id]
        src = DATA / r["file"]
        if src.exists():
            REMOVED.mkdir(parents=True, exist_ok=True)
            src.rename(REMOVED / r["file"])
        r["removed"] = True
        r["removed_at"] = datetime.now().isoformat(timespec="seconds")
        self._save_rows()
        entry = {"time": datetime.now().isoformat(timespec="seconds"), "photo_id": photo_id,
                 "decision": "moved to bin", "reviewer": reviewer, "before": {}, "after": {}, "note": note}
        self._log(entry)
        return entry

    # The bin: removed photos wait in data/removed until they are restored or deleted for good

    def bin(self):
        return [r for r in self.rows if r.get("removed") and not r.get("deleted")]

    def restore(self, photo_id, reviewer="staff"):
        r = self.rows[photo_id]
        src = REMOVED / r["file"]
        if src.exists():
            src.rename(DATA / r["file"])
        r.pop("removed", None)
        r.pop("removed_at", None)
        self._save_rows()
        self._log({"time": datetime.now().isoformat(timespec="seconds"), "photo_id": photo_id,
                   "decision": "restored", "reviewer": reviewer, "before": {}, "after": {}, "note": ""})

    def delete_forever(self, photo_id, reviewer="staff", note=""):
        """Deletes the file for good. The record stays (marked deleted) so photo numbers never shift."""
        r = self.rows[photo_id]
        (REMOVED / r["file"]).unlink(missing_ok=True)
        r["deleted"] = True
        self._save_rows()
        self._log({"time": datetime.now().isoformat(timespec="seconds"), "photo_id": photo_id,
                   "decision": "deleted forever", "reviewer": reviewer, "before": {}, "after": {}, "note": note})

    def empty_old(self, days, reviewer="automatic"):
        """Deletes bin photos older than the number of days the admin chose. Returns how many."""
        cutoff = datetime.now().timestamp() - days * 86400
        old = [r for r in self.bin() if datetime.fromisoformat(r.get("removed_at", "1970-01-01")).timestamp() < cutoff]
        for r in old:
            self.delete_forever(r["id"], reviewer, f"older than {days} days")
        return len(old)

    def _save_rows(self):
        (OUT / "photos.json").write_text(json.dumps(self.rows, indent=1), encoding="utf-8")

    def _log(self, entry):
        with open(OUT / "audit_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

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
