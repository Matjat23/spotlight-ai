"""spotlight.io: photo search app for Dreams Come True staff.

NiceGUI for the screens, FastAPI (the same server) for the REST API under /api.
Runs only on this computer (127.0.0.1), photos never leave the laptop.

    python pipeline.py      # once, writes outputs/
    python app.py           # then open http://127.0.0.1:8080
"""
import csv
import inspect
import io
import json
import re
import zipfile
import time
from datetime import datetime
from urllib.parse import quote
from pathlib import Path

from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from nicegui import app, run, ui
from pillow_heif import register_heif_opener

import accounts
import applog
import pipeline
from search import SearchEngine

register_heif_opener()

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "raw"
THUMBS = ROOT / "outputs" / "thumbs"
THUMBS.mkdir(parents=True, exist_ok=True)

app.add_static_files("/static", ROOT / "static")  # logo
engine = SearchEngine()
# Any error in a page is written to logs/app.log with its traceback, not only to the terminal
app.on_exception(lambda e: applog.log.error("unhandled_error", exc_info=e, extra={"data": {}}))
FACES_FILE = ROOT / "outputs" / "faces.json"
FACES = json.loads(FACES_FILE.read_text(encoding="utf-8")) if FACES_FILE.exists() else []


def focus(photo_id):
    """Crop point for a thumbnail, as a CSS object-position. Thumbnails are cropped to fill the card,
    so this centres the crop on the faces found in the photo. Photos without faces keep the middle."""
    boxes = [f["box"] for f in FACES if f["photo"] == photo_id and f.get("score", 1) > 0.6]
    if not boxes:
        return "50% 50%"
    x = (min(b[0] for b in boxes) + max(b[2] for b in boxes)) / 2
    y = (min(b[1] for b in boxes) + max(b[3] for b in boxes)) / 2
    return f"{x * 100:.0f}% {min(y * 100 + 10, 100):.0f}%"  # a bit below the faces so heads are not cut
# Favorites only live in memory for now (the look is done, saving comes later)
FAVORITES = set()


# REST API (FastAPI)

REMOVED = ROOT / "data" / "removed"
SETTINGS = ROOT / "outputs" / "settings.json"
BIN_CHOICES = {7: "after 7 days", 15: "after 15 days", 30: "after 30 days", 60: "after 60 days", 0: "never (admin empties it)"}


def photo_path(r):
    """Where the photo file is now: the archive, or the bin (data/removed)."""
    return DATA / r["file"] if (DATA / r["file"]).exists() else REMOVED / r["file"]


def settings():
    return json.loads(SETTINGS.read_text(encoding="utf-8")) if SETTINGS.exists() else {"bin_days": 30}


def save_settings(**changes):
    SETTINGS.write_text(json.dumps({**settings(), **changes}, indent=1), encoding="utf-8")


def empty_old_bin():
    """Deletes bin photos older than the admin's choice. Runs at start and when the bin is opened."""
    undated = [r for r in engine.bin() if "removed_at" not in r]
    for r in undated:  # removed before the bin existed, their days start counting now
        r["removed_at"] = datetime.now().isoformat(timespec="seconds")
    if undated:
        engine._save_rows()
    days = settings().get("bin_days", 30)
    if days:
        n = engine.empty_old(days)
        if n:
            applog.info("bin_emptied_automatically", photos=n, days=days)

VIDEO_DIR = ROOT / "data" / "videos" / "web"
VIDEO_INFO = ROOT / "outputs" / "videos.json"
VIDEOS = json.loads(VIDEO_INFO.read_text(encoding="utf-8")) if VIDEO_INFO.exists() else {}
FRAME_NAME = re.compile(r"^zz_video_(V[0-9]+)_([0-9]+)[.]jpg$")


def video_of(r):
    """(video id, second) when this row is a moment from a video (made by videos.py), else None."""
    m = FRAME_NAME.match(r["file"])
    return (m.group(1), int(m.group(2))) if m and m.group(1) in VIDEOS else None


def clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def item_name(r):
    """Photo 12, or Video 3 at 0:17 for a moment in a video."""
    v = video_of(r)
    if not v:
        return f"Photo {r['id'] + 1}"
    return f"Video {int(v[0][1:])}" + (f" at {clock(v[1])}" if v[1] else "")


def one_per_video(rows):
    """Like Google Photos: a video shows once, at its best matching moment (the first one in the list)."""
    seen, out = set(), []
    for r in rows:
        v = video_of(r)
        if v and v[0] in seen:
            continue
        if v:
            seen.add(v[0])
        out.append(r)
    return out


def video_moments(vid):
    rows = [r for r in engine.rows if not r.get("removed") and (video_of(r) or ("",))[0] == vid]
    return sorted(rows, key=lambda r: video_of(r)[1])


@app.get("/video/{vid}")
def video_file(vid: str):
    """The browser-friendly copy. FileResponse answers range requests, so the player can jump to a moment."""
    if vid not in VIDEOS:
        raise HTTPException(404, "No video with that id")
    return FileResponse(VIDEO_DIR / VIDEOS[vid]["web"], media_type="video/mp4",
                        filename=f"spotlight-video-{int(vid[1:])}.mp4", content_disposition_type="inline")


def photo_or_404(photo_id):
    if not 0 <= photo_id < len(engine.rows):
        raise HTTPException(404, "No photo with that id")
    return engine.rows[photo_id]


@app.get("/photo/{photo_id}/{size}")
def photo_file(photo_id: int, size: str):
    """JPEG copy of a photo, made once and cached. size: thumb or large."""
    r = photo_or_404(photo_id)
    side = 400 if size == "thumb" else 1400
    turn = pipeline.rotations().get(r["file"], 0)
    path = THUMBS / f"{photo_id}_{side}_{turn}.jpg"  # a new file after a rotation
    if not path.exists():
        img = pipeline.open_upright(photo_path(r))
        img.thumbnail((side, side))
        img.save(path, quality=85)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/face/{person}")
def face_cover(person: str):
    """Round cover for the People page: the clearest face of that person group, cut on the fly.
    Nothing is written to disk."""
    faces = [f for f in FACES if f["person"] == person and not engine.rows[f["photo"]].get("removed")]
    if not faces:
        raise HTTPException(404, "No face for that person group")
    best = max(faces, key=lambda f: f["score"] * (f["box"][2] - f["box"][0]))
    img = pipeline.open_upright(DATA / engine.rows[best["photo"]]["file"])
    w, h = img.size
    x1, y1, x2, y2 = best["box"]
    cx, cy = (x1 + x2) / 2 * w, (y1 + y2) / 2 * h
    side = max((x2 - x1) * w, (y2 - y1) * h) * 1.6
    crop = img.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))
    crop.thumbnail((240, 240))
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=85)
    return Response(buf.getvalue(), media_type="image/jpeg")


@app.get("/api/search")
def api_search(q: str = "", activity: str = "", location: str = "", attribute: str = "", year: str = "",
               person: str = "", top: int = 40):
    filters = {"activity": activity, "location": location, "attribute": attribute, "year": year, "person": person}
    hits, took = engine.search(q, filters, top)
    return {"query": q, "ms": round(took * 1000, 1),
            "results": [{**public(r), "score": round(s, 4), "ranks": why} for r, s, why in hits]}


@app.get("/api/photos/{photo_id}")
def api_photo(photo_id: int):
    return public(photo_or_404(photo_id))


@app.get("/api/photos/{photo_id}/pivot/{kind}")
def api_pivot(photo_id: int, kind: str):
    photo_or_404(photo_id)
    if kind not in {"event", "location", "person"}:
        raise HTTPException(400, "kind must be event, location or person")
    return [public(r) for r in engine.pivot(photo_id, kind)]


@app.post("/api/photos/{photo_id}/review")
def api_review(photo_id: int, body: dict):
    photo_or_404(photo_id)
    return engine.review(photo_id, body.get("decision", "accepted"), body.get("changes"), body.get("reviewer", "staff"), body.get("note", ""))


@app.get("/api/audit")
def api_audit(last: int = 50):
    return engine.audit_log(last)


@app.get("/api/export.csv")
def api_export():
    """The structured output file: one row per photo with event, location and attributes."""
    buf = io.StringIO()
    fields = ["id", "file", "caption", "activity", "location", "person_attributes", "objects", "moments", "person_ids", "event_id", "event_name", "taken", "consent", "status"]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(engine.rows)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=spotlight_index.csv"})


CONSENT = ["unknown", "internal only", "OK for marketing"]


def consent_of(r):
    return r.get("consent", "unknown")


def jpeg_bytes(r, size):
    """The photo as a JPEG: original size, or web size (max 1600 px) for newsletters and social media.
    iPhone HEIC photos are converted, so every download opens everywhere."""
    img = pipeline.open_upright(photo_path(r))
    if size == "web":
        img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90 if size == "web" else 95)
    return buf.getvalue()


def log_download(ids, size, by):
    applog.info("download", photos=len(ids), size=size)
    for i in ids:
        engine._log({"time": datetime.now().isoformat(timespec="seconds"), "photo_id": i, "decision": "downloaded",
                     "reviewer": by or "staff", "before": {}, "after": {}, "note": f"{size}, consent: {consent_of(engine.rows[i])}"})


@app.get("/download/{photo_id}")
def download_one(photo_id: int, size: str = "original", by: str = "staff"):
    """One photo. The file name is the photo number, original file names can contain children's names."""
    r = photo_or_404(photo_id)
    log_download([photo_id], size, by)
    return Response(jpeg_bytes(r, size), media_type="image/jpeg", headers={
        "Content-Disposition": f"attachment; filename=spotlight_photo_{photo_id + 1}_{size}.jpg"})


@app.get("/download-zip")
def download_zip(ids: str, size: str = "web", by: str = "staff"):
    """Several photos in one ZIP, with captions.csv (caption, event, location, consent) for each photo."""
    chosen = [int(x) for x in ids.split(",") if x.strip().isdigit()]
    chosen = [i for i in chosen if 0 <= i < len(engine.rows) and not engine.rows[i].get("removed")]
    if not chosen:
        raise HTTPException(400, "No photos selected")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        table = io.StringIO()
        writer = csv.writer(table)
        writer.writerow(["file", "photo", "caption", "activity", "location", "consent"])
        for i in chosen:
            r = engine.rows[i]
            name = f"spotlight_photo_{i + 1}.jpg"
            z.writestr(name, jpeg_bytes(r, size))
            writer.writerow([name, i + 1, r["caption"], r["activity"], r["location"], consent_of(r)])
        z.writestr("captions.csv", table.getvalue())
    log_download(chosen, size, by)
    return Response(buf.getvalue(), media_type="application/zip", headers={
        "Content-Disposition": f"attachment; filename=spotlight_{len(chosen)}_photos.zip"})


def public(r):
    """Record without the file name, file names can contain children's names."""
    return {k: v for k, v in r.items() if k != "file"}


# Screens (NiceGUI)

# Photos are numbered from 1 for people. Inside the code (and the /api) they count from 0.
STATUS_COLOR = {"ai": "grey", "accepted": "positive", "edited": "primary"}
PAGE_SIZE = 24
NAV = [("Library", [("Photos", "photo_library", "/"), ("Videos", "smart_display", "/?kind=video"),
                    ("People", "face", "/people"),
                    ("Places", "place", "/places"), ("Favorites", "favorite", "/favorites")]),
       ("Manage", [("Add photos", "add_photo_alternate", "/add"), ("Duplicates", "content_copy", "/duplicates"),
                   ("Bin", "delete", "/bin"), ("System", "monitor_heart", "/system"),
                   ("Accounts", "manage_accounts", "/accounts")])]
ADMIN_PAGES = {"/duplicates", "/bin", "/system", "/accounts"}
# The sidebar has the same dark blue as the header, in light and dark mode
CSS = """
body.body--light { background: #f6f5fa; }
body.body--dark { background: #1b1838; }  /* deep blue, same family as the header, not black */
body.body--dark .q-card, body.body--dark .q-menu, body.body--dark .q-dialog .q-card { background: #26224a !important; color: #ecebf6; }
body.body--dark .text-grey-7, body.body--dark .text-grey-8 { color: #b3afd6 !important; }
body.body--dark .q-field__native, body.body--dark .q-field__label { color: #ecebf6 !important; }
body.body--dark .q-table__card, body.body--dark .q-table tbody td { background: #26224a; }
.q-drawer--right { background: transparent !important; }
.q-drawer--left, .q-drawer--left .q-drawer__content { background: #24214f !important; }
.q-drawer--left .q-item { color: #ffffff; }
.spot-header { height: 64px; }
.spot-brand { cursor: pointer; width: 176px; }
.spot-wordmark { font-size: 21px; font-weight: 700; letter-spacing: -.2px; line-height: 1; }
/* No grey scrollbar strip next to the sidebar */
.q-drawer--left .q-scrollarea__bar, .q-drawer--left .q-scrollarea__thumb { display: none !important; }
.q-drawer--left ::-webkit-scrollbar, .q-drawer--left::-webkit-scrollbar { width: 0; height: 0; }
.q-drawer--left .q-drawer__content { scrollbar-width: none; }
.spot-nav-group { color: #8f8bbd; font-size: 11px; letter-spacing: .12em; padding: 18px 20px 6px; }
.spot-nav-item { margin: 2px 10px; border-radius: 10px; min-height: 40px !important; padding: 6px 12px !important;
                 transition: background .15s; }
.spot-nav-item .q-item__section--avatar { min-width: 36px !important; }
.spot-nav-item:hover { background: rgba(255, 255, 255, .07); }
.spot-nav-active { background: rgba(242, 192, 99, .16) !important; font-weight: 600;
                   box-shadow: inset 3px 0 0 #f2c063; }
.spot-nav-footer { margin: 12px; padding: 12px; border-radius: 10px; font-size: 12px; color: #e6e3ff;
                   background: rgba(255, 255, 255, .05); }
body.body--light .spot-panel { background: #ffffff !important; }
body.body--dark .spot-panel { background: #221e42 !important; }
.spot-caption { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
                overflow: hidden; min-height: 2.9em; }
body.body--light .spot-star { background: rgba(255, 255, 255, .9); }
body.body--dark .spot-star { background: rgba(27, 24, 56, .85); }
"""
UPLOAD_PREFIX = "~upload_"  # "~" sorts after every other name, so existing photo numbers never shift
GRID = "display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px"


def confirm(question, action):
    """Small yes/no box so one accidental click does not change a record."""
    with ui.dialog() as dlg, ui.card():
        ui.label(question)
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=dlg.close).props("flat")
            ui.button("Yes", on_click=lambda: (dlg.close(), action())).props("unelevated color=primary")
    dlg.open()


def who():
    """The logged in account. Goes into every review log entry."""
    return app.storage.user.get("username", "unknown")


def is_admin():
    return app.storage.user.get("role") == "admin"


def admin_only():
    """Sends staff without admin rights back to the photos."""
    if not is_admin():
        ui.navigate.to("/")
        return True
    return False


def logout():
    applog.info("logout", user=who())
    app.storage.user.update(authenticated=False, username=None, role=None)
    ui.navigate.to("/login")


def layout(active):
    """Header and left sidebar shared by every page, like Google Photos."""
    ui.colors(primary="#24214f", secondary="#b9770e", accent="#b9770e")
    ui.add_css(CSS)
    dark = ui.dark_mode(app.storage.general.get("dark", False))

    def toggle_dark():
        dark.value = not dark.value
        app.storage.general["dark"] = dark.value
        mode_button.props(f"icon={'light_mode' if dark.value else 'dark_mode'}")

    with ui.left_drawer(value=True, fixed=True).props("width=230").classes("q-pt-sm spot-nav column no-wrap") as nav:
        for group, items in NAV:
            ui.label(group.upper()).classes("spot-nav-group")
            for name, icon, path in items:
                if path in ADMIN_PAGES and not is_admin():
                    continue
                selected = path == active
                with ui.item(on_click=lambda path=path: ui.navigate.to(path)).props("dense").classes(
                        "spot-nav-item" + (" spot-nav-active" if selected else "")):
                    with ui.item_section().props("avatar"):
                        ui.icon(icon, size="20px", color="amber-5" if selected else "grey-5")
                    with ui.item_section():
                        ui.label(name)
                    n = duplicate_count() if path == "/duplicates" else len(engine.bin()) if path == "/bin" else 0
                    if n:
                        with ui.item_section().props("side"):
                            ui.badge(str(n)).props("color=amber-8 rounded")
        ui.space()
        visible = sum(1 for r in engine.rows if not r.get("removed"))
        with ui.column().classes("spot-nav-footer gap-1"):
            with ui.row().classes("items-center gap-2 no-wrap"):
                ui.icon("lock", size="16px", color="amber-5")
                ui.label("Private · on this laptop")
            ui.label(f"{visible} photos · no cloud · no names").classes("text-grey-5")
    # Compact header like Google Photos: 64 px, logo the same height as the name, all on one line
    with ui.header().classes("items-center bg-primary q-px-md no-wrap spot-header"):
        ui.button(icon="menu", on_click=nav.toggle).props("flat round dense color=white")
        with ui.row().classes("items-center no-wrap gap-2 spot-brand").on("click", lambda: ui.navigate.to("/")):
            ui.image("/static/logo.svg").style("width:34px;height:32px").props("no-spinner")
            with ui.row().classes("items-baseline no-wrap gap-0"):
                ui.label("spotlight").classes("spot-wordmark")
                ui.label(".io").classes("spot-wordmark").style("color:#f2c063")
        ui.label("Photo search for Dreams Come True").classes("text-caption gt-md").style("opacity:.65")
        ui.space()
        mode_button = ui.button(icon="light_mode" if dark.value else "dark_mode", on_click=toggle_dark).props(
            "flat round color=white").tooltip("Light / dark mode")
        # Account menu top right, like Google and GitHub: click the avatar for name, export and log out
        with ui.button().props("flat round dense").classes("q-ml-sm").tooltip(who()):
            ui.avatar(who()[:1].upper(), color="amber-8", text_color="white", size="38px")
            with ui.menu().props("anchor='bottom right' self='top right'").classes("q-pa-sm"):
                with ui.row().classes("items-center no-wrap q-pa-sm gap-3"):
                    ui.avatar(who()[:1].upper(), color="amber-8", text_color="white", size="44px")
                    with ui.column().classes("gap-0"):
                        ui.label(who()).classes("text-subtitle1 text-weight-bold")
                        ui.label(app.storage.user.get("role", "")).classes("text-caption text-grey-7")
                ui.separator()
                ui.menu_item("Export data (CSV)", on_click=lambda: ui.download("/api/export.csv"))
                if is_admin():
                    ui.menu_item("Manage accounts", on_click=lambda: ui.navigate.to("/accounts"))
                ui.separator()
                ui.menu_item("Log out", on_click=logout).classes("text-negative")


def heart_button(photo_id):
    """Favorite heart. Only kept in memory until the app restarts, saving comes later."""
    def toggle():
        FAVORITES.symmetric_difference_update({photo_id})
        btn.props(f"icon={'favorite' if photo_id in FAVORITES else 'favorite_border'}")
    btn = ui.button(icon="favorite" if photo_id in FAVORITES else "favorite_border").props("flat round dense size=sm color=pink-6")
    btn.on("click.stop", toggle)
    return btn


def select_button(photo_id, selection, on_change):
    """Tick box on a photo card, to download several photos at once."""
    def toggle():
        selection.symmetric_difference_update({photo_id})
        btn.props(f"icon={'check_box' if photo_id in selection else 'check_box_outline_blank'}")
        on_change()
    btn = ui.button(icon="check_box" if photo_id in selection else "check_box_outline_blank").props(
        "flat round dense size=sm color=primary").tooltip("Select for download")
    btn.on("click.stop", toggle)


def photo_card(r, on_open, selection=None, on_select=None):
    copies = duplicates()["copies"].get(r["id"], [])
    with ui.card().tight().classes("cursor-pointer relative full-height").on("click", lambda r=r: on_open(r["id"])):
        ui.image(f"/photo/{r['id']}/thumb").classes("h-44").props(f"fit=cover position='{focus(r['id'])}'")
        video = video_of(r)
        if video:  # play button and length, like Google Photos
            ui.icon("play_circle", size="44px", color="white").classes("absolute").style(
                "top:66px;left:50%;transform:translateX(-50%);opacity:.9;filter:drop-shadow(0 1px 4px rgba(0,0,0,.6))")
            ui.badge(clock(VIDEOS[video[0]]["duration"])).props("color=grey-10").classes("absolute").style(
                "top:146px;right:6px;opacity:.85")
        if copies:
            ui.badge(f"{len(copies) + 1} copies").props("color=grey-9").classes("absolute q-ma-xs").style(
                "top:150px;left:4px").tooltip("The same photo is in the archive more than once (for example "
                                              f"iPhone and JPG). Also photo {', '.join(str(c + 1) for c in copies)}")
        with ui.element("div").classes("absolute-top-right q-ma-xs rounded-full spot-star"):
            heart_button(r["id"])
        if selection is not None and not video:
            with ui.element("div").classes("absolute-top-left q-ma-xs rounded-full spot-star"):
                select_button(r["id"], selection, on_select)
        with ui.card_section().classes("q-pa-sm"):
            with ui.row().classes("items-center gap-1"):
                ui.label(item_name(r)).classes("text-caption text-grey-7")
                if consent_of(r) == "OK for marketing":
                    ui.icon("verified", color="positive", size="xs").tooltip("Photo release signed: OK for marketing")
            ui.label(r["caption"]).classes("text-body2 spot-caption").tooltip(r["caption"])


SAME_PHOTO = 0.985  # CLIP similarity above which two photos are copies of each other
SIMILAR = 0.93      # above this two photos are very similar (same moment, scanned twice), staff decide
NOT_DUPES = ROOT / "outputs" / "not_duplicates.json"  # pairs staff marked as "not duplicates"
_dupes = {"key": None, "copies": {}, "hidden": set(), "similar": []}


def not_duplicates():
    return {tuple(x) for x in json.loads(NOT_DUPES.read_text(encoding="utf-8"))} if NOT_DUPES.exists() else set()


def _groups(ids, linked):
    """Joins photos into groups: a and b in one group when linked(a, b), also through other photos."""
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            if linked(ids[x], ids[y]):
                parent[find(ids[y])] = find(ids[x])
    groups = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def duplicates():
    """Exact copies (same file name with another extension, like an iPhone .heic and its .jpg, or a
    practically identical photo) and very similar photos. One card per copy group is shown in the grid,
    the other copies are hidden. Pairs marked as not duplicates are never grouped again."""
    rows = [r for r in engine.rows if not r.get("removed") and not video_of(r)]
    skip = not_duplicates()
    key = (len(rows), len(skip))
    if _dupes["key"] == key:
        return _dupes
    sims = engine.clip_emb @ engine.clip_emb.T
    stem = {r["id"]: Path(r["file"]).stem.lower() for r in rows}

    def allowed(a, b):
        return (min(a, b), max(a, b)) not in skip

    exact = _groups([r["id"] for r in rows], lambda a, b: allowed(a, b) and (
        stem[a] == stem[b] or sims[a, b] >= SAME_PHOTO))
    copies, hidden = {}, set()
    for members in exact:
        # Keep the copy with a date (the original camera file), else the lowest number
        keep = min(members, key=lambda i: (not engine.rows[i]["taken"], i))
        copies[keep] = [i for i in members if i != keep]
        hidden.update(copies[keep])
    shown = [r["id"] for r in rows if r["id"] not in hidden]
    similar = _groups(shown, lambda a, b: allowed(a, b) and sims[a, b] >= SIMILAR)
    _dupes.update(key=key, copies=copies, hidden=hidden, similar=similar)
    return _dupes


def download_photos(ids, size):
    """Downloads one photo or a ZIP. Warns first if a photo has no signed release for marketing."""
    ids = sorted(ids)

    def go():
        by = quote(who())
        ui.download(f"/download/{ids[0]}?size={size}&by={by}" if len(ids) == 1
                    else f"/download-zip?ids={','.join(map(str, ids))}&size={size}&by={by}")

    missing = [i for i in ids if consent_of(engine.rows[i]) != "OK for marketing"]
    if missing:
        numbers = ", ".join(str(i + 1) for i in missing[:10]) + (" ..." if len(missing) > 10 else "")
        confirm(f"{len(missing)} of these photos have no signed release for marketing (photo {numbers}). "
                "Use them only inside DCT. Download anyway?", go)
    else:
        go()


def download_video(vid):
    applog.info("download_video", video=int(vid[1:]))
    ui.download(f"/video/{vid}", f"spotlight-video-{int(vid[1:])}.mp4")


def open_big(photo_id):
    with ui.dialog() as big, ui.card().classes("q-pa-none").style("max-width:95vw"):
        ui.image(f"/photo/{photo_id}/large").props("fit=contain").style("width:90vw;height:85vh")
    big.open()


@ui.page("/")
def index(person: str = "", location: str = "", kind: str = ""):
    layout("/?kind=video" if kind == "video" else "/")
    # Person and place come from the People and Places pages and show as removable chips
    state = {"pivot": None, "selected": None, "rows": [], "page": 1, "person": person, "location": location}
    chosen = set()  # photos ticked for download
    facets = engine.facets()

    with ui.column().classes("w-full q-pa-lg gap-3"):
        with ui.row().classes("w-full items-center gap-2 no-wrap"):
            search_box = ui.input(placeholder="Try: child meeting a mascot at the theme park").props(
                "outlined clearable dense").classes("flex-grow text-lg")
            ui.button("Search", on_click=lambda: run_search()).props("unelevated color=secondary")

        with ui.row().classes("gap-3 items-center"):
            f_activity = ui.select(facets["activity"], label="Event / activity", clearable=True,
                                   on_change=lambda: run_search()).classes("w-48")
            f_moment = ui.select(facets["moment"] or pipeline.MOMENTS, label="Moment", clearable=True,
                                 on_change=lambda: run_search()).classes("w-36")
            f_year = ui.select(facets["year"], label="Year", clearable=True,
                               on_change=lambda: run_search()).classes("w-28")
            f_consent = ui.select(CONSENT, label="Photo release", clearable=True,
                                  on_change=lambda: run_search()).classes("w-44")

        with ui.row().classes("w-full items-center gap-3 q-pa-sm rounded-borders").style(
                "background: rgba(185,119,14,.15)") as download_bar:
            chosen_label = ui.label().classes("text-weight-bold")
            ui.button("Download ZIP (web size)", icon="download",
                      on_click=lambda: download_photos(chosen, "web")).props("unelevated dense color=secondary")
            ui.button("Download ZIP (original)", icon="download",
                      on_click=lambda: download_photos(chosen, "original")).props("outline dense color=secondary")
            ui.button("Clear selection", on_click=lambda: clear_chosen()).props("flat dense")
        download_bar.set_visibility(False)

        with ui.row().classes("w-full items-center justify-between"):
            status = ui.label().classes("text-grey-8")
            pager_top = ui.row()
        pivot_bar = ui.row().classes("items-center gap-2")
        grid = ui.element("div").classes("w-full").style(GRID)
        pager_bottom = ui.row().classes("w-full justify-center")

    # fixed: the panel stays on screen while the photo grid scrolls
    with ui.right_drawer(value=False, fixed=True).props("width=420 bordered").classes("q-pa-md spot-panel") as drawer:
        detail = ui.column().classes("w-full gap-2")


    def filters():
        return {"activity": f_activity.value, "location": state["location"], "attribute": None,
                "year": f_year.value, "person": state["person"], "consent": f_consent.value,
                "moment": f_moment.value}

    def update_chosen():
        chosen_label.text = f"{len(chosen)} selected"
        download_bar.set_visibility(bool(chosen))

    def clear_chosen():
        chosen.clear()
        update_chosen()
        render_page()

    def clear_person():
        state["person"] = ""
        run_search()

    def clear_location():
        state["location"] = ""
        run_search()

    def run_search():
        state["pivot"] = None
        query = search_box.value or ""
        hits, took = engine.search(query, filters(), top=len(engine.rows))
        if query.strip() or any(filters().values()):
            # Only counts and timing, never the search words (they could contain a child's name)
            applog.info("search", words=len(query.split()), results=len(hits), ms=round(took * 1000, 1),
                        filters=sorted(k for k, v in filters().items() if v))
        hidden = duplicates()["hidden"]
        hits = [h for h in hits if h[0]["id"] not in hidden]
        if kind == "video":
            hits = [h for h in hits if video_of(h[0])]
        rows = one_per_video([r for r, _, _ in hits])
        n_videos = sum(1 for r in rows if video_of(r))
        counts = [f"{len(rows) - n_videos} photos"] if kind != "video" else []
        if n_videos or kind == "video":
            counts.append(f"{n_videos} video{'s' if n_videos != 1 else ''}")
        status.text = f"{' and '.join(counts)} in {took * 1000:.0f} ms"
        pivot_bar.clear()
        if state["person"]:
            with pivot_bar:
                ui.chip(f"Person {state['person']}", icon="face", removable=True,
                        on_value_change=clear_person).props("color=secondary text-color=white")
        if state["location"]:
            with pivot_bar:
                ui.chip(f"Place: {state['location']}", icon="place", removable=True,
                        on_value_change=clear_location).props("color=secondary text-color=white")
        show(rows)

    def run_pivot(photo_id, kind):
        hidden = duplicates()["hidden"]
        rows = [x for x in engine.pivot(photo_id, kind) if x["id"] not in hidden]
        state["pivot"] = (photo_id, kind)
        r = engine.rows[photo_id]
        label = {"event": f"same event ({r['event_name']}, {r['event_id']})",
                 "location": f"same location ({r['location']})",
                 "person": "same person"}[kind]
        status.text = f"{len(rows)} photos with the {label}"
        pivot_bar.clear()
        with pivot_bar:
            ui.chip(f"Pivot: {label}", icon="hub", removable=True,
                    on_value_change=lambda: run_search()).props("color=secondary text-color=white")
        show(one_per_video([r] + rows))

    def show(rows):
        state["rows"] = rows
        state["page"] = 1
        render_page()

    def go_to(page):
        state["page"] = page
        render_page()

    def render_page():
        rows = state["rows"]
        pages = max(1, -(-len(rows) // PAGE_SIZE))
        start = (state["page"] - 1) * PAGE_SIZE
        for pager in (pager_top, pager_bottom):
            pager.clear()
            if pages > 1:
                with pager:
                    ui.pagination(1, pages, direction_links=True, value=state["page"],
                                  on_change=lambda e: go_to(e.value)).props(
                                      "boundary-links max-pages=7 color=secondary active-color=secondary")
        grid.clear()
        with grid:
            if not rows:
                ui.label("No photos match. Try fewer words or clear a filter.").classes("text-grey-7")
            for r in rows[start : start + PAGE_SIZE]:
                photo_card(r, open_detail, chosen, update_chosen)

    def rotate(photo_id):
        """Turns a photo saved sideways. Kept in outputs/rotations.json, so it stays turned after a restart
        and pipeline.py describes it the right way up next time."""
        r = engine.rows[photo_id]
        degrees = (pipeline.rotations().get(r["file"], 0) + 90) % 360
        pipeline.set_rotation(r["file"], degrees)
        engine._log({"time": datetime.now().isoformat(timespec="seconds"), "photo_id": photo_id,
                     "decision": "rotated", "reviewer": who(), "before": {}, "after": {}, "note": f"{degrees} degrees"})
        applog.info("rotate", photo=photo_id + 1, degrees=degrees)
        state["selected"] = None
        open_detail(photo_id)
        render_page()

    def close_detail():
        state["selected"] = None
        drawer.hide()

    def open_detail(photo_id):
        if state["selected"] == photo_id and drawer.value:
            close_detail()  # clicking the same photo again closes the panel
            return
        r = engine.rows[photo_id]
        video = video_of(r)
        state["selected"] = photo_id
        detail.clear()
        with detail:
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Video details" if video else "Photo details").classes("text-subtitle1 text-weight-bold")
                ui.button(icon="close", on_click=close_detail).props("flat round dense").tooltip("Close (Esc)")
            if video:
                vid, second = video
                # #t= starts the player at the moment the search found
                player = ui.video(f"/video/{vid}#t={second}", autoplay=False).classes("w-full rounded-borders")
                ui.label("Moments in this video (click to jump)").classes("text-overline q-mt-sm")
                with ui.row().classes("gap-1"):
                    for m in video_moments(vid):
                        t = video_of(m)[1]
                        ui.chip(clock(t), icon="play_arrow", on_click=lambda t=t: (player.seek(t), player.play())).props(
                            "dense clickable" + ("" if t != second else " color=secondary text-color=white")).tooltip(
                            m["caption"])
                said = VIDEOS[vid].get("speech", [])
                if said:
                    ui.label("What is said (click a line to jump)").classes("text-overline q-mt-sm")
                    with ui.scroll_area().classes("w-full rounded-borders").style("max-height:180px"):
                        for start, _, text in said:
                            with ui.row().classes("no-wrap items-start gap-2 cursor-pointer w-full").on(
                                    "click", lambda t=start: (player.seek(t), player.play())):
                                ui.label(clock(start)).classes("text-caption text-secondary text-weight-bold")
                                ui.label(text).classes("text-body2")
                    ui.label("Written down by Whisper on this laptop. It can mishear words.").classes(
                        "text-caption text-grey-7")
                with ui.row().classes("gap-2 q-mt-sm"):
                    ui.button("Download video", icon="download",
                              on_click=lambda: download_video(vid)).props("unelevated dense color=secondary")
                with ui.row().classes("items-center gap-2"):
                    ui.label(f"Video {int(vid[1:])}").classes("text-subtitle2")
                    heart_button(photo_id)
                    ui.label(f"{clock(VIDEOS[vid]['duration'])} long").classes("text-caption text-grey-7")
                    if r["taken"]:
                        ui.label(r["taken"][:10]).classes("text-caption text-grey-7")
                ui.label(f"The description below is for the moment at {clock(second)}.").classes(
                    "text-caption text-grey-7")
            else:
                with ui.dialog() as big, ui.card().classes("q-pa-none").style("max-width:95vw"):
                    ui.image(f"/photo/{photo_id}/large").props("fit=contain").style("width:90vw;height:85vh")
                ui.image(f"/photo/{photo_id}/large").classes("w-full rounded-borders cursor-pointer").on("click", big.open)
                ui.label("Click the photo to enlarge it").classes("text-caption text-grey-7")
            with ui.row().classes("gap-2") as photo_buttons:
                ui.button("Download", icon="download", on_click=lambda: download_photos([photo_id], "original")).props(
                    "unelevated dense color=secondary").tooltip("Original quality, for print")
                ui.button("Rotate", icon="rotate_right", on_click=lambda: rotate(photo_id)).props(
                    "outline dense color=secondary").tooltip("Turn the photo 90 degrees clockwise")
            with ui.row().classes("items-center gap-2") as photo_info:
                ui.label(f"Photo {photo_id + 1}").classes("text-subtitle2")
                heart_button(photo_id)
                ui.badge("AI suggestion" if r["status"] == "ai" else f"staff {r['status']}").props(
                    f"color={STATUS_COLOR[r['status']]}")
                if r["taken"]:
                    ui.label(r["taken"][:10]).classes("text-caption text-grey-7")
            photo_buttons.set_visibility(not video)
            photo_info.set_visibility(not video)

            ui.label("Show me more").classes("text-overline q-mt-sm")
            with ui.row().classes("gap-2"):
                ui.button("Same event", icon="event", on_click=lambda: run_pivot(photo_id, "event")).props("outline dense")
                ui.button("Same location", icon="place", on_click=lambda: run_pivot(photo_id, "location")).props("outline dense")
                ui.button("Same person", icon="face", on_click=lambda: run_pivot(photo_id, "person")).props("outline dense")
            people = [p for p in r.get("person_ids", "").split(";") if p]
            if people:
                with ui.row().classes("items-center gap-2"):
                    ui.label("People in this photo:").classes("text-caption")
                    for p in people:
                        ui.chip(p, icon="face", on_click=lambda p=p: ui.navigate.to(f"/?person={p}")).props(
                            "dense outline clickable")

            ui.label("AI description").classes("text-overline q-mt-sm")
            ui.label("You edit the text that describes the photo. The photo itself is never changed.").classes(
                "text-caption text-grey-7")
            caption = ui.textarea("Caption", value=r["caption"]).props("outlined dense autogrow").classes("w-full")
            activity = ui.select(sorted(set(facets["activity"]) | {r["activity"]}), value=r["activity"],
                                 label=f"Event / activity (AI {r['activity_conf']:.0%})", new_value_mode="add-unique").classes("w-full")
            location = ui.select(sorted(set(facets["location"]) | {r["location"]}), value=r["location"],
                                 label=f"Location (AI {r['location_conf']:.0%})", new_value_mode="add-unique").classes("w-full")
            attrs = ui.select(facets["attribute"], value=[a for a in r["person_attributes"].split(";") if a],
                              label="Person attributes", multiple=True, new_value_mode="add-unique").props(
                "use-chips").classes("w-full")
            objects = ui.input("Visible objects (separated by ;)", value=r.get("objects", "")).props("dense").classes("w-full")
            moments = ui.select(pipeline.MOMENTS, value=[m for m in r.get("moments", "").split(";") if m],
                                label="Moments (what is visible, not feelings)", multiple=True).props(
                "use-chips").classes("w-full")
            consent = ui.select(CONSENT, value=consent_of(r), label="Photo release (signed by the family?)").classes(
                "w-full")
            ui.label("Set OK for marketing only when DCT has a signed photo release for this photo.").classes(
                "text-caption text-grey-7")
            note = ui.input("Note (optional)").props("dense").classes("w-full")

            def save(decision):
                changes = {"caption": caption.value, "activity": activity.value, "location": location.value,
                           "person_attributes": ";".join(attrs.value or []), "objects": objects.value or "",
                           "consent": consent.value, "moments": ";".join(moments.value or [])}
                entry = engine.review(photo_id, decision, changes, reviewer=who(), note=note.value or "")
                applog.info("review", photo=photo_id + 1, decision=entry["decision"], fields=sorted(entry["after"]))
                ui.notify(f"Saved as {entry['decision']}", type="positive")
                open_detail(photo_id)
                if state["pivot"]:
                    run_pivot(*state["pivot"])
                else:
                    run_search()

            with ui.row().classes("gap-2"):
                ui.button("Accept description", icon="check", on_click=lambda: confirm(
                    f"Mark the AI description of photo {photo_id + 1} as correct?", lambda: save("accepted"))).props(
                    "unelevated color=positive")
                ui.button("Save my changes", icon="edit", on_click=lambda: confirm(
                    f"Save your changes to the description of photo {photo_id + 1}?", lambda: save("edited"))).props(
                    "unelevated color=primary")

            def remove():
                for m in (video_moments(video[0]) if video else [r]):  # a video goes to the bin with all its moments
                    engine.remove(m["id"], reviewer=who())
                applog.info("remove", photo=photo_id + 1)
                FAVORITES.discard(photo_id)
                drawer.hide()
                ui.notify(f"{item_name(r)} moved to the bin. Restore it from the Bin page.", type="warning")
                run_search()

            ui.separator().classes("q-my-sm")
            ui.button("Move to bin", icon="delete", on_click=lambda: None if not is_admin() else confirm(
                f"Move {item_name(r).lower()} to the bin? You can restore it from the Bin page until it is "
                "emptied.", remove)).props("outline color=negative").set_visibility(is_admin())
        drawer.show()

    search_box.on("keydown.enter", lambda: run_search())
    # Esc closes the photo panel from anywhere on the page
    ui.keyboard(on_key=lambda e: close_detail() if e.key.name == "Escape" and e.action.keydown else None,
                ignore=[])
    run_search()


@ui.page("/people")
def people_page():
    layout("/people")
    counts = {}
    for r in engine.rows:
        if r.get("removed"):
            continue
        for p in r.get("person_ids", "").split(";"):
            if p:
                counts[p] = counts.get(p, 0) + 1
    with ui.column().classes("w-full q-pa-lg gap-3"):
        ui.label("People").classes("text-h5")
        ui.label("Anonymous groups made from faces (P01, P02...). No names and no identity lookup. "
                 "Click a person to see every photo they are in.").classes("text-grey-7")
        if not counts:
            ui.label("No person groups yet. Run python pipeline.py to create them.").classes("text-grey-7")
        with ui.element("div").classes("w-full").style(
                "display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:20px"):
            for p, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
                with ui.column().classes("items-center cursor-pointer").on(
                        "click", lambda p=p: ui.navigate.to(f"/?person={p}")):
                    ui.image(f"/face/{p}").classes("rounded-full").style("width:130px;height:130px").props("fit=cover")
                    ui.label(p).classes("text-weight-bold")
                    ui.label(f"{n} photos").classes("text-caption text-grey-7")


@ui.page("/places")
def places_page():
    layout("/places")
    places = {}
    for r in engine.rows:
        if not r.get("removed"):
            places.setdefault(r["location"], []).append(r)
    with ui.column().classes("w-full q-pa-lg gap-3"):
        ui.label("Places").classes("text-h5")
        ui.label("Where the photos were taken, as recognized by the AI. Click a place to see its photos.").classes(
            "text-grey-7")
        with ui.element("div").classes("w-full").style(
                "display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px"):
            for place, rows in sorted(places.items(), key=lambda x: -len(x[1])):
                cover = max(rows, key=lambda r: r["location_conf"])
                with ui.card().tight().classes("cursor-pointer relative").on(
                        "click", lambda place=place: ui.navigate.to(f"/?location={place}")):
                    ui.image(f"/photo/{cover['id']}/thumb").classes("h-48").props(f"fit=cover position='{focus(cover['id'])}'")
                    with ui.element("div").classes("absolute-bottom q-pa-sm").style(
                            "background:linear-gradient(transparent,rgba(0,0,0,.7))"):
                        ui.label(place.capitalize()).classes("text-white text-weight-bold")
                        ui.label(f"{len(rows)} photos").classes("text-white text-caption")


@ui.page("/favorites")
def favorites_page():
    layout("/favorites")
    with ui.column().classes("w-full q-pa-lg gap-3"):
        ui.label("Favorites").classes("text-h5")
        ui.label("Click the heart on any photo to add it here. For now favorites are kept until the app "
                 "restarts, saving them comes later.").classes("text-grey-7")
        favs = [engine.rows[i] for i in sorted(FAVORITES) if not engine.rows[i].get("removed")]
        if favs:
            with ui.row().classes("gap-2"):
                ui.button("Download all (web size)", icon="download",
                          on_click=lambda: download_photos([r["id"] for r in favs], "web")).props(
                    "unelevated color=secondary")
                ui.button("Download all (original)", icon="download",
                          on_click=lambda: download_photos([r["id"] for r in favs], "original")).props(
                    "outline color=secondary")
        with ui.element("div").classes("w-full").style(GRID):
            if not favs:
                ui.label("No favorites yet.").classes("text-grey-7")
            for r in favs:
                photo_card(r, open_big)


def waiting_uploads():
    known = {r["file"] for r in engine.rows}
    return sorted(f for f in DATA.iterdir() if f.name.startswith(UPLOAD_PREFIX) and f.name not in known)


@ui.page("/add")
def add_page():
    layout("/add")
    with ui.column().classes("w-full q-pa-lg gap-3"):
        ui.label("Add photos").classes("text-h5")
        ui.label("1. Drop new event photos below (jpg, png or iPhone heic). "
                 "2. Click Describe new photos. The AI writes the caption, event, location and clothing "
                 "on this laptop, about 6 seconds per photo. Nothing leaves the computer.").classes("text-grey-7")

        waiting = ui.column().classes("gap-1")
        status = ui.label().classes("text-grey-8")

        def refresh():
            waiting.clear()
            files = waiting_uploads()
            with waiting:
                for f in files:
                    ui.label(f"Waiting: {f.name[len(UPLOAD_PREFIX) + 16:]}").classes("text-body2")
            status.text = f"{len(files)} photos waiting to be described" if files else "No photos waiting."
            describe.set_enabled(bool(files))

        async def received(e):
            name = re.sub(r"[^\w.\- ]", "_", e.file.name)
            path = DATA / f"{UPLOAD_PREFIX}{datetime.now():%Y%m%d_%H%M%S}_{name}"
            saved = e.file.save(path)
            if inspect.isawaitable(saved):
                await saved
            refresh()

        async def describe_new():
            files = waiting_uploads()
            if not files:
                return
            describe.set_enabled(False)
            spinner.set_visibility(True)
            status.text = f"Describing {len(files)} photos, please wait..."
            started = time.time()
            try:
                rows, clip_emb, text_emb, attr_emb = await run.io_bound(
                    pipeline.process_photos, files, len(engine.rows))
                engine.add(rows, clip_emb, text_emb, attr_emb, reviewer=who())
                applog.info("upload_described", photos=len(rows), seconds=round(time.time() - started, 1))
                ui.notify(f"{len(rows)} photos added and searchable", type="positive")
            except Exception as err:
                applog.error("upload_failed", photos=len(files))
                ui.notify(f"Could not describe the photos: {err}", type="negative", multi_line=True)
            spinner.set_visibility(False)
            refresh()

        ui.upload(label="Drop photos here or click +", multiple=True, auto_upload=True,
                  on_upload=received).props("accept=.jpg,.jpeg,.png,.heic").classes("w-full max-w-xl")
        with ui.row().classes("items-center gap-3"):
            describe = ui.button("Describe new photos", icon="auto_awesome", on_click=describe_new).props(
                "unelevated color=secondary")
            spinner = ui.spinner(size="lg")
            spinner.set_visibility(False)
        ui.label("New photos join the People groups at the next full run of pipeline.py.").classes(
            "text-caption text-grey-7")
        refresh()


def audit_rows(action="", person=""):
    rows = []
    for e in engine.audit_log(100000):
        if action and e["decision"] != action or person and e["reviewer"] != person:
            continue
        rows.append({"time": e["time"], "photo": e["photo_id"] + 1, "action": e["decision"], "who": e["reviewer"],
                     "changed": ", ".join(f"{k}: {e['before'].get(k, '')} → {v}" for k, v in e["after"].items()) or "-",
                     "note": e["note"]})
    return rows


@app.get("/api/audit.csv")
def audit_csv(action: str = "", person: str = ""):
    """The review log as a spreadsheet, for DCT's records."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["time", "photo", "action", "who", "changed", "note"])
    writer.writeheader()
    writer.writerows(audit_rows(action, person))
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=spotlight_review_log.csv"})


def duplicate_count():
    d = duplicates()
    return len(d["copies"]) + len(d["similar"])


@ui.page("/duplicates")
def duplicates_page():
    if admin_only():
        return
    layout("/duplicates")
    d = duplicates()

    def keep_only(keep, group):
        def go():
            for i in group:
                if i != keep:
                    engine.remove(i, reviewer=who(), note=f"duplicate of photo {keep + 1}")
            applog.info("duplicates_merged", kept=keep + 1, removed=len(group) - 1)
            ui.notify(f"Kept photo {keep + 1}, the others are in the bin", type="positive")
            ui.navigate.reload()
        confirm(f"Keep photo {keep + 1} and remove the other {len(group) - 1}? "
                "They go to the bin, where you can restore them.", go)

    def not_same(group):
        pairs = not_duplicates() | {(min(a, b), max(a, b)) for a in group for b in group if a != b}
        NOT_DUPES.write_text(json.dumps(sorted(pairs)), encoding="utf-8")
        engine._log({"time": datetime.now().isoformat(timespec="seconds"), "photo_id": group[0],
                     "decision": "not duplicates", "reviewer": who(), "before": {}, "after": {},
                     "note": "photos " + ", ".join(str(i + 1) for i in group)})
        ui.notify("Marked as different photos", type="positive")
        ui.navigate.reload()

    def group_card(title, hint, group, color):
        sizes = {i: pipeline.Image.open(DATA / engine.rows[i]["file"]).size for i in group}
        best = max(group, key=lambda i: sizes[i][0] * sizes[i][1])
        with ui.card().classes("w-full q-pa-md"):
            with ui.row().classes("items-center gap-2"):
                ui.badge(title).props(f"color={color}")
                ui.label(hint).classes("text-caption text-grey-7")
            with ui.row().classes("gap-4 q-mt-sm"):
                for i in sorted(group):
                    r = engine.rows[i]
                    w, h = sizes[i]
                    with ui.column().classes("items-start gap-1"):
                        ui.image(f"/photo/{i}/thumb").classes("rounded-borders cursor-pointer").style(
                            "width:220px;height:165px").props(f"fit=cover position='{focus(i)}'").on("click", lambda i=i: open_big(i))
                        with ui.row().classes("items-center gap-1"):
                            ui.label(f"Photo {i + 1}").classes("text-weight-bold")
                            if i == best:
                                ui.badge("best quality").props("color=positive")
                        ui.label(f"{w} x {h} px · {r['taken'][:10] or 'no date'} · "
                                 f"{Path(r['file']).suffix.lstrip('.').upper()}").classes("text-caption text-grey-7")
                        ui.button("Keep this one", icon="check", on_click=lambda i=i: keep_only(i, group)).props(
                            "dense unelevated color=secondary")
            ui.button("Not duplicates", icon="call_split", on_click=lambda: not_same(group)).props(
                "flat dense").classes("q-mt-sm")

    with ui.column().classes("w-full q-pa-lg gap-3"):
        ui.label("Duplicates").classes("text-h5")
        ui.label("Photos that are in the archive more than once. Keep the best one, or mark them as "
                 "different photos. Removed files are moved to data/removed and logged with your name.").classes(
            "text-grey-7")
        if not d["copies"] and not d["similar"]:
            ui.label("No duplicates found.").classes("text-grey-7")
        for keep, others in d["copies"].items():
            group_card("Exact copies", "The same photo twice, for example the iPhone file and a JPG copy",
                       [keep] + others, "red-7")
        for group in d["similar"]:
            group_card("Very similar", "Probably the same moment, check if both are worth keeping",
                       group, "amber-8")


@ui.page("/bin")
def bin_page():
    if admin_only():
        return
    layout("/bin")
    empty_old_bin()
    days = settings().get("bin_days", 30)

    def days_left(r):
        if not days:
            return "kept until you empty the bin"
        gone = datetime.fromisoformat(r["removed_at"]).timestamp() + days * 86400
        left = max(0, int((gone - datetime.now().timestamp()) // 86400) + 1)
        return f"deleted for good in {left} day{'s' if left != 1 else ''}"

    def restore(r):
        engine.restore(r["id"], reviewer=who())
        applog.info("restore", photo=r["id"] + 1)
        ui.notify(f"Photo {r['id'] + 1} is back in the archive", type="positive")
        ui.navigate.reload()

    def delete_forever(ids):
        def go():
            for i in ids:
                engine.delete_forever(i, reviewer=who())
            applog.info("deleted_forever", photos=len(ids))
            ui.notify(f"{len(ids)} photo{'s' if len(ids) != 1 else ''} deleted for good", type="warning")
            ui.navigate.reload()
        confirm(f"Delete {len(ids)} photo{'s' if len(ids) != 1 else ''} for good? This cannot be undone.", go)

    def set_days(e):
        save_settings(bin_days=e.value)
        applog.info("bin_setting_changed", days=e.value)
        ui.navigate.reload()

    items = sorted(engine.bin(), key=lambda r: r["removed_at"], reverse=True)
    with ui.column().classes("w-full q-pa-lg gap-3"):
        ui.label("Bin").classes("text-h5")
        ui.label("Photos moved to the bin are hidden everywhere. Restore them, or they are deleted for good "
                 "after the time you choose.").classes("text-grey-7")
        with ui.row().classes("items-center gap-3"):
            ui.select(BIN_CHOICES, value=days, label="Empty the bin automatically", on_change=set_days).props(
                "outlined dense").classes("w-64")
            if items:
                ui.button("Restore all", icon="restore", on_click=lambda: [engine.restore(r["id"], reviewer=who())
                                                                            for r in items] and ui.navigate.reload()
                          ).props("outline color=secondary")
                ui.button("Empty bin now", icon="delete_forever",
                          on_click=lambda: delete_forever([r["id"] for r in items])).props("outline color=negative")
        if not items:
            ui.label("The bin is empty.").classes("text-grey-7 q-mt-md")
        with ui.element("div").classes("w-full").style(GRID):
            for r in items:
                with ui.card().tight():
                    ui.image(f"/photo/{r['id']}/thumb").classes("h-40").props(f"fit=cover position='{focus(r['id'])}'").style("opacity:.6")
                    with ui.card_section().classes("q-pa-sm"):
                        ui.label(item_name(r)).classes("text-weight-bold")
                        ui.label(f"Moved {r['removed_at'][:10]} · {days_left(r)}").classes("text-caption text-grey-7")
                        with ui.row().classes("gap-1 q-mt-xs"):
                            ui.button("Restore", icon="restore", on_click=lambda r=r: restore(r)).props(
                                "dense unelevated color=secondary")
                            ui.button(icon="delete_forever", on_click=lambda r=r: delete_forever([r["id"]])).props(
                                "dense flat color=negative").tooltip("Delete for good")


@ui.page("/log")
def log_page():
    if admin_only():
        return
    layout("/system")  # the review log lives under System
    everything = engine.audit_log(100000)
    with ui.column().classes("w-full q-pa-lg gap-3"):
        with ui.row().classes("items-center gap-2"):
            ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/system")).props("flat round dense").tooltip(
                "Back to System")
            ui.label("Review log").classes("text-h5")
        ui.label("Every accept, edit, download, upload and removal, newest first, with who did it.").classes(
            "text-grey-7")
        with ui.row().classes("items-center gap-3"):
            action = ui.select(sorted({e["decision"] for e in everything}), label="Action", clearable=True,
                               on_change=lambda: refresh()).classes("w-44")
            person = ui.select(sorted({e["reviewer"] for e in everything}), label="Who", clearable=True,
                               on_change=lambda: refresh()).classes("w-44")
            ui.button("Export CSV", icon="download", on_click=lambda: ui.download(
                f"/api/audit.csv?action={quote(action.value or '')}&person={quote(person.value or '')}")).props(
                "outline color=secondary")
        count = ui.label().classes("text-grey-7")
        table = ui.table(columns=[{"name": c, "label": c.title(), "field": c, "align": "left", "sortable": True}
                                  for c in ["time", "photo", "action", "who", "changed", "note"]],
                         rows=[], pagination=25).classes("w-full")

        def refresh():
            table.rows = audit_rows(action.value or "", person.value or "")
            table.update()
            count.text = f"{len(table.rows)} entries"

        refresh()


@ui.page("/system")
def system_page():
    if admin_only():
        return
    layout("/system")
    recent = list(applog.RECENT)
    searches = [e for e in recent if e["event"] == "search"]
    problems = [e for e in recent if e["level"] in ("WARNING", "ERROR")][::-1][:50]
    log_size = sum(f.stat().st_size for f in applog.LOG_DIR.glob("app.log*")) / 1e6 if applog.LOG_DIR.exists() else 0
    stats = [
        ("Photos", sum(1 for r in engine.rows if not r.get("removed") and not video_of(r))),
        ("Videos", len({video_of(r)[0] for r in engine.rows if not r.get("removed") and video_of(r)})),
        ("Removed", sum(1 for r in engine.rows if r.get("removed"))),
        ("Person groups", len({p for r in engine.rows for p in r.get("person_ids", "").split(";") if p})),
        ("Avg search", f"{sum(e['ms'] for e in searches) / len(searches):.0f} ms" if searches else "no searches yet"),
        ("Errors since start", sum(1 for e in recent if e["level"] == "ERROR")),
        ("Log files", f"{log_size:.1f} MB"),
    ]
    with ui.column().classes("w-full q-pa-lg gap-3"):
        ui.label("System").classes("text-h5")
        ui.label(f"How the app is doing. The full log is in {applog.LOG_FILE} (one JSON line per event, "
                 "no file names, captions or search words).").classes("text-grey-7")
        with ui.row().classes("gap-3"):
            for label, value in stats:
                with ui.card().classes("q-pa-md").style("min-width:150px"):
                    ui.label(str(value)).classes("text-h6")
                    ui.label(label).classes("text-caption text-grey-7")
        with ui.card().classes("w-full q-pa-md cursor-pointer").on("click", lambda: ui.navigate.to("/log")):
            with ui.row().classes("items-center no-wrap gap-3 w-full"):
                ui.icon("history", size="28px").classes("text-secondary")
                with ui.column().classes("gap-0"):
                    ui.label("Review log").classes("text-subtitle1 text-weight-bold")
                    ui.label(f"{len(engine.audit_log(100000))} actions: every accept, edit, download, upload and "
                             "removal, with who did it").classes("text-caption text-grey-7")
                ui.space()
                ui.icon("chevron_right").classes("text-grey-6")
        ui.label("Warnings and errors").classes("text-subtitle1 q-mt-md")
        if not problems:
            ui.label("None since the app started.").classes("text-grey-7")
        for e in problems:
            with ui.expansion(f"{e['time']}  {e['level']}  {e['event']}", icon="error_outline").classes("w-full"):
                ui.code(json.dumps(e, indent=1), language="json").classes("w-full")
        ui.label("Latest events").classes("text-subtitle1 q-mt-md")
        ui.table(columns=[{"name": c, "label": c.title(), "field": c, "align": "left"}
                          for c in ["time", "level", "event", "details"]],
                 rows=[{"time": e["time"], "level": e["level"], "event": e["event"],
                        "details": ", ".join(f"{k}={v}" for k, v in e.items()
                                             if k not in ("time", "level", "event", "error"))}
                       for e in recent[::-1][:100]], pagination=20).classes("w-full")


# Accounts and login

class LoginRequired(BaseHTTPMiddleware):
    """Every page, photo and API call needs a logged in account. Only the login page, the first-run
    setup page and the files NiceGUI itself needs are open."""
    OPEN = ("/login", "/setup", "/_nicegui", "/static", "/favicon")

    async def dispatch(self, request: Request, call_next):
        if not app.storage.user.get("authenticated", False) and not request.url.path.startswith(self.OPEN):
            return RedirectResponse("/setup" if not accounts.any_users() else "/login")
        return await call_next(request)


app.add_middleware(LoginRequired)


def login_frame(title, subtitle):
    ui.colors(primary="#24214f", secondary="#b9770e", accent="#b9770e")
    ui.add_css(CSS + "body { background: #14122e !important; }")
    card = ui.card().classes("absolute-center q-pa-xl items-center").style("width: 380px; max-width: 92vw")
    with card:
        with ui.element("div").classes("q-pa-md").style("background:#24214f;border-radius:18px"):
            ui.image("/static/logo.svg").style("width:84px;height:80px").props("no-spinner")
        with ui.row().classes("items-baseline gap-0"):
            ui.label("spotlight").classes("text-h5 text-weight-bold")
            ui.label(".io").classes("text-h5 text-weight-bold").style("color:#b9770e")
        ui.label(title).classes("text-subtitle1 q-mt-sm")
        ui.label(subtitle).classes("text-caption text-grey-7 text-center")
    return card


@ui.page("/login")
def login_page():
    if not accounts.any_users():
        ui.navigate.to("/setup")
        return
    if app.storage.user.get("authenticated"):
        ui.navigate.to("/")
        return

    def try_login():
        role = accounts.check(username.value or "", password.value or "")
        if role:
            app.storage.user.update(authenticated=True, username=username.value.strip().lower(), role=role)
            applog.info("login", user=username.value.strip().lower())
            ui.navigate.to("/")
        else:
            applog.warning("login_failed")
            message.text = "Wrong user name or password"

    with login_frame("Log in", "Dreams Come True staff only"):
        username = ui.input("User name").props("outlined dense").classes("w-full")
        password = ui.input("Password", password=True, password_toggle_button=True).props(
            "outlined dense").classes("w-full").on("keydown.enter", try_login)
        message = ui.label().classes("text-negative text-caption")
        ui.button("Log in", on_click=try_login).props("unelevated color=secondary").classes("w-full")


@ui.page("/setup")
def setup_page():
    """Only used once: when there are no accounts yet, the first person makes the admin account."""
    if accounts.any_users():
        ui.navigate.to("/login")
        return

    def create():
        if password.value != repeat.value:
            message.text = "The passwords are not the same"
            return
        error = accounts.add(username.value or "", password.value or "", "admin")
        if error:
            message.text = error
            return
        applog.info("first_admin_created")
        ui.navigate.to("/login")

    with login_frame("Create the admin account", "First start: this account can add staff accounts later"):
        username = ui.input("User name").props("outlined dense").classes("w-full")
        password = ui.input(f"Password (at least {accounts.MIN_PASSWORD} characters)", password=True,
                            password_toggle_button=True).props("outlined dense").classes("w-full")
        repeat = ui.input("Repeat password", password=True).props("outlined dense").classes("w-full")
        message = ui.label().classes("text-negative text-caption")
        ui.button("Create admin account", on_click=create).props("unelevated color=secondary").classes("w-full")


@ui.page("/accounts")
def accounts_page():
    if admin_only():
        return
    layout("/accounts")
    with ui.column().classes("w-full q-pa-lg gap-3"):
        ui.label("Accounts").classes("text-h5")
        ui.label("Staff can search, edit, download and add photos. Admins can also remove photos, "
                 "clean up duplicates, see the System page and manage accounts.").classes("text-grey-7")
        listing = ui.column().classes("w-full gap-2")

        def refresh():
            listing.clear()
            with listing:
                for name, u in sorted(accounts.users().items()):
                    with ui.card().classes("w-full q-pa-sm"):
                        with ui.row().classes("items-center w-full gap-3"):
                            ui.avatar(name[:1].upper(), color="amber-8", text_color="white", size="32px")
                            ui.label(name).classes("text-weight-bold")
                            ui.badge(u["role"]).props("color=primary" if u["role"] == "admin" else "color=grey-7")
                            ui.label(f"since {u['created'][:10]}").classes("text-caption text-grey-7")
                            ui.space()
                            ui.button("New password", icon="key", on_click=lambda n=name: new_password(n)).props(
                                "flat dense")
                            if name != who():
                                ui.button("Remove", icon="person_remove", on_click=lambda n=name: confirm(
                                    f"Remove the account {n}? Their past actions stay in the review log.",
                                    lambda: drop(n))).props("flat dense color=negative")

        def drop(name):
            accounts.remove(name)
            applog.info("account_removed", by=who())
            refresh()

        def new_password(name):
            with ui.dialog() as dlg, ui.card().style("min-width:320px"):
                ui.label(f"New password for {name}").classes("text-subtitle1")
                pw = ui.input("New password", password=True, password_toggle_button=True).classes("w-full")
                msg = ui.label().classes("text-negative text-caption")

                def save():
                    error = accounts.set_password(name, pw.value or "")
                    if error:
                        msg.text = error
                        return
                    applog.info("password_changed", by=who())
                    dlg.close()
                    ui.notify(f"Password changed for {name}", type="positive")

                with ui.row().classes("w-full justify-end"):
                    ui.button("Cancel", on_click=dlg.close).props("flat")
                    ui.button("Save", on_click=save).props("unelevated color=secondary")
            dlg.open()

        refresh()
        ui.label("Add an account").classes("text-subtitle1 q-mt-md")
        with ui.row().classes("items-start gap-3"):
            new_name = ui.input("User name").props("outlined dense")
            new_pw = ui.input(f"Password (min {accounts.MIN_PASSWORD})", password=True,
                              password_toggle_button=True).props("outlined dense")
            new_role = ui.select(accounts.ROLES, value="staff", label="Role").props("outlined dense").classes("w-32")

            def add():
                error = accounts.add(new_name.value or "", new_pw.value or "", new_role.value)
                if error:
                    ui.notify(error, type="negative")
                    return
                applog.info("account_added", by=who(), role=new_role.value)
                ui.notify(f"Account {new_name.value.strip().lower()} added", type="positive")
                new_name.value = new_pw.value = ""
                refresh()

            ui.button("Add", icon="person_add", on_click=add).props("unelevated color=secondary")


if __name__ in {"__main__", "__mp_main__"}:
    empty_old_bin()
    applog.info("startup", photos=sum(1 for r in engine.rows if not r.get("removed")), faces=len(FACES))
    ui.run(host="127.0.0.1", port=8080, title="spotlight.io", favicon=ROOT / "static" / "logo.svg",
           storage_secret="spotlight-local-only",
           reload=False, show=False, fastapi_docs=True)
