"""Spotlight AI: photo search app for Dreams Come True staff.

NiceGUI for the screens, FastAPI (the same server) for the REST API under /api.
Runs only on this computer (127.0.0.1), photos never leave the laptop.

    python pipeline.py      # once, writes outputs/
    python app.py           # then open http://127.0.0.1:8080
"""
import csv
import io
import json
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from nicegui import app, ui
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from search import SearchEngine

register_heif_opener()

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "raw"
THUMBS = ROOT / "outputs" / "thumbs"
THUMBS.mkdir(parents=True, exist_ok=True)

engine = SearchEngine()
FACES_FILE = ROOT / "outputs" / "faces.json"
FACES = json.loads(FACES_FILE.read_text(encoding="utf-8")) if FACES_FILE.exists() else []
# Favorites only live in memory for now (the look is done, saving comes later)
FAVORITES = set()


# REST API (FastAPI)

def photo_or_404(photo_id):
    if not 0 <= photo_id < len(engine.rows):
        raise HTTPException(404, "No photo with that id")
    return engine.rows[photo_id]


@app.get("/photo/{photo_id}/{size}")
def photo_file(photo_id: int, size: str):
    """JPEG copy of a photo, made once and cached. size: thumb or large."""
    r = photo_or_404(photo_id)
    side = 400 if size == "thumb" else 1400
    path = THUMBS / f"{photo_id}_{side}.jpg"
    if not path.exists():
        img = ImageOps.exif_transpose(Image.open(DATA / r["file"])).convert("RGB")
        img.thumbnail((side, side))
        img.save(path, quality=85)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/face/{person}")
def face_cover(person: str):
    """Round cover for the People page: the clearest face of that person group, cut on the fly.
    Nothing is written to disk."""
    faces = [f for f in FACES if f["person"] == person]
    if not faces:
        raise HTTPException(404, "No face for that person group")
    best = max(faces, key=lambda f: f["score"] * (f["box"][2] - f["box"][0]))
    img = ImageOps.exif_transpose(Image.open(DATA / engine.rows[best["photo"]]["file"])).convert("RGB")
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
    fields = ["id", "file", "caption", "activity", "location", "person_attributes", "objects", "person_ids", "event_id", "event_name", "taken", "status"]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(engine.rows)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=spotlight_index.csv"})


def public(r):
    """Record without the file name, file names can contain children's names."""
    return {k: v for k, v in r.items() if k != "file"}


# Screens (NiceGUI)

STATUS_COLOR = {"ai": "grey", "accepted": "positive", "edited": "primary"}
PAGE_SIZE = 24
NAV = [("Photos", "photo_library", "/"), ("People", "face", "/people"), ("Places", "place", "/places"),
       ("Favorites", "star", "/favorites"), ("Review log", "history", "/log")]
GRID = "display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px"


def confirm(question, action):
    """Small yes/no box so one accidental click does not change a record."""
    with ui.dialog() as dlg, ui.card():
        ui.label(question)
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=dlg.close).props("flat")
            ui.button("Yes", on_click=lambda: (dlg.close(), action())).props("unelevated color=primary")
    dlg.open()


def layout(active):
    """Header and left sidebar shared by every page, like Google Photos."""
    ui.colors(primary="#24214f", secondary="#b9770e", accent="#b9770e")
    ui.query("body").style("background:#f6f5fa")
    with ui.left_drawer(value=True, fixed=True).props("width=220 bordered").classes("bg-white q-pt-md") as nav:
        for name, icon, path in NAV:
            selected = path == active
            with ui.item(on_click=lambda path=path: ui.navigate.to(path)).classes(
                    "rounded-r-full q-mr-sm" + (" bg-amber-1 text-weight-bold" if selected else "")):
                with ui.item_section().props("avatar"):
                    ui.icon(icon, color="secondary" if selected else "grey-7")
                with ui.item_section():
                    ui.label(name)
    with ui.header().classes("items-center bg-primary q-px-md"):
        ui.button(icon="menu", on_click=nav.toggle).props("flat round color=white")
        ui.label("Spotlight AI").classes("text-h6 text-weight-bold")
        ui.label("photo search for Dreams Come True staff").classes("text-caption opacity-80 gt-sm")
        ui.space()
        ui.link("Download index (CSV)", "/api/export.csv").classes("text-white")


def star_button(photo_id):
    """Favorite star. Only kept in memory until the app restarts, saving comes later."""
    def toggle():
        FAVORITES.symmetric_difference_update({photo_id})
        btn.props(f"icon={'star' if photo_id in FAVORITES else 'star_border'}")
    btn = ui.button(icon="star" if photo_id in FAVORITES else "star_border").props("flat round dense color=amber")
    btn.on("click.stop", toggle)
    return btn


def photo_card(r, on_open):
    with ui.card().tight().classes("cursor-pointer relative").on("click", lambda r=r: on_open(r["id"])):
        ui.image(f"/photo/{r['id']}/thumb").classes("h-40").props("fit=cover")
        with ui.element("div").classes("absolute-top-right q-ma-xs bg-white rounded-full"):
            star_button(r["id"])
        with ui.card_section().classes("q-pa-sm"):
            ui.label(f"Photo {r['id']}").classes("text-caption text-grey-7")
            ui.label(r["caption"]).classes("text-body2")
            with ui.row().classes("gap-1 q-mt-xs"):
                ui.badge(r["activity"]).props("color=primary")
                ui.badge(r["location"]).props("color=secondary")
                if r["status"] != "ai":
                    ui.badge(r["status"]).props(f"color={STATUS_COLOR[r['status']]}")


def open_big(photo_id):
    with ui.dialog() as big, ui.card().classes("q-pa-none").style("max-width:95vw"):
        ui.image(f"/photo/{photo_id}/large").props("fit=contain").style("width:90vw;height:85vh")
    big.open()


@ui.page("/")
def index(person: str = "", location: str = ""):
    layout("/")
    state = {"pivot": None, "selected": None, "rows": [], "page": 1, "person": person}
    facets = engine.facets()

    with ui.column().classes("w-full q-pa-lg gap-3"):
        with ui.row().classes("w-full items-center gap-2 no-wrap"):
            search_box = ui.input(placeholder="Try: child meeting a mascot at the theme park").props(
                "outlined clearable dense").classes("flex-grow text-lg")
            ui.button("Search", on_click=lambda: run_search()).props("unelevated color=secondary")

        with ui.row().classes("gap-2 items-center"):
            for example in ["birthday cake", "kids with mouse ears", "family dinner at a restaurant",
                            "volunteers in matching shirts", "child with a stuffed animal"]:
                ui.chip(example, on_click=lambda e=example: set_query(e)).props("outline clickable")

        with ui.row().classes("gap-3 items-center"):
            f_activity = ui.select(facets["activity"], label="Event / activity", clearable=True,
                                   on_change=lambda: run_search()).classes("w-48")
            f_location = ui.select(facets["location"], label="Location", clearable=True,
                                   value=location if location in facets["location"] else None,
                                   on_change=lambda: run_search()).classes("w-44")
            f_attribute = ui.select(facets["attribute"], label="Person attribute", clearable=True,
                                    on_change=lambda: run_search()).classes("w-48")
            f_year = ui.select(facets["year"], label="Year", clearable=True,
                               on_change=lambda: run_search()).classes("w-28")

        with ui.row().classes("w-full items-center justify-between"):
            status = ui.label().classes("text-grey-8")
            pager_top = ui.row()
        pivot_bar = ui.row().classes("items-center gap-2")
        grid = ui.element("div").classes("w-full").style(GRID)
        pager_bottom = ui.row().classes("w-full justify-center")

    # fixed: the panel stays on screen while the photo grid scrolls
    with ui.right_drawer(value=False, fixed=True).props("width=420 bordered").classes("bg-white q-pa-md") as drawer:
        detail = ui.column().classes("w-full gap-2")

    def set_query(text):
        search_box.value = text
        run_search()

    def filters():
        return {"activity": f_activity.value, "location": f_location.value,
                "attribute": f_attribute.value, "year": f_year.value, "person": state["person"]}

    def clear_person():
        state["person"] = ""
        run_search()

    def run_search():
        state["pivot"] = None
        hits, took = engine.search(search_box.value or "", filters(), top=len(engine.rows))
        status.text = f"{len(hits)} photos in {took * 1000:.0f} ms"
        pivot_bar.clear()
        if state["person"]:
            with pivot_bar:
                ui.chip(f"Person {state['person']}", icon="face", removable=True,
                        on_value_change=clear_person).props("color=secondary text-color=white")
        show([r for r, _, _ in hits])

    def run_pivot(photo_id, kind):
        rows = engine.pivot(photo_id, kind)
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
        show([r] + rows)

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
                                  on_change=lambda e: go_to(e.value)).props("boundary-links max-pages=7")
        grid.clear()
        with grid:
            if not rows:
                ui.label("No photos match. Try fewer words or clear a filter.").classes("text-grey-7")
            for r in rows[start : start + PAGE_SIZE]:
                photo_card(r, open_detail)

    def open_detail(photo_id):
        r = engine.rows[photo_id]
        state["selected"] = photo_id
        detail.clear()
        with detail:
            with ui.dialog() as big, ui.card().classes("q-pa-none").style("max-width:95vw"):
                ui.image(f"/photo/{photo_id}/large").props("fit=contain").style("width:90vw;height:85vh")
            ui.image(f"/photo/{photo_id}/large").classes("w-full rounded-borders cursor-pointer").on("click", big.open)
            ui.label("Click the photo to enlarge it").classes("text-caption text-grey-7")
            with ui.row().classes("items-center gap-2"):
                ui.label(f"Photo {photo_id}").classes("text-subtitle2")
                star_button(photo_id)
                ui.badge("AI suggestion" if r["status"] == "ai" else f"staff {r['status']}").props(
                    f"color={STATUS_COLOR[r['status']]}")
                if r["taken"]:
                    ui.label(r["taken"][:10]).classes("text-caption text-grey-7")

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
            else:
                ui.label("No person group (no clear face)").classes("text-caption")
            ui.label("Same person uses anonymous face groups first, then clothing. No names.").classes(
                "text-caption text-grey-7")

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
            note = ui.input("Note (optional)").props("dense").classes("w-full")

            def save(decision):
                changes = {"caption": caption.value, "activity": activity.value, "location": location.value,
                           "person_attributes": ";".join(attrs.value or []), "objects": objects.value or ""}
                entry = engine.review(photo_id, decision, changes, note=note.value or "")
                ui.notify(f"Saved as {entry['decision']}", type="positive")
                open_detail(photo_id)
                if state["pivot"]:
                    run_pivot(*state["pivot"])
                else:
                    run_search()

            with ui.row().classes("gap-2"):
                ui.button("Accept description", icon="check", on_click=lambda: confirm(
                    f"Mark the AI description of photo {photo_id} as correct?", lambda: save("accepted"))).props(
                    "unelevated color=positive")
                ui.button("Save my changes", icon="edit", on_click=lambda: confirm(
                    f"Save your changes to the description of photo {photo_id}?", lambda: save("edited"))).props(
                    "unelevated color=primary")
        drawer.show()

    search_box.on("keydown.enter", lambda: run_search())
    run_search()


@ui.page("/people")
def people_page():
    layout("/people")
    counts = {}
    for r in engine.rows:
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
                    ui.image(f"/photo/{cover['id']}/thumb").classes("h-48").props("fit=cover")
                    with ui.element("div").classes("absolute-bottom q-pa-sm").style(
                            "background:linear-gradient(transparent,rgba(0,0,0,.7))"):
                        ui.label(place.capitalize()).classes("text-white text-weight-bold")
                        ui.label(f"{len(rows)} photos").classes("text-white text-caption")


@ui.page("/favorites")
def favorites_page():
    layout("/favorites")
    with ui.column().classes("w-full q-pa-lg gap-3"):
        ui.label("Favorites").classes("text-h5")
        ui.label("Click the star on any photo to add it here. For now favorites are kept until the app "
                 "restarts, saving them comes later.").classes("text-grey-7")
        with ui.element("div").classes("w-full").style(GRID):
            favs = [engine.rows[i] for i in sorted(FAVORITES)]
            if not favs:
                ui.label("No favorites yet.").classes("text-grey-7")
            for r in favs:
                photo_card(r, open_big)


@ui.page("/log")
def log_page():
    layout("/log")
    with ui.column().classes("w-full q-pa-lg"):
        ui.label("Review log").classes("text-h5")
        ui.label("Every accept and edit by staff, newest first").classes("text-grey-7")
        rows = [{"time": e["time"], "photo": e["photo_id"], "decision": e["decision"], "reviewer": e["reviewer"],
                 "changed": ", ".join(f"{k}: {e['before'][k]} → {v}" for k, v in e["after"].items()) or "-",
                 "note": e["note"]} for e in engine.audit_log(200)]
        ui.table(columns=[{"name": c, "label": c.title(), "field": c, "align": "left"} for c in
                          ["time", "photo", "decision", "reviewer", "changed", "note"]], rows=rows).classes("w-full")


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host="127.0.0.1", port=8080, title="Spotlight AI", reload=False, show=False, fastapi_docs=True)
