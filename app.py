"""Spotlight AI: photo search app for Dreams Come True staff.

NiceGUI for the screens, FastAPI (the same server) for the REST API under /api.
Runs only on this computer (127.0.0.1), photos never leave the laptop.

    python pipeline.py      # once, writes outputs/
    python app.py           # then open http://127.0.0.1:8080
"""
import csv
import io
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse, StreamingResponse
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


@app.get("/api/search")
def api_search(q: str = "", activity: str = "", location: str = "", attribute: str = "", year: str = "", top: int = 40):
    filters = {"activity": activity, "location": location, "attribute": attribute, "year": year}
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
    fields = ["id", "file", "caption", "activity", "location", "person_attributes", "event_id", "event_name", "taken", "status"]
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


@ui.page("/")
def index():
    state = {"query": "", "pivot": None, "selected": None}
    facets = engine.facets()

    ui.colors(primary="#24214f", secondary="#b9770e", accent="#b9770e")
    ui.query("body").style("background:#f6f5fa")

    with ui.header().classes("items-center bg-primary q-px-lg"):
        ui.label("Spotlight AI").classes("text-h6 text-weight-bold")
        ui.label("photo search for Dreams Come True staff").classes("text-caption opacity-80")
        ui.space()
        ui.link("Review log", "/log").classes("text-white")
        ui.link("Download index (CSV)", "/api/export.csv").classes("text-white q-ml-md")

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
                                   on_change=lambda: run_search()).classes("w-44")
            f_attribute = ui.select(facets["attribute"], label="Person attribute", clearable=True,
                                    on_change=lambda: run_search()).classes("w-48")
            f_year = ui.select(facets["year"], label="Year", clearable=True,
                               on_change=lambda: run_search()).classes("w-28")

        status = ui.label().classes("text-grey-8")
        pivot_bar = ui.row().classes("items-center gap-2")
        grid = ui.element("div").classes("w-full").style(
            "display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px")

    with ui.right_drawer(value=False, fixed=False).props("width=420 bordered").classes("bg-white q-pa-md") as drawer:
        detail = ui.column().classes("w-full gap-2")

    def set_query(text):
        search_box.value = text
        run_search()

    def filters():
        return {"activity": f_activity.value, "location": f_location.value,
                "attribute": f_attribute.value, "year": f_year.value}

    def run_search():
        state["pivot"] = None
        hits, took = engine.search(search_box.value or "", filters())
        status.text = f"{len(hits)} photos in {took * 1000:.0f} ms"
        pivot_bar.clear()
        show([r for r, _, _ in hits])

    def run_pivot(photo_id, kind):
        rows = engine.pivot(photo_id, kind)
        state["pivot"] = (photo_id, kind)
        r = engine.rows[photo_id]
        label = {"event": f"same event ({r['event_name']}, {r['event_id']})",
                 "location": f"same location ({r['location']})",
                 "person": f"same person attributes ({r['person_attributes'].replace(';', ', ')})"}[kind]
        status.text = f"{len(rows)} photos with the {label}"
        pivot_bar.clear()
        with pivot_bar:
            ui.chip(f"Pivot: {label}", icon="hub", removable=True,
                    on_value_change=lambda: run_search()).props("color=secondary text-color=white")
        show([r] + rows)

    def show(rows):
        grid.clear()
        with grid:
            if not rows:
                ui.label("No photos match. Try fewer words or clear a filter.").classes("text-grey-7")
            for r in rows:
                with ui.card().tight().classes("cursor-pointer").on("click", lambda r=r: open_detail(r["id"])):
                    ui.image(f"/photo/{r['id']}/thumb").classes("h-40").props("fit=cover")
                    with ui.card_section().classes("q-pa-sm"):
                        ui.label(f"Photo {r['id']}").classes("text-caption text-grey-7")
                        ui.label(r["caption"]).classes("text-body2")
                        with ui.row().classes("gap-1 q-mt-xs"):
                            ui.badge(r["activity"]).props("color=primary")
                            ui.badge(r["location"]).props("color=secondary")
                            if r["status"] != "ai":
                                ui.badge(r["status"]).props(f"color={STATUS_COLOR[r['status']]}")

    def open_detail(photo_id):
        r = engine.rows[photo_id]
        state["selected"] = photo_id
        detail.clear()
        with detail:
            ui.image(f"/photo/{photo_id}/large").classes("w-full rounded-borders")
            with ui.row().classes("items-center gap-2"):
                ui.label(f"Photo {photo_id}").classes("text-subtitle2")
                ui.badge("AI suggestion" if r["status"] == "ai" else f"staff {r['status']}").props(
                    f"color={STATUS_COLOR[r['status']]}")
                if r["taken"]:
                    ui.label(r["taken"][:10]).classes("text-caption text-grey-7")

            ui.label("Show me more").classes("text-overline q-mt-sm")
            with ui.row().classes("gap-2"):
                ui.button("Same event", icon="event", on_click=lambda: run_pivot(photo_id, "event")).props("outline dense")
                ui.button("Same location", icon="place", on_click=lambda: run_pivot(photo_id, "location")).props("outline dense")
                ui.button("Same person", icon="checkroom", on_click=lambda: run_pivot(photo_id, "person")).props("outline dense")
            ui.label("Same person uses clothing and gear only, never faces.").classes("text-caption text-grey-7")

            ui.label("Record").classes("text-overline q-mt-sm")
            caption = ui.textarea("Caption", value=r["caption"]).props("outlined dense autogrow").classes("w-full")
            activity = ui.select(sorted(set(facets["activity"]) | {r["activity"]}), value=r["activity"],
                                 label=f"Event / activity (AI {r['activity_conf']:.0%})", new_value_mode="add-unique").classes("w-full")
            location = ui.select(sorted(set(facets["location"]) | {r["location"]}), value=r["location"],
                                 label=f"Location (AI {r['location_conf']:.0%})", new_value_mode="add-unique").classes("w-full")
            attrs = ui.select(facets["attribute"], value=[a for a in r["person_attributes"].split(";") if a],
                              label="Person attributes", multiple=True, new_value_mode="add-unique").props(
                "use-chips").classes("w-full")
            note = ui.input("Note (optional)").props("dense").classes("w-full")

            def save(decision):
                changes = {"caption": caption.value, "activity": activity.value, "location": location.value,
                           "person_attributes": ";".join(attrs.value or [])}
                entry = engine.review(photo_id, decision, changes, note=note.value or "")
                ui.notify(f"Saved as {entry['decision']}", type="positive")
                open_detail(photo_id)
                if state["pivot"]:
                    run_pivot(*state["pivot"])
                else:
                    run_search()

            with ui.row().classes("gap-2"):
                ui.button("Accept", icon="check", on_click=lambda: save("accepted")).props("unelevated color=positive")
                ui.button("Save edits", icon="edit", on_click=lambda: save("edited")).props("unelevated color=primary")
        drawer.show()

    search_box.on("keydown.enter", lambda: run_search())
    run_search()


@ui.page("/log")
def log_page():
    ui.colors(primary="#24214f", secondary="#b9770e")
    with ui.header().classes("items-center bg-primary q-px-lg"):
        ui.link("Spotlight AI", "/").classes("text-white text-h6 text-weight-bold")
        ui.label("Review log").classes("text-caption opacity-80")
    with ui.column().classes("w-full q-pa-lg"):
        ui.label("Every accept and edit by staff, newest first").classes("text-h6")
        rows = [{"time": e["time"], "photo": e["photo_id"], "decision": e["decision"], "reviewer": e["reviewer"],
                 "changed": ", ".join(f"{k}: {e['before'][k]} → {v}" for k, v in e["after"].items()) or "-",
                 "note": e["note"]} for e in engine.audit_log(200)]
        ui.table(columns=[{"name": c, "label": c.title(), "field": c, "align": "left"} for c in
                          ["time", "photo", "decision", "reviewer", "changed", "note"]], rows=rows).classes("w-full")


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host="127.0.0.1", port=8080, title="Spotlight AI", reload=False, show=False)
