from html import escape
from math import isfinite
from urllib.parse import quote, urlencode

from fastapi.responses import HTMLResponse

from app.notes.models import Note


def page(title: str, content: str, *, status_code: int = 200) -> HTMLResponse:
    return HTMLResponse(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} · Personal Knowledge Agent</title>
<style>
  body {{ max-width: 44rem; margin: 2rem auto; padding: 0 1rem; font: 1rem/1.55 system-ui; }}
  label {{ display: block; margin-bottom: .3rem; font-weight: 600; }}
  textarea {{ display: block; width: 100%; box-sizing: border-box; margin-bottom: .5rem; }}
  input, button, textarea {{ font: inherit; }}
  img {{ display: block; max-width: 100%; height: auto; }}
  article {{ border-top: 1px solid #888; padding: .7rem 0; }}
  article p {{ margin: .3rem 0; }}
  .note-text {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
</style>
</head>
<body>
<main>{content}</main>
</body>
</html>""",
        status_code=status_code,
        headers={
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": "frame-ancestors 'none'",
        },
    )


NOTE_FORM = """
<form data-note-form data-note-path="{root}/notes/" method="post" action="{root}/api/notes">
  <label for="value">Text note or link</label>
  <textarea id="value" name="value" rows="3" required></textarea>
  <button type="submit">Save note</button>
  <p role="status"></p>
</form>
"""

PHOTO_FORM = """
<form data-note-form data-note-path="{root}/notes/" method="post" action="{root}/api/photos" enctype="multipart/form-data">
  <label for="photo">Photo (JPEG or PNG, up to 8 MiB)</label>
  <input id="photo" name="photo" type="file" accept="image/jpeg,image/png" required>
  <label for="caption">Caption (optional)</label>
  <textarea id="caption" name="caption" rows="2" maxlength="12000"></textarea>
  <button type="submit">Save photo</button>
  <p role="status"></p>
</form>
"""

NOTE_FORM_SCRIPT = """
<script>
  for (const form of document.querySelectorAll("[data-note-form]")) {
    const button = form.querySelector("button");
    const status = form.querySelector("[role=status]");
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      button.disabled = true;
      status.textContent = "Saving…";
      try {
        const data = new FormData(form);
        const isPhoto = form.enctype === "multipart/form-data";
        const headers = isPhoto ? { "X-PKA-Upload": "1" } : { "Content-Type": "application/json" };
        const body = isPhoto ? data : JSON.stringify({ value: data.get("value") });
        const response = await fetch(form.action, { method: "POST", headers, body });
        const result = await response.json();
        if (!response.ok) {
          const message = typeof result.detail === "string" ? result.detail : "Invalid note.";
          throw new Error(message);
        }
        window.location.assign(form.dataset.notePath + encodeURIComponent(result.id));
      } catch (error) {
        status.textContent = error.message;
        button.disabled = false;
      }
    });
  }
</script>
"""


def render_home(
    notes: list[Note],
    q: str,
    *,
    root_path: str,
    result_limit: int,
    semantic_available: bool,
    semantic_selected: bool,
    photos_available: bool = False,
    search_error: str | None = None,
) -> HTMLResponse:
    root = quote(root_path.rstrip("/"))
    rows = []
    score_scale = ""
    for note in notes:
        preview = note.summary or note.value or "Photo"
        if len(preview) > 160:
            preview = preview[:157] + "…"
        metadata = escape(note.status)
        distance = note.cosine_distance if semantic_selected else None
        if distance is not None and isfinite(distance):
            metadata += f" · Similarity {1 - distance:z.2f}"
            score_scale = "<p>Similarity ranges from −1 to 1. Higher scores indicate closer matches.</p>"
        rows.append(
            f'<article><a href="{root}/notes/{note.id}">Note {note.id}</a> · {metadata}'
            f'<p class="note-text">{escape(preview)}</p></article>'
        )

    results = "".join(rows) if rows else "<p>No notes found.</p>"
    if search_error:
        keyword_query = escape(urlencode({"q": q, "mode": "keyword"}))
        results = (
            f'<p role="alert">{escape(search_error)} '
            f'<a href="{root}/?{keyword_query}">Try keyword search</a></p>'
        )
    mode_control = ""
    if semantic_available:
        selected = " selected" if semantic_selected else ""
        limits = "".join(
            f'<option value="{value}"{" selected" if value == result_limit else ""}>{value}</option>'
            for value in range(1, 51)
        )
        mode_control = f"""
  <label for="mode">Search mode</label>
  <select id="mode" name="mode">
    <option value="keyword">Keyword</option>
    <option value="semantic"{selected}>Semantic</option>
  </select>
  <label for="k">Results (semantic search)</label>
  <select id="k" name="k">{limits}</select>"""
    return page(
        "Notes",
        f"""<h1>Personal Knowledge Agent</h1>
<p>Save sources with English summaries and topic tags.</p>
{NOTE_FORM.format(root=root)}
{PHOTO_FORM.format(root=root) if photos_available else ""}
{NOTE_FORM_SCRIPT}
<form action="{root}/" method="get">
  <label for="q">Search notes</label>
  <input id="q" name="q" value="{escape(q)}" maxlength="200">{mode_control}
  <button type="submit">Search</button>
</form>
<h2>Notes</h2>
{score_scale}{results}""",
        status_code=503 if search_error else 200,
    )


def render_note(note: Note, *, root_path: str) -> HTMLResponse:
    root = quote(root_path.rstrip("/"))
    note_id = note.id
    label = {"url": "Submitted link", "image": "Caption", "text": "Submitted text"}
    fields = (
        (label[note.kind], note.value),
        ("Extracted text", note.text if note.kind != "image" else None),
        ("Summary", note.summary),
        ("Tags", ", ".join(note.tags)),
        ("Error", note.error),
    )
    sections = "".join(
        f'<h2>{label}</h2><p class="note-text">{escape(value)}</p>'
        for label, value in fields
        if value
    )
    if note.status in {"queued", "processing"}:
        sections += f'<p><a href="{root}/notes/{note_id}">Refresh status</a></p>'
    image = (
        f'<img src="{root}/notes/{note_id}/photo" alt="Uploaded photo">'
        if note.kind == "image"
        else ""
    )
    return page(
        f"Note {note_id}",
        f"""<p><a href="{root}/">All notes</a></p>
<h1>Note {note_id}</h1>
<p>Status <strong>{escape(note.status)}</strong></p>
{image}
{sections}""",
    )
