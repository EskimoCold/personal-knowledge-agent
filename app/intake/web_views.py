from html import escape

from fastapi.responses import HTMLResponse

from app.notes.models import Note


def page(title: str, content: str) -> HTMLResponse:
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
  article {{ border-top: 1px solid #888; padding: .7rem 0; }}
  article p {{ margin: .3rem 0; }}
  .note-text {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
</style>
</head>
<body>
<main>{content}</main>
</body>
</html>""",
        headers={
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": "frame-ancestors 'none'",
        },
    )


NOTE_FORM = """
<form id="note-form" method="post" action="/api/notes">
  <label for="value">Text note or link</label>
  <textarea id="value" name="value" rows="3" required></textarea>
  <button type="submit">Save note</button>
  <p id="form-status" role="status"></p>
</form>
<script>
  const form = document.getElementById("note-form");
  const button = form.querySelector("button");
  const status = document.getElementById("form-status");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    button.disabled = true;
    status.textContent = "Saving…";
    try {
      const value = new FormData(form).get("value");
      const response = await fetch(form.action, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ value }),
      });
      const result = await response.json();
      if (!response.ok) {
        const message = typeof result.detail === "string" ? result.detail : "Invalid note.";
        throw new Error(message);
      }
      window.location.assign("/notes/" + encodeURIComponent(result.id));
    } catch (error) {
      status.textContent = error.message;
      button.disabled = false;
    }
  });
</script>
"""


def render_home(
    notes: list[Note], q: str, *, semantic_available: bool, semantic_selected: bool
) -> HTMLResponse:
    rows = []
    for note in notes:
        preview = note.summary or note.value
        if len(preview) > 160:
            preview = preview[:157] + "…"
        rows.append(
            f'<article><a href="/notes/{note.id}">Note {note.id}</a> · {escape(note.status)}'
            f'<p class="note-text">{escape(preview)}</p></article>'
        )

    results = "".join(rows) if rows else "<p>No notes found.</p>"
    mode_control = ""
    if semantic_available:
        selected = " selected" if semantic_selected else ""
        mode_control = f"""
  <label for="mode">Search mode</label>
  <select id="mode" name="mode">
    <option value="keyword">Keyword</option>
    <option value="semantic"{selected}>Semantic</option>
  </select>"""
    return page(
        "Notes",
        f"""<h1>Personal Knowledge Agent</h1>
<p>Local demo with stub processing.</p>
{NOTE_FORM}
<form action="/" method="get">
  <label for="q">Search notes</label>
  <input id="q" name="q" value="{escape(q)}" maxlength="200">{mode_control}
  <button type="submit">Search</button>
</form>
<h2>Notes</h2>
{results}""",
    )


def render_note(note: Note) -> HTMLResponse:
    note_id = note.id
    fields = (
        ("Submitted link" if note.kind == "url" else "Submitted text", note.value),
        ("Extracted text", note.text),
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
        sections += f'<p><a href="/notes/{note_id}">Refresh status</a></p>'
    return page(
        f"Note {note_id}",
        f"""<p><a href="/">All notes</a></p>
<h1>Note {note_id}</h1>
<p>Status <strong>{escape(note.status)}</strong> · Stub processing</p>
{sections}""",
    )
