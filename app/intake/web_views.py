from dataclasses import dataclass
from html import escape
from math import isfinite
from pathlib import Path
from urllib.parse import quote, urlencode

from fastapi.responses import HTMLResponse

from app.notes.models import Note

NOTEBOOK_STYLE = Path(__file__).with_name("notebook.css").read_text(encoding="utf-8")
NOTE_TYPES = {"text": "Text", "url": "Link", "image": "Photo"}


@dataclass(frozen=True)
class SearchExplanation:
    text: str
    sources: tuple[Note, ...] = ()
    reranked: bool = False


def page(title: str, content: str, *, status_code: int = 200) -> HTMLResponse:
    return HTMLResponse(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} · Personal Knowledge Agent</title>
<style>{NOTEBOOK_STYLE}</style>
</head>
<body>
{content}
</body>
</html>""",
        status_code=status_code,
        headers={
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": "frame-ancestors 'none'",
        },
    )


HEADER = """
<header class="site-header">
  <div class="site-header-inner">
    <a class="brand" href="{root}/"><span aria-hidden="true">✳</span> Personal Knowledge Agent</a>
    {shortcut}
  </div>
</header>
"""

INPUT_SWITCHER = """
<div class="input-switcher" role="group" aria-label="What to save" hidden>
  <button type="button" aria-controls="text-form" aria-pressed="true">Text or link</button>
  <button type="button" aria-controls="photo-form" aria-pressed="false">Photo</button>
</div>
"""

NOTE_FORM = """
<form id="text-form" data-note-form data-note-path="{root}/notes/" method="post" action="{root}/api/notes">
  <label for="value">Note or link</label>
  <textarea id="value" name="value" rows="3"
    placeholder="Write something or paste a link…" required></textarea>
  <div class="form-actions">
    <button type="submit">Save note</button>
    <p role="status"></p>
  </div>
</form>
"""

PHOTO_FORM = """
<form id="photo-form" data-note-form data-note-path="{root}/notes/" method="post"
  action="{root}/api/photos" enctype="multipart/form-data">
  <div class="photo-picker">
    <label for="photo">Choose a photo</label>
    <input id="photo" name="photo" type="file" accept="image/jpeg,image/png" aria-describedby="photo-hint" required>
    <p id="photo-hint" class="hint">JPEG or PNG · Up to 8 MiB and 16 megapixels</p>
  </div>
  <label for="caption">Caption (optional)</label>
  <textarea id="caption" name="caption" rows="2" maxlength="12000"
    placeholder="Anything you want to remember about it?"></textarea>
  <div class="form-actions">
    <button type="submit">Save photo</button>
    <p role="status"></p>
  </div>
</form>
"""

HOME_SCRIPT = """
<script>
  const forms = document.querySelectorAll("[data-note-form]");
  const switcher = document.querySelector(".input-switcher");
  const inputTypes = switcher?.querySelectorAll("button") ?? [];
  if (switcher) {
    for (const button of inputTypes) {
      button.addEventListener("click", () => {
        for (const choice of inputTypes) choice.setAttribute("aria-pressed", String(choice === button));
        for (const form of forms) form.hidden = form.id !== button.getAttribute("aria-controls");
      });
    }
    inputTypes[0].click();
    switcher.hidden = false;
  }

  for (const form of forms) {
    const button = form.querySelector("button");
    const status = form.querySelector("[role=status]");
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      button.disabled = true;
      for (const choice of inputTypes) choice.disabled = true;
      status.textContent = "Saving…";
      try {
        const data = new FormData(form);
        const isPhoto = form.enctype === "multipart/form-data";
        const headers = isPhoto ? { "X-PKA-Upload": "1" } : { "Content-Type": "application/json" };
        const body = isPhoto ? data : JSON.stringify({ value: data.get("value") });
        const response = await fetch(form.action, { method: "POST", headers, body });
        const result = await response.json().catch(() => null);
        if (!response.ok) {
          status.textContent = typeof result?.detail === "string" ? result.detail : "Couldn't save. Try again.";
        } else if (result?.id) {
          window.location.assign(form.dataset.notePath + encodeURIComponent(result.id));
          return;
        } else {
          status.textContent = "Couldn't confirm the save. Check your recent notes.";
        }
      } catch {
        status.textContent = "Connection interrupted. Check your recent notes before retrying.";
      }
      button.disabled = false;
      for (const choice of inputTypes) choice.disabled = false;
    });
  }

  const mode = document.getElementById("mode");
  if (mode) {
    const count = document.getElementById("result-count");
    const updateCount = () => { count.hidden = mode.value !== "semantic"; };
    mode.addEventListener("change", updateCount);
    window.addEventListener("pageshow", updateCount);
    updateCount();
  }
</script>
"""


def note_preview(note: Note) -> str:
    preview = note.summary or note.value or "Photo"
    return preview if len(preview) <= 160 else preview[:157] + "…"


def render_search_explanation(explanation: SearchExplanation, root: str) -> str:
    sources = ""
    if explanation.sources:
        links = "".join(
            f'<li><a href="{root}/notes/{note.id}">{escape(note_preview(note))}</a></li>'
            for note in explanation.sources
        )
        sources = f"<h4>Sources</h4><ul>{links}</ul>"
    return f"""
    <section class="search-answer" aria-labelledby="answer-title">
      <h3 id="answer-title">Answer from your notes</h3>
      <p class="note-text">{escape(explanation.text)}</p>
      {sources}
    </section>"""


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
    explanation: SearchExplanation | None = None,
    answers_available: bool = False,
) -> HTMLResponse:
    root = quote(root_path.rstrip("/"))
    searching = bool(q.strip())
    ranked = searching and semantic_selected
    reranked = explanation is not None and explanation.reranked
    rows = []
    ranking_hints = []
    has_similarity = False
    for index, note in enumerate(notes):
        distance = note.cosine_distance if ranked else None
        metadata = NOTE_TYPES[note.kind]
        if note.status != "ready":
            metadata += f" · {escape(note.status.capitalize())}"
        top_match = ""
        if ranked and index == 0:
            top_match = '<span class="top-match">Top match</span> '
        if distance is not None and isfinite(distance):
            score_label = "Vector similarity" if reranked else "Similarity"
            metadata += f" · {score_label} {1 - distance:z.2f}"
            has_similarity = True
        rows.append(
            '<li class="note-row">'
            f'<p class="note-meta">{top_match}{metadata}</p>'
            f'<a class="note-text" href="{root}/notes/{note.id}">{escape(note_preview(note))}</a></li>'
        )

    if reranked:
        ranking_hints.append("The LLM reranks shortlisted notes with text.")
    if has_similarity:
        ranking_hints.append(
            "Similarity ranges from −1 to 1. Higher scores indicate closer vector matches."
        )
    ranking_hint = (
        f'<p class="hint">{" ".join(ranking_hints)}</p>' if ranking_hints else ""
    )
    title = "Search results" if searching else "Recent notes"
    empty = (
        "No matches. Try another description or keyword."
        if searching
        else "Your saved notes will appear here."
    )
    results = (
        f'<ul class="note-list">{"".join(rows)}</ul>' if rows else f"<p>{empty}</p>"
    )
    if search_error:
        keyword_query = escape(urlencode({"q": q, "mode": "keyword"}))
        results = (
            f'<p class="search-error" role="alert">{escape(search_error)} '
            f'<a href="{root}/?{keyword_query}#find">Try keyword search</a></p>'
        )
    mode_control = ""
    if semantic_available:
        meaning = " selected" if semantic_selected else ""
        keyword = "" if semantic_selected else " selected"
        limits = "".join(
            f'<option value="{value}"{" selected" if value == result_limit else ""}>{value}</option>'
            for value in range(1, 51)
        )
        mode_control = f"""
  <div class="search-options">
    <label for="mode">Match by
      <select id="mode" name="mode">
        <option value="semantic"{meaning}>Meaning</option>
        <option value="keyword"{keyword}>Keyword</option>
      </select>
    </label>
    <label for="k" id="result-count">Results
      <select id="k" name="k">{limits}</select>
    </label>
  </div>"""
    answer_panel = render_search_explanation(explanation, root) if explanation else ""
    query_hint = (
        "Try ‘What did I save about the blue door?’"
        if answers_available
        else "Try ‘a blue door’"
    )
    header = HEADER.format(root=root, shortcut='<a href="#find">Find a note ↓</a>')
    return page(
        "Notes",
        f"""{header}
<main class="notebook">
  <section class="composer" aria-labelledby="keep">
    <h1 id="keep">What do you want to keep?</h1>
    <p class="intro">Write a note, paste a link{", or add a photo" if photos_available else ""}.
      {"Find it later in your own words." if semantic_available else "Find it later by searching your notes."}</p>
    {INPUT_SWITCHER if photos_available else ""}
    {NOTE_FORM.format(root=root)}
    {PHOTO_FORM.format(root=root) if photos_available else ""}
  </section>
  <section id="find" class="search" aria-labelledby="find-title">
    <h2 id="find-title">Find a note</h2>
    <form action="{root}/#find" method="get">
      <label for="q">Search your notes</label>
      <div class="search-row">
        <input type="search" id="q" name="q" value="{escape(q)}" maxlength="200" placeholder="{query_hint}">
        <button type="submit">Search</button>
      </div>{mode_control}
    </form>
    {answer_panel}
    <h3>{title}</h3>
    {ranking_hint}{results}
  </section>
</main>
{HOME_SCRIPT}""",
        status_code=503 if search_error else 200,
    )


def render_note(note: Note, *, root_path: str) -> HTMLResponse:
    root = quote(root_path.rstrip("/"))
    header = HEADER.format(root=root, shortcut=f'<a href="{root}/">All notes</a>')
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
    refresh = (
        f' · <a href="{root}/notes/{note_id}">Refresh status</a>'
        if note.status in {"queued", "processing"}
        else ""
    )
    image = (
        f'<img src="{root}/notes/{note_id}/photo" alt="Uploaded photo">'
        if note.kind == "image"
        else ""
    )
    return page(
        f"Note {note_id}",
        f"""{header}
<main class="notebook detail">
  <h1>Note {note_id}</h1>
  <p class="note-meta">{NOTE_TYPES[note.kind]} · <strong>{escape(note.status.capitalize())}</strong>{refresh}</p>
  {image}
  {sections}
</main>""",
    )
