from app.notes.models import NoteInput, ProcessedNote


DEMO_URL = "https://example.com/"


class ProcessingError(Exception):
    """Expected processing failure with a message safe to display to the owner."""


def process(note: NoteInput) -> ProcessedNote:
    """Run deterministic demo processing without network access."""
    if note.kind == "url":
        if note.value not in (DEMO_URL, DEMO_URL.rstrip("/")):
            raise ProcessingError(f"The demo processor supports only {DEMO_URL}.")
        return ProcessedNote(
            text="Учебная страница показывает, как сохранить и найти публичную ссылку.",
            summary="Пример сохранения ссылки.",
            tags=["demo", "link"],
        )

    excerpt = " ".join(note.value.split())
    summary = excerpt[:180] + ("…" if len(excerpt) > 180 else "")
    return ProcessedNote(text=note.value, summary=summary, tags=["text"])
