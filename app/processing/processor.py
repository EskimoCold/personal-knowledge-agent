from app.notes.models import NoteInput, ProcessedNote
from app.processing.exceptions import ProcessingError
from app.processing.webpage.exceptions import PageError
from app.processing.webpage.fetcher import PageFetcher


async def process(note: NoteInput, fetcher: PageFetcher | None = None) -> ProcessedNote:
    """Extract URL text, then produce a deterministic excerpt until a model exists."""
    text = note.value
    if note.kind == "url":
        try:
            if fetcher is None:
                async with PageFetcher.open() as fetcher:
                    page = await fetcher.fetch(note.value)
            else:
                page = await fetcher.fetch(note.value)
        except PageError as exc:
            raise ProcessingError(str(exc)) from None
        text = page.text

    excerpt = " ".join(text.split())
    summary = excerpt[:180] + ("…" if len(excerpt) > 180 else "")
    return ProcessedNote(
        text=text, summary=summary, tags=["text"] if note.kind == "text" else []
    )
