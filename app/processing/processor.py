import asyncio

from app.notes.models import NoteInput, ProcessedNote
from app.processing.exceptions import ProcessingError
from app.processing.summarizer import Summarizer
from app.processing.webpage.exceptions import PageError
from app.processing.webpage.fetcher import PageFetcher


async def extract_text(note: NoteInput, fetcher: PageFetcher | None = None) -> str:
    text = note.value
    if note.kind == "url":
        try:
            if fetcher is None:
                async with PageFetcher.open() as owned_fetcher:
                    page = await owned_fetcher.fetch(note.value)
            else:
                page = await fetcher.fetch(note.value)
        except PageError as exc:
            raise ProcessingError(str(exc)) from None
        text = page.text

    return text


async def process(
    note: NoteInput,
    fetcher: PageFetcher | None = None,
    summarizer: Summarizer | None = None,
) -> ProcessedNote:
    text = await extract_text(note, fetcher)
    if summarizer is None:
        raise ProcessingError("Local summarizer is not configured.")
    task = asyncio.create_task(asyncio.to_thread(summarizer.summarize, text))
    try:
        summary, tags = await asyncio.shield(task)
    except asyncio.CancelledError:
        drained = asyncio.gather(task, return_exceptions=True)
        while not drained.done():
            try:
                await asyncio.shield(drained)
            except asyncio.CancelledError:
                continue
        raise
    return ProcessedNote(text=text, summary=summary, tags=tags)
