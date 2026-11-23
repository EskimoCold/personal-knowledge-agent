import asyncio
from collections.abc import Callable
from threading import Event, Lock
from typing import Protocol

from app.processing.exceptions import ProcessingError
from app.processing.search import SearchRequest, SearchResult, search_notes
from app.processing.summarizer import Summarizer


class NoteSummarizer(Protocol):
    def summarize(self, text: str) -> tuple[str, list[str]]: ...


async def run_native[T](call: Callable[[], T], stop_event: Event | None = None) -> T:
    task = asyncio.create_task(asyncio.to_thread(call))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        if stop_event is not None:
            stop_event.set()
        # Native inference must finish before the model can be reused or unloaded.
        drained = asyncio.gather(task, return_exceptions=True)
        while not drained.done():
            try:
                await asyncio.shield(drained)
            except asyncio.CancelledError:
                continue
        raise


class InferenceBusy(Exception):
    pass


class LocalInference:
    def __init__(self, summarizer: Summarizer):
        self._summarizer = summarizer
        self._lock = Lock()
        self._stopped = Event()

    def summarize(self, text: str) -> tuple[str, list[str]]:
        with self._lock:
            if self._stopped.is_set():
                raise ProcessingError("Local model is shutting down.")
            return self._summarizer.summarize(text)

    def search(self, request: SearchRequest, stop_event: Event) -> SearchResult:
        if self._stopped.is_set() or not self._lock.acquire(blocking=False):
            raise InferenceBusy
        try:
            if self._stopped.is_set():
                raise InferenceBusy
            return search_notes(self._summarizer, request, stop_event=stop_event)
        finally:
            self._lock.release()

    def stop(self) -> None:
        self._stopped.set()
        with self._lock:
            # Wait for the active call before its model context exits.
            pass
