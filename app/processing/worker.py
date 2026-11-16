import asyncio
import logging

from app.notes.models import Job, NoteInput
from app.processing.embeddings import Embedder, EmbeddingError, open_embedder
from app.processing.exceptions import ProcessingError
from app.processing.processor import extract_text, process
from app.processing.summarizer import Summarizer, open_summarizer
from app.processing.webpage.fetcher import PageFetcher
from app.storage.postgres import Postgres, open_postgres

logger = logging.getLogger(__name__)
PROCESSING_TIMEOUT_SECONDS = 150
EMBEDDING_TIMEOUT_SECONDS = 30
LEASE_RENEWAL_SECONDS = 60


async def run_once(
    db: Postgres,
    embedder: Embedder | None = None,
    fetcher: PageFetcher | None = None,
    summarizer: Summarizer | None = None,
) -> bool:
    job = await db.claim_job()
    if job is None:
        return False

    async with asyncio.TaskGroup() as group:
        renewal = group.create_task(_renew_lease(db, job))
        try:
            result, error, embedding = await _process_job(
                db, job, embedder, fetcher, summarizer
            )
        finally:
            renewal.cancel()
    await db.finish_job(job, result, error=error, embedding=embedding)
    return True


async def _renew_lease(db: Postgres, job: Job) -> None:
    while True:
        await asyncio.sleep(LEASE_RENEWAL_SECONDS)
        if not await db.renew_job(job):
            raise RuntimeError("The worker lost its job claim.")


async def _process_job(db, job, embedder, fetcher, summarizer):
    result = None
    embedding = None
    try:
        async with asyncio.timeout(PROCESSING_TIMEOUT_SECONDS):
            text = await extract_text(NoteInput(job.kind, job.value), fetcher)
        if not await db.renew_job(job, text=text):
            raise RuntimeError("The worker lost its job claim.")
        result = await process(NoteInput("text", text), summarizer=summarizer)
        if embedder is not None:
            try:
                async with asyncio.timeout(EMBEDDING_TIMEOUT_SECONDS):
                    embedding = await embedder.embed(result.text)
            except (EmbeddingError, TimeoutError):
                logger.warning("Embedding failed for note %s", job.id)
    except TimeoutError:
        return None, "Note processing timed out.", None
    except ProcessingError as exc:
        return None, str(exc), None
    except Exception:  # noqa: BLE001 - isolate unexpected failures to this job.
        logger.error("Processor failed for note %s", job.id)
        return None, "Processing failed.", None
    else:
        return result, None, embedding


async def main() -> None:
    with open_summarizer() as summarizer:
        async with (
            open_embedder() as embedder,
            open_postgres() as db,
            PageFetcher.open() as fetcher,
        ):
            while True:
                if not await run_once(db, embedder, fetcher, summarizer):
                    await asyncio.sleep(1)


if __name__ == "__main__":
    logging.basicConfig()
    asyncio.run(main())
