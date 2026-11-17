import asyncio
import time

import structlog

from app.logging import configure_logging
from app.notes.models import Job, NoteInput
from app.processing.embeddings import Embedder, EmbeddingError, open_embedder
from app.processing.exceptions import ProcessingError
from app.processing.processor import extract_text, process
from app.processing.summarizer import Summarizer, open_summarizer
from app.processing.webpage.fetcher import PageFetcher
from app.storage.postgres import Postgres, open_postgres

logger = structlog.get_logger(__name__)
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

    with structlog.contextvars.bound_contextvars(note_id=job.id, kind=job.kind):
        started = time.perf_counter()
        logger.info("job_started")
        try:
            async with asyncio.TaskGroup() as group:
                renewal = group.create_task(_renew_lease(db, job))
                try:
                    result, error, embedding = await _process_job(db, job, embedder, fetcher, summarizer)
                finally:
                    renewal.cancel()
            saved = await db.finish_job(job, result, error=error, embedding=embedding)
        except asyncio.CancelledError:
            logger.warning("job_cancelled")
            raise
        except Exception as exc:
            logger.error("job_aborted", error_type=type(exc).__name__)
            raise
        logger.info("job_finished", status="failed" if error else "ready", persisted=bool(saved),
                    duration_seconds=round(time.perf_counter() - started, 3))
    return True


async def _renew_lease(db: Postgres, job: Job) -> None:
    while True:
        await asyncio.sleep(LEASE_RENEWAL_SECONDS)
        if not await db.renew_job(job):
            logger.error("job_lease_lost")
            raise RuntimeError("The worker lost its job claim.")
        logger.info("job_lease_renewed")


async def _process_job(db, job, embedder, fetcher, summarizer):
    result = None
    embedding = None
    stage = "extraction"
    try:
        logger.info("source_extraction_started")
        async with asyncio.timeout(PROCESSING_TIMEOUT_SECONDS):
            text = await extract_text(NoteInput(job.kind, job.value), fetcher)
        logger.info("source_extracted", source_characters=len(text))
        stage = "source_checkpoint"
        if not await db.renew_job(job, text=text):
            raise RuntimeError("The worker lost its job claim.")
        stage = "generation"
        result = await process(NoteInput("text", text), summarizer=summarizer)
        if embedder is not None:
            stage = "embedding"
            logger.info("embedding_started")
            try:
                async with asyncio.timeout(EMBEDDING_TIMEOUT_SECONDS):
                    embedding = await embedder.embed(result.text)
                logger.info("embedding_completed")
            except (EmbeddingError, TimeoutError):
                logger.warning("embedding_failed")
    except TimeoutError:
        logger.error("job_processing_failed", stage=stage, reason="timed_out")
        return None, "Note processing timed out.", None
    except ProcessingError as exc:
        logger.warning("job_processing_failed", stage=stage, reason=str(exc))
        return None, str(exc), None
    except Exception as exc:  # noqa: BLE001 - isolate unexpected failures to this job.
        logger.error("job_processing_failed", stage=stage, error_type=type(exc).__name__)
        return None, "Processing failed.", None
    else:
        return result, None, embedding


async def main() -> None:
    configure_logging()
    with open_summarizer() as summarizer:
        async with (
            open_embedder() as embedder,
            open_postgres() as db,
            PageFetcher.open() as fetcher,
        ):
            logger.info("worker_started")
            while True:
                if not await run_once(db, embedder, fetcher, summarizer):
                    await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
