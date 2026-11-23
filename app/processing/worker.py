import asyncio
import os
import time
from contextlib import asynccontextmanager

import structlog
import uvicorn

from app.logging import configure_logging
from app.notes.models import Job, NoteInput, ProcessedNote
from app.processing.embeddings import Embedder, EmbeddingError, open_embedder
from app.processing.exceptions import ProcessingError
from app.processing.inference import LocalInference, NoteSummarizer, run_native
from app.processing.processor import extract_text, process
from app.processing.search_service import create_search_app
from app.processing.summarizer import open_summarizer
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
    summarizer: NoteSummarizer | None = None,
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
                    result, error, embedding = await _process_job(
                        db, job, embedder, fetcher, summarizer
                    )
                finally:
                    renewal.cancel()
            saved = await db.finish_job(job, result, error=error, embedding=embedding)
        except asyncio.CancelledError:
            logger.warning("job_cancelled")
            raise
        except Exception as exc:
            logger.error("job_aborted", error_type=type(exc).__name__)
            raise
        logger.info(
            "job_finished",
            status="failed" if error else "ready",
            persisted=bool(saved),
            duration_seconds=round(time.perf_counter() - started, 3),
        )
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
        image = job.image if job.kind == "image" else None
        if job.kind == "image" and (not image or embedder is None):
            return None, "Photo embedding is unavailable.", None
        if image and not text:
            result = ProcessedNote(text="", summary=None, tags=[])
        else:
            result = await process(NoteInput("text", text), summarizer=summarizer)
        if embedder is not None:
            stage = "embedding"
            logger.info("embedding_started")
            try:
                async with asyncio.timeout(EMBEDDING_TIMEOUT_SECONDS):
                    if image:
                        embedding = await embedder.embed(
                            result.summary or "", document=True, image=image
                        )
                    else:
                        embedding = await embedder.embed(
                            result.summary or "", document=True
                        )
                logger.info("embedding_completed")
            except (EmbeddingError, TimeoutError):
                logger.warning("embedding_failed")
                if image:
                    return None, "Photo embedding is unavailable.", None
    except TimeoutError:
        logger.error("job_processing_failed", stage=stage, reason="timed_out")
        return None, "Note processing timed out.", None
    except ProcessingError as exc:
        logger.warning("job_processing_failed", stage=stage, reason=str(exc))
        return None, str(exc), None
    except Exception as exc:  # noqa: BLE001 - isolate unexpected failures to this job.
        logger.error(
            "job_processing_failed", stage=stage, error_type=type(exc).__name__
        )
        return None, "Processing failed.", None
    else:
        return result, None, embedding


@asynccontextmanager
async def open_worker():
    with open_summarizer() as summarizer:
        async with (
            open_embedder() as embedder,
            open_postgres() as db,
            PageFetcher.open() as fetcher,
        ):
            inference = LocalInference(summarizer)
            try:
                yield db, embedder, fetcher, inference
            finally:
                await run_native(inference.stop)


async def poll(db, embedder, fetcher, inference: LocalInference) -> None:
    logger.info("worker_started")
    while True:
        if not await run_once(db, embedder, fetcher, inference):
            await asyncio.sleep(1)


async def serve_search_and_poll(host: str) -> None:
    failure = None

    def polling_stopped(task: asyncio.Task) -> None:
        nonlocal failure
        if not task.cancelled():
            failure = task.exception()
            server.should_exit = True

    @asynccontextmanager
    async def lifespan(app):
        async with open_worker() as (db, embedder, fetcher, inference):
            polling = asyncio.create_task(poll(db, embedder, fetcher, inference))
            polling.add_done_callback(polling_stopped)
            try:
                yield {"inference": inference}
            finally:
                polling.cancel()
                await asyncio.gather(polling, return_exceptions=True)

    server = uvicorn.Server(
        uvicorn.Config(
            create_search_app(lifespan),
            host=host,
            port=8000,
            access_log=False,
            limit_concurrency=8,
            log_config=None,
        )
    )
    await server.serve()
    if failure is not None:
        raise failure


async def main() -> None:
    configure_logging()
    host = os.environ.get("SEARCH_INFERENCE_HOST", "").strip()
    if host:
        await serve_search_and_poll(host)
    else:
        async with open_worker() as resources:
            await poll(*resources)


if __name__ == "__main__":
    asyncio.run(main())
