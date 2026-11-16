import asyncio
import logging

from app.notes.models import NoteInput
from app.processing.embeddings import Embedder, EmbeddingError, open_embedder
from app.processing.exceptions import ProcessingError
from app.processing.processor import process
from app.processing.webpage.fetcher import PageFetcher
from app.storage.postgres import Postgres, open_postgres

logger = logging.getLogger(__name__)
PROCESSING_TIMEOUT_SECONDS = 150


async def run_once(
    db: Postgres, embedder: Embedder | None = None, fetcher: PageFetcher | None = None
) -> bool:
    job = await db.claim_job()
    if job is None:
        return False

    result = None
    embedding = None
    try:
        async with asyncio.timeout(PROCESSING_TIMEOUT_SECONDS):
            result = await process(NoteInput(job.kind, job.value), fetcher)
            if embedder is not None:
                try:
                    embedding = await embedder.embed(result.text)
                except EmbeddingError:
                    logger.warning("Embedding failed for note %s", job.id)
    except TimeoutError:
        if result is None:
            await db.finish_job(job, None, error="Note processing timed out.")
        else:
            logger.warning("Embedding timed out for note %s", job.id)
            await db.finish_job(job, result, embedding=None)
    except ProcessingError as exc:
        await db.finish_job(job, None, error=str(exc))
    except Exception:
        logger.error("Processor failed for note %s", job.id)
        await db.finish_job(job, None, error="Processing failed.")
    else:
        await db.finish_job(job, result, embedding=embedding)
    return True


async def main() -> None:
    async with (
        open_embedder() as embedder,
        open_postgres() as db,
        PageFetcher.open() as fetcher,
    ):
        while True:
            if not await run_once(db, embedder, fetcher):
                await asyncio.sleep(1)


if __name__ == "__main__":
    logging.basicConfig()
    asyncio.run(main())
