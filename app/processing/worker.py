import asyncio
import logging

from app.notes.models import NoteInput
from app.processing.embeddings import Embedder, EmbeddingError, open_embedder
from app.processing.processor import ProcessingError, process
from app.storage.postgres import Postgres, open_postgres


logger = logging.getLogger(__name__)


async def run_once(db: Postgres, embedder: Embedder | None = None) -> bool:
    job = await db.claim_job()
    if job is None:
        return False

    try:
        result = process(NoteInput(job.kind, job.value))
    except ProcessingError as exc:
        await db.finish_job(job, None, error=str(exc))
    except Exception:
        logger.error("Processor failed for note %s", job.id)
        await db.finish_job(job, None, error="Processing failed.")
    else:
        embedding = None
        if embedder is not None:
            try:
                embedding = await embedder.embed(result.text)
            except EmbeddingError:
                logger.warning("Embedding failed for note %s", job.id)
        await db.finish_job(job, result, embedding=embedding)
    return True


async def main() -> None:
    async with open_embedder() as embedder, open_postgres() as db:
        while True:
            if not await run_once(db, embedder):
                await asyncio.sleep(1)


if __name__ == "__main__":
    logging.basicConfig()
    asyncio.run(main())
