import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import asyncpg

from app.notes.models import Embedding, Job, Note, NoteInput, ProcessedNote


@asynccontextmanager
async def open_postgres(dsn: str | None = None) -> AsyncIterator["Postgres"]:
    async with asyncpg.create_pool(
        dsn if dsn is not None else os.environ["DATABASE_URL"],
        min_size=1,
        max_size=4,
        timeout=5,
        command_timeout=15,
    ) as pool:
        yield Postgres(pool)


class Postgres:
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def enqueue(
        self,
        owner_id: str,
        note_input: NoteInput,
        *,
        telegram_message_id: int | None = None,
    ) -> int:
        return await self._pool.fetchval(
            """
            INSERT INTO notes (owner_id, kind, value, telegram_message_id)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (owner_id, telegram_message_id)
            DO UPDATE SET telegram_message_id = EXCLUDED.telegram_message_id
            RETURNING id
            """,
            owner_id,
            note_input.kind,
            note_input.value,
            telegram_message_id,
        )

    async def list_notes(self, owner_id: str, query: str = "") -> list[Note]:
        rows = await self._pool.fetch(
            """
            SELECT id, kind, value, status, text, summary, tags, error
            FROM notes
            WHERE owner_id = $1
              AND ($2 = '' OR (
                  (status = 'ready' OR kind = 'url') AND strpos(
                      lower(concat_ws(' ', value, text, summary, array_to_string(tags, ' '))),
                      lower($2)
                  ) > 0
              ))
            ORDER BY id DESC
            LIMIT 50
            """,
            owner_id,
            query.strip(),
        )
        return [Note(**row) for row in rows]

    async def get_note(self, owner_id: str, note_id: int) -> Note | None:
        row = await self._pool.fetchrow(
            """
            SELECT id, kind, value, status, text, summary, tags, error
            FROM notes
            WHERE owner_id = $1 AND id = $2
            """,
            owner_id,
            note_id,
        )
        return Note(**row) if row is not None else None

    async def claim_job(self) -> Job | None:
        row = await self._pool.fetchrow(
            """
            WITH next_note AS (
                SELECT id FROM notes
                WHERE status = 'queued' OR (status = 'processing' AND lease_until <= now())
                ORDER BY id
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE notes
            SET status = 'processing', claim_token = $1, lease_until = now() + interval '2 minutes'
            FROM next_note
            WHERE notes.id = next_note.id
            RETURNING notes.id, notes.kind, notes.value, notes.claim_token
            """,
            uuid4(),
        )
        return Job(**row) if row is not None else None

    async def finish_job(
        self,
        job: Job,
        result: ProcessedNote | None,
        *,
        error: str | None = None,
        embedding: Embedding | None = None,
    ) -> bool:
        if result is None:
            status, text, summary, tags = "failed", None, None, []
            embedding = None
        else:
            status, text, summary, tags = (
                "ready",
                result.text,
                result.summary,
                result.tags,
            )

        note_id = await self._pool.fetchval(
            """
            UPDATE notes
            SET status = $1, text = $2, summary = $3, tags = $4, error = $5,
                embedding = $6::double precision[]::vector, embedding_model = $7,
                claim_token = NULL, lease_until = NULL
            WHERE id = $8 AND claim_token = $9
            RETURNING id
            """,
            status,
            text,
            summary,
            tags,
            error,
            embedding.values if embedding is not None else None,
            embedding.model if embedding is not None else None,
            job.id,
            job.claim_token,
        )
        return note_id is not None

    async def search_similar(self, owner_id: str, embedding: Embedding) -> list[Note]:
        rows = await self._pool.fetch(
            """
            SELECT id, kind, value, status, text, summary, tags, error
            FROM notes
            WHERE owner_id = $1 AND status = 'ready'
              AND embedding_model = $2 AND public.vector_dims(embedding) = $3
            ORDER BY embedding <=> $4::double precision[]::vector, id DESC
            LIMIT 20
            """,
            owner_id,
            embedding.model,
            len(embedding.values),
            embedding.values,
        )
        return [Note(**row) for row in rows]
