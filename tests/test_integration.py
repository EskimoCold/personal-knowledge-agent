import asyncio
import os
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import asyncpg
import httpx
from fastapi.testclient import TestClient

from app.intake import web
from app.intake.telegram import handle_update
from app.notes.models import Embedding, NoteInput
from app.processing.processor import DEMO_URL, process
from app.processing.worker import run_once
from app.storage.postgres import open_postgres


@asynccontextmanager
async def connection(dsn):
    conn = await asyncpg.connect(dsn)
    try:
        yield conn
    finally:
        await conn.close()


@unittest.skipUnless(
    os.environ.get("PKA_TEST_DATABASE_URL"), "requires Postgres with pgvector"
)
class PostgresIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        base_dsn = os.environ["PKA_TEST_DATABASE_URL"]
        self.schema_name = f"test_{uuid4().hex}"
        async with connection(base_dsn) as conn:
            await conn.execute(f"CREATE SCHEMA {self.schema_name}")
        self.addAsyncCleanup(self.drop_schema, base_dsn, self.schema_name)
        separator = "&" if "?" in base_dsn else "?"
        self.dsn = f"{base_dsn}{separator}search_path={self.schema_name},public"
        self.enterContext(
            patch.dict(
                os.environ,
                {
                    "DATABASE_URL": self.dsn,
                    "PKA_WEB_USERS": '{"1":"first","2":"second"}',
                    "EMBEDDINGS_URL": "",
                    "EMBEDDINGS_MODEL": "",
                    "EMBEDDINGS_DIMENSIONS": "",
                    "EMBEDDINGS_API_KEY": "",
                },
            )
        )

    async def drop_schema(self, dsn, schema):
        async with connection(dsn) as conn:
            await conn.execute(f"DROP SCHEMA {schema} CASCADE")

    async def start_database(self):
        await self.upgrade_database()
        return await self.enterAsyncContext(open_postgres(self.dsn))

    async def upgrade_database(self):
        migration = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "alembic",
            "upgrade",
            "head",
            cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ, DATABASE_URL=self.dsn),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await migration.communicate()
        self.assertEqual(migration.returncode, 0, output.decode(errors="replace"))

    async def test_submission_processing_and_owner_scoped_search(self):
        db = await self.start_database()
        with TestClient(web.app) as client:
            owner = ("1", "first")
            other = ("2", "second")
            paths = []
            for value in (
                "Нужно настроить межсетевой экран.",
                DEMO_URL,
                "http://127.0.0.1/",
            ):
                response = client.post("/api/notes", auth=owner, json={"value": value})
                self.assertEqual(response.status_code, 201)
                paths.append(f"/notes/{response.json()['id']}")

            self.assertIn("queued", client.get(paths[0], auth=owner).text)
            self.assertTrue(await run_once(db))
            self.assertTrue(await run_once(db))
            self.assertTrue(await run_once(db))
            self.assertFalse(await run_once(db))

            search = {"q": "МЕЖСЕТЕВОЙ"}
            self.assertIn(paths[0], client.get("/", auth=owner, params=search).text)
            self.assertNotIn(paths[0], client.get("/", auth=other, params=search).text)
            self.assertEqual(client.get(paths[0], auth=other).status_code, 404)
            self.assertIn(
                "Пример сохранения ссылки.", client.get(paths[1], auth=owner).text
            )
            self.assertIn("failed", client.get(paths[2], auth=owner).text)

    async def test_failed_and_pending_links_are_searchable_only_by_the_owner(self):
        db = await self.start_database()
        failed = await db.enqueue("1", NoteInput.from_text("https://example.org/report"))
        self.assertTrue(await run_once(db))
        self.assertEqual((await db.get_note("1", failed)).status, "failed")
        pending = await db.enqueue("1", NoteInput.from_text("https://example.org/draft"))
        await db.enqueue("1", NoteInput.from_text("Text note about example.org"))

        matches = await db.list_notes("1", "EXAMPLE.ORG")
        self.assertEqual([note.id for note in matches], [pending, failed])
        self.assertEqual(await db.list_notes("2", "example.org"), [])

    async def test_deduplication_and_recovery_from_an_expired_claim(self):
        db = await self.start_database()
        note = NoteInput.from_text("Original note")
        note_id = await db.enqueue("1", note, telegram_message_id=42)
        self.assertEqual(
            await db.enqueue(
                "1", NoteInput.from_text("Redelivery"), telegram_message_id=42
            ),
            note_id,
        )
        self.assertEqual((await db.get_note("1", note_id)).value, note.value)

        claims = [
            job for job in await asyncio.gather(db.claim_job(), db.claim_job()) if job
        ]
        self.assertEqual(len(claims), 1)
        original_claim = claims[0]
        self.assertEqual((await db.get_note("1", note_id)).status, "processing")

        async with connection(self.dsn) as conn:
            await conn.execute(
                "UPDATE notes SET lease_until = now() WHERE id = $1", note_id
            )

        replacement = await db.claim_job()
        self.assertEqual(replacement.id, note_id)
        stale_embedding = Embedding("stale", [1.0, 0.0])
        self.assertFalse(
            await db.finish_job(
                original_claim, process(note), embedding=stale_embedding
            )
        )
        self.assertTrue(await db.finish_job(replacement, process(note)))
        self.assertEqual((await db.get_note("1", note_id)).status, "ready")
        self.assertEqual(await db.search_similar("1", stale_embedding), [])
        self.assertIsNone(await db.claim_job())

    async def test_telegram_intake_is_searchable_in_web(self):
        db = await self.start_database()

        def respond(request):
            self.assertEqual(request.url.path, "/sendMessage")
            return httpx.Response(200, json={"ok": True, "result": {}})

        update = {
            "message": {
                "message_id": 42,
                "chat": {"id": 1, "type": "private"},
                "from": {"id": 1},
                "text": "Сохранено через Telegram",
            },
        }
        async with httpx.AsyncClient(
            base_url="https://telegram.test", transport=httpx.MockTransport(respond)
        ) as client:
            await handle_update(client, db, {1}, update)
            await handle_update(client, db, {1}, update)

        [note] = await db.list_notes("1")
        self.assertTrue(await run_once(db))
        with TestClient(web.app) as client:
            response = client.get("/", auth=("1", "first"), params={"q": "Сохранено"})
            self.assertIn(f"/notes/{note.id}", response.text)

    async def test_semantic_search_ranks_compatible_embeddings_for_the_owner(self):
        db = await self.start_database()
        query = "network perimeter"
        query_embedding = Embedding("test-model", [1.0, 0.0])
        vectors = {
            query: query_embedding,
            "Office firewall": query_embedding,
            "Garden tools": Embedding("test-model", [0.0, 1.0]),
            "Different model": Embedding("other-model", [1.0, 0.0]),
            "Different dimensions": Embedding("test-model", [1.0, 0.0, 0.0]),
            "Private firewall": query_embedding,
        }

        class TestEmbedder:
            async def embed(self, text):
                return vectors[text]

        embedder = TestEmbedder()
        note_ids = {}
        for text in (
            "Office firewall",
            "Garden tools",
            "Different model",
            "Different dimensions",
            "Private firewall",
            "Keyword-only note",
        ):
            owner_id = "2" if text == "Private firewall" else "1"
            note_ids[text] = await db.enqueue(owner_id, NoteInput.from_text(text))
            self.assertTrue(await run_once(db, embedder if text in vectors else None))

        async with connection(self.dsn) as conn:
            stored = await conn.fetchrow(
                "SELECT embedding_model, embedding::text FROM notes WHERE id = $1",
                note_ids["Office firewall"],
            )
            self.assertEqual(tuple(stored), ("test-model", "[1,0]"))
            unembedded = await conn.fetchrow(
                "SELECT embedding_model, embedding::text FROM notes WHERE id = $1",
                note_ids["Keyword-only note"],
            )
            self.assertEqual(tuple(unembedded), (None, None))

        @asynccontextmanager
        async def configured_embedder():
            yield embedder

        with (
            patch.object(web, "open_embedder", configured_embedder),
            TestClient(web.app) as client,
        ):
            links = {
                text: f'href="/notes/{note_id}"' for text, note_id in note_ids.items()
            }
            search = {"q": query, "mode": "semantic"}
            response = client.get("/", auth=("1", "first"), params=search)
            self.assertEqual(response.status_code, 200)
            self.assertLess(
                response.text.index(links["Office firewall"]),
                response.text.index(links["Garden tools"]),
            )
            for text in (
                "Different model",
                "Different dimensions",
                "Private firewall",
                "Keyword-only note",
            ):
                self.assertNotIn(links[text], response.text)

            private_results = client.get("/", auth=("2", "second"), params=search)
            self.assertIn(links["Private firewall"], private_results.text)
            self.assertNotIn(links["Office firewall"], private_results.text)

            keyword_results = client.get(
                "/", auth=("1", "first"), params={"q": "Keyword-only"}
            )
            self.assertIn(links["Keyword-only note"], keyword_results.text)

    async def test_concurrent_migrations_create_the_fresh_schema(self):
        await asyncio.gather(self.upgrade_database(), self.upgrade_database())

        async with connection(self.dsn) as conn:
            self.assertEqual(
                await conn.fetchval(
                    f"SELECT version_num FROM {self.schema_name}.alembic_version"
                ),
                "002_embeddings",
            )
            self.assertIsNone(
                await conn.fetchval(
                    "SELECT to_regclass($1)", f"{self.schema_name}.note_embeddings"
                )
            )
