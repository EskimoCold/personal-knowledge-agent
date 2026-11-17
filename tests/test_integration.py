import asyncio
import os
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import asyncpg
import httpx
from fastapi.testclient import TestClient

from app.intake import web
from app.intake.telegram import handle_update
from app.notes.models import Embedding, NoteInput
from app.processing.processor import process
from app.processing.webpage.config import WebpageConfig
from app.processing.webpage.fetcher import PageFetcher
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
        self.summarizer = SimpleNamespace(
            summarize=Mock(return_value=("An English model summary.", []))
        )
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
        article = (
            Path(__file__).with_name("fixtures").joinpath("article.html").read_bytes()
        )

        def respond(request):
            body = article if request.url.path == "/garden" else b"<html></html>"
            return httpx.Response(
                200,
                headers={"Content-Type": "text/html"},
                stream=httpx.ByteStream(body),
            )

        self.page_client = await self.enterAsyncContext(
            httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False)
        )
        self.fetcher = PageFetcher(self.page_client, WebpageConfig())

    async def run_job(self, db, embedder=None, fetcher=None):
        return await run_once(db, embedder, fetcher, summarizer=self.summarizer)

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
                "https://article.test/garden",
                "http://127.0.0.1/",
            ):
                response = client.post("/api/notes", auth=owner, json={"value": value})
                self.assertEqual(response.status_code, 201)
                paths.append(f"/notes/{response.json()['id']}")

            self.assertIn("queued", client.get(paths[0], auth=owner).text)
            self.assertTrue(await self.run_job(db, fetcher=self.fetcher))
            self.assertTrue(await self.run_job(db, fetcher=self.fetcher))
            self.assertTrue(await self.run_job(db, fetcher=self.fetcher))
            self.assertFalse(await self.run_job(db))

            search = {"q": "МЕЖСЕТЕВОЙ"}
            self.assertIn(paths[0], client.get("/", auth=owner, params=search).text)
            self.assertNotIn(paths[0], client.get("/", auth=other, params=search).text)
            self.assertEqual(client.get(paths[0], auth=other).status_code, 404)
            self.assertIn("fresh vegetables", client.get(paths[1], auth=owner).text)
            self.assertIn("failed", client.get(paths[2], auth=owner).text)

    async def test_failed_and_pending_links_are_searchable_only_by_the_owner(self):
        db = await self.start_database()
        failed = await db.enqueue(
            "1", NoteInput.from_text("https://example.org/report")
        )
        self.assertTrue(await self.run_job(db, fetcher=self.fetcher))
        self.assertEqual((await db.get_note("1", failed)).status, "failed")
        pending = await db.enqueue(
            "1", NoteInput.from_text("https://example.org/draft")
        )
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
                original_claim,
                await process(note, summarizer=self.summarizer),
                embedding=stale_embedding,
            )
        )
        self.assertTrue(
            await db.finish_job(
                replacement, await process(note, summarizer=self.summarizer)
            )
        )
        self.assertEqual((await db.get_note("1", note_id)).status, "ready")
        self.assertEqual(await db.search_similar("1", stale_embedding), [])
        self.assertIsNone(await db.claim_job())

    async def test_extraction_failure_does_not_stop_later_article_persistence(self):
        db = await self.start_database()
        failed_id = await db.enqueue(
            "1", NoteInput("url", "https://article.test/empty")
        )
        article_id = await db.enqueue(
            "1", NoteInput("url", "https://article.test/garden")
        )
        self.assertTrue(await self.run_job(db, fetcher=self.fetcher))
        self.assertTrue(await self.run_job(db, fetcher=self.fetcher))
        self.assertFalse(await self.run_job(db, fetcher=self.fetcher))
        failed = await db.get_note("1", failed_id)
        self.assertEqual(failed.status, "failed")
        self.assertIn("article", failed.error)
        article = await db.get_note("1", article_id)
        self.assertEqual(article.status, "ready")
        self.assertIn("fresh vegetables", article.text)
        self.assertNotIn("Navigation marker", article.text)
        self.assertEqual(article.tags, [])
        self.assertEqual(article.summary, "An English model summary.")

    async def test_claim_covers_fetch_retries_and_optional_embedding_budget(self):
        db = await self.start_database()
        note_id = await db.enqueue("1", NoteInput("url", "https://article.test/garden"))
        self.assertEqual((await db.claim_job()).id, note_id)
        # Simulate 90 seconds fetching, 10 seconds backoff and 30 seconds embedding.
        async with connection(self.dsn) as conn:
            await conn.execute(
                "UPDATE notes SET lease_until = lease_until - interval '130 seconds' WHERE id = $1",
                note_id,
            )
        self.assertIsNone(await db.claim_job())

    async def test_model_failure_retains_searchable_source_and_later_job_runs(self):
        from app.processing.exceptions import ProcessingError

        db = await self.start_database()
        failed_id = await db.enqueue("1", NoteInput("text", "Retained failure marker"))
        later_id = await db.enqueue("1", NoteInput("text", "Next source"))
        self.summarizer.summarize.side_effect = [
            ProcessingError("Local model generation failed."),
            ("English result.", ["science"]),
        ]
        await self.run_job(db)
        failed = await db.get_note("1", failed_id)
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.text, "Retained failure marker")
        self.assertIsNone(failed.summary)
        self.assertEqual(failed.tags, [])
        self.assertEqual(
            [n.id for n in await db.list_notes("1", "failure marker")], [failed_id]
        )
        self.assertEqual(await db.list_notes("2", "failure marker"), [])
        await self.run_job(db)
        self.assertEqual((await db.get_note("1", later_id)).status, "ready")

    async def test_renewal_saves_source_and_old_claim_cannot_renew_or_finish(self):
        db = await self.start_database()
        note_id = await db.enqueue("1", NoteInput("url", "https://article.test/garden"))
        original = await db.claim_job()
        self.assertTrue(await db.renew_job(original, text="Extracted source"))
        self.assertEqual((await db.get_note("1", note_id)).text, "Extracted source")
        async with connection(self.dsn) as conn:
            await conn.execute(
                "UPDATE notes SET lease_until = now() WHERE id = $1", note_id
            )
        self.assertFalse(await db.renew_job(original, text="Stale source"))
        replacement = await db.claim_job()
        self.assertFalse(await db.renew_job(original))
        self.assertFalse(await db.finish_job(original, None, error="stale"))
        self.assertTrue(await db.finish_job(replacement, None, error="model failed"))
        self.assertEqual((await db.get_note("1", note_id)).text, "Extracted source")

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
        self.assertTrue(await self.run_job(db))
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
            self.assertTrue(
                await self.run_job(db, embedder if text in vectors else None)
            )

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
