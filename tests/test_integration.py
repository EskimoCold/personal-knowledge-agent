import asyncio
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import asyncpg
import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.intake import web
from app.intake.telegram import handle_update
from app.notes.models import Embedding, NoteInput
from app.processing.embeddings import EMBEDDINGGEMMA_MODEL, open_embedder
from app.processing.processor import process
from app.processing.summarizer import open_summarizer
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


@pytest.mark.skipif(
    not os.environ.get("PKA_TEST_DATABASE_URL"),
    reason="requires Postgres with pgvector",
)
class TestPostgresIntegration:
    @pytest.fixture(autouse=True)
    async def database_resources(self):
        async with AsyncExitStack() as self.stack:
            self.summarizer = SimpleNamespace(
                summarize=Mock(return_value=("An English model summary.", []))
            )
            base_dsn = os.environ["PKA_TEST_DATABASE_URL"]
            self.schema_name = f"test_{uuid4().hex}"
            async with connection(base_dsn) as conn:
                await conn.execute(f"CREATE SCHEMA {self.schema_name}")
            self.stack.push_async_callback(self.drop_schema, base_dsn, self.schema_name)
            separator = "&" if "?" in base_dsn else "?"
            self.dsn = f"{base_dsn}{separator}search_path={self.schema_name},public"
            self.stack.enter_context(
                patch.dict(
                    os.environ,
                    {
                        "DATABASE_URL": self.dsn,
                        "PKA_WEB_USERS": '{"1":"first","2":"second"}',
                        "EMBEDDINGS_URL": "",
                        "EMBEDDINGS_MODEL": "",
                        "EMBEDDINGS_DIMENSIONS": "",
                        "EMBEDDINGS_API_KEY": "",
                        "LOCAL_EMBEDDINGS_URL": "",
                    },
                )
            )
            article = (
                Path(__file__)
                .with_name("fixtures")
                .joinpath("article.html")
                .read_bytes()
            )

            def respond(request):
                if request.url.host == "api.keenable.ai":
                    return httpx.Response(422)
                body = article if request.url.path == "/garden" else b"<html></html>"
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/html"},
                    stream=httpx.ByteStream(body),
                )

            self.page_client = await self.stack.enter_async_context(
                httpx.AsyncClient(
                    transport=httpx.MockTransport(respond), trust_env=False
                )
            )
            self.fetcher = PageFetcher(self.page_client, WebpageConfig())

            yield

    async def run_job(self, db, embedder=None, fetcher=None):
        return await run_once(db, embedder, fetcher, summarizer=self.summarizer)

    async def drop_schema(self, dsn, schema):
        async with connection(dsn) as conn:
            await conn.execute(f"DROP SCHEMA {schema} CASCADE")

    async def start_database(self):
        await self.upgrade_database()
        return await self.stack.enter_async_context(open_postgres(self.dsn))

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
        assert migration.returncode == 0, output.decode(errors="replace")

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
                assert response.status_code == 201
                paths.append(f"/notes/{response.json()['id']}")

            assert "queued" in client.get(paths[0], auth=owner).text
            assert await self.run_job(db, fetcher=self.fetcher)
            assert await self.run_job(db, fetcher=self.fetcher)
            assert await self.run_job(db, fetcher=self.fetcher)
            assert not await self.run_job(db)

            search = {"q": "МЕЖСЕТЕВОЙ"}
            assert paths[0] in client.get("/", auth=owner, params=search).text
            assert paths[0] not in client.get("/", auth=other, params=search).text
            assert client.get(paths[0], auth=other).status_code == 404
            assert "fresh vegetables" in client.get(paths[1], auth=owner).text
            assert "failed" in client.get(paths[2], auth=owner).text

    async def test_failed_and_pending_links_are_searchable_only_by_the_owner(self):
        db = await self.start_database()
        failed = await db.enqueue(
            "1", NoteInput.from_text("https://example.org/report")
        )
        assert await self.run_job(db, fetcher=self.fetcher)
        assert (await db.get_note("1", failed)).status == "failed"
        pending = await db.enqueue(
            "1", NoteInput.from_text("https://example.org/draft")
        )
        await db.enqueue("1", NoteInput.from_text("Text note about example.org"))

        matches = await db.list_notes("1", "EXAMPLE.ORG")
        assert [note.id for note in matches] == [pending, failed]
        assert await db.list_notes("2", "example.org") == []

    async def test_deduplication_and_recovery_from_an_expired_claim(self):
        db = await self.start_database()
        note = NoteInput.from_text("Original note")
        note_id = await db.enqueue("1", note, telegram_message_id=42)
        assert (
            await db.enqueue(
                "1", NoteInput.from_text("Redelivery"), telegram_message_id=42
            )
            == note_id
        )
        assert (await db.get_note("1", note_id)).value == note.value

        claims = [
            job for job in await asyncio.gather(db.claim_job(), db.claim_job()) if job
        ]
        assert len(claims) == 1
        original_claim = claims[0]
        assert (await db.get_note("1", note_id)).status == "processing"

        async with connection(self.dsn) as conn:
            await conn.execute(
                "UPDATE notes SET lease_until = now() WHERE id = $1", note_id
            )

        replacement = await db.claim_job()
        assert replacement.id == note_id
        stale_embedding = Embedding("stale", [1.0, 0.0])
        assert not await db.finish_job(
            original_claim,
            await process(note, summarizer=self.summarizer),
            embedding=stale_embedding,
        )
        assert await db.finish_job(
            replacement, await process(note, summarizer=self.summarizer)
        )
        assert (await db.get_note("1", note_id)).status == "ready"
        assert await db.search_similar("1", stale_embedding) == []
        assert await db.claim_job() is None

    async def test_extraction_failure_does_not_stop_later_article_persistence(self):
        db = await self.start_database()
        failed_id = await db.enqueue(
            "1", NoteInput("url", "https://article.test/empty")
        )
        article_id = await db.enqueue(
            "1", NoteInput("url", "https://article.test/garden")
        )
        assert await self.run_job(db, fetcher=self.fetcher)
        assert await self.run_job(db, fetcher=self.fetcher)
        assert not await self.run_job(db, fetcher=self.fetcher)
        failed = await db.get_note("1", failed_id)
        assert failed.status == "failed"
        assert "article" in failed.error
        article = await db.get_note("1", article_id)
        assert article.status == "ready"
        assert "fresh vegetables" in article.text
        assert "Navigation marker" not in article.text
        assert article.tags == []
        assert article.summary == "An English model summary."

    async def test_claim_covers_fetch_retries_and_optional_embedding_budget(self):
        db = await self.start_database()
        note_id = await db.enqueue("1", NoteInput("url", "https://article.test/garden"))
        assert (await db.claim_job()).id == note_id
        # Simulate 90 seconds fetching, 10 seconds backoff and 30 seconds embedding.
        async with connection(self.dsn) as conn:
            await conn.execute(
                "UPDATE notes SET lease_until = lease_until - interval '130 seconds' WHERE id = $1",
                note_id,
            )
        assert await db.claim_job() is None

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
        assert failed.status == "failed"
        assert failed.text == "Retained failure marker"
        assert failed.summary is None
        assert failed.tags == []
        assert [n.id for n in await db.list_notes("1", "failure marker")] == [failed_id]
        assert await db.list_notes("2", "failure marker") == []
        await self.run_job(db)
        assert (await db.get_note("1", later_id)).status == "ready"

    async def test_renewal_saves_source_and_old_claim_cannot_renew_or_finish(self):
        db = await self.start_database()
        note_id = await db.enqueue("1", NoteInput("url", "https://article.test/garden"))
        original = await db.claim_job()
        assert await db.renew_job(original, text="Extracted source")
        assert (await db.get_note("1", note_id)).text == "Extracted source"
        async with connection(self.dsn) as conn:
            await conn.execute(
                "UPDATE notes SET lease_until = now() WHERE id = $1", note_id
            )
        assert not await db.renew_job(original, text="Stale source")
        replacement = await db.claim_job()
        assert not await db.renew_job(original)
        assert not await db.finish_job(original, None, error="stale")
        assert await db.finish_job(replacement, None, error="model failed")
        assert (await db.get_note("1", note_id)).text == "Extracted source"

    async def test_telegram_intake_is_searchable_in_web(self):
        db = await self.start_database()

        def respond(request):
            assert request.url.path == "/sendMessage"
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
        assert await self.run_job(db)
        with TestClient(web.app) as client:
            response = client.get("/", auth=("1", "first"), params={"q": "Сохранено"})
            assert f"/notes/{note.id}" in response.text

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
            async def embed(self, text, *, document=False):
                return vectors[text.removeprefix("Summary of ")]

        embedder = TestEmbedder()
        self.summarizer.summarize.side_effect = lambda text: (f"Summary of {text}", [])
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
            assert await self.run_job(db, embedder if text in vectors else None)

        async with connection(self.dsn) as conn:
            stored = await conn.fetchrow(
                "SELECT embedding_model, embedding::text FROM notes WHERE id = $1",
                note_ids["Office firewall"],
            )
            assert tuple(stored) == ("test-model", "[1,0]")
            unembedded = await conn.fetchrow(
                "SELECT embedding_model, embedding::text FROM notes WHERE id = $1",
                note_ids["Keyword-only note"],
            )
            assert tuple(unembedded) == (None, None)

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
            assert response.status_code == 200
            assert response.text.index(links["Office firewall"]) < response.text.index(
                links["Garden tools"]
            )
            for text in (
                "Different model",
                "Different dimensions",
                "Private firewall",
                "Keyword-only note",
            ):
                assert links[text] not in response.text

            private_results = client.get("/", auth=("2", "second"), params=search)
            assert links["Private firewall"] in private_results.text
            assert links["Office firewall"] not in private_results.text

            keyword_results = client.get(
                "/", auth=("1", "first"), params={"q": "Keyword-only"}
            )
            assert links["Keyword-only note"] in keyword_results.text

    async def test_concurrent_migrations_create_the_fresh_schema(self):
        await asyncio.gather(self.upgrade_database(), self.upgrade_database())

        async with connection(self.dsn) as conn:
            assert (
                await conn.fetchval(
                    f"SELECT version_num FROM {self.schema_name}.alembic_version"
                )
                == "003_note_images"
            )
            assert (
                await conn.fetchval(
                    "SELECT to_regclass($1)", f"{self.schema_name}.note_embeddings"
                )
                is None
            )

    @pytest.mark.skipif(
        not all(
            os.environ.get(name)
            for name in ("PKA_TEST_LOCAL_EMBEDDINGS_URL", "PKA_TEST_MODEL_PATH")
        ),
        reason="requires the real EmbeddingGemma 2 service and local summarizer weights",
    )
    async def test_real_models_store_text_and_photos_and_find_visual_details(self):
        db = await self.start_database()
        caption = "An illustration from the personal reference collection."

        def make_photo(shape, color):
            image = Image.new("RGB", (384, 384), "white")
            draw = ImageDraw.Draw(image)
            shape_draw = draw.rectangle if shape == "square" else draw.ellipse
            shape_draw((52, 52, 332, 332), fill=color)
            output = BytesIO()
            image.save(output, "PNG")
            return output.getvalue()

        photos = (make_photo("square", "red"), make_photo("circle", "green"))
        owner = ("1", "first")
        with (
            patch.dict(
                os.environ,
                {
                    "LOCAL_EMBEDDINGS_URL": os.environ["PKA_TEST_LOCAL_EMBEDDINGS_URL"],
                    "MODEL_PATH": os.environ["PKA_TEST_MODEL_PATH"],
                    "MODEL_GPU_LAYERS": "0",
                },
            ),
            TestClient(web.app) as client,
            open_summarizer() as summarizer,
        ):
            response = client.post(
                "/api/notes",
                auth=owner,
                json={"value": "The garden needs watering every Tuesday morning."},
            )
            assert response.status_code == 201
            note_ids = [response.json()["id"]]
            for image, text in zip(photos, (caption, ""), strict=True):
                response = client.post(
                    "/api/photos",
                    auth=owner,
                    files={"photo": ("note.png", image, "image/png")},
                    data={"caption": text},
                    headers={"X-PKA-Upload": "1"},
                )
                assert response.status_code == 201, response.text
                note_ids.append(response.json()["id"])

            async with open_embedder() as embedder:
                for _ in note_ids:
                    assert await run_once(db, embedder, summarizer=summarizer)

            async with connection(self.dsn) as conn:
                rows = await conn.fetch(
                    """
                    SELECT status, summary, embedding_model AS model,
                           public.vector_dims(embedding) AS dimensions,
                           embedding <#> embedding AS neg_norm_squared
                    FROM notes
                    WHERE id = ANY($1::bigint[])
                    ORDER BY id
                    """,
                    note_ids,
                )
            assert len(rows) == 3
            assert all(row["status"] == "ready" for row in rows)
            assert all(row["model"] == EMBEDDINGGEMMA_MODEL for row in rows)
            assert all(row["dimensions"] == 768 for row in rows)
            assert [row["neg_norm_squared"] for row in rows] == pytest.approx(
                [-1.0, -1.0, -1.0], abs=1e-5
            )
            assert rows[0]["summary"]
            assert rows[1]["summary"]
            assert rows[2]["summary"] is None

            for query, matching_id in zip(
                ("a bright red square", "a green circle"), note_ids[1:], strict=True
            ):
                response = client.get(
                    "/", auth=owner, params={"q": query, "mode": "semantic"}
                )
                assert response.status_code == 200, response.text
                assert response.text.index(f'href="/notes/{matching_id}"') == min(
                    response.text.index(f'href="/notes/{note_id}"')
                    for note_id in note_ids
                )
                assert (
                    client.get(
                        f"/notes/{matching_id}/photo", auth=("2", "second")
                    ).status_code
                    == 404
                )

            assert (
                client.get(f"/notes/{note_ids[2]}/photo", auth=owner).content
                == photos[1]
            )
