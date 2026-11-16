import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.notes.models import NoteInput
from app.processing.processor import process
from app.processing.worker import run_once


class ProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_plain_text_keeps_original_and_deterministic_excerpt(self):
        result = await process(NoteInput("text", "  My\nplain text  "))
        self.assertEqual(result.text, "  My\nplain text  ")
        self.assertEqual(result.summary, "My plain text")
        self.assertEqual(result.tags, ["text"])

    async def test_url_uses_extracted_text_for_excerpt_without_topic_tags(self):
        text = "Readable article. " * 30
        fetcher = SimpleNamespace(fetch=AsyncMock(return_value=SimpleNamespace(text=text)))
        result = await process(NoteInput("url", "https://article.test/"), fetcher)
        self.assertEqual(result.text, text)
        self.assertEqual(result.summary, ("Readable article. " * 30).strip()[:180] + "…")
        self.assertEqual(result.tags, [])

    async def test_worker_persists_article_and_embeds_extracted_text(self):
        from app.notes.models import Embedding
        from app.processing.webpage.models import ExtractionResult

        job = SimpleNamespace(id=1, kind="url", value="https://article.test/")
        db = SimpleNamespace(claim_job=AsyncMock(return_value=job), finish_job=AsyncMock())
        fetcher = SimpleNamespace(fetch=AsyncMock(return_value=ExtractionResult(
            "Article body", "Article title", "https://article.test/final"
        )))
        embedding = Embedding("test", [1.0])
        embedder = SimpleNamespace(embed=AsyncMock(return_value=embedding))
        self.assertTrue(await run_once(db, embedder, fetcher))
        finish = db.finish_job.await_args
        self.assertEqual(finish.args[1].text, "Article body")
        self.assertEqual(finish.kwargs["embedding"], embedding)
        embedder.embed.assert_awaited_once_with("Article body")

    async def test_extraction_failure_fails_job_and_later_note_still_runs(self):
        from app.processing.webpage.exceptions import PageError

        failed = SimpleNamespace(id=1, kind="url", value="https://article.test/?token=secret")
        later = SimpleNamespace(id=2, kind="text", value="Later note")
        db = SimpleNamespace(
            claim_job=AsyncMock(side_effect=[failed, later, None]), finish_job=AsyncMock()
        )
        fetcher = SimpleNamespace(fetch=AsyncMock(side_effect=PageError("No readable article text found.")))
        self.assertTrue(await run_once(db, fetcher=fetcher))
        self.assertEqual(db.finish_job.await_args.args, (failed, None))
        self.assertIn("article", db.finish_job.await_args.kwargs["error"])
        self.assertTrue(await run_once(db, fetcher=fetcher))
        self.assertEqual(db.finish_job.await_args.args[1].text, "Later note")
        self.assertFalse(await run_once(db, fetcher=fetcher))

    async def test_processing_deadline_fails_slow_extraction_and_worker_continues(self):
        async def slow_fetch(_):
            await asyncio.sleep(0.05)
            return SimpleNamespace(text="Article")

        job = SimpleNamespace(id=1, kind="url", value="https://article.test/")
        later = SimpleNamespace(id=2, kind="text", value="Later note")
        db = SimpleNamespace(
            claim_job=AsyncMock(side_effect=[job, later]), finish_job=AsyncMock()
        )
        fetcher = SimpleNamespace(fetch=slow_fetch)
        with patch("app.processing.worker.PROCESSING_TIMEOUT_SECONDS", 0.01, create=True):
            self.assertTrue(await run_once(db, fetcher=fetcher))
            self.assertIsNone(db.finish_job.await_args.args[1])
            self.assertIn("timed out", db.finish_job.await_args.kwargs["error"])
            self.assertTrue(await run_once(db, fetcher=fetcher))
            self.assertEqual(db.finish_job.await_args.args[1].text, "Later note")

    async def test_deadline_during_embedding_preserves_extracted_article(self):
        async def slow_embed(_):
            from app.notes.models import Embedding

            await asyncio.sleep(0.05)
            return Embedding("late", [1.0])

        job = SimpleNamespace(id=1, kind="url", value="https://article.test/")
        db = SimpleNamespace(claim_job=AsyncMock(return_value=job), finish_job=AsyncMock())
        fetcher = SimpleNamespace(fetch=AsyncMock(return_value=SimpleNamespace(text="Article")))
        embedder = SimpleNamespace(embed=slow_embed)
        with patch("app.processing.worker.PROCESSING_TIMEOUT_SECONDS", 0.01, create=True):
            self.assertTrue(await run_once(db, embedder, fetcher))
        self.assertEqual(db.finish_job.await_args.args[1].text, "Article")
        self.assertIsNone(db.finish_job.await_args.kwargs["embedding"])
