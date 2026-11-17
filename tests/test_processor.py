import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import structlog

from app.notes.models import NoteInput
from app.processing.processor import process
from app.processing.worker import run_once


class TestProcessor:
    @pytest.fixture(autouse=True)
    def processor_dependencies(self):
        self.summarizer = SimpleNamespace(
            summarize=Mock(return_value=("An English model summary.", ["science"]))
        )
        with structlog.testing.capture_logs(
            processors=[structlog.contextvars.merge_contextvars]
        ) as self.events:
            yield

    async def run_job(self, db, embedder=None, fetcher=None):
        db.renew_job = AsyncMock(return_value=True)
        return await run_once(db, embedder, fetcher, summarizer=self.summarizer)

    async def test_plain_text_keeps_original_and_uses_model(self):
        result = await process(
            NoteInput("text", "  My\nplain text  "), summarizer=self.summarizer
        )
        assert result.text == "  My\nplain text  "
        assert result.summary == "An English model summary."
        assert result.tags == ["science"]

    async def test_generation_thread_logs_keep_job_id_without_leaking_between_jobs(
        self,
    ):
        def generate(_):
            structlog.get_logger("test_model").info("test_generation")
            return "An English result.", []

        self.summarizer.summarize.side_effect = generate
        db = SimpleNamespace(
            claim_job=AsyncMock(
                side_effect=[
                    SimpleNamespace(id=1, kind="text", value="Private first note"),
                    SimpleNamespace(id=2, kind="text", value="Private second note"),
                ]
            ),
            finish_job=AsyncMock(),
        )
        await self.run_job(db)
        await self.run_job(db)
        assert [
            e["note_id"] for e in self.events if e["event"] == "test_generation"
        ] == [1, 2]
        assert "note_id" not in structlog.contextvars.get_contextvars()
        assert "Private first note" not in str(self.events)

    async def test_url_summarizes_extracted_text(self):
        text = "Readable article. " * 30
        fetcher = SimpleNamespace(
            fetch=AsyncMock(return_value=SimpleNamespace(text=text))
        )
        result = await process(
            NoteInput("url", "https://article.test/"), fetcher, self.summarizer
        )
        assert result.text == text
        assert result.summary == "An English model summary."
        assert result.tags == ["science"]

    async def test_worker_persists_article_and_embeds_extracted_text(self):
        from app.notes.models import Embedding
        from app.processing.webpage.models import ExtractionResult

        job = SimpleNamespace(id=1, kind="url", value="https://article.test/")
        db = SimpleNamespace(
            claim_job=AsyncMock(return_value=job), finish_job=AsyncMock()
        )
        fetcher = SimpleNamespace(
            fetch=AsyncMock(
                return_value=ExtractionResult(
                    "Article body", "Article title", "https://article.test/final"
                )
            )
        )
        embedding = Embedding("test", [1.0])
        embedder = SimpleNamespace(embed=AsyncMock(return_value=embedding))
        assert await self.run_job(db, embedder, fetcher)
        finish = db.finish_job.await_args
        assert finish.args[1].text == "Article body"
        assert finish.kwargs["embedding"] == embedding
        embedder.embed.assert_awaited_once_with("Article body")

    async def test_extraction_failure_fails_job_and_later_note_still_runs(self):
        from app.processing.webpage.exceptions import PageError

        failed = SimpleNamespace(
            id=1, kind="url", value="https://article.test/?token=secret"
        )
        later = SimpleNamespace(id=2, kind="text", value="Later note")
        db = SimpleNamespace(
            claim_job=AsyncMock(side_effect=[failed, later, None]),
            finish_job=AsyncMock(),
        )
        fetcher = SimpleNamespace(
            fetch=AsyncMock(side_effect=PageError("No readable article text found."))
        )
        assert await self.run_job(db, fetcher=fetcher)
        assert db.finish_job.await_args.args == (failed, None)
        assert "article" in db.finish_job.await_args.kwargs["error"]
        assert await self.run_job(db, fetcher=fetcher)
        assert db.finish_job.await_args.args[1].text == "Later note"
        assert not await self.run_job(db, fetcher=fetcher)

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
        with patch(
            "app.processing.worker.PROCESSING_TIMEOUT_SECONDS", 0.01, create=True
        ):
            assert await self.run_job(db, fetcher=fetcher)
            assert db.finish_job.await_args.args[1] is None
            assert "timed out" in db.finish_job.await_args.kwargs["error"]
            assert await self.run_job(db, fetcher=fetcher)
            assert db.finish_job.await_args.args[1].text == "Later note"

    async def test_deadline_during_embedding_preserves_extracted_article(self):
        async def slow_embed(_):
            from app.notes.models import Embedding

            await asyncio.sleep(0.05)
            return Embedding("late", [1.0])

        job = SimpleNamespace(id=1, kind="url", value="https://article.test/")
        db = SimpleNamespace(
            claim_job=AsyncMock(return_value=job), finish_job=AsyncMock()
        )
        fetcher = SimpleNamespace(
            fetch=AsyncMock(return_value=SimpleNamespace(text="Article"))
        )
        embedder = SimpleNamespace(embed=slow_embed)
        with patch(
            "app.processing.worker.EMBEDDING_TIMEOUT_SECONDS", 0.01, create=True
        ):
            assert await self.run_job(db, embedder, fetcher)
        assert db.finish_job.await_args.args[1].text == "Article"
        assert db.finish_job.await_args.kwargs["embedding"] is None

    async def test_model_failure_saves_source_and_continues_next_job(self):
        from app.processing.exceptions import ProcessingError

        failed = SimpleNamespace(id=1, kind="url", value="https://article.test/")
        later = SimpleNamespace(id=2, kind="text", value="Later text")
        db = SimpleNamespace(
            claim_job=AsyncMock(side_effect=[failed, later]), finish_job=AsyncMock()
        )
        fetcher = SimpleNamespace(
            fetch=AsyncMock(return_value=SimpleNamespace(text="Full source"))
        )
        self.summarizer.summarize.side_effect = [
            ProcessingError("Local model generation failed."),
            ("A summary.", []),
        ]
        assert await self.run_job(db, fetcher=fetcher)
        db.renew_job.assert_awaited_once_with(failed, text="Full source")
        assert db.finish_job.await_args.args == (failed, None)
        assert "model" in db.finish_job.await_args.kwargs["error"]
        assert await self.run_job(db, fetcher=fetcher)
        assert db.finish_job.await_args.args[1].summary == "A summary."

    async def test_cancellation_waits_for_native_inference_to_finish(self):
        import threading

        started, release = threading.Event(), threading.Event()

        def blocking(_):
            started.set()
            release.wait(2)
            return "Done.", []

        self.summarizer.summarize.side_effect = blocking
        task = asyncio.create_task(
            process(NoteInput("text", "Source"), summarizer=self.summarizer)
        )
        await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_slow_inference_renews_lease_without_fetch_timeout(self):
        import time

        def slow(_):
            time.sleep(0.04)
            return "Done.", []

        self.summarizer.summarize.side_effect = slow
        job = SimpleNamespace(id=1, kind="text", value="Source")
        db = SimpleNamespace(
            claim_job=AsyncMock(return_value=job),
            renew_job=AsyncMock(return_value=True),
            finish_job=AsyncMock(),
        )
        with (
            patch("app.processing.worker.PROCESSING_TIMEOUT_SECONDS", 0.01),
            patch("app.processing.worker.LEASE_RENEWAL_SECONDS", 0.005),
        ):
            await run_once(db, summarizer=self.summarizer)
        assert db.renew_job.await_count > 2
        assert db.finish_job.await_args.args[1].summary == "Done."

    async def test_lost_lease_waits_for_native_call_and_does_not_publish(self):
        import threading
        import time

        finished = threading.Event()

        def slow(_):
            time.sleep(0.04)
            finished.set()
            return "Done.", []

        self.summarizer.summarize.side_effect = slow
        job = SimpleNamespace(id=1, kind="text", value="Source")
        db = SimpleNamespace(
            claim_job=AsyncMock(return_value=job),
            renew_job=AsyncMock(side_effect=[True, False]),
            finish_job=AsyncMock(),
        )
        with (
            patch("app.processing.worker.LEASE_RENEWAL_SECONDS", 0.01),
            pytest.raises(ExceptionGroup),
        ):
            await run_once(db, summarizer=self.summarizer)
        assert finished.is_set()
        db.finish_job.assert_not_awaited()

    async def test_repeated_cancellation_still_waits_for_native_inference(self):
        import threading

        started, release = threading.Event(), threading.Event()

        def blocking(_):
            started.set()
            release.wait(2)
            return "Done.", []

        self.summarizer.summarize.side_effect = blocking
        task = asyncio.create_task(
            process(NoteInput("text", "Source"), summarizer=self.summarizer)
        )
        await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.01)
        ended_before_native = task.done()
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        assert not ended_before_native

    async def test_finishing_job_stops_heartbeat_before_clearing_claim(self):
        finished = False

        async def renew(*args, **kwargs):
            return not finished

        async def finish(*args, **kwargs):
            nonlocal finished
            finished = True
            await asyncio.sleep(0.03)

        job = SimpleNamespace(id=1, kind="text", value="Source")
        db = SimpleNamespace(
            claim_job=AsyncMock(return_value=job),
            renew_job=renew,
            finish_job=AsyncMock(side_effect=finish),
        )
        with patch("app.processing.worker.LEASE_RENEWAL_SECONDS", 0.005):
            assert await run_once(db, summarizer=self.summarizer)
