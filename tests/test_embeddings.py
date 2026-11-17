import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
import structlog

from app.processing.embeddings import Embedder, EmbeddingError, open_embedder
from app.processing.worker import run_once


class TestEmbedder:
    async def test_valid_embedding_is_normalized_and_tagged_with_model(self):
        def respond(request):
            assert request.method == "POST"
            assert request.url.path == "/v1/embeddings"
            assert json.loads(request.content) == {
                "model": "model-a",
                "input": "Some note",
                "encoding_format": "float",
            }
            return httpx.Response(200, json={"data": [{"embedding": [3, 4]}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            embedder = Embedder(
                client, httpx.URL("http://embed/v1/embeddings"), "model-a", 2
            )
            result = await embedder.embed("Some note")

        assert result.model == "model-a"
        assert result.values[0] == pytest.approx(0.6, abs=5e-8, rel=0)
        assert result.values[1] == pytest.approx(0.8, abs=5e-8, rel=0)

    @pytest.mark.parametrize(
        "values", ([1], [False, 1], ["1", 2], [0, 0], [1e40, 1], [1e-80, 0])
    )
    async def test_rejects_invalid_vectors(self, values):
        response = httpx.Response(200, json={"data": [{"embedding": values}]})
        transport = httpx.MockTransport(lambda _: response)
        async with httpx.AsyncClient(transport=transport) as client:
            embedder = Embedder(
                client, httpx.URL("http://embed/v1/embeddings"), "model-a", 2
            )
            with pytest.raises(EmbeddingError, match="invalid vector"):
                await embedder.embed("Some note")

    async def test_status_failure_does_not_expose_endpoint_or_response(self):
        transport = httpx.MockTransport(
            lambda _: httpx.Response(503, text="Private upstream error")
        )
        async with httpx.AsyncClient(transport=transport) as client:
            embedder = Embedder(
                client, httpx.URL("http://embed/v1/embeddings"), "model-a", 2
            )
            with pytest.raises(
                EmbeddingError, match="^Embedding service request failed\\.$"
            ):
                await embedder.embed("Some note")

    async def test_unconfigured_endpoint_keeps_embeddings_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            async with open_embedder() as embedder:
                assert embedder is None

    async def test_configured_endpoint_reads_dimensions_and_authorization(self):
        with patch.dict(
            os.environ,
            {
                "EMBEDDINGS_URL": "http://embed/v1/embeddings",
                "EMBEDDINGS_MODEL": "model-a",
                "EMBEDDINGS_DIMENSIONS": "384",
                "EMBEDDINGS_API_KEY": "test-token",
            },
            clear=True,
        ):
            async with open_embedder() as embedder:
                assert embedder.model == "model-a"
                assert embedder.dimensions == 384
                assert embedder.client.headers["Authorization"] == "Bearer test-token"


class TestEmbeddingWorker:
    async def test_embedder_failure_leaves_note_ready_for_keyword_search(self):
        job = SimpleNamespace(id=1, kind="text", value="Some private note")
        db = SimpleNamespace(
            claim_job=AsyncMock(return_value=job),
            finish_job=AsyncMock(),
            renew_job=AsyncMock(return_value=True),
        )
        embedder = SimpleNamespace(
            embed=AsyncMock(side_effect=EmbeddingError("Provider unavailable."))
        )

        with structlog.testing.capture_logs() as logs:
            assert await run_once(
                db,
                embedder,
                summarizer=SimpleNamespace(
                    summarize=Mock(return_value=("A summary.", []))
                ),
            )

        embedder.embed.assert_awaited_once_with(job.value)
        db.finish_job.assert_awaited_once()
        finish = db.finish_job.await_args
        assert finish.args[0] is job
        assert finish.args[1].text == job.value
        assert finish.kwargs["embedding"] is None
        assert job.value not in str(logs)
        assert "embedding_failed" in [event["event"] for event in logs]
