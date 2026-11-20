import json
import os
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
import structlog

from app.notes.models import Embedding
from app.processing import embedding_service
from app.processing.embeddings import Embedder, EmbeddingError, open_embedder
from app.processing.worker import run_once


@pytest.mark.parametrize(
    ("cuda_available", "cuda_oom", "expected_devices"),
    [
        (False, False, ["cpu"]),
        (True, False, ["cuda"]),
        (True, True, ["cuda", "cpu"]),
    ],
)
async def test_model_uses_cuda_when_available(
    monkeypatch, cuda_available, cuda_oom, expected_devices
):
    class OutOfMemoryError(RuntimeError):
        pass

    model = object()
    loader = Mock(side_effect=[OutOfMemoryError(), model] if cuda_oom else [model])
    empty_cache = Mock()
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            OutOfMemoryError=OutOfMemoryError,
            empty_cache=empty_cache,
        ),
        set_num_threads=Mock(),
        get_num_threads=lambda: 4,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=loader),
    )

    async with embedding_service.lifespan(embedding_service.app) as state:
        assert state["model"] is model
        assert state["device"] == expected_devices[-1]

    assert [call.kwargs["device"] for call in loader.call_args_list] == expected_devices
    assert empty_cache.call_count == int(cuda_oom)


class TestEmbedder:
    async def test_local_endpoint_sends_photo_and_document_purpose(self):
        def respond(request):
            payload = json.loads(request.content)
            assert payload["input"] == "Caption summary."
            assert payload["image"] == "cGhvdG8="
            assert payload["purpose"] == "document"
            return httpx.Response(200, json={"data": [{"embedding": [1.0] * 768}]})

        with patch.dict(
            os.environ,
            {"LOCAL_EMBEDDINGS_URL": "http://embed/v1/embeddings"},
            clear=True,
        ):
            async with open_embedder() as embedder:
                assert embedder.supports_images
                assert embedder.client.timeout.read == 30
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(respond)
                ) as client:
                    result = await replace(embedder, client=client).embed(
                        "Caption summary.", document=True, image=b"photo"
                    )
                    assert result.model == embedder.model
                    assert len(result.values) == 768

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
    @pytest.mark.parametrize("caption", ["", "Photo caption"])
    @pytest.mark.parametrize("embedding_fails", [False, True])
    async def test_photo_worker_uses_image_and_optional_summary(
        self, caption, embedding_fails
    ):
        job = SimpleNamespace(id=1, kind="image", value=caption, image=b"photo")
        db = SimpleNamespace(
            claim_job=AsyncMock(return_value=job),
            finish_job=AsyncMock(),
            renew_job=AsyncMock(return_value=True),
        )
        summarizer = SimpleNamespace(summarize=Mock(return_value=("Summary.", [])))
        vector = Embedding("test", [1.0])
        embedder = SimpleNamespace(
            embed=AsyncMock(
                return_value=vector,
                side_effect=EmbeddingError("Offline") if embedding_fails else None,
            )
        )

        assert await run_once(db, embedder, summarizer=summarizer)

        if caption:
            summarizer.summarize.assert_called_once_with(caption)
        else:
            summarizer.summarize.assert_not_called()
        embedder.embed.assert_awaited_once_with(
            "Summary." if caption else "", document=True, image=b"photo"
        )
        finish = db.finish_job.await_args
        if embedding_fails:
            assert finish.args[1] is None
            assert "Photo embedding" in finish.kwargs["error"]
            assert finish.kwargs["embedding"] is None
        else:
            assert finish.args[1].summary == ("Summary." if caption else None)
            assert finish.kwargs["embedding"] == vector

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

        embedder.embed.assert_awaited_once_with("A summary.", document=True)
        db.finish_job.assert_awaited_once()
        finish = db.finish_job.await_args
        assert finish.args[0] is job
        assert finish.args[1].text == job.value
        assert finish.kwargs["embedding"] is None
        assert job.value not in str(logs)
        assert "embedding_failed" in [event["event"] for event in logs]
