import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from app.processing.embeddings import Embedder, EmbeddingError, open_embedder
from app.processing.worker import run_once


class EmbedderTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_embedding_is_normalized_and_tagged_with_model(self):
        def respond(request):
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.path, "/v1/embeddings")
            self.assertEqual(
                json.loads(request.content),
                {"model": "model-a", "input": "Some note", "encoding_format": "float"},
            )
            return httpx.Response(200, json={"data": [{"embedding": [3, 4]}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            embedder = Embedder(
                client, httpx.URL("http://embed/v1/embeddings"), "model-a", 2
            )
            result = await embedder.embed("Some note")

        self.assertEqual(result.model, "model-a")
        self.assertAlmostEqual(result.values[0], 0.6)
        self.assertAlmostEqual(result.values[1], 0.8)

    async def test_rejects_invalid_vectors(self):
        for values in ([1], [False, 1], ["1", 2], [0, 0], [1e40, 1], [1e-80, 0]):
            with self.subTest(values=values):
                response = httpx.Response(200, json={"data": [{"embedding": values}]})
                transport = httpx.MockTransport(lambda _: response)
                async with httpx.AsyncClient(transport=transport) as client:
                    embedder = Embedder(
                        client, httpx.URL("http://embed/v1/embeddings"), "model-a", 2
                    )
                    with self.assertRaisesRegex(EmbeddingError, "invalid vector"):
                        await embedder.embed("Some note")

    async def test_status_failure_does_not_expose_endpoint_or_response(self):
        transport = httpx.MockTransport(
            lambda _: httpx.Response(503, text="Private upstream error")
        )
        async with httpx.AsyncClient(transport=transport) as client:
            embedder = Embedder(
                client, httpx.URL("http://embed/v1/embeddings"), "model-a", 2
            )
            with self.assertRaisesRegex(
                EmbeddingError, r"^Embedding service request failed\.$"
            ):
                await embedder.embed("Some note")

    async def test_unconfigured_endpoint_keeps_embeddings_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            async with open_embedder() as embedder:
                self.assertIsNone(embedder)

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
                self.assertEqual(embedder.model, "model-a")
                self.assertEqual(embedder.dimensions, 384)
                self.assertEqual(
                    embedder.client.headers["Authorization"], "Bearer test-token"
                )


class EmbeddingWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedder_failure_leaves_note_ready_for_keyword_search(self):
        job = SimpleNamespace(id=1, kind="text", value="Some private note")
        db = SimpleNamespace(
            claim_job=AsyncMock(return_value=job), finish_job=AsyncMock()
        )
        embedder = SimpleNamespace(
            embed=AsyncMock(side_effect=EmbeddingError("Provider unavailable."))
        )

        with self.assertLogs("app.processing.worker", level="WARNING") as logs:
            self.assertTrue(await run_once(db, embedder))

        embedder.embed.assert_awaited_once_with(job.value)
        db.finish_job.assert_awaited_once()
        finish = db.finish_job.await_args
        self.assertIs(finish.args[0], job)
        self.assertEqual(finish.args[1].text, job.value)
        self.assertIsNone(finish.kwargs["embedding"])
        self.assertNotIn(job.value, "\n".join(logs.output))
