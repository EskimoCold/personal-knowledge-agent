import asyncio
import math
import os
from array import array
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx

from app.notes.models import Embedding


class EmbeddingError(Exception):
    """Embedding service error safe to report without request details."""


@dataclass(frozen=True)
class Embedder:
    client: httpx.AsyncClient
    url: httpx.URL
    model: str
    dimensions: int

    async def embed(self, text: str) -> Embedding:
        try:
            async with asyncio.timeout(30):
                response = await self.client.post(
                    self.url,
                    json={
                        "model": self.model,
                        "input": text,
                        "encoding_format": "float",
                    },
                )
                response.raise_for_status()
        except (TimeoutError, httpx.HTTPError):
            raise EmbeddingError("Embedding service request failed.") from None

        try:
            rows = response.json()["data"]
            if not isinstance(rows, list) or len(rows) != 1:
                raise ValueError
            values = rows[0]["embedding"]
            if (
                not isinstance(values, list)
                or len(values) != self.dimensions
                or any(type(value) not in (int, float) for value in values)
            ):
                raise ValueError

            components = array("f", values)
            norm = math.hypot(*components)
            if not math.isfinite(norm) or norm == 0:
                raise ValueError
        except (KeyError, OverflowError, TypeError, ValueError):
            raise EmbeddingError(
                "Embedding service returned an invalid vector."
            ) from None

        # Unit vectors preserve cosine ranking and avoid overflow in float32 norms.
        normalized = array("f", (value / norm for value in components))
        return Embedding(model=self.model, values=normalized.tolist())


@asynccontextmanager
async def open_embedder() -> AsyncIterator[Embedder | None]:
    url = os.environ.get("EMBEDDINGS_URL", "").strip()
    model = os.environ.get("EMBEDDINGS_MODEL", "").strip()
    size = os.environ.get("EMBEDDINGS_DIMENSIONS", "").strip()
    api_key = os.environ.get("EMBEDDINGS_API_KEY", "").strip()
    if not any((url, model, size, api_key)):
        yield None
        return
    if not all((url, model, size)):
        raise ValueError(
            "Set EMBEDDINGS_URL, EMBEDDINGS_MODEL and EMBEDDINGS_DIMENSIONS."
        )
    try:
        endpoint = httpx.URL(url)
    except httpx.InvalidURL:
        raise ValueError("EMBEDDINGS_URL must be an HTTP(S) endpoint.") from None
    if (
        endpoint.scheme not in {"http", "https"}
        or not endpoint.host
        or endpoint.userinfo
    ):
        raise ValueError(
            "EMBEDDINGS_URL must be an HTTP(S) endpoint without credentials."
        )
    try:
        dimensions = int(size)
        if not 1 <= dimensions <= 16_000:
            raise ValueError
    except ValueError:
        raise ValueError(
            "EMBEDDINGS_DIMENSIONS must be an integer from 1 to 16000."
        ) from None
    if api_key and (
        not api_key.isascii() or not api_key.isprintable() or " " in api_key
    ):
        raise ValueError("EMBEDDINGS_API_KEY must be an ASCII bearer token.")

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(headers=headers, trust_env=False) as client:
        yield Embedder(client, endpoint, model, dimensions)
