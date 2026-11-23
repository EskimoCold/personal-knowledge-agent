import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from pydantic import ValidationError

from app.processing.search import SearchRequest, SearchResult

SEARCH_TIMEOUT_SECONDS = 30.0


class SearchUnavailable(Exception):
    """Search inference failed without exposing request or provider details."""


@dataclass(frozen=True)
class SearchClient:
    client: httpx.AsyncClient
    url: httpx.URL

    async def search(self, request: SearchRequest) -> SearchResult:
        try:
            # Give the worker's generation deadline time to return an error.
            async with asyncio.timeout(SEARCH_TIMEOUT_SECONDS + 2):
                response = await self.client.post(self.url, json=request.model_dump())
                response.raise_for_status()
            result = SearchResult.model_validate_json(response.content)
            if not set(result.ranked_ids).issubset(
                note.id for note in request.candidates
            ):
                raise SearchUnavailable("Search returned an unknown source.")
        except (TimeoutError, httpx.HTTPError, ValidationError):
            raise SearchUnavailable("Search inference is unavailable.") from None
        return result


@asynccontextmanager
async def open_search_client() -> AsyncIterator[SearchClient | None]:
    value = os.environ.get("SEARCH_INFERENCE_URL", "").strip()
    if not value:
        yield None
        return
    try:
        url = httpx.URL(value)
    except httpx.InvalidURL:
        raise ValueError("SEARCH_INFERENCE_URL must be an HTTP(S) endpoint.") from None
    if url.scheme not in {"http", "https"} or not url.host or url.userinfo:
        raise ValueError(
            "SEARCH_INFERENCE_URL must be an HTTP(S) endpoint without credentials."
        )

    timeout = httpx.Timeout(SEARCH_TIMEOUT_SECONDS + 2, connect=2.0)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        yield SearchClient(client, url)
