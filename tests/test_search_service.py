import asyncio
from contextlib import asynccontextmanager
from functools import partial
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from app.processing.exceptions import InferenceBusy, ProcessingError, SearchUnavailable
from app.processing.inference import LocalInference, run_native
from app.processing.search import SearchRequest, SearchResult
from app.processing.search_client import SearchClient
from app.processing.search_service import create_search_app


@pytest.fixture
def search_request():
    return SearchRequest(
        query="When is the meeting?", candidates=[{"id": 14, "text": "At noon."}]
    )


def test_search_endpoint_is_post_only_and_uses_the_worker_model(search_request):
    result = SearchResult(ranked_ids=[14], answer="At noon.", citations=[14])
    inference = SimpleNamespace(search=Mock(return_value=result))

    @asynccontextmanager
    async def lifespan(app):
        yield {"inference": inference}

    with TestClient(create_search_app(lifespan)) as client:
        assert client.get("/search").status_code == 405
        assert client.get("/docs").status_code == 404
        response = client.post("/search", json=search_request.model_dump())

    assert response.status_code == 200
    assert response.json() == result.model_dump()
    assert inference.search.call_args.args[0] == search_request


def test_search_endpoint_bounds_body_before_parsing():
    inference = SimpleNamespace(search=Mock())

    @asynccontextmanager
    async def lifespan(app):
        yield {"inference": inference}

    with TestClient(create_search_app(lifespan)) as client:
        response = client.post(
            "/search",
            headers={"Content-Type": "application/json"},
            content=iter([b" " * 32768] * 3),
        )

    assert response.status_code == 413
    inference.search.assert_not_called()


def test_search_deadline_signals_native_inference(monkeypatch, search_request):
    monkeypatch.setattr("app.processing.search_service.SEARCH_TIMEOUT_SECONDS", 0.02)

    def timed_search(request, stop_event):
        assert stop_event.wait(2)
        raise ProcessingError("Canceled.")

    @asynccontextmanager
    async def lifespan(app):
        yield {"inference": SimpleNamespace(search=timed_search)}

    with TestClient(create_search_app(lifespan)) as client:
        response = client.post("/search", json=search_request.model_dump())

    assert response.status_code == 503


@pytest.mark.parametrize("status,id", [(200, 14), (200, 99), (503, 14)])
async def test_client_rejects_unknown_sources_and_unavailable_workers(
    search_request, status, id
):
    result = SearchResult(ranked_ids=[id], answer="At noon.", citations=[id])
    transport = httpx.MockTransport(
        lambda request: httpx.Response(status, json=result.model_dump())
    )
    async with httpx.AsyncClient(transport=transport) as client:
        search = SearchClient(client, httpx.URL("http://worker:8000/search"))
        if status == 200 and id == 14:
            assert await search.search(search_request) == result
        else:
            with pytest.raises(SearchUnavailable):
                await search.search(search_request)


async def test_cancellation_does_not_release_the_model_during_summarization(
    search_request,
):
    started, release = Event(), Event()

    def summarize(text):
        started.set()
        assert release.wait(2)
        return "Summary.", []

    inference = LocalInference(SimpleNamespace(summarize=summarize))
    call = asyncio.create_task(run_native(partial(inference.summarize, "Note")))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        call.cancel()
        with pytest.raises(InferenceBusy):
            await run_native(partial(inference.search, search_request, Event()))
        assert not call.done()
    finally:
        release.set()
        await asyncio.gather(call, return_exceptions=True)
    assert call.cancelled()
