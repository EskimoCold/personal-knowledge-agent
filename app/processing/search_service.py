import asyncio
from functools import partial
from threading import Event

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError
from starlette.requests import ClientDisconnect

from app.processing.exceptions import ProcessingError
from app.processing.inference import InferenceBusy, run_native
from app.processing.search import SearchRequest, SearchResult
from app.processing.search_client import SEARCH_TIMEOUT_SECONDS

MAX_REQUEST_BYTES = 64 * 1024


async def read_search_request(request: Request) -> SearchRequest:
    media_type = request.headers.get("content-type", "").partition(";")[0]
    if media_type.strip().lower() != "application/json":
        raise HTTPException(415, "Send application/json.")

    body = bytearray()
    try:
        async with asyncio.timeout(2):
            async for chunk in request.stream():
                if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                    raise HTTPException(413, "Search request is too large.")
                body.extend(chunk)
    except (TimeoutError, ClientDisconnect):
        raise HTTPException(408, "Search request was interrupted.") from None

    try:
        return SearchRequest.model_validate_json(body)
    except ValidationError:
        raise HTTPException(422, "Invalid search request.") from None


def create_search_app(lifespan) -> FastAPI:
    app = FastAPI(
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        redirect_slashes=False,
    )

    @app.post("/search")
    async def search(request: Request) -> SearchResult:
        payload = await read_search_request(request)
        stop_event = Event()
        deadline = asyncio.get_running_loop().call_later(
            SEARCH_TIMEOUT_SECONDS, stop_event.set
        )
        call = partial(request.state.inference.search, payload, stop_event)
        try:
            return await run_native(call, stop_event)
        except (InferenceBusy, ProcessingError):
            raise HTTPException(503, "Search inference is unavailable.") from None
        finally:
            deadline.cancel()

    return app
