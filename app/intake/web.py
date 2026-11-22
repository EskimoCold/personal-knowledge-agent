import json
import os
import re
import secrets
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from python_multipart.exceptions import MultipartParseError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import ClientDisconnect

from app.intake.web_views import render_home, render_note
from app.notes.images import MAX_IMAGE_BYTES, decode_photo
from app.notes.models import NoteInput
from app.processing.embeddings import EmbeddingError, open_embedder
from app.storage.postgres import open_postgres


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with open_postgres() as db, open_embedder() as embedder:
        yield {"db": db, "embedder": embedder}


app = FastAPI(
    title="Personal Knowledge Agent",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
basic = HTTPBasic()


async def require_owner(
    credentials: Annotated[HTTPBasicCredentials, Depends(basic)],
) -> str:
    try:
        users = json.loads(os.environ.get("PKA_WEB_USERS", "{}"))
    except ValueError:
        users = {}

    owner_id = credentials.username
    password = users.get(owner_id) if isinstance(users, dict) else None
    if (
        re.fullmatch(r"[1-9][0-9]*", owner_id)
        and isinstance(password, str)
        and password
        and password.isascii()
        and secrets.compare_digest(password.encode(), credentials.password.encode())
    ):
        return owner_id

    raise HTTPException(
        401, "Invalid credentials", headers={"WWW-Authenticate": "Basic"}
    )


Owner = Annotated[str, Depends(require_owner)]
search_limit = TypeAdapter(Annotated[int, Field(ge=1, le=50)])


@app.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    owner_id: Owner,
    q: Annotated[str, Query(max_length=200)] = "",
    mode: Literal["keyword", "semantic"] = "keyword",
    k: str = "20",
) -> HTMLResponse:
    if "\0" in q:
        raise HTTPException(422, "Search terms cannot contain NUL bytes.")

    try:
        result_limit = search_limit.validate_python(k)
    except ValidationError:
        if mode == "semantic":
            raise HTTPException(422, "Choose a result count from 1 to 50.") from None
        result_limit = 20

    query = q.strip()
    db = request.state.db
    embedder = request.state.embedder
    search_error = None
    notes = []
    if query and mode == "semantic":
        if embedder is None:
            search_error = "Semantic search is not configured."
        else:
            try:
                embedding = await embedder.embed(query)
            except EmbeddingError:
                search_error = "Semantic search is unavailable."
            else:
                notes = await db.search_similar(owner_id, embedding, limit=result_limit)
    else:
        notes = await db.list_notes(owner_id, query)

    return render_home(
        notes,
        q,
        root_path=request.scope.get("root_path", ""),
        result_limit=result_limit,
        semantic_available=embedder is not None,
        semantic_selected=mode == "semantic",
        photos_available=getattr(embedder, "supports_images", False),
        search_error=search_error,
    )


@app.get("/notes/{note_id}", response_class=HTMLResponse)
async def detail(
    request: Request,
    note_id: Annotated[int, Path(gt=0, le=2**63 - 1)],
    owner_id: Owner,
) -> HTMLResponse:
    note = await request.state.db.get_note(owner_id, note_id)
    if note is None:
        raise HTTPException(404, "Note not found")
    return render_note(note, root_path=request.scope.get("root_path", ""))


@app.get("/notes/{note_id}/photo")
async def photo(
    request: Request,
    note_id: Annotated[int, Path(gt=0, le=2**63 - 1)],
    owner_id: Owner,
) -> Response:
    saved = await request.state.db.get_photo(owner_id, note_id)
    if saved is None:
        raise HTTPException(404, "Photo not found")
    image, media_type = saved
    return Response(
        image,
        media_type=media_type,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'none'; sandbox",
            "Cross-Origin-Resource-Policy": "same-origin",
            "X-Content-Type-Options": "nosniff",
        },
    )


async def require_json(request: Request) -> None:
    content_type = (
        request.headers.get("content-type", "").partition(";")[0].strip().lower()
    )
    if content_type != "application/json":
        raise HTTPException(415, "Send application/json")


class NoteSubmission(BaseModel):
    value: str


@app.post("/api/notes", status_code=201, dependencies=[Depends(require_json)])
async def create_note(
    request: Request, payload: NoteSubmission, owner_id: Owner
) -> dict[str, int]:
    try:
        note_input = NoteInput.from_text(payload.value)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return {"id": await request.state.db.enqueue(owner_id, note_input)}


class PhotoUploadTooLarge(MultiPartException):
    """The upload exceeded its total request size limit."""


class PhotoFormParser(MultiPartParser):
    complete = False

    def on_end(self) -> None:
        self.complete = True


async def photo_upload_stream(request: Request) -> AsyncGenerator[bytes, None]:
    size = 0
    try:
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_IMAGE_BYTES + 64 * 1024:
                raise PhotoUploadTooLarge("Upload a photo up to 8 MiB.")
            yield chunk
    except ClientDisconnect:
        raise MultiPartException("Photo upload was interrupted.") from None


@app.post("/api/photos", status_code=201)
async def create_photo(request: Request, owner_id: Owner) -> dict[str, int]:
    if request.headers.get("x-pka-upload") != "1":
        raise HTTPException(403, "Missing photo upload header.")
    if not getattr(request.state.embedder, "supports_images", False):
        raise HTTPException(503, "Photo uploads require local embeddings.")
    content_type = request.headers.get("content-type", "").partition(";")[0]
    if content_type.strip().lower() != "multipart/form-data":
        raise HTTPException(415, "Send multipart/form-data.")
    parser = PhotoFormParser(
        request.headers,
        photo_upload_stream(request),
        max_files=1,
        max_fields=1,
        max_part_size=48_000,
    )
    try:
        form = await parser.parse()
    except PhotoUploadTooLarge as exc:
        raise HTTPException(413, exc.message) from None
    except (MultiPartException, MultipartParseError):
        raise HTTPException(422, "Upload one photo and an optional caption.") from None
    try:
        upload = form.get("photo")
        caption = form.get("caption", "")
        if (
            not parser.complete
            or not isinstance(upload, UploadFile)
            or not isinstance(caption, str)
            or set(form) - {"photo", "caption"}
        ):
            raise HTTPException(422, "Upload one photo and an optional caption.")
        image = await upload.read(MAX_IMAGE_BYTES + 1)
        if len(image) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Upload a photo up to 8 MiB.")
        try:
            _, media_type = await run_in_threadpool(decode_photo, image)
            note_input = NoteInput.from_photo(caption, image, media_type)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
    finally:
        await form.close()
    return {"id": await request.state.db.enqueue(owner_id, note_input)}
