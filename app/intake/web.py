import json
import os
import re
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from app.intake.web_views import render_home, render_note
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


@app.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    owner_id: Owner,
    q: Annotated[str, Query(max_length=200)] = "",
    mode: Literal["keyword", "semantic"] = "keyword",
) -> HTMLResponse:
    if "\0" in q:
        raise HTTPException(422, "Search terms cannot contain NUL bytes.")

    query = q.strip()
    db = request.state.db
    embedder = request.state.embedder
    if query and mode == "semantic":
        if embedder is None:
            raise HTTPException(503, "Semantic search is not configured.")
        try:
            embedding = await embedder.embed(query)
        except EmbeddingError as exc:
            raise HTTPException(503, "Semantic search is unavailable.") from exc
        notes = await db.search_similar(owner_id, embedding)
    else:
        notes = await db.list_notes(owner_id, query)

    return render_home(
        notes,
        q,
        semantic_available=embedder is not None,
        semantic_selected=mode == "semantic",
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
    return render_note(note)


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
