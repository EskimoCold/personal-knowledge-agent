import asyncio
from base64 import b64decode
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.notes.images import MAX_IMAGE_BYTES, decode_photo
from app.processing.embeddings import (
    EMBEDDINGGEMMA_MODEL,
    EMBEDDINGGEMMA_REPO,
    EMBEDDINGGEMMA_REVISION,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(min(4, torch.get_num_threads()))

    def load_model(device: str):
        return SentenceTransformer(
            EMBEDDINGGEMMA_REPO,
            revision=EMBEDDINGGEMMA_REVISION,
            device=device,
            config_kwargs={"audio_config": None},
            model_kwargs={"dtype": "float32"},
            trust_remote_code=False,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = load_model(device)
    except torch.cuda.OutOfMemoryError:
        if device == "cpu":
            raise
        device, model = "cpu", None
    if model is None:
        torch.cuda.empty_cache()
        model = load_model(device)
    yield {"model": model, "device": device, "inference_lock": asyncio.Lock()}


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


class EmbeddingRequest(BaseModel):
    model: str
    input: str = Field(max_length=12_000)
    purpose: Literal["document", "query"]
    image: str | None = Field(default=None, max_length=4 * ((MAX_IMAGE_BYTES + 2) // 3))


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    return {"status": "ready", "device": request.state.device}


def encode(model, payload: EmbeddingRequest) -> list[float]:
    text = payload.input.strip()
    for placeholder in ("image", "video", "audio"):
        text = text.replace(f"<|{placeholder}|>", f"[{placeholder}]")
    prompt = payload.purpose
    if payload.image is None:
        content = text
    else:
        try:
            photo, _ = decode_photo(b64decode(payload.image, validate=True))
        except ValueError:
            raise HTTPException(422, "Invalid photo.") from None
        content = {"image": photo}
        if text:
            content["text"] = text
        else:
            prompt = None
    return model.encode(
        content, prompt_name=prompt, normalize_embeddings=True, show_progress_bar=False
    ).tolist()


@app.post("/v1/embeddings")
async def create_embedding(request: Request, payload: EmbeddingRequest) -> dict:
    if payload.model != EMBEDDINGGEMMA_MODEL:
        raise HTTPException(422, "Unsupported model version.")
    if not payload.input.strip() and payload.image is None:
        raise HTTPException(422, "Provide text or a photo.")
    if payload.purpose == "query" and payload.image is not None:
        raise HTTPException(422, "Search queries must be text.")
    async with request.state.inference_lock:
        embedding = await run_in_threadpool(encode, request.state.model, payload)
    return {"data": [{"embedding": embedding}]}
