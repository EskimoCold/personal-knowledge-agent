from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit
from uuid import UUID

NoteKind = Literal["text", "url", "image"]


def _note_text(value: str, *, required: bool = True) -> str:
    value = value.strip()
    if (required and not value) or len(value) > 12_000 or "\0" in value:
        minimum = 1 if required else 0
        raise ValueError(f"Enter {minimum}–12,000 characters without NUL bytes.")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("Enter valid Unicode text.") from None
    return value


@dataclass(frozen=True)
class NoteInput:
    kind: NoteKind
    value: str
    image: bytes | None = None
    image_media_type: str | None = None

    @classmethod
    def from_text(cls, value: str) -> "NoteInput":
        value = _note_text(value)

        if value.lower().startswith(("https://", "http://")) and not any(
            char.isspace() for char in value
        ):
            try:
                url = urlsplit(value)
            except ValueError:
                raise ValueError("Enter a valid HTTP or HTTPS URL.") from None
            if not url.hostname or url.username is not None:
                raise ValueError("Enter a valid HTTP or HTTPS URL without credentials.")
            return cls("url", value)

        return cls("text", value)

    @classmethod
    def from_photo(cls, caption: str, image: bytes, media_type: str) -> "NoteInput":
        return cls("image", _note_text(caption, required=False), image, media_type)


@dataclass(frozen=True)
class ProcessedNote:
    text: str
    summary: str | None
    tags: list[str]


@dataclass(frozen=True)
class Note:
    id: int
    kind: NoteKind
    value: str
    status: str
    text: str | None
    summary: str | None
    tags: list[str]
    error: str | None


@dataclass(frozen=True)
class Job:
    id: int
    kind: NoteKind
    value: str
    claim_token: UUID
    image: bytes | None = None


@dataclass(frozen=True)
class Embedding:
    model: str
    values: list[float]
