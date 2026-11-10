from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit
from uuid import UUID


@dataclass(frozen=True)
class NoteInput:
    kind: Literal["text", "url"]
    value: str

    @classmethod
    def from_text(cls, value: str) -> "NoteInput":
        value = value.strip()
        if not value or len(value) > 12_000 or "\0" in value:
            raise ValueError("Enter 1–12,000 characters without NUL bytes.")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("Enter valid Unicode text.") from None

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


@dataclass(frozen=True)
class ProcessedNote:
    text: str
    summary: str
    tags: list[str]


@dataclass(frozen=True)
class Note:
    id: int
    kind: Literal["text", "url"]
    value: str
    status: str
    text: str | None
    summary: str | None
    tags: list[str]
    error: str | None


@dataclass(frozen=True)
class Job:
    id: int
    kind: Literal["text", "url"]
    value: str
    claim_token: UUID


@dataclass(frozen=True)
class Embedding:
    model: str
    values: list[float]
