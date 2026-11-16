from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractionResult:
    text: str
    title: str | None
    final_url: str
