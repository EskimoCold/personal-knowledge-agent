import trafilatura

from app.processing.webpage.exceptions import PageError
from app.processing.webpage.models import ExtractionResult


def extract_article(body: bytes, encoding: str | None, final_url: str) -> ExtractionResult:
    try:
        html = body.decode(encoding, errors="replace") if encoding else body
        article = trafilatura.bare_extraction(
            html,
            url=final_url,
            include_comments=False,
            favor_precision=True,
            with_metadata=True,
        )
    except Exception:
        raise PageError("Could not extract readable article text from the page.") from None

    text = (article.text or "").strip() if article is not None else ""
    if not text:
        raise PageError("No readable article text found in the page.")
    return ExtractionResult(text, article.title or None, final_url)
