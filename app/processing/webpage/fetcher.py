import asyncio
import json
import math
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

from app.processing.webpage.config import WebpageConfig
from app.processing.webpage.exceptions import PageError, RetryablePageError
from app.processing.webpage.extractor import extract_article
from app.processing.webpage.models import ExtractionResult


@dataclass(frozen=True)
class PageFetcher:
    client: httpx.AsyncClient
    settings: WebpageConfig

    @classmethod
    @asynccontextmanager
    async def open(cls) -> AsyncIterator["PageFetcher"]:
        settings = WebpageConfig.from_env()
        async with httpx.AsyncClient(
            trust_env=False,
            timeout=settings.timeout_seconds,
            headers={
                "User-Agent": "PersonalKnowledgeAgent/0.1 (article text extraction)",
                "Accept": "text/html, application/xhtml+xml",
                "Accept-Encoding": "gzip, deflate",
            },
        ) as client:
            yield cls(client, settings)

    async def fetch(self, value: str) -> ExtractionResult:
        try:
            source = await self._with_retries(self._attempt, value)
            if source is not None:
                body, encoding, final_url = source
                try:
                    return await asyncio.to_thread(
                        extract_article, body, encoding, final_url
                    )
                except PageError:
                    pass
            self.client.cookies.clear()
            return await self._with_retries(self._keenable, value)
        finally:
            self.client.cookies.clear()

    async def _with_retries[T](
        self, request: Callable[[str], Awaitable[T]], url: str
    ) -> T:
        for attempt in range(self.settings.retries + 1):
            retry_after = None
            try:
                async with asyncio.timeout(self.settings.timeout_seconds):
                    return await request(url)
            except RetryablePageError as exc:
                message, retry_after = str(exc), exc.retry_after
            except (TimeoutError, httpx.TimeoutException):
                message = "Webpage request timed out."
            except httpx.HTTPError:
                raise PageError("Webpage request failed.") from None
            if attempt == self.settings.retries:
                raise PageError(message) from None
            await asyncio.sleep(self._retry_delay(retry_after, attempt))
        raise PageError("Webpage request failed.")

    async def _attempt(self, value: str) -> tuple[bytes, str | None, str] | None:
        url = self._validate_url(value)
        for redirects in range(self.settings.max_redirects + 1):
            # Fetching stays stateless across redirects and sequential jobs.
            self.client.cookies.clear()
            async with self.client.stream(
                "GET", url, follow_redirects=False
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    if redirects == self.settings.max_redirects:
                        raise PageError("Webpage redirect limit exceeded.")
                    location = response.headers.get("Location")
                    if not location:
                        raise PageError("Webpage redirect has no destination.")
                    # Validate the raw location as well as the resolved URL: URL
                    # normalization must not silently remove control characters.
                    if "\\" in location or any(
                        char.isspace() or ord(char) < 32 for char in location
                    ):
                        raise PageError("Invalid webpage redirect destination.")
                    try:
                        url = self._validate_url(str(response.url.join(location)))
                    except (ValueError, httpx.InvalidURL):
                        raise PageError(
                            "Invalid webpage redirect destination."
                        ) from None
                    continue
                if response.status_code in {429, 502, 503, 504}:
                    raise RetryablePageError(
                        f"Webpage server returned HTTP {response.status_code}.",
                        response.headers.get("Retry-After"),
                    )
                if not 200 <= response.status_code < 300:
                    raise PageError(
                        f"Webpage server returned HTTP {response.status_code}."
                    )
                media_type = (
                    response.headers.get("Content-Type", "")
                    .partition(";")[0]
                    .strip()
                    .lower()
                )
                if media_type not in {"text/html", "application/xhtml+xml"}:
                    return None
                body = await self._read_body(response)
                encoding = response.charset_encoding
                return body, encoding, str(response.url)
        raise PageError("Webpage redirect limit exceeded.")

    async def _keenable(self, url: str) -> ExtractionResult:
        async with self.client.stream(
            "GET",
            "https://api.keenable.ai/v1/fetch/public",
            headers={
                "X-Keenable-Title": "Personal Knowledge Agent",
                "Accept": "application/json",
            },
            params={"url": url, "live": "true", "max_chars": self.settings.max_bytes},
        ) as response:
            if response.status_code in {429, 502, 503, 504}:
                raise RetryablePageError(
                    f"Keenable returned HTTP {response.status_code}.",
                    response.headers.get("Retry-After"),
                )
            if response.status_code != 200:
                raise PageError(
                    f"Keenable could not fetch the article (HTTP {response.status_code})."
                )
            body = await self._read_body(response)
        try:
            data = json.loads(body)
        except (ValueError, RecursionError):
            raise PageError("Invalid response from Keenable.") from None
        if not isinstance(data, dict) or not isinstance(data.get("content"), str):
            raise PageError("Invalid response from Keenable.")
        if data.get("truncated"):
            raise PageError("Article exceeds the size limit.")
        text = data["content"].strip()
        if not text or "\0" in text:
            raise PageError("No readable article text found.")
        return ExtractionResult(text, data.get("title"), data.get("url") or url)

    async def _read_body(self, response: httpx.Response) -> bytes:
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > self.settings.max_bytes:
                raise PageError("Webpage exceeds the download size limit.")
        return bytes(body)

    @staticmethod
    def _validate_url(value: str) -> httpx.URL:
        try:
            url = httpx.URL(value)
            if (
                url.scheme not in {"http", "https"}
                or not url.host
                or url.userinfo
                or (url.port is not None and not 1 <= url.port <= 65535)
            ):
                raise ValueError
        except (ValueError, httpx.InvalidURL):
            raise PageError("Enter a valid HTTP(S) URL without credentials.") from None
        return url.copy_with(fragment=None)

    @staticmethod
    def _retry_delay(header: str | None, attempt: int) -> float:
        fallback = 0.5 * 2**attempt
        if header is None:
            return fallback
        try:
            delay = float(header)
        except ValueError:
            try:
                date = parsedate_to_datetime(header)
                delay = (date - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                return fallback
        return min(5, max(0, delay)) if math.isfinite(delay) else fallback
