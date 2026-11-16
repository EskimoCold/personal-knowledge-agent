import asyncio
import gzip
import os
import ssl
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from app.processing.webpage.config import WebpageConfig
from app.processing.webpage.exceptions import PageError
from app.processing.webpage.fetcher import PageFetcher


ARTICLE = Path(__file__).with_name("fixtures").joinpath("article.html").read_bytes()


class Chunks(httpx.AsyncByteStream):
    def __init__(self, *chunks, delay=0):
        self.chunks = chunks
        self.delay = delay
        self.closed = False
        self.reads = 0

    async def __aiter__(self):
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            self.reads += 1
            yield chunk

    async def aclose(self):
        self.closed = True


def html_response(body=ARTICLE, **kwargs):
    return httpx.Response(
        200, headers={"Content-Type": "text/html; charset=utf-8", **kwargs},
        stream=Chunks(body),
    )


class FetchTests(unittest.IsolatedAsyncioTestCase):
    async def fetch(self, handler, url="https://article.test/start", **settings):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), trust_env=False
        ) as client:
            return await PageFetcher(client, WebpageConfig(**settings)).fetch(url)

    async def test_extracts_article_and_title_without_navigation_or_comments(self):
        page = await self.fetch(lambda _: html_response())
        self.assertIn("A city garden can provide fresh vegetables", page.text)
        self.assertNotIn("Navigation marker", page.text)
        self.assertNotIn("Footer marker", page.text)
        self.assertNotIn("Comment marker", page.text)
        self.assertEqual(page.title, "Growing a city garden")
        self.assertEqual(page.final_url, "https://article.test/start")

    async def test_title_is_optional_and_xhtml_is_supported(self):
        body = ARTICLE.replace(b"<title>Growing a city garden</title>", b"").replace(
            b"<h1>Growing a city garden</h1>", b""
        )
        page = await self.fetch(
            lambda _: html_response(body, **{"Content-Type": "application/xhtml+xml"})
        )
        self.assertTrue(page.text)
        self.assertIsNone(page.title)

    async def test_honors_response_charset_and_html_declared_encoding(self):
        body = ARTICLE.replace(b"A city garden", "Un jardin d'été".encode("cp1252"))
        for content, headers in (
            (body, {"Content-Type": "text/html; charset=windows-1252"}),
            (body.replace(b"<head>", b'<head><meta charset="windows-1252">'),
             {"Content-Type": "text/html"}),
        ):
            with self.subTest(headers=headers):
                page = await self.fetch(lambda _: html_response(content, **headers))
                self.assertIn("Un jardin d'été", page.text)

    async def test_follows_relative_redirects_up_to_limit(self):
        paths = []

        def respond(request):
            paths.append(request.url.path)
            if len(paths) <= 5:
                return httpx.Response(302, headers={"Location": f"next{len(paths)}"})
            return html_response()

        page = await self.fetch(respond)
        self.assertEqual(len(paths), 6)
        self.assertEqual(page.final_url, "https://article.test/next5")

    async def test_cookies_are_never_sent_between_redirects_or_jobs(self):
        paths = []

        def respond(request):
            paths.append(request.url.path)
            self.assertNotIn("cookie", request.headers)
            if request.url.path == "/first":
                return httpx.Response(302, headers={
                    "Location": "/article", "Set-Cookie": "session=owner-one; Path=/; Secure",
                })
            return html_response(**{"Set-Cookie": "session=owner-one; Path=/; Secure"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            fetcher = PageFetcher(client, WebpageConfig())
            await fetcher.fetch("https://article.test/first")
            await fetcher.fetch("https://article.test/second")
        self.assertEqual(paths, ["/first", "/article", "/second"])

    async def test_redirect_loop_stops_without_retrying(self):
        calls = 0

        def respond(_):
            nonlocal calls
            calls += 1
            return httpx.Response(302, headers={"Location": "/again?secret=hidden"})

        with self.assertRaisesRegex(PageError, "redirect") as caught:
            await self.fetch(respond)
        self.assertEqual(calls, 6)
        self.assertNotIn("hidden", str(caught.exception))

    async def test_rejects_empty_html_and_unsupported_content_without_retries(self):
        for response, message in (
            (html_response(b"<html><body><script>run()</script></body></html>"), "article"),
            (httpx.Response(200, headers={"Content-Type": "application/pdf"}), "HTML"),
            (httpx.Response(200, headers={"Content-Type": "image/png"}), "HTML"),
            (httpx.Response(200), "HTML"),
        ):
            calls = []
            with self.subTest(message=message):
                with self.assertRaisesRegex(PageError, message):
                    await self.fetch(lambda _: calls.append(1) or response)
                self.assertEqual(len(calls), 1)

    async def test_stream_limit_ignores_absent_or_misleading_content_length(self):
        for headers in ({}, {"Content-Length": "1"}):
            stream = Chunks(b"1234", b"5678", b"never read")
            response = httpx.Response(
                200, headers={"Content-Type": "text/html", **headers}, stream=stream
            )
            with self.subTest(headers=headers):
                with self.assertRaisesRegex(PageError, "size"):
                    await self.fetch(lambda _: response, max_bytes=7)
                self.assertEqual(stream.reads, 2)
                self.assertTrue(stream.closed)

    async def test_compressed_body_limit_counts_decoded_bytes(self):
        stream = Chunks(gzip.compress(b"x" * 100_000))
        response = httpx.Response(200, headers={
            "Content-Type": "text/html", "Content-Encoding": "gzip",
        }, stream=stream)
        with self.assertRaisesRegex(PageError, "size"):
            await self.fetch(lambda _: response, max_bytes=100)
        self.assertTrue(stream.closed)

    async def test_extracts_gzip_html(self):
        page = await self.fetch(lambda _: html_response(
            gzip.compress(ARTICLE), **{"Content-Encoding": "gzip"}
        ))
        self.assertIn("fresh vegetables", page.text)

    async def test_timeout_covers_whole_redirect_and_body_attempt(self):
        streams = []

        async def respond(request):
            await asyncio.sleep(0.02)
            if request.url.path == "/start":
                return httpx.Response(302, headers={"Location": "/article"})
            stream = Chunks(ARTICLE, delay=0.02)
            streams.append(stream)
            return httpx.Response(200, headers={"Content-Type": "text/html"}, stream=stream)

        with self.assertRaisesRegex(PageError, "timed out"):
            await self.fetch(respond, timeout_seconds=0.05, retries=0)
        self.assertTrue(streams[0].closed)

    async def test_transient_failures_retry_at_most_twice_with_bounded_backoff(self):
        for failure in (httpx.ConnectTimeout("private data"), 429, 502, 503, 504):
            calls = []

            def respond(_):
                calls.append(1)
                if isinstance(failure, Exception):
                    raise failure
                return httpx.Response(failure, headers={"Retry-After": "9999"})

            with self.subTest(failure=type(failure).__name__):
                with patch("app.processing.webpage.fetcher.asyncio.sleep", new_callable=AsyncMock) as sleep:
                    with self.assertRaises(PageError) as caught:
                        await self.fetch(respond)
                self.assertEqual(len(calls), 3)
                self.assertEqual(sleep.await_count, 2)
                self.assertTrue(all(0 <= call.args[0] <= 5 for call in sleep.await_args_list))
                self.assertNotIn("private data", str(caught.exception))

    async def test_transient_status_can_recover_and_respects_retry_after(self):
        responses = iter((httpx.Response(429, headers={"Retry-After": "2"}), html_response()))
        with patch("app.processing.webpage.fetcher.asyncio.sleep", new_callable=AsyncMock) as sleep:
            page = await self.fetch(lambda _: next(responses))
        self.assertIn("fresh vegetables", page.text)
        self.assertEqual(sleep.await_args.args[0], 2)

    async def test_permanent_errors_and_tls_failures_are_not_retried(self):
        for failure in (404, 403, httpx.ConnectError("certificate", request=None)):
            calls = []

            def respond(_):
                calls.append(1)
                if isinstance(failure, Exception):
                    raise failure from ssl.SSLCertVerificationError("secret certificate")
                return httpx.Response(failure, text="private server content")

            with self.subTest(failure=failure):
                with self.assertRaises(PageError) as caught:
                    await self.fetch(respond)
                self.assertEqual(len(calls), 1)
                self.assertNotIn("private", str(caught.exception))

    async def test_network_errors_other_than_timeouts_are_not_retried(self):
        for error in (
            httpx.ConnectError("Connection failed"),
            httpx.ReadError("Connection closed"),
            httpx.RemoteProtocolError("Invalid response"),
        ):
            calls = []

            def respond(_):
                calls.append(1)
                raise error

            with self.subTest(error=type(error).__name__):
                with patch("app.processing.webpage.fetcher.asyncio.sleep", new_callable=AsyncMock):
                    with self.assertRaises(PageError):
                        await self.fetch(respond)
                self.assertEqual(len(calls), 1)

    async def test_http_fetch_supports_hostnames_and_internal_ip_urls(self):
        for url in (
            "https://article.test/", "http://localhost/", "http://127.0.0.1/",
            "http://10.0.0.1/", "http://[::1]/",
        ):
            with self.subTest(url=url):
                page = await self.fetch(lambda _: html_response(), url)
                self.assertIn("fresh vegetables", page.text)
                self.assertEqual(page.final_url, url)

    async def test_invalid_urls_never_reach_http_transport(self):
        for url in (
            "ftp://example.org/file", "https:///missing", "http://user:pass@example.org/",
            "https://example.org:bad/", "https://example.org:-1/",
            "https://example.org:65536/",
        ):
            calls = []
            with self.subTest(url=url):
                with self.assertRaises(PageError):
                    await self.fetch(lambda _: calls.append(1) or html_response(), url)
                self.assertEqual(calls, [])

    async def test_redirect_can_point_to_an_internal_address(self):
        def respond(request):
            if request.url.path == "/start":
                return httpx.Response(302, headers={"Location": "http://127.0.0.1/article"})
            return html_response()

        page = await self.fetch(respond)
        self.assertIn("fresh vegetables", page.text)
        self.assertEqual(page.final_url, "http://127.0.0.1/article")

    async def test_invalid_redirect_target_is_rejected_before_request(self):
        for location in ("ftp://article.test/file", "http://user:pass@article.test/", "http://article.test:65536/"):
            calls = []

            def respond(_):
                calls.append(1)
                return httpx.Response(302, headers={"Location": location})

            with self.subTest(location=location):
                with self.assertRaises(PageError):
                    await self.fetch(respond)
                self.assertEqual(len(calls), 1)

    async def test_cancellation_closes_response(self):
        stream = Chunks(ARTICLE, delay=10)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(
            200, headers={"Content-Type": "text/html"}, stream=stream
        ))) as client:
            task = asyncio.create_task(PageFetcher(client, WebpageConfig()).fetch("https://article.test/"))
            await asyncio.sleep(0.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(stream.closed)


class ConfigurationTests(unittest.IsolatedAsyncioTestCase):


    async def test_environment_config_and_limits(self):
        with patch.dict(os.environ, {
            "PAGE_FETCH_TIMEOUT_SECONDS": "10", "PAGE_FETCH_MAX_REDIRECTS": "3",
            "PAGE_FETCH_MAX_BYTES": "1234", "PAGE_FETCH_RETRIES": "1",
        }, clear=True):
            async with PageFetcher.open() as fetcher:
                self.assertEqual(fetcher.settings, WebpageConfig(10, 3, 1234, 1))
                self.assertIn("PersonalKnowledgeAgent", fetcher.client.headers["User-Agent"])
        for settings in ({"retries": 3}, {"max_redirects": 6}, {"max_bytes": 0}, {"timeout_seconds": float("nan")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                replace(WebpageConfig(), **settings)
