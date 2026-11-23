import re
from contextlib import asynccontextmanager
from html import unescape
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.intake import web
from app.notes.models import Note
from app.processing.exceptions import SearchUnavailable
from app.processing.search import SearchResult
from app.processing.search_config import MAX_SEARCH_CANDIDATES, MAX_SOURCE_CHARS


class TestWeb:
    @pytest.fixture(autouse=True)
    def web_dependencies(self, monkeypatch):
        self.db = SimpleNamespace(
            list_notes=AsyncMock(return_value=[]),
            get_note=AsyncMock(),
            get_photo=AsyncMock(return_value=None),
            enqueue=AsyncMock(return_value=23),
            search_similar=AsyncMock(return_value=[]),
        )
        self.embedder = None
        self.search_client = None

        @asynccontextmanager
        async def open_postgres():
            yield self.db

        @asynccontextmanager
        async def open_embedder():
            yield self.embedder

        @asynccontextmanager
        async def open_search_client():
            yield self.search_client

        monkeypatch.setattr(web, "open_postgres", open_postgres)
        monkeypatch.setattr(web, "open_embedder", open_embedder)
        monkeypatch.setattr(web, "open_search_client", open_search_client)
        monkeypatch.setenv("PKA_WEB_USERS", '{"101":"first","202":"second"}')
        self.auth = ("101", "first")

    def test_access_requires_configured_credentials(self):
        with TestClient(web.app) as client:
            assert client.get("/").status_code == 401
            assert client.get("/", auth=("101", "wrong")).status_code == 401
            with patch.dict("os.environ", {"PKA_WEB_USERS": "null"}):
                assert client.get("/", auth=self.auth).status_code == 401
        self.db.list_notes.assert_not_awaited()

    def test_search_and_detail_are_owner_scoped_and_escape_content(self):
        payload = "<script>alert(1)</script>"
        note = SimpleNamespace(
            id=23,
            kind="text",
            value=payload,
            status="ready",
            text=payload,
            summary=payload,
            tags=[payload],
            error=None,
        )
        self.db.list_notes.return_value = [note]
        self.db.get_note.side_effect = lambda owner_id, note_id: (
            note if (owner_id, note_id) == ("101", 23) else None
        )

        with TestClient(web.app) as client:
            response = client.get("/", params={"q": payload}, auth=self.auth)
            self.db.list_notes.assert_awaited_once_with("101", payload)
            assert response.status_code == 200
            assert payload not in response.text
            assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text
            assert 'name="mode"' not in response.text

            response = client.get("/notes/23", auth=self.auth)
            assert response.status_code == 200
            assert payload not in response.text
            assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text
            assert client.get("/notes/23", auth=("202", "second")).status_code == 404

    def test_only_json_creates_notes_for_authenticated_owner(self):
        with TestClient(web.app) as client:
            response = client.post(
                "/api/notes",
                auth=self.auth,
                content='{"value":"should not enqueue"}',
                headers={"Content-Type": "text/plain"},
            )
            assert response.status_code == 415
            self.db.enqueue.assert_not_awaited()

            response = client.post(
                "/api/notes", auth=self.auth, json={"value": "Заметка"}
            )
            assert response.status_code == 201
            assert response.json() == {"id": 23}
            assert self.db.enqueue.await_args.args[0] == "101"
            assert self.db.enqueue.await_args.args[1].value == "Заметка"

    def test_invalid_unicode_is_rejected_before_database_access(self):
        with TestClient(web.app) as client:
            response = client.post(
                "/api/notes",
                auth=self.auth,
                content=r'{"value":"\ud800"}',
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 422
            assert client.get("/?q=%00", auth=self.auth).status_code == 422
        self.db.enqueue.assert_not_awaited()
        self.db.list_notes.assert_not_awaited()

    def test_photo_upload_validates_bytes_and_uses_authenticated_owner(self):
        self.embedder = SimpleNamespace(supports_images=True)
        output = BytesIO()
        Image.new("RGB", (16, 16), "red").save(output, "PNG")
        photo = output.getvalue()

        with TestClient(web.app) as client:
            assert 'name="photo"' in client.get("/", auth=self.auth).text
            files = {"photo": ("note.png", photo, "application/octet-stream")}
            assert client.post("/api/photos", files=files).status_code == 401
            response = client.post(
                "/api/photos",
                auth=self.auth,
                files=files,
                data={"caption": "  Red  "},
                headers={"X-PKA-Upload": "1"},
            )
            assert response.status_code == 201
            assert response.json() == {"id": 23}
            owner, note = self.db.enqueue.await_args.args
            assert owner == "101"
            assert (note.kind, note.value, note.image, note.image_media_type) == (
                "image",
                "Red",
                photo,
                "image/png",
            )
            response = client.post(
                "/api/photos",
                auth=self.auth,
                files={"photo": ("fake.png", b"invalid", "image/png")},
                headers={"X-PKA-Upload": "1"},
            )
            assert response.status_code == 422
            self.db.enqueue.assert_awaited_once()

    def test_incomplete_photo_upload_does_not_drop_the_caption(self):
        self.embedder = SimpleNamespace(supports_images=True)
        output = BytesIO()
        Image.new("RGB", (16, 16)).save(output, "PNG")
        with TestClient(web.app) as client:
            request = client.build_request(
                "POST",
                "/api/photos",
                files=[
                    ("photo", ("note.png", output.getvalue(), "image/png")),
                    ("caption", (None, "Incomplete caption.")),
                ],
            )
            truncated = (
                request.read().split(b"Incomplete caption.", 1)[0] + b"Incomplete"
            )
            response = client.post(
                "/api/photos",
                auth=self.auth,
                content=truncated,
                headers={
                    "Content-Type": request.headers["content-type"],
                    "X-PKA-Upload": "1",
                },
            )
            assert response.status_code == 422
            self.db.enqueue.assert_not_awaited()

    def test_photo_response_uses_owner_and_disables_caching(self):
        self.db.get_photo.side_effect = lambda owner, note_id: (
            (b"photo", "image/jpeg") if (owner, note_id) == ("101", 23) else None
        )
        with TestClient(web.app) as client:
            response = client.get("/notes/23/photo", auth=self.auth)
            assert response.status_code == 200
            assert response.content == b"photo"
            assert response.headers["content-type"] == "image/jpeg"
            assert response.headers["cache-control"] == "no-store"
            assert (
                client.get("/notes/23/photo", auth=("202", "second")).status_code == 404
            )

    def test_notebook_defaults_to_meaning_without_running_a_search(self):
        self.embedder = SimpleNamespace(embed=AsyncMock())

        with TestClient(web.app) as client:
            response = client.get("/", auth=self.auth)

        assert response.status_code == 200
        assert '<option value="semantic" selected>Meaning</option>' in response.text
        assert '<option value="20" selected>20</option>' in response.text
        self.embedder.embed.assert_not_awaited()
        self.db.list_notes.assert_awaited_once_with("101", "")

    def test_existing_search_url_without_a_mode_uses_keyword(self):
        self.embedder = SimpleNamespace(embed=AsyncMock())

        with TestClient(web.app) as client:
            response = client.get("/", params={"q": "blue door"}, auth=self.auth)

        assert response.status_code == 200
        assert '<option value="keyword" selected>Keyword</option>' in response.text
        self.embedder.embed.assert_not_awaited()
        self.db.list_notes.assert_awaited_once_with("101", "blue door")

    def test_meaning_search_marks_only_the_first_ranked_result(self):
        self.embedder = SimpleNamespace(embed=AsyncMock(return_value=object()))
        self.db.search_similar.return_value = [
            Note(23, "image", "Blue door", "ready", None, None, [], None, 0.2),
            Note(42, "text", "Door in the garden", "ready", None, None, [], None, 0.4),
        ]

        with TestClient(web.app) as client:
            response = client.get(
                "/", params={"q": "door", "mode": "semantic"}, auth=self.auth
            )

        assert response.status_code == 200
        assert response.text.count("Top match") == 1
        assert (
            response.text.index("Top match")
            < response.text.index('href="/notes/23"')
            < response.text.index('href="/notes/42"')
        )

    def test_semantic_search_embeds_query_and_scopes_owner(self):
        embedding = object()
        self.embedder = SimpleNamespace(embed=AsyncMock(return_value=embedding))

        with TestClient(web.app) as client:
            response = client.get(
                "/", params={"q": " сетевой экран ", "mode": "semantic"}, auth=self.auth
            )
            assert response.status_code == 200
            assert '<option value="semantic" selected>' in response.text

        self.embedder.embed.assert_awaited_once_with("сетевой экран")
        self.db.search_similar.assert_awaited_once_with("101", embedding, limit=20)
        self.db.list_notes.assert_not_awaited()

    def test_semantic_search_reranks_a_larger_pool_and_cites_owner_notes(self):
        embedding = object()
        self.embedder = SimpleNamespace(embed=AsyncMock(return_value=embedding))
        self.search_client = SimpleNamespace(
            search=AsyncMock(
                return_value=SearchResult(
                    ranked_ids=[42, 23],
                    answer="The door was <blue>.",
                    citations=[42, 23],
                )
            )
        )
        link = Note(
            id=23,
            kind="url",
            value="https://example.com/article",
            status="ready",
            text="The door was photographed in April. " * 100,
            summary="Background. " * 60,
            tags=[],
            error=None,
            cosine_distance=0.1,
        )
        note = Note(42, "text", "The <blue> door", "ready", None, None, [], None, 0.3)
        self.db.search_similar.return_value = [link, note]
        self.db.get_note.side_effect = lambda owner_id, note_id: (
            note if (owner_id, note_id) == ("101", 42) else None
        )
        parent = FastAPI(lifespan=web.lifespan)
        parent.mount("/team#1/pka", web.app)

        with TestClient(parent) as client:
            response = client.get(
                "/team%231/pka/",
                params={"q": " What color was the door? ", "mode": "semantic", "k": 1},
                auth=self.auth,
            )
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store"
            assert "Answer from your notes" in response.text
            assert "The door was &lt;blue&gt;." in response.text
            assert "The door was <blue>." not in response.text
            assert "The &lt;blue&gt; door" in response.text
            assert "The <blue> door" not in response.text
            answer, results = response.text.split("<h3>Search results</h3>")
            assert 'href="/team%231/pka/notes/42"' in answer
            assert 'href="/team%231/pka/notes/23"' in answer
            assert 'href="/team%231/pka/notes/42"' in results
            assert 'href="/team%231/pka/notes/23"' not in results
            assert "Vector similarity 0.70" in results
            assert "LLM reranks" in results

            source_url = "/team%231/pka/notes/42"
            assert client.get(source_url, auth=self.auth).status_code == 200
            assert client.get(source_url, auth=("202", "second")).status_code == 404

        self.db.search_similar.assert_awaited_once_with(
            "101", embedding, limit=MAX_SEARCH_CANDIDATES
        )
        self.search_client.search.assert_awaited_once()
        request = self.search_client.search.await_args.args[0]
        assert request.query == "What color was the door?"
        assert [candidate.id for candidate in request.candidates] == [23, 42]
        assert len(request.candidates[0].text) == MAX_SOURCE_CHARS
        assert link.value not in request.candidates[0].text
        assert "The door was photographed in April." in request.candidates[0].text
        assert request.candidates[1].text == note.value

    def test_reranking_preserves_image_position_and_can_abstain(self):
        self.embedder = SimpleNamespace(embed=AsyncMock(), supports_images=True)
        self.search_client = SimpleNamespace(
            search=AsyncMock(
                return_value=SearchResult(
                    ranked_ids=[31, 23], answer=None, citations=[]
                )
            )
        )
        self.db.search_similar.return_value = [
            Note(23, "text", "First", "ready", None, None, [], None, 0.1),
            Note(42, "image", "", "ready", None, None, [], None, 0.2),
            Note(31, "text", "Third", "ready", None, None, [], None, 0.3),
        ]

        with TestClient(web.app) as client:
            response = client.get(
                "/",
                params={"q": "When did I save the photo?", "mode": "semantic"},
                auth=self.auth,
            )

        assert response.status_code == 200
        assert "did not provide enough evidence" in response.text
        assert "<h4>Sources</h4>" not in response.text
        assert re.findall(r'href="/notes/(\d+)"', response.text) == ["31", "42", "23"]
        request = self.search_client.search.await_args.args[0]
        assert [candidate.id for candidate in request.candidates] == [23, 31]

    def test_unavailable_llm_search_preserves_vector_results(self):
        self.embedder = SimpleNamespace(embed=AsyncMock())
        self.search_client = SimpleNamespace(
            search=AsyncMock(side_effect=SearchUnavailable("private detail"))
        )
        self.db.search_similar.return_value = [
            Note(23, "text", "First", "ready", None, None, [], None, 0.1),
            Note(42, "text", "Second", "ready", None, None, [], None, 0.2),
        ]

        with TestClient(web.app) as client:
            response = client.get(
                "/",
                params={"q": "first", "mode": "semantic", "k": 1},
                auth=self.auth,
            )

        assert response.status_code == 200
        assert "LLM search is unavailable. Showing vector matches." in response.text
        assert "private detail" not in response.text
        assert re.findall(r'href="/notes/(\d+)"', response.text) == ["23"]

    def test_image_only_search_does_not_call_the_text_model(self):
        self.embedder = SimpleNamespace(embed=AsyncMock(), supports_images=True)
        self.search_client = SimpleNamespace(search=AsyncMock())
        self.db.search_similar.return_value = [
            Note(42, "image", "", "ready", None, None, [], None, 0.1),
        ]

        with TestClient(web.app) as client:
            response = client.get(
                "/", params={"q": "red circle", "mode": "semantic"}, auth=self.auth
            )

        assert response.status_code == 200
        assert 'href="/notes/42"' in response.text
        assert "Matching notes have no text to answer from." in response.text
        self.search_client.search.assert_not_awaited()

    @pytest.mark.parametrize(
        ("mount_path", "url_path"),
        (
            ("/pka", "/pka"),
            ("/team#1/pka", "/team%231/pka"),
            ("/team?2/pka", "/team%3F2/pka"),
        ),
    )
    def test_search_and_note_navigation_use_the_mount_path(self, mount_path, url_path):
        note = SimpleNamespace(
            id=23,
            kind="image",
            value="Reference photo",
            status="queued",
            summary=None,
            tags=[],
            error=None,
            cosine_distance=0.186,
        )
        self.embedder = SimpleNamespace(
            embed=AsyncMock(return_value=object()), supports_images=True
        )
        self.db.search_similar.return_value = [note]
        self.db.get_note.return_value = note
        self.db.get_photo.return_value = (b"photo", "image/jpeg")
        parent = FastAPI(lifespan=web.lifespan)
        parent.mount(mount_path, web.app)
        with TestClient(parent) as client:
            results = client.get(
                f"{url_path}/",
                params={"q": "photo", "mode": "semantic"},
                auth=self.auth,
            )
            assert results.status_code == 200
            assert f'href="{url_path}/notes/23"' in results.text
            assert "Similarity 0.81" in results.text
            assert f'action="{url_path}/#find" method="get"' in results.text
            assert f'action="{url_path}/api/notes"' in results.text
            assert f'action="{url_path}/api/photos"' in results.text
            assert f'data-note-path="{url_path}/notes/"' in results.text

            detail = client.get(f"{url_path}/notes/23", auth=self.auth)
            assert detail.status_code == 200
            assert f'href="{url_path}/">All notes' in detail.text
            assert f'href="{url_path}/notes/23">Refresh status' in detail.text
            assert f'src="{url_path}/notes/23/photo"' in detail.text
            assert (
                client.get(f"{url_path}/notes/23/photo", auth=self.auth).content
                == b"photo"
            )

    @pytest.mark.parametrize("limit", (1, 12, 50))
    def test_semantic_search_honors_and_preserves_result_limit(self, limit):
        embedding = object()
        self.embedder = SimpleNamespace(embed=AsyncMock(return_value=embedding))
        with TestClient(web.app) as client:
            response = client.get(
                "/",
                params={"q": "network", "mode": "semantic", "k": limit},
                auth=self.auth,
            )
        assert response.status_code == 200
        choices = re.search(
            r'<select id="k" name="k">(.*?)</select>', response.text, re.S
        )
        assert choices is not None
        assert re.findall(r'<option value="([^"]+)"', choices[1]) == [
            str(value) for value in range(1, 51)
        ]
        assert f'<option value="{limit}" selected>' in choices[1]
        self.db.search_similar.assert_awaited_once_with("101", embedding, limit=limit)

    @pytest.mark.parametrize("limit", ("0", "51", "1.5", "many"))
    def test_invalid_result_limit_is_rejected_before_embedding(self, limit):
        self.embedder = SimpleNamespace(embed=AsyncMock())
        with TestClient(web.app) as client:
            response = client.get(
                "/",
                params={"q": "network", "mode": "semantic", "k": limit},
                auth=self.auth,
            )
        assert response.status_code == 422
        self.embedder.embed.assert_not_awaited()
        self.db.search_similar.assert_not_awaited()

    @pytest.mark.parametrize("limit", ("", "51", "1e1"))
    def test_keyword_search_ignores_an_invalid_semantic_result_limit(self, limit):
        self.embedder = SimpleNamespace(embed=AsyncMock())
        self.search_client = SimpleNamespace(search=AsyncMock())
        with TestClient(web.app) as client:
            response = client.get(
                "/",
                params={"q": "network", "mode": "keyword", "k": limit},
                auth=self.auth,
            )
        assert response.status_code == 200
        self.embedder.embed.assert_not_awaited()
        self.search_client.search.assert_not_awaited()
        self.db.list_notes.assert_awaited_once_with("101", "network")

    @pytest.mark.parametrize("configured", (True, False))
    def test_unavailable_semantic_search_links_to_keyword_results(self, configured):
        if configured:
            self.embedder = SimpleNamespace(
                embed=AsyncMock(
                    side_effect=web.EmbeddingError("internal provider error")
                )
            )

        query = "экран & сеть"
        parent = FastAPI(lifespan=web.lifespan)
        parent.mount("/team#1/pka", web.app)
        with TestClient(parent) as client:
            response = client.get(
                "/team%231/pka/",
                params={"q": query, "mode": "semantic"},
                auth=self.auth,
            )
            assert response.status_code == 503
            assert response.headers["content-type"].startswith("text/html")
            assert "internal provider error" not in response.text
            match = re.search(r'href="([^"]+)">Try keyword search</a>', response.text)
            assert match is not None
            response = client.get(unescape(match[1]), auth=self.auth)
            assert response.status_code == 200

        self.db.search_similar.assert_not_awaited()
        self.db.list_notes.assert_awaited_once_with("101", query)
