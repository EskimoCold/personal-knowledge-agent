from contextlib import asynccontextmanager
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.intake import web


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

        @asynccontextmanager
        async def open_postgres():
            yield self.db

        @asynccontextmanager
        async def open_embedder():
            yield self.embedder

        monkeypatch.setattr(web, "open_postgres", open_postgres)
        monkeypatch.setattr(web, "open_embedder", open_embedder)
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
        self.db.search_similar.assert_awaited_once_with("101", embedding)
        self.db.list_notes.assert_not_awaited()

    def test_unavailable_embedder_does_not_prevent_keyword_search(self):
        self.embedder = SimpleNamespace(
            embed=AsyncMock(side_effect=web.EmbeddingError("internal provider error"))
        )

        with TestClient(web.app) as client:
            response = client.get(
                "/", params={"q": "экран", "mode": "semantic"}, auth=self.auth
            )
            assert response.status_code == 503
            assert "internal provider error" not in response.text
            response = client.get("/", params={"q": "экран"}, auth=self.auth)
            assert response.status_code == 200

        self.db.search_similar.assert_not_awaited()
        self.db.list_notes.assert_awaited_once_with("101", "экран")
