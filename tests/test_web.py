from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.intake import web


class TestWeb:
    @pytest.fixture(autouse=True)
    def web_dependencies(self, monkeypatch):
        self.db = SimpleNamespace(
            list_notes=AsyncMock(return_value=[]),
            get_note=AsyncMock(),
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
