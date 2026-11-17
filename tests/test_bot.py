import json
import logging
import os
from asyncio import CancelledError
from unittest.mock import AsyncMock, Mock, patch

import asyncpg
import httpx
import pytest

from app.intake import telegram


def message_update(
    update_id: int, sender_id: int = 42, chat_type: str = "private"
) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id + 100,
            "chat": {"id": sender_id, "type": chat_type},
            "from": {"id": sender_id},
            "text": "Заметка про информационную безопасность",
        },
    }


class TestBot:
    @pytest.fixture(autouse=True)
    def bot_dependencies(self, tmp_path, monkeypatch):
        self.heartbeat = tmp_path / "heartbeat"
        monkeypatch.setattr(telegram, "HEARTBEAT", self.heartbeat)

    async def test_ignores_unapproved_senders_and_group_chats(self):
        db = Mock(enqueue=AsyncMock())
        with patch.object(telegram, "telegram_request") as request:
            await telegram.handle_update(
                Mock(), db, {42}, message_update(10, sender_id=43)
            )
            await telegram.handle_update(
                Mock(), db, {42}, message_update(11, chat_type="group")
            )
        db.enqueue.assert_not_called()
        request.assert_not_called()

    @pytest.mark.parametrize("value", ("", "-42", "invalid"))
    async def test_requires_explicit_allowlist_before_opening_client(self, value):
        self.heartbeat.touch()
        with patch.dict(
            "os.environ",
            {"TELEGRAM_BOT_TOKEN": "test", "TELEGRAM_ALLOWED_USER_IDS": value},
        ):
            with (
                patch.object(httpx, "AsyncClient") as client,
                patch.object(telegram, "open_postgres") as open_postgres,
            ):
                with pytest.raises(SystemExit):
                    await telegram.main()
                assert not self.heartbeat.exists()
                client.assert_not_called()
                open_postgres.assert_not_called()

    async def test_retries_from_last_saved_update_and_replies_after_persisting(
        self, caplog
    ):
        first, second = message_update(10), message_update(11)
        timeline = []
        attempts = 0
        poll_responses = iter(
            [(0, [first, second]), (11, [second]), (12, []), (0, None)]
        )

        async def enqueue(owner_id, note_input, *, telegram_message_id):
            nonlocal attempts
            assert owner_id == "42"
            assert note_input.value == first["message"]["text"]
            timeline.append(("save", telegram_message_id))
            attempts += 1
            if attempts == 2:
                raise asyncpg.ConnectionDoesNotExistError("temporary database outage")
            return telegram_message_id

        def respond(request):
            data = json.loads(request.content)
            if request.url.path.endswith("/sendMessage"):
                assert data["chat_id"] == 42
                timeline.append(("reply", data["text"]))
                return httpx.Response(200, json={"ok": True, "result": {}})
            offset = data["offset"]
            timeline.append(("poll", offset))
            expected_offset, updates = next(poll_responses)
            assert offset == expected_offset
            if updates is None:
                raise CancelledError
            return httpx.Response(200, json={"ok": True, "result": updates})

        db = Mock(enqueue=AsyncMock(side_effect=enqueue))
        async with httpx.AsyncClient(
            base_url="https://telegram.test/bot/",
            transport=httpx.MockTransport(respond),
        ) as client:
            with patch.object(telegram, "sleep") as sleep:
                with (
                    caplog.at_level("WARNING", logger=telegram.logger.name),
                    pytest.raises(CancelledError),
                ):
                    await telegram.poll(client, db, {42})
                assert any(
                    record.name == telegram.logger.name
                    and record.levelno >= logging.WARNING
                    for record in caplog.records
                )
                sleep.assert_awaited_once_with(2)

        assert timeline == [
            ("poll", 0),
            ("save", 110),
            ("reply", "Received note #110. Check its status in the web app."),
            ("save", 111),
            ("poll", 11),
            ("save", 111),
            ("reply", "Received note #111. Check its status in the web app."),
            ("poll", 12),
            ("poll", 0),
        ]

    async def test_permanent_database_error_hides_submitted_content(self):
        db = Mock(
            enqueue=AsyncMock(side_effect=asyncpg.CheckViolationError("private note"))
        )
        with patch.object(
            telegram, "telegram_request", return_value=[message_update(10)]
        ) as request:
            with pytest.raises(RuntimeError, match="CheckViolationError") as error:
                await telegram.poll(Mock(), db, {42})

        assert "private note" not in str(error.value)
        assert error.value.__suppress_context__
        request.assert_awaited_once()
        assert not self.heartbeat.exists()

    async def test_health_tracks_capture_and_recovers_after_poll_failure(self, caplog):
        def check_health():
            with patch("builtins.print"):
                return telegram.healthcheck()

        assert check_health() == 1

        calls = 0

        def respond(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(200, json={"ok": True, "result": []})
            if calls == 2:
                assert check_health() == 0
                os.utime(self.heartbeat, (0, 0))
                return httpx.Response(503)
            if calls == 3:
                assert check_health() == 1
                return httpx.Response(200, json={"ok": True, "result": []})
            assert check_health() == 0
            raise CancelledError

        async with httpx.AsyncClient(
            base_url="https://telegram.test", transport=httpx.MockTransport(respond)
        ) as client:
            with (
                patch.object(telegram, "sleep"),
                caplog.at_level("WARNING", logger=telegram.logger.name),
                pytest.raises(CancelledError),
            ):
                await telegram.poll(client, Mock(), {42})
        assert any(
            record.name == telegram.logger.name and record.levelno >= logging.WARNING
            for record in caplog.records
        )

    @pytest.mark.parametrize(
        "status, reason", [(401, "bot token"), (404, "bot token"), (409, "webhook")]
    )
    async def test_poll_reports_invalid_token_and_delivery_conflict(
        self, status, reason
    ):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(status, text="private response")
        )
        async with httpx.AsyncClient(
            base_url="https://telegram.test/botSECRET/", transport=transport
        ) as client:
            with (
                patch.object(telegram, "sleep", side_effect=CancelledError),
                pytest.raises(RuntimeError, match=reason) as error,
            ):
                await telegram.poll(client, Mock(), {42})
        assert "SECRET" not in str(error.value)
        assert "private response" not in str(error.value)
        assert not self.heartbeat.exists()
