import json
import unittest
from asyncio import CancelledError
from unittest.mock import AsyncMock, Mock, patch

import asyncpg
import httpx

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


class BotTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_requires_explicit_allowlist_before_opening_client(self):
        for value in ("", "-42", "invalid"):
            with self.subTest(value=value):
                with patch.dict(
                    "os.environ",
                    {"TELEGRAM_BOT_TOKEN": "test", "TELEGRAM_ALLOWED_USER_IDS": value},
                ):
                    with (
                        patch.object(httpx, "AsyncClient") as client,
                        patch.object(telegram, "open_postgres") as open_postgres,
                    ):
                        with self.assertRaises(SystemExit):
                            await telegram.main()
                        client.assert_not_called()
                        open_postgres.assert_not_called()

    async def test_retries_from_last_saved_update_and_replies_after_persisting(self):
        first, second = message_update(10), message_update(11)
        timeline = []
        attempts = 0
        poll_responses = iter(
            [(0, [first, second]), (11, [second]), (12, []), (0, None)]
        )

        async def enqueue(owner_id, note_input, *, telegram_message_id):
            nonlocal attempts
            self.assertEqual(owner_id, "42")
            self.assertEqual(note_input.value, first["message"]["text"])
            timeline.append(("save", telegram_message_id))
            attempts += 1
            if attempts == 2:
                raise asyncpg.ConnectionDoesNotExistError("temporary database outage")
            return telegram_message_id

        def respond(request):
            data = json.loads(request.content)
            if request.url.path.endswith("/sendMessage"):
                self.assertEqual(data["chat_id"], 42)
                timeline.append(("reply", data["text"]))
                return httpx.Response(200, json={"ok": True, "result": {}})
            offset = data["offset"]
            timeline.append(("poll", offset))
            expected_offset, updates = next(poll_responses)
            self.assertEqual(offset, expected_offset)
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
                    self.assertLogs(telegram.logger, "WARNING"),
                    self.assertRaises(CancelledError),
                ):
                    await telegram.poll(client, db, {42})
                sleep.assert_awaited_once_with(2)

        self.assertEqual(
            timeline,
            [
                ("poll", 0),
                ("save", 110),
                ("reply", "Received note #110. Check its status in the web app."),
                ("save", 111),
                ("poll", 11),
                ("save", 111),
                ("reply", "Received note #111. Check its status in the web app."),
                ("poll", 12),
                ("poll", 0),
            ],
        )

    async def test_permanent_database_error_hides_submitted_content(self):
        db = Mock(
            enqueue=AsyncMock(side_effect=asyncpg.CheckViolationError("private note"))
        )
        with patch.object(
            telegram, "telegram_request", return_value=[message_update(10)]
        ) as request:
            with self.assertRaisesRegex(RuntimeError, "CheckViolationError") as error:
                await telegram.poll(Mock(), db, {42})

        self.assertNotIn("private note", str(error.exception))
        self.assertTrue(error.exception.__suppress_context__)
        request.assert_awaited_once()
