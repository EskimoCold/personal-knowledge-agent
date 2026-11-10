import logging
import os
from asyncio import run, sleep
from typing import Any

import asyncpg
import httpx

from app.notes.models import NoteInput
from app.storage.postgres import Postgres, open_postgres


logger = logging.getLogger(__name__)
HELP = "Send a text note or a standalone HTTP(S) link."


class TelegramError(Exception):
    pass


async def telegram_request(
    client: httpx.AsyncClient, method: str, **data: object
) -> Any:
    try:
        response = await client.post(method, json=data)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        # Telegram URLs contain the bot token, and responses can contain notes.
        raise TelegramError from None
    if (
        not isinstance(payload, dict)
        or payload.get("ok") is not True
        or "result" not in payload
    ):
        raise TelegramError
    return payload["result"]


async def handle_update(
    client: httpx.AsyncClient,
    db: Postgres,
    allowed_user_ids: set[int],
    update: dict[str, Any],
) -> None:
    message = update.get("message")
    if message is None or message["chat"]["type"] != "private":
        return
    sender_id = message.get("from", {}).get("id")
    if type(sender_id) is not int or sender_id not in allowed_user_ids:
        return

    text = message.get("text")
    reply = HELP
    if text is not None and text != "/start":
        try:
            note_input = NoteInput.from_text(text)
        except ValueError:
            reply = "Could not accept that note. Send a shorter text note or a valid HTTP(S) link."
        else:
            note_id = await db.enqueue(
                str(sender_id), note_input, telegram_message_id=message["message_id"]
            )
            reply = f"Received note #{note_id}. Check its status in the web app."

    try:
        await telegram_request(
            client, "sendMessage", chat_id=message["chat"]["id"], text=reply
        )
    except TelegramError:
        # A failed reply must not block later notes after handling the update.
        logger.warning("Could not send Telegram reply")


async def poll(
    client: httpx.AsyncClient, db: Postgres, allowed_user_ids: set[int]
) -> None:
    offset = 0
    while True:
        try:
            updates = await telegram_request(
                client,
                "getUpdates",
                offset=offset,
                timeout=20,
                allowed_updates=["message"],
            )
            if not updates:
                # Telegram can assign a lower update ID after a week of inactivity.
                offset = 0
            for update in updates:
                await handle_update(client, db, allowed_user_ids, update)
                offset = update["update_id"] + 1
        except TelegramError:
            logger.warning("Could not poll Telegram; retrying")
            await sleep(2)
        except (
            asyncpg.PostgresConnectionError,
            asyncpg.AdminShutdownError,
            asyncpg.CrashShutdownError,
            asyncpg.CannotConnectNowError,
            asyncpg.TooManyConnectionsError,
            OSError,
        ) as error:
            logger.warning(
                "Could not save Telegram note (%s); retrying", type(error).__name__
            )
            await sleep(2)
        except asyncpg.PostgresError as error:
            # PostgreSQL error details can contain the submitted note.
            raise RuntimeError(
                f"Could not save Telegram note ({type(error).__name__})."
            ) from None


async def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN to start the bot.")
    try:
        allowed_user_ids = {
            int(value)
            for value in os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").split(",")
        }
        if min(allowed_user_ids) < 1:
            raise ValueError
    except ValueError:
        raise SystemExit(
            "Set TELEGRAM_ALLOWED_USER_IDS to comma-separated positive IDs."
        ) from None

    logging.getLogger("httpx").setLevel(logging.WARNING)
    async with (
        open_postgres() as db,
        httpx.AsyncClient(
            base_url=f"https://api.telegram.org/bot{token}/", timeout=30.0
        ) as client,
    ):
        await poll(client, db, allowed_user_ids)


if __name__ == "__main__":
    run(main())
