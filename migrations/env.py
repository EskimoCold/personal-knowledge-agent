import asyncio
import os
from logging.config import fileConfig

import asyncpg
from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool


if context.config.config_file_name:
    fileConfig(context.config.config_file_name)


def run_migrations_offline() -> None:
    context.configure(url="postgresql+asyncpg://", literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        version_table_schema=connection.dialect.default_schema_name,
    )
    with context.begin_transaction():
        connection.exec_driver_sql(
            "SELECT pg_advisory_xact_lock(hashtext('personal-knowledge-agent/migrations'))"
        )
        context.run_migrations()


async def connect() -> asyncpg.Connection:
    # Use the same asyncpg DSN as the app, including any PostgreSQL session settings.
    return await asyncpg.connect(os.environ["DATABASE_URL"], timeout=5)


async def run_migrations_online() -> None:
    engine = create_async_engine(
        "postgresql+asyncpg://",
        async_creator=connect,
        poolclass=NullPool,
        isolation_level="READ COMMITTED",
    )
    try:
        async with engine.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
