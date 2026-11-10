"""Create notes and queue indexes."""

from alembic import op


revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The first Compose demo created these objects before Alembic was introduced.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            owner_id text NOT NULL,
            kind text NOT NULL CHECK (kind IN ('text', 'url')),
            value text NOT NULL,
            telegram_message_id bigint,
            status text NOT NULL DEFAULT 'queued'
                CHECK (status IN ('queued', 'processing', 'ready', 'failed')),
            claim_token uuid,
            lease_until timestamptz,
            text text,
            summary text,
            tags text[] NOT NULL DEFAULT '{}',
            error text,
            UNIQUE (owner_id, telegram_message_id)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS notes_owner_id_idx ON notes (owner_id, id DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS notes_queue_idx ON notes (id) "
        "WHERE status IN ('queued', 'processing')"
    )


def downgrade() -> None:
    op.drop_table("notes")
