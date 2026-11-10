"""Enable pgvector and store one embedding per note."""

from alembic import op


revision = "002_embeddings"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
    op.execute(
        """
        ALTER TABLE notes
            ADD COLUMN embedding public.vector,
            ADD COLUMN embedding_model text,
            ADD CONSTRAINT notes_embedding_pair_check
                CHECK ((embedding IS NULL) = (embedding_model IS NULL))
        """
    )


def downgrade() -> None:
    # Other schemas in this database can use the same extension.
    op.execute("ALTER TABLE notes DROP COLUMN embedding, DROP COLUMN embedding_model")
