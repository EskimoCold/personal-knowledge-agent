"""Store a bounded image and its media type with each photo note."""

from alembic import op

revision = "003_note_images"
down_revision = "002_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE notes
            DROP CONSTRAINT notes_kind_check,
            ADD COLUMN image_data bytea,
            ADD COLUMN image_media_type text,
            ADD CONSTRAINT notes_kind_check CHECK (kind IN ('text', 'url', 'image')),
            ADD CONSTRAINT notes_image_check CHECK (
                (kind = 'image') = (image_data IS NOT NULL)
                AND (image_data IS NULL) = (image_media_type IS NULL)
                AND octet_length(image_data) BETWEEN 1 AND 8388608
                AND image_media_type IN ('image/jpeg', 'image/png')
            )
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE notes
            DROP CONSTRAINT notes_kind_check,
            ADD CONSTRAINT notes_kind_check CHECK (kind IN ('text', 'url')),
            DROP CONSTRAINT notes_image_check,
            DROP COLUMN image_data,
            DROP COLUMN image_media_type
        """
    )
