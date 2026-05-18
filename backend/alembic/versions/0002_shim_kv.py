"""V.4 Wave 4: shim_kv table backing the PersistenceBackend Protocol.

Revision ID: 0002_shim_kv
Revises: 0001_initial_schema
Create Date: 2026-05-18

The PostgresBackend in ``services/persistence.py`` stores arbitrary JSONB
rows keyed by (table_name, key). This lets every existing service swap
from the file shim to Postgres without rewriting any read/write call,
while the typed ORM schema from 0001 stays available for services that
migrate explicitly (a follow-up after Wave 4).

Why a separate table rather than reusing 0001:
  * 0001 doesn't yet carry V.4 columns (campaign_year, advertiser_size,
    sample_1_* / sample_2_* per-arm counts, holdout_distributions, cost
    on feature_store rows). Aligning 0001 with V.4 means rewriting every
    service to use ORM models — out of scope for Wave 4.
  * The shim_kv approach is one table + 30 lines of Python and gets us
    real persistence today; the typed migration is the right long-term
    direction but is a Wave 4B / 4C refactor.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0002_shim_kv"
down_revision: Union[str, None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "shim_kv",
        sa.Column("table_name", sa.String(length=128), nullable=False),
        sa.Column("key", sa.String(length=256), nullable=False),
        sa.Column("row", JSONB, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("table_name", "key", name="pk_shim_kv"),
    )
    # Most queries scope to a single table_name; ordered by key for read_table.
    op.create_index("ix_shim_kv_table_name", "shim_kv", ["table_name"])


def downgrade() -> None:
    op.drop_index("ix_shim_kv_table_name", table_name="shim_kv")
    op.drop_table("shim_kv")
