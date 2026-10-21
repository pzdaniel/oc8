"""handoff.idempotency_key: exact-once handoffs from an agent's emit_event

An agent emits through the department contract (control tool emit_event),
and a resumed run re-runs from the start -- so the same emission can arrive
twice, and two runs of the same task can emit at the same moment. A
read-then-insert check cannot stop the second one; a unique key can. The key
(source task, binding, mapped payload) is set only for agent emissions, so
every other handoff keeps NULL and is untouched by the partial index.

Revision ID: 0097
Revises: 0096
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0097"
down_revision: str | None = "0096"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("handoff", sa.Column("idempotency_key", sa.Text(), nullable=True))
    op.create_index(
        "uq_handoff_idempotency",
        "handoff",
        ["tenant_id", "idempotency_key"],
        unique=True,
        postgresql_where=sa.text("idempotency_key IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_handoff_idempotency", table_name="handoff")
    op.drop_column("handoff", "idempotency_key")
