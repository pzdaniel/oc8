"""Emit a business event -> resolve contract bindings -> create handoffs (§14a.3)."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import cast, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from oc8 import models as m
from oc8.audit import append_event
from oc8.collab.contracts import (
    ContractViolation,
    apply_payload_map,
    department_emits,
    department_intake,
)
from oc8.collab.handoff import create_handoff


async def emit_event(
    db: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    source_department: m.Department,
    event_type: str,
    payload: dict[str, Any],
    created_by: uuid.UUID,
    source_task_id: uuid.UUID | None = None,
) -> list[m.Handoff]:
    """Create one pending handoff per matching binding whose target takes it in.

    `source_task_id` is set when an AGENT emits (control tool `emit_event`):
    it is recorded on the handoff, and an identical handoff this same task
    already produced is returned instead of being created again. A run that is
    resumed re-runs from the start, so without this every resume of a run that
    emitted before parking would hand the same work over a second time.
    """
    if event_type not in department_emits(source_department.frame):
        await append_event(
            db,
            tenant_id=tenant_id,
            actor_type="agent",
            actor_id=created_by,
            category="contract",
            action=f"emit:{event_type}",
            resource={"department_id": str(source_department.id)},
            decision="deny",
            reason="event not declared in department emits",
        )
        raise ContractViolation(f"department does not emit {event_type}")

    bindings = (
        (
            await db.execute(
                select(m.ContractBinding).where(
                    m.ContractBinding.event_type == event_type,
                    m.ContractBinding.source_department_id == source_department.id,
                )
            )
        )
        .scalars()
        .all()
    )

    created: list[m.Handoff] = []
    for binding in bindings:
        target = await db.get(m.Department, binding.target_department_id)
        handoff_type = await db.get(m.HandoffType, binding.handoff_type_id)
        if target is None or handoff_type is None:
            continue
        if department_intake(target.frame, handoff_type.name) is None:
            await append_event(
                db,
                tenant_id=tenant_id,
                actor_type="agent",
                actor_id=created_by,
                category="contract",
                action=f"intake:{handoff_type.name}",
                resource={"target_department_id": str(target.id)},
                decision="deny",
                reason="target department does not intake this handoff type",
            )
            continue
        mapped = apply_payload_map(binding.payload_map, payload)
        if source_task_id is not None:
            existing = await db.scalar(
                select(m.Handoff).where(
                    m.Handoff.source_task_id == source_task_id,
                    m.Handoff.source_department_id == source_department.id,
                    m.Handoff.target_department_id == target.id,
                    m.Handoff.handoff_type_id == handoff_type.id,
                    m.Handoff.payload == cast(mapped, JSONB),
                )
            )
            if existing is not None:
                created.append(existing)
                continue
        handoff = await create_handoff(
            db,
            tenant_id=tenant_id,
            handoff_type=handoff_type,
            source_department_id=source_department.id,
            target_department_id=target.id,
            payload=mapped,
            created_by=created_by,
            source_task_id=source_task_id,
            gate=binding.gate,
        )
        created.append(handoff)
    return created
