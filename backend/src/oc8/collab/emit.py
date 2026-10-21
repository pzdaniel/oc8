"""Emit a business event -> resolve contract bindings -> create handoffs (§14a.3)."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
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
    it is recorded on the handoff, and the same emission along the same
    binding from this task is returned instead of being created again
    (`idempotency_key`, enforced by a unique index). A run that is
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
                select(m.ContractBinding)
                .where(
                    m.ContractBinding.event_type == event_type,
                    m.ContractBinding.source_department_id == source_department.id,
                )
                # Rows the session already holds would otherwise come back as
                # they were when first loaded; the contract is read as it is NOW.
                .execution_options(populate_existing=True)
            )
        )
        .scalars()
        .all()
    )

    created: list[m.Handoff] = []
    for binding in bindings:
        target = await db.get(m.Department, binding.target_department_id, populate_existing=True)
        handoff_type = await db.get(m.HandoffType, binding.handoff_type_id, populate_existing=True)
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
        if source_task_id is None:
            handoff = await create_handoff(
                db,
                tenant_id=tenant_id,
                handoff_type=handoff_type,
                source_department_id=source_department.id,
                target_department_id=target.id,
                payload=mapped,
                created_by=created_by,
                gate=binding.gate,
            )
        else:
            handoff = await _create_once(
                db,
                tenant_id=tenant_id,
                key=idempotency_key(
                    source_task_id=source_task_id,
                    binding_id=binding.id,
                    event_type=event_type,
                    mapped_payload=mapped,
                ),
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


def idempotency_key(
    *,
    source_task_id: uuid.UUID,
    binding_id: uuid.UUID,
    event_type: str,
    mapped_payload: dict[str, Any],
) -> str:
    """Identity of ONE emission along ONE contract route.

    The binding is part of it (and, for readability, the event it is bound
    to): two different events of the same task that happen to map to the same
    target, handoff type and payload are two routes, not a replay, and must
    both arrive. The payload is the MAPPED one -- what the receiver actually
    gets -- canonicalised so key order cannot make one emission look like two.
    """
    digest = hashlib.sha256(
        json.dumps(mapped_payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    return f"emit:{source_task_id}:{binding_id}:{event_type}:{digest}"


async def _by_key(db: AsyncSession, key: str) -> m.Handoff | None:
    found: m.Handoff | None = await db.scalar(
        select(m.Handoff).where(m.Handoff.idempotency_key == key)
    )
    return found


async def _create_once(
    db: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    key: str,
    handoff_type: m.HandoffType,
    source_department_id: uuid.UUID,
    target_department_id: uuid.UUID,
    payload: dict[str, Any],
    created_by: uuid.UUID,
    source_task_id: uuid.UUID,
    gate: str,
) -> m.Handoff:
    """Create the handoff for `key`, or return the one that already exists.

    The lookup first is only the cheap path for a replay; what makes it
    exact-once is `uq_handoff_idempotency`. Two runs racing past the lookup
    both insert, the second blocks on the index until the first commits, gets
    the unique violation, and -- its savepoint rolled back, nothing else of its
    transaction lost -- returns the first one's row instead.
    """
    existing = await _by_key(db, key)
    if existing is not None:
        return existing
    try:
        async with db.begin_nested():
            return await create_handoff(
                db,
                tenant_id=tenant_id,
                handoff_type=handoff_type,
                source_department_id=source_department_id,
                target_department_id=target_department_id,
                payload=payload,
                created_by=created_by,
                source_task_id=source_task_id,
                gate=gate,
                idempotency_key=key,
            )
    except IntegrityError:
        winner = await _by_key(db, key)
        if winner is None:
            raise
        return winner
