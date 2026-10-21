"""Handoff lifecycle: payload validation + governed state transitions (§14a.2)."""

from __future__ import annotations

import uuid
from typing import Any

import jsonschema
from sqlalchemy.ext.asyncio import AsyncSession

from oc8.hooks.bus import dispatch_action
from oc8.hooks.executor import InProcessExecutor
from oc8.hooks.types import HookCtx
from oc8.models.collab import Handoff, HandoffType

_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"accepted", "rejected", "expired"}),
    "accepted": frozenset({"in_progress", "completed", "rejected"}),
    "in_progress": frozenset({"completed", "rejected"}),
    "completed": frozenset(),
    "rejected": frozenset(),
    "expired": frozenset(),
}


class HandoffError(ValueError):
    """A handoff operation failed."""


class PayloadInvalid(HandoffError):
    """The payload does not satisfy the handoff type's schema."""


class HandoffStateError(HandoffError):
    """An illegal handoff state transition was attempted."""


def _transition(handoff: Handoff, dst: str) -> None:
    if dst not in _TRANSITIONS[handoff.status]:
        raise HandoffStateError(f"{handoff.status} -> {dst} is not allowed")
    handoff.status = dst


async def _emit_status_changed(handoff: Handoff) -> None:
    await dispatch_action(
        HookCtx(tenant_id=handoff.tenant_id),
        "handoff.status.changed",
        executor_default=InProcessExecutor(),
        handoff_id=str(handoff.id),
        status=handoff.status,
    )
    from oc8.realtime.bus import get_event_bus

    await get_event_bus().publish_event(
        handoff.tenant_id,
        "handoff.status",
        {"handoff_id": str(handoff.id), "status": handoff.status},
        source=f"oc8/handoff/{handoff.id}",
    )


async def create_handoff_type(
    db: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    name: str,
    payload_schema: dict[str, Any],
    classification: str = "internal",
) -> HandoffType:
    if "type" not in payload_schema:
        raise HandoffError("payload_schema must be a JSON Schema (missing 'type')")
    ht = HandoffType(
        tenant_id=tenant_id,
        name=name,
        payload_schema=payload_schema,
        classification=classification,
    )
    db.add(ht)
    await db.flush()
    return ht


async def create_handoff(
    db: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    handoff_type: HandoffType,
    source_department_id: uuid.UUID,
    target_department_id: uuid.UUID,
    payload: dict[str, Any],
    created_by: uuid.UUID,
    source_task_id: uuid.UUID | None = None,
    gate: str = "auto",
    flow_run_id: uuid.UUID | None = None,
    idempotency_key: str | None = None,
) -> Handoff:
    try:
        jsonschema.validate(payload, handoff_type.payload_schema)
    except jsonschema.ValidationError as exc:
        raise PayloadInvalid(exc.message) from exc
    handoff = Handoff(
        tenant_id=tenant_id,
        handoff_type_id=handoff_type.id,
        source_department_id=source_department_id,
        target_department_id=target_department_id,
        source_task_id=source_task_id,
        flow_run_id=flow_run_id,
        payload=payload,
        gate=gate,
        created_by=created_by,
        status="pending",
        idempotency_key=idempotency_key,
    )
    db.add(handoff)
    await db.flush()
    return handoff


async def accept_handoff(
    db: AsyncSession, handoff: Handoff, *, target_task_id: uuid.UUID | None = None
) -> Handoff:
    _transition(handoff, "accepted")
    if target_task_id is not None:
        handoff.target_task_id = target_task_id
    await db.flush()
    await _emit_status_changed(handoff)
    return handoff


async def reject_handoff(db: AsyncSession, handoff: Handoff) -> Handoff:
    _transition(handoff, "rejected")
    await db.flush()
    await _emit_status_changed(handoff)
    return handoff


async def complete_handoff(db: AsyncSession, handoff: Handoff) -> Handoff:
    _transition(handoff, "completed")
    await db.flush()
    await _emit_status_changed(handoff)
    return handoff
