"""An agent hands work to ANOTHER department through the department contracts.

Delegation stops at the department border on purpose (only the tenant
Assistant may cross it). Work that belongs to another department -- a buyer's
agent that found an article missing, say, and the department that creates
articles -- crosses through the contract instead: the source department
declares what it EMITS, the target what it TAKES IN, a binding connects the
two, and the result is a PENDING handoff someone in the target department
accepts. These tests pin that the tool can do exactly that and nothing more.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest
from sqlalchemy import select

from oc8 import models as m
from oc8.agent.control_tools import (
    EMIT_EVENT,
    MAX_EMIT_PAYLOAD_BYTES,
    TEAM_LEAD_ONLY_REASON,
    authorize_emit,
    execute_control_tool,
    offered_tools,
)
from oc8.agent.engine import _authorize
from oc8.agent.tool_idempotency import REPLAY_NOTE
from oc8.authz.pdp import Decision, Effect
from oc8.collab.emit import emit_event
from oc8.collab.handoff import create_handoff_type
from oc8.modelrouter import ToolCall
from tests.conftest import AppSessionFactory

pytestmark = pytest.mark.asyncio

EVENT = "einkauf.artikel.fehlt"
HANDOFF = "artikel.anlegen"


async def _setup(
    db: Any,
    tenant: uuid.UUID,
    *,
    target_intakes: bool = True,
    schema: dict[str, Any] | None = None,
) -> tuple[m.Department, m.Department, m.Agent, m.Task]:
    """Einkauf (emits EVENT) -> Stammdaten (takes HANDOFF in), one binding."""
    einkauf = m.Department(tenant_id=tenant, name="Einkauf", frame={"emits": [EVENT]})
    stamm = m.Department(
        tenant_id=tenant,
        name="Stammdaten",
        frame=(
            {"intakes": [{"handoff_type": HANDOFF, "route": "Team Lead", "gate": "auto"}]}
            if target_intakes
            else {}
        ),
    )
    db.add_all([einkauf, stamm])
    await db.flush()
    ht = await create_handoff_type(
        db,
        tenant_id=tenant,
        name=HANDOFF,
        payload_schema=schema or {"type": "object", "required": ["artikel"]},
    )
    db.add(
        m.ContractBinding(
            tenant_id=tenant,
            event_type=EVENT,
            handoff_type_id=ht.id,
            source_department_id=einkauf.id,
            target_department_id=stamm.id,
            payload_map={"artikel": "$.artikel", "lieferant": "$.lieferant"},
        )
    )
    agent = m.Agent(tenant_id=tenant, department_id=einkauf.id, name="Angebote", is_team_lead=True)
    db.add(agent)
    await db.flush()
    task = m.Task(
        tenant_id=tenant,
        department_id=einkauf.id,
        assigned_agent_id=agent.id,
        title="Angebot Lieferant X",
        state="in_progress",
    )
    db.add(task)
    await db.flush()
    return einkauf, stamm, agent, task


def _call(**arguments: Any) -> ToolCall:
    return ToolCall(id=str(uuid.uuid4()), name=EMIT_EVENT.name, arguments=arguments)


ARGS: dict[str, Any] = {
    "event_type": EVENT,
    "payload": {"artikel": "Rüttelplatte BP 25", "lieferant": "X GmbH", "intern": "nicht weiter"},
}


async def _emit(
    db: Any,
    tenant: uuid.UUID,
    agent: m.Agent,
    task: m.Task,
    frame: dict[str, Any],
    args: dict[str, Any] | None = None,
) -> str:
    tc = _call(**(args if args is not None else ARGS))
    outcome = await execute_control_tool(
        db,
        tenant_id=tenant,
        agent=agent,
        task=task,
        tc=tc,
        decision=authorize_emit(frame, tc.arguments, is_team_lead=bool(agent.is_team_lead)),
        assigned_skills=[],
        active_skills=[],
        mcp_conn=None,
        originating_operator=None,
    )
    assert outcome is not None
    assert outcome.suspend is None  # it hands over and carries on, never parks
    assert outcome.pending_run is None  # no run is started anywhere
    return outcome.output


async def _handoffs(db: Any) -> list[m.Handoff]:
    return list((await db.execute(select(m.Handoff))).scalars().all())


async def test_the_event_becomes_a_pending_handoff_in_the_other_department(
    app_session: AppSessionFactory,
) -> None:
    tenant = uuid.uuid4()
    async with app_session(tenant) as db:
        einkauf, stamm, agent, task = await _setup(db, tenant)
        output = await _emit(db, tenant, agent, task, einkauf.frame)

        assert output.startswith("Übergeben an Stammdaten")
        [handoff] = await _handoffs(db)
        assert handoff.status == "pending"  # waits for the target to accept it
        assert handoff.target_department_id == stamm.id
        assert handoff.source_department_id == einkauf.id
        assert handoff.source_task_id == task.id
        assert handoff.created_by == agent.id
        # The contract's payload map decides what crosses, not the agent.
        assert handoff.payload == {"artikel": "Rüttelplatte BP 25", "lieferant": "X GmbH"}
        # Nothing was put on anybody's desk yet: that happens on acceptance.
        assert handoff.target_task_id is None
        audit = (
            await db.execute(
                select(m.AuditEvent).where(
                    m.AuditEvent.category == "contract", m.AuditEvent.action == f"emit:{EVENT}"
                )
            )
        ).scalar_one()
        assert audit.decision == "allow"
        assert audit.actor_id == agent.id
        assert audit.resource["handoff_ids"] == [str(handoff.id)]


async def test_an_event_outside_the_contract_is_refused(
    app_session: AppSessionFactory,
) -> None:
    tenant = uuid.uuid4()
    async with app_session(tenant) as db:
        einkauf, _stamm, agent, task = await _setup(db, tenant)
        args = {**ARGS, "event_type": "einkauf.bestellung.freigeben"}
        decision = authorize_emit(einkauf.frame, args, is_team_lead=True)
        assert decision.effect is Effect.DENY
        assert "not in your department's contract" in decision.reason
        output = await _emit(db, tenant, agent, task, einkauf.frame, args)
        assert output.startswith("ERROR:")
        assert await _handoffs(db) == []


async def test_a_department_without_emits_cannot_emit_at_all(
    app_session: AppSessionFactory,
) -> None:
    tenant = uuid.uuid4()
    async with app_session(tenant) as db:
        _einkauf, stamm, _agent, _task = await _setup(db, tenant)
        decision = authorize_emit(stamm.frame, ARGS, is_team_lead=True)
        assert decision.effect is Effect.DENY
        assert decision.reason == "your department declares no events it may emit"
        assert authorize_emit({}, ARGS, is_team_lead=True).effect is Effect.DENY


async def test_a_target_that_does_not_take_it_in_gets_nothing(
    app_session: AppSessionFactory,
) -> None:
    tenant = uuid.uuid4()
    async with app_session(tenant) as db:
        einkauf, _stamm, agent, task = await _setup(db, tenant, target_intakes=False)
        output = await _emit(db, tenant, agent, task, einkauf.frame)
        assert output.startswith("ERROR: no department takes")
        assert await _handoffs(db) == []
        denied = (
            await db.execute(select(m.AuditEvent).where(m.AuditEvent.action == f"intake:{HANDOFF}"))
        ).scalar_one()
        assert denied.decision == "deny"


async def test_another_department_declaring_the_same_event_does_not_use_this_binding(
    app_session: AppSessionFactory,
) -> None:
    """The binding belongs to its SOURCE department: an agent elsewhere that
    declares the same event name does not get to ride on it."""
    tenant = uuid.uuid4()
    async with app_session(tenant) as db:
        _einkauf, _stamm, _agent, _task = await _setup(db, tenant)
        vertrieb = m.Department(tenant_id=tenant, name="Vertrieb", frame={"emits": [EVENT]})
        db.add(vertrieb)
        await db.flush()
        other = m.Agent(
            tenant_id=tenant, department_id=vertrieb.id, name="Vertrieb-Agent", is_team_lead=True
        )
        db.add(other)
        await db.flush()
        other_task = m.Task(
            tenant_id=tenant,
            department_id=vertrieb.id,
            assigned_agent_id=other.id,
            title="x",
            state="in_progress",
        )
        db.add(other_task)
        await db.flush()
        output = await _emit(db, tenant, other, other_task, vertrieb.frame)
        assert output.startswith("ERROR: no department takes")
        assert await _handoffs(db) == []


async def test_the_frame_is_checked_again_against_the_department_as_it_is_now(
    app_session: AppSessionFactory,
) -> None:
    """A long run authorised against the frame it read at the start must not
    keep emitting after the event was taken out of the contract."""
    tenant = uuid.uuid4()
    async with app_session(tenant) as db:
        einkauf, _stamm, agent, task = await _setup(db, tenant)
        stale_frame = dict(einkauf.frame)
        einkauf.frame = {"emits": []}
        await db.flush()
        output = await _emit(db, tenant, agent, task, stale_frame)
        assert output.startswith("ERROR:")
        assert await _handoffs(db) == []
        refused = (
            await db.execute(select(m.AuditEvent).where(m.AuditEvent.action == f"emit:{EVENT}"))
        ).scalar_one()
        assert refused.decision == "deny"


async def test_a_too_large_payload_is_refused(app_session: AppSessionFactory) -> None:
    tenant = uuid.uuid4()
    async with app_session(tenant) as db:
        einkauf, _stamm, agent, task = await _setup(db, tenant)
        args = {"event_type": EVENT, "payload": {"artikel": "x" * (MAX_EMIT_PAYLOAD_BYTES + 1)}}
        decision = authorize_emit(einkauf.frame, args, is_team_lead=True)
        assert decision.effect is Effect.DENY
        assert "payload too large" in decision.reason
        output = await _emit(db, tenant, agent, task, einkauf.frame, args)
        assert output.startswith("ERROR: payload too large")
        assert await _handoffs(db) == []


async def test_a_payload_the_target_contract_rejects_leaves_no_handoff_behind(
    app_session: AppSessionFactory,
) -> None:
    tenant = uuid.uuid4()
    async with app_session(tenant) as db:
        einkauf, _stamm, agent, task = await _setup(
            db, tenant, schema={"type": "object", "required": ["artikel", "menge"]}
        )
        output = await _emit(db, tenant, agent, task, einkauf.frame)
        assert output.startswith("ERROR: the receiving department's contract rejects")
        assert await _handoffs(db) == []


async def test_emitting_the_same_thing_twice_from_one_task_hands_it_over_once(
    app_session: AppSessionFactory,
) -> None:
    """A resumed run re-runs from the start; the work must not arrive twice."""
    tenant = uuid.uuid4()
    async with app_session(tenant) as db:
        einkauf, _stamm, agent, task = await _setup(db, tenant)
        first = await _emit(db, tenant, agent, task, einkauf.frame)
        second = await _emit(db, tenant, agent, task, einkauf.frame)
        [handoff] = await _handoffs(db)
        assert str(handoff.id) in first
        assert str(handoff.id) in second
        # A different article from the same task is a different handoff.
        await _emit(
            db,
            tenant,
            agent,
            task,
            einkauf.frame,
            {"event_type": EVENT, "payload": {"artikel": "Stampfer"}},
        )
        assert len(await _handoffs(db)) == 2


async def test_offered_only_with_emits_and_with_exactly_those_events() -> None:
    agent = m.Agent(tenant_id=uuid.uuid4(), department_id=uuid.uuid4(), name="a", is_team_lead=True)
    without = offered_tools(agent, assigned_skills=[], active_skills=[], mcp_tools=[])
    assert EMIT_EVENT.name not in {t.name for t in without}
    offered = offered_tools(
        agent, assigned_skills=[], active_skills=[], mcp_tools=[], emits=[EVENT]
    )
    [tool] = [t for t in offered if t.name == EMIT_EVENT.name]
    assert tool.parameters["properties"]["event_type"]["enum"] == [EVENT]


async def test_the_engine_pep_decides_it_against_the_contract_not_a_connection() -> None:
    """Without its own branch the call would fall through to the connection
    check and be DENIED (and audited as such) for want of a connection."""
    agent = m.Agent(tenant_id=uuid.uuid4(), department_id=uuid.uuid4(), name="a", is_team_lead=True)
    common: dict[str, Any] = {
        "tool_policies": {},
        "connection_key": None,
        "tool_scopes": None,
    }
    allowed = _authorize(agent, _call(**ARGS), frame={"emits": [EVENT]}, **common)
    assert allowed == Decision(Effect.ALLOW)
    denied = _authorize(agent, _call(**ARGS), frame={"emits": ["anderes"]}, **common)
    assert denied.effect is Effect.DENY


async def test_two_different_events_to_the_same_place_are_two_handoffs(
    app_session: AppSessionFactory,
) -> None:
    """Same task, same target, same handoff type, same mapped payload -- but two
    different events along two bindings. That is two routes, not a replay."""
    tenant = uuid.uuid4()
    async with app_session(tenant) as db:
        einkauf, stamm, agent, task = await _setup(db, tenant)
        other_event = "einkauf.artikel.veraltet"
        einkauf.frame = {"emits": [EVENT, other_event]}
        ht = (
            await db.execute(select(m.HandoffType).where(m.HandoffType.name == HANDOFF))
        ).scalar_one()
        db.add(
            m.ContractBinding(
                tenant_id=tenant,
                event_type=other_event,
                handoff_type_id=ht.id,
                source_department_id=einkauf.id,
                target_department_id=stamm.id,
                payload_map={"artikel": "$.artikel", "lieferant": "$.lieferant"},
            )
        )
        await db.flush()
        await _emit(db, tenant, agent, task, einkauf.frame)
        await _emit(db, tenant, agent, task, einkauf.frame, {**ARGS, "event_type": other_event})
        handoffs = await _handoffs(db)
        assert len(handoffs) == 2
        assert handoffs[0].payload == handoffs[1].payload
        assert handoffs[0].idempotency_key != handoffs[1].idempotency_key


async def test_two_runs_emitting_at_the_same_moment_create_one_handoff(
    app_session: AppSessionFactory,
) -> None:
    """A read-then-insert check alone lets both through; the unique index does
    not. The second transaction waits on the first, then gets its row back."""
    tenant = uuid.uuid4()
    async with app_session(tenant) as db:
        einkauf, _stamm, agent, task = await _setup(db, tenant)
        dept_id, agent_id, task_id = einkauf.id, agent.id, task.id

    entered = asyncio.Event()

    async def one_run(hold: float) -> uuid.UUID:
        async with app_session(tenant) as db:
            dept = await db.get(m.Department, dept_id)
            assert dept is not None
            [h] = await emit_event(
                db,
                tenant_id=tenant,
                source_department=dept,
                event_type=EVENT,
                payload=ARGS["payload"],
                created_by=agent_id,
                source_task_id=task_id,
            )
            entered.set()
            await asyncio.sleep(hold)  # keep the first transaction open
            return h.id

    async def second_run() -> uuid.UUID:
        await entered.wait()  # start while the first is inserted but uncommitted
        return await one_run(0)

    first, second = await asyncio.gather(one_run(0.5), second_run())
    assert first == second
    async with app_session(tenant) as db:
        assert len(await _handoffs(db)) == 1


async def test_an_event_revoked_while_the_run_is_going_is_refused(
    app_session: AppSessionFactory,
) -> None:
    """The run's session loaded the department at its start (engine.py does);
    an operator then takes the event out of the contract in another
    transaction. The re-check must read the row as it is NOW, not the copy the
    session's identity map still holds."""
    tenant = uuid.uuid4()
    async with app_session(tenant) as db:
        einkauf, _stamm, agent, task = await _setup(db, tenant)
        dept_id = einkauf.id

    async with app_session(tenant) as run_db:
        dept = await run_db.get(m.Department, dept_id)  # as the engine does
        assert dept is not None
        frame_at_start = dict(dept.frame)
        run_agent = await run_db.get(m.Agent, agent.id)
        run_task = await run_db.get(m.Task, task.id)
        assert run_agent is not None and run_task is not None

        async with app_session(tenant) as operator_db:
            changed = await operator_db.get(m.Department, dept_id)
            assert changed is not None
            changed.frame = {"emits": []}

        output = await _emit(run_db, tenant, run_agent, run_task, frame_at_start)
        assert output.startswith("ERROR:"), output
        assert await _handoffs(run_db) == []


async def test_a_replay_says_so_and_is_not_recorded_as_a_second_handover(
    app_session: AppSessionFactory,
) -> None:
    """Same convention as tool_idempotency: the model is told explicitly that
    NO second action was taken, and audit/activity count the handover once."""
    tenant = uuid.uuid4()
    async with app_session(tenant) as db:
        einkauf, _stamm, agent, task = await _setup(db, tenant)
        first = await _emit(db, tenant, agent, task, einkauf.frame)
        second = await _emit(db, tenant, agent, task, einkauf.frame)

        assert REPLAY_NOTE not in first
        assert first.startswith("Übergeben an")
        assert REPLAY_NOTE in second
        assert not second.startswith("Übergeben an")
        allows = (
            (
                await db.execute(
                    select(m.AuditEvent).where(
                        m.AuditEvent.action == f"emit:{EVENT}", m.AuditEvent.decision == "allow"
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(allows) == 1
        activities = (
            (
                await db.execute(
                    select(m.ActivityEvent).where(m.ActivityEvent.message.like("Übergabe an%"))
                )
            )
            .scalars()
            .all()
        )
        assert len(activities) == 1


async def test_only_the_team_lead_is_offered_the_tool() -> None:
    """A member reports to its lead; the lead decides whether the work goes to
    another department. Offering it to a member would only invite calls that
    are always denied."""
    member = m.Agent(tenant_id=uuid.uuid4(), department_id=uuid.uuid4(), name="b")
    offered = offered_tools(
        member, assigned_skills=[], active_skills=[], mcp_tools=[], emits=[EVENT]
    )
    assert EMIT_EVENT.name not in {t.name for t in offered}


async def test_a_member_is_refused_by_the_engine_pep_even_with_emits() -> None:
    member = m.Agent(tenant_id=uuid.uuid4(), department_id=uuid.uuid4(), name="b")
    decision = _authorize(
        member,
        _call(**ARGS),
        frame={"emits": [EVENT]},
        tool_policies={},
        connection_key=None,
        tool_scopes=None,
    )
    assert decision.effect is Effect.DENY
    assert decision.reason == TEAM_LEAD_ONLY_REASON
    assert authorize_emit({"emits": [EVENT]}, ARGS, is_team_lead=False) == decision


async def test_a_lead_role_taken_away_during_the_run_is_refused(
    app_session: AppSessionFactory,
) -> None:
    """The PEP saw a lead at the start of the run; an operator then takes the
    lead role away. The dispatch reads the flag as it is NOW, hands nothing
    over and leaves the refusal in the audit trail."""
    tenant = uuid.uuid4()
    async with app_session(tenant) as db:
        einkauf, _stamm, agent, task = await _setup(db, tenant)
        dept_id = einkauf.id

    async with app_session(tenant) as run_db:
        dept = await run_db.get(m.Department, dept_id)
        assert dept is not None
        frame_at_start = dict(dept.frame)
        run_agent = await run_db.get(m.Agent, agent.id)
        run_task = await run_db.get(m.Task, task.id)
        assert run_agent is not None and run_task is not None
        assert run_agent.is_team_lead

        async with app_session(tenant) as operator_db:
            changed = await operator_db.get(m.Agent, agent.id)
            assert changed is not None
            changed.is_team_lead = False

        tc = _call(**ARGS)
        outcome = await execute_control_tool(
            run_db,
            tenant_id=tenant,
            agent=run_agent,
            task=run_task,
            tc=tc,
            decision=authorize_emit(frame_at_start, tc.arguments, is_team_lead=True),
            assigned_skills=[],
            active_skills=[],
            mcp_conn=None,
            originating_operator=None,
        )
        assert outcome is not None
        assert outcome.output == f"ERROR: {TEAM_LEAD_ONLY_REASON}"
        assert await _handoffs(run_db) == []
        denied = (
            await run_db.execute(select(m.AuditEvent).where(m.AuditEvent.action == f"emit:{EVENT}"))
        ).scalar_one()
        assert denied.decision == "deny"
        assert denied.reason == TEAM_LEAD_ONLY_REASON


async def test_an_archived_lead_is_refused_even_though_the_flag_is_still_set(
    app_session: AppSessionFactory,
) -> None:
    """delete_agent soft-deletes an agent that still has runs and leaves
    is_team_lead untouched; the fresh check must treat an archived lead as no
    lead at all, so archiving stops the handover at once."""
    import datetime as dt

    tenant = uuid.uuid4()
    async with app_session(tenant) as db:
        einkauf, _stamm, agent, task = await _setup(db, tenant)
        dept_id = einkauf.id

    async with app_session(tenant) as run_db:
        dept = await run_db.get(m.Department, dept_id)
        assert dept is not None
        frame_at_start = dict(dept.frame)
        run_agent = await run_db.get(m.Agent, agent.id)
        run_task = await run_db.get(m.Task, task.id)
        assert run_agent is not None and run_task is not None

        async with app_session(tenant) as operator_db:
            archived = await operator_db.get(m.Agent, agent.id)
            assert archived is not None
            archived.deleted_at = dt.datetime.now(dt.UTC)
            assert archived.is_team_lead

        tc = _call(**ARGS)
        outcome = await execute_control_tool(
            run_db,
            tenant_id=tenant,
            agent=run_agent,
            task=run_task,
            tc=tc,
            decision=authorize_emit(frame_at_start, tc.arguments, is_team_lead=True),
            assigned_skills=[],
            active_skills=[],
            mcp_conn=None,
            originating_operator=None,
        )
        assert outcome is not None
        assert outcome.output == f"ERROR: {TEAM_LEAD_ONLY_REASON}"
        assert await _handoffs(run_db) == []
