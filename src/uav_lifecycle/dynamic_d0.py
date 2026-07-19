"""Execute and aggregate the frozen deterministic D0 witness contracts.

The scenario module remains the immutable fixture source.  This module is the
single executable adapter between those contracts and the canonical simulator.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from enum import Enum
from fractions import Fraction
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from .artifacts import sha256_file, write_json_atomic
from .dynamic_planning import PlannedPath, build_planning_problem
from .dynamic_policies import make_policy
from .dynamic_scenarios import (
    AbsenceKind,
    D0Action,
    D0Fault,
    D0Run,
    ExpectedPrivateRecord,
    ExpectedPublicRecord,
    PrivateRecordKind,
    PublicRecordKind,
    RejectionReason,
    d0_contract_digest,
    d0_scenarios,
)
from .dynamic_simulator import (
    advance_public_belief,
    commit_batch,
    complete_event_batch,
    initialize_state,
    run_episode,
)
from .dynamic_types import (
    DynamicConfig,
    PlanningClock,
    PublicActionAck,
    PublicObservation,
    PublicSnapshot,
    quantize_tick,
)
from .mode_allocation import Mode as AllocationMode, evaluate_mode_path
from .mode_cbba import screen_modes


class _TickScriptedPolicy:
    """Canonical fixture adapter whose only runtime input is public state."""

    def __init__(self, run: D0Run) -> None:
        self.decisions: dict[int, list[tuple[int, int, str]]] = {}
        self.planning_clock = PlanningClock.PERIODIC
        self.calls: list[tuple[PublicSnapshot, tuple[tuple[int, int, str], ...], int]] = []
        self.future_decision_ticks = tuple(
            sorted({step.commit_tick for step in run.script if step.committed})
        )
        self.force_stall = run.expected_gate is not None
        for step in run.script:
            if step.committed:
                assert step.agent_id is not None and step.target_id is not None
                assert step.action is not None
                self.decisions.setdefault(step.commit_tick, []).append(
                    (step.agent_id, step.target_id, step.action.value)
                )

    def decide(self, snapshot: PublicSnapshot) -> tuple[tuple[int, int, str], ...]:
        if not isinstance(snapshot, PublicSnapshot):
            raise TypeError("D0 policy requires PublicSnapshot")
        decisions = tuple(self.decisions.get(snapshot.tick, ()))
        self.calls.append((snapshot, decisions, self.positive_pair_count(snapshot)))
        return decisions

    def positive_pair_count(self, snapshot: PublicSnapshot) -> int:
        if self.force_stall and snapshot.tick == 0:
            return 1
        return max(
            len(self.decisions.get(snapshot.tick, ())),
            int(any(tick > snapshot.tick for tick in self.future_decision_ticks)),
        )


def _positive_task_targets(snapshot: PublicSnapshot, max_tick: int) -> list[int]:
    problem = build_planning_problem(snapshot, DynamicConfig(), max_tick)
    screened = screen_modes(problem.instance)
    positive: list[int] = []
    for local_target, task in enumerate(screened):
        if task is None:
            continue
        if any(
            evaluation.feasible and evaluation.score > 1e-12
            for agent_id in range(len(problem.instance.agents))
            for evaluation in (
                evaluate_mode_path(problem.instance, agent_id, (task.key,)),
            )
        ):
            positive.append(problem.global_target_ids[local_target])
    return positive


class _ObservedRealPolicy:
    """Transparent diagnostics around an approved production policy."""

    def __init__(self, method: str, max_tick: int) -> None:
        self.delegate = make_policy(method, DynamicConfig())
        self.method = self.delegate.method
        self.planning_clock = self.delegate.planning_clock
        self.max_tick = max_tick
        self.planning_calls = 0
        self.auto_next_calls = 0
        self.auto_next_commit_count = 0
        self.tick0_frozen_suffix_count = 0
        self.decisions: list[dict[str, Any]] = []

    def bind_horizon(self, max_tick: int) -> None:
        self.delegate.bind_horizon(max_tick)

    def decide(self, snapshot: PublicSnapshot) -> Any:
        decision = self.delegate.decide(snapshot)
        self.planning_calls += 1
        proposals = list(getattr(decision, "proposals", ()))
        self.decisions.append({
            "tick": snapshot.tick,
            "policy": type(self.delegate).__name__,
            "positive_task_targets": _positive_task_targets(snapshot, self.max_tick),
            "committed_targets": [proposal.target_id for proposal in proposals],
            "planned_paths": [
                {
                    "agent_id": path.agent_id,
                    "tasks": [
                        [target_id, str(getattr(mode, "value", mode))]
                        for target_id, mode in path.tasks
                    ],
                }
                for path in decision.planned_paths
            ],
            "planning_sha256": sha256(decision.planning_bytes).hexdigest(),
        })
        if snapshot.tick == 0:
            self.tick0_frozen_suffix_count = sum(
                bool(suffix) for _, suffix in decision.pending_suffixes
            )
        return decision

    def auto_next(
        self, snapshot: PublicSnapshot, completed_agent_ids: tuple[int, ...]
    ) -> Any:
        self.auto_next_calls += 1
        decision = self.delegate.auto_next(snapshot, completed_agent_ids)
        self.auto_next_commit_count += int(bool(decision.proposals))
        return decision

    def positive_pair_count(self, snapshot: PublicSnapshot) -> int:
        return self.delegate.positive_pair_count(snapshot)

    def has_pending_suffix(self) -> bool:
        probe = getattr(self.delegate, "has_pending_suffix", None)
        return bool(probe()) if probe is not None else False


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical(asdict(value))
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("D0 canonical records forbid NaN and infinity")
    return value


def canonical_d0_bytes(value: Any) -> bytes:
    """Return strict, stable JSON bytes for replay and artifact verification."""

    return json.dumps(
        _canonical(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(canonical_d0_bytes(value)).hexdigest()


def _source_step_index(
    run: D0Run, *, tick: int, target_id: int, agent_id: int, mode: str
) -> int:
    matches = tuple(
        index
        for index, step in enumerate(run.script)
        if step.committed
        and step.finish_tick == tick
        and step.target_id == target_id
        and step.agent_id == agent_id
        and step.action is not None
        and step.action.value == mode
    )
    if len(matches) != 1:
        raise AssertionError("completion must map to exactly one canonical step")
    return matches[0]


def _logical_public_trace(run: D0Run, result: Any, policy: Any) -> tuple[ExpectedPublicRecord, ...]:
    beliefs = [target.belief for target in run.scenario.targets]
    records: list[ExpectedPublicRecord] = []
    for event in result.public_events:
        source = _source_step_index(
            run, tick=event.tick, target_id=event.target_id,
            agent_id=event.agent_id, mode=event.mode,
        )
        observation = event.observation if isinstance(event, PublicObservation) else None
        posterior, gate = advance_public_belief(
            beliefs[event.target_id], event.mode, observation, DynamicConfig(), tick=event.tick
        )
        if gate is not None:
            raise AssertionError(f"unexpected belief Gate: {gate}")
        beliefs[event.target_id] = posterior
        records.append(ExpectedPublicRecord(
            PublicRecordKind.ATTACK_ACK if isinstance(event, PublicActionAck) else PublicRecordKind.OBSERVATION,
            0, event.tick, event.target_id, event.agent_id, D0Action(event.mode), observation,
            source, posterior_belief=posterior,
        ))
    for expected in (item for item in run.expected_public_trace if item.kind is PublicRecordKind.SCHEDULER):
        calls = tuple(call for call in policy.calls if call[0].tick == expected.tick)
        if len(calls) != 1:
            raise AssertionError("scheduler witness must correspond to one planning call")
        snapshot, decisions, positive = calls[0]
        records.append(ExpectedPublicRecord(
            PublicRecordKind.SCHEDULER, 0, snapshot.tick, 0, 0, None, None,
            expected.source_step_index,
            sum(agent.busy_action is not None for agent in snapshot.agents),
            positive, len(decisions),
        ))
    if any(item.kind is PublicRecordKind.TERMINATION for item in run.expected_public_trace):
        terminal_tick = (
            run.scenario.t_max_tick
            if result.record.termination == "horizon"
            else quantize_tick(result.record.makespan, DynamicConfig().tick_size)
        )
        records.append(ExpectedPublicRecord(
            PublicRecordKind.TERMINATION, 0, terminal_tick,
            0, 0, None, None, None, 0, 0, 0,
        ))
    rank = {
        PublicRecordKind.OBSERVATION: 0, PublicRecordKind.ATTACK_ACK: 0,
        PublicRecordKind.SCHEDULER: 1, PublicRecordKind.TERMINATION: 2,
    }
    return tuple(
        replace(record, event_id=index)
        for index, record in enumerate(sorted(records, key=lambda item: (item.tick, rank[item.kind])))
    )


def _state_before_step(run: D0Run, step_index: int) -> Any:
    state = initialize_state(run.scenario)
    config = DynamicConfig()
    for step in run.script[:step_index]:
        if not step.committed:
            continue
        due = tuple(event for event in state.completion_events if event.finish_tick <= step.commit_tick)
        for tick in sorted({event.finish_tick for event in due}):
            batch = tuple(event for event in state.completion_events if event.finish_tick == tick)
            state = complete_event_batch(state, batch, config).state
        assert step.agent_id is not None and step.target_id is not None and step.action is not None
        outcome = commit_batch(
            state, ((step.agent_id, step.target_id, step.action.value),), config,
            run.scenario.t_max_tick, tick=step.commit_tick,
        )
        if outcome.gates:
            raise AssertionError(f"canonical prefix produced Gate: {outcome.gates}")
        state = outcome.state
    return state


def _logical_rejection(run: D0Run, expected: ExpectedPrivateRecord) -> ExpectedPrivateRecord:
    if expected.source_step_index is None:
        raise AssertionError("rejection requires a source step")
    step = run.script[expected.source_step_index]
    state = _state_before_step(run, expected.source_step_index)
    config = DynamicConfig()
    direct_faults = {
        D0Fault.REJECT_LOCKED, D0Fault.REJECT_HORIZON,
        D0Fault.REJECT_RANGE, D0Fault.REJECT_AMMO,
    }
    if step.fault is D0Fault.EXCLUDE_BUSY:
        unlocked = next(target.target_id for target in state.targets if target.target_id not in dict(state.target_locks))
        assert step.agent_id is not None
        candidates = ((step.agent_id, unlocked, "attack"),)
    elif step.fault in direct_faults:
        assert step.agent_id is not None and step.target_id is not None and step.action is not None
        candidates = ((step.agent_id, step.target_id, step.action.value),)
    else:
        raise AssertionError("rejection witness is not backed by commit_batch")
    outcome = commit_batch(
        state, candidates, config, run.scenario.t_max_tick, tick=step.commit_tick
    )
    if outcome.state != state or outcome.committed:
        raise AssertionError("rejection must be atomic")
    if len(outcome.gates) != 1:
        raise AssertionError("rejection witness requires exactly one simulator Gate")
    gate_reason = outcome.gates[0].reason
    by_gate = {
        "horizon": RejectionReason.HORIZON, "range": RejectionReason.RANGE,
        "ammo": RejectionReason.AMMO, "target_locked": RejectionReason.TARGET_LOCKED,
        "busy_agent": RejectionReason.BUSY_AGENT,
    }
    reason = by_gate[gate_reason]
    target_id = 0 if step.target_id is None else step.target_id
    agent_id = 0 if step.agent_id is None else step.agent_id
    truth = state.private_targets[target_id]
    return ExpectedPrivateRecord(
        PrivateRecordKind.REJECTION, 0, step.commit_tick, target_id, agent_id,
        step.action, truth.true_category, truth.true_damage, truth.true_damage,
        None, None, 0.0, expected.source_step_index, reason,
    )


def _logical_private_trace(run: D0Run, result: Any) -> tuple[ExpectedPrivateRecord, ...]:
    records: list[ExpectedPrivateRecord] = []
    for event in result.private_audit_events:
        source = _source_step_index(
            run, tick=event.tick, target_id=event.target_id,
            agent_id=event.agent_id, mode=event.mode,
        )
        records.append(ExpectedPrivateRecord(
            PrivateRecordKind.COMPLETION, 0, event.tick, event.target_id,
            event.agent_id, D0Action(event.mode), event.true_category,
            event.damage_before, event.damage_after, event.draw,
            event.physical_success if event.mode == "attack" else None,
            event.realized_reward, source, None, event.realized_reward > 0.0,
        ))
    for expected in run.expected_private_audit_trace:
        if expected.kind is PrivateRecordKind.REJECTION:
            assert expected.source_step_index is not None
            fault = run.script[expected.source_step_index].fault
            if fault not in {
                D0Fault.REJECT_LOCKED,
                D0Fault.REJECT_HORIZON,
                D0Fault.REJECT_RANGE,
                D0Fault.REJECT_AMMO,
                D0Fault.EXCLUDE_BUSY,
            }:
                raise AssertionError("actual rejection must originate in commit_batch")
            records.append(_logical_rejection(run, expected))
    if any(item.kind is PrivateRecordKind.TERMINATION for item in run.expected_private_audit_trace):
        truth = run.scenario.private_targets[0]
        terminal_tick = (
            run.scenario.t_max_tick
            if result.record.termination == "horizon"
            else quantize_tick(result.record.makespan, DynamicConfig().tick_size)
        )
        records.append(ExpectedPrivateRecord(
            PrivateRecordKind.TERMINATION, 0,
            terminal_tick,
            0, 0, None, truth.true_category, truth.true_damage, truth.true_damage,
            None, None, 0.0,
        ))
    rank = {
        PrivateRecordKind.COMPLETION: 0, PrivateRecordKind.REJECTION: 1,
        PrivateRecordKind.ALLOCATION: 1, PrivateRecordKind.SCHEDULER: 2,
        PrivateRecordKind.REPLAY: 2, PrivateRecordKind.TERMINATION: 3,
    }
    return tuple(
        replace(record, event_id=index)
        for index, record in enumerate(sorted(records, key=lambda item: (item.tick, rank[item.kind])))
    )


def _expected_private_trace(run: D0Run) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in run.expected_private_audit_trace:
        item = asdict(record)
        if record.draw is not None:
            item["draw"] = float(record.draw)
        records.append(_canonical(item))
    return records


def _expected_utility(run: D0Run) -> tuple[dict[str, float], dict[str, float], list[dict[str, Any]]]:
    """Independent oracle from frozen fixture timelines, never evaluator output."""

    config = DynamicConfig()
    service_by_mode = {
        D0Action.RECON: config.recon_service_cost,
        D0Action.ATTACK: config.attack_service_cost,
        D0Action.BDA: config.bda_service_cost,
    }
    completions = {
        event.source_step_index: event
        for event in run.expected_private_audit_trace
        if event.kind is PrivateRecordKind.COMPLETION
    }
    service_terms: list[float] = []
    reward_terms: list[float] = []
    distance_consumed = 0.0
    ammo_consumed = 0.0
    discount_terms: list[dict[str, Any]] = []
    for index, step in enumerate(run.script):
        event = completions.get(index)
        if event is None:
            continue
        if step.start_tick is None or step.finish_tick is None or step.action is None:
            raise AssertionError("completion oracle requires a full expected timeline")
        service_discount = math.exp(
            -config.discount_rate * step.start_tick * config.tick_size
        )
        reward_discount = math.exp(
            -config.discount_rate * step.finish_tick * config.tick_size
        )
        service_terms.append(service_discount * service_by_mode[step.action])
        reward_terms.append(reward_discount * event.realized_reward)
        distance_consumed += (
            step.start_tick - step.commit_tick
        ) * config.tick_size * config.speed
        ammo_consumed += float(step.action is D0Action.ATTACK)
        discount_terms.append({
            "source_step_index": index,
            "start_tick": step.start_tick,
            "completion_tick": step.finish_tick,
            "service_discount": service_discount,
            "reward_discount": reward_discount,
            "uses_absolute_tick": True,
        })
    destroyed_value = math.fsum(reward_terms)
    service_cost = math.fsum(service_terms)
    distance_cost = config.distance_cost_rate * distance_consumed
    ammo_cost = config.ammo_cost_rate * ammo_consumed
    gross = math.fsum(
        config.value_high if target.true_category == "H" else config.value_low
        for target in run.scenario.private_targets
    )
    realized = destroyed_value - service_cost - distance_cost - ammo_cost
    utility = {
        "destroyed_value": destroyed_value,
        "service_cost": service_cost,
        "distance_cost": distance_cost,
        "ammo_cost": ammo_cost,
        "realized_utility": realized,
        "normalized_utility": realized / gross,
        "gross_scenario_value": gross,
    }
    resources = {
        "distance_consumed": distance_consumed,
        "ammo_consumed": ammo_consumed,
    }
    return utility, resources, discount_terms


def _commit_next_probe(run: D0Run) -> dict[str, Any]:
    first, suffix = run.script[:2]
    if not first.committed or suffix.committed:
        raise AssertionError("commit-next fixture requires one committed leg and one suffix")
    if any(
        value is None
        for value in (
            first.agent_id, first.target_id, first.action,
            suffix.target_id, suffix.action,
        )
    ):
        raise AssertionError("commit-next fixture lacks typed path fields")
    config = DynamicConfig()
    initial = initialize_state(run.scenario)
    problem = build_planning_problem(
        initial.snapshot(), config, run.scenario.t_max_tick
    )
    local_agent = problem.global_agent_ids.index(first.agent_id)
    local_path = (
        (
            problem.global_target_ids.index(first.target_id),
            AllocationMode(first.action.value),
        ),
        (
            problem.global_target_ids.index(suffix.target_id),
            AllocationMode(suffix.action.value),
        ),
    )
    evaluation = evaluate_mode_path(problem.instance, local_agent, local_path)
    if evaluation.start_ticks is None:
        raise AssertionError("commit-next path lacks start ticks")
    completion_ticks = tuple(
        int(evaluate_mode_path(problem.instance, local_agent, local_path[:index]).completion_tick)
        for index in range(1, 3)
    )
    path = PlannedPath(
        first.agent_id,
        (
            (first.target_id, AllocationMode(first.action.value)),
            (suffix.target_id, AllocationMode(suffix.action.value)),
        ),
        evaluation.score,
        evaluation.start_ticks,
        completion_ticks,
    )
    before = canonical_d0_bytes(path)
    outcome = commit_batch(
        initial, (path,), config, run.scenario.t_max_tick, tick=first.commit_tick
    )
    after = canonical_d0_bytes(path)
    state = outcome.state
    committed = outcome.committed
    agent = state.agents[first.agent_id]
    suffix_target = suffix.target_id
    first_target = first.target_id
    first_mode = first.action.value
    first_leg_only = (
        len(committed) == 1
        and committed[0].target_id == first_target
        and len(state.actions) == 1
    )
    return {
        "input_path": _canonical(path),
        "input_task_count": len(path.tasks),
        "input_path_unmutated": before == after,
        "returned_committed_count": len(committed),
        "first_leg_only_lock": state.target_locks == ((first_target, first.agent_id),),
        "first_leg_only_resources": (
            first_leg_only
            and agent.active_action == committed[0]
            and agent.ammo.reserved == committed[0].reserved_ammo == 1.0
            and agent.distance.reserved == committed[0].reserved_distance
        ),
        "first_leg_only_ordinal": (
            len(state.ordinals) == 1
            and state.ordinals[0][0] == first_target
            and str(getattr(state.ordinals[0][1], "value", state.ordinals[0][1])) == first_mode
        ),
        "first_leg_only_event": (
            len(state.completion_events) == 1
            and state.completion_events[0].target_id == first_target
        ),
        "suffix_has_no_lock_action_or_event": (
            suffix_target not in dict(state.target_locks)
            and all(action.target_id != suffix_target for action in state.actions)
            and all(event.target_id != suffix_target for event in state.completion_events)
        ),
    }


def _consume_focused_assertions(fixture: Any, runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(fixture.focused_assertions) != 1:
        raise AssertionError("each D0 witness must register exactly one focused assertion")
    assertion = fixture.focused_assertions[0]
    operands = {operand.name.value: operand.value for operand in assertion.operands}
    if len(operands) != len(assertion.operands):
        raise AssertionError("focused assertion operand names must be unique")
    required = {"expected_count", "expected_reward", "tolerance"}
    if not required.issubset(operands):
        raise AssertionError("focused assertion is missing required operands")
    extras_by_assertion = {
        "recon_joint_bayes": {"expected_posterior"},
        "busy_exclusion": {"expected_bidders", "expected_gate_agents"},
    }
    supported = {
        "initial_wreck_zero_reward", "first_destroyed_paid_once",
        "continuous_attack", "ack_nonleakage", "recon_joint_bayes",
        "bda_before_attack", "target_lock", "busy_exclusion",
        "completion_batch", "commit_next", "handoff", "hard_rejection",
        "completion_before_planning", "defer_reactivation", "no_positive",
        "counter_replay", "busy_prevents_termination", "horizon_settlement",
        "allocation_stall", "event_nonleakage", "b1m_auto_next",
        "absolute_discount",
    }
    assertion_name = assertion.assertion_id.value
    if assertion_name not in supported:
        raise AssertionError(f"unknown focused assertion: {assertion_name}")
    allowed_operands = required | extras_by_assertion.get(assertion_name, set())
    if set(operands) != allowed_operands:
        raise AssertionError("unknown or unconsumed focused assertion operand")
    tolerance = float(operands["tolerance"])
    first_contract = fixture.runs[0]
    completions = [
        event for event in first_contract.expected_private_audit_trace
        if event.kind is PrivateRecordKind.COMPLETION
    ]
    rejections = [
        event for event in first_contract.expected_private_audit_trace
        if event.kind is PrivateRecordKind.REJECTION
    ]
    count_by_assertion = {
        "initial_wreck_zero_reward": sum(
            step.committed and step.action is D0Action.ATTACK
            and step.physical_control is not None
            and step.physical_control.damage_before == "D"
            for step in first_contract.script
        ),
        "target_lock": len(rejections),
        "busy_exclusion": len(rejections),
        "hard_rejection": len(rejections),
        "no_positive": int(first_contract.expected_terminal.value == "no_positive"),
        "allocation_stall": int(first_contract.expected_gate is not None),
    }
    actual_count = count_by_assertion.get(assertion.assertion_id.value, len(completions))
    if actual_count != operands["expected_count"]:
        raise AssertionError("focused expected_count was not satisfied")
    expected_reward = math.fsum(event.realized_reward for event in completions)
    if not math.isclose(
        expected_reward, float(operands["expected_reward"]),
        rel_tol=0.0, abs_tol=tolerance,
    ):
        raise AssertionError("focused expected_reward was not satisfied")
    if assertion_name == "initial_wreck_zero_reward" and expected_reward != 0.0:
        raise AssertionError("initial wreck produced reward")
    if assertion_name == "first_destroyed_paid_once":
        if sum(event.first_destroyed_payment for event in completions) != 1:
            raise AssertionError("first destruction payment was not unique")
    if "expected_posterior" in operands:
        posterior = next(
            event.posterior_belief for event in first_contract.expected_public_trace
            if event.kind is PublicRecordKind.OBSERVATION
        )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)
            for left, right in zip(posterior, operands["expected_posterior"], strict=True)
        ):
            raise AssertionError("focused posterior mismatch")
    if "expected_bidders" in operands or "expected_gate_agents" in operands:
        busy_step = next(step for step in first_contract.script if step.fault is D0Fault.EXCLUDE_BUSY)
        if tuple(busy_step.precondition.idle_agents) != tuple(operands["expected_bidders"]):
            raise AssertionError("focused bidder set mismatch")
        if tuple(busy_step.precondition.idle_agents) != tuple(operands["expected_gate_agents"]):
            raise AssertionError("focused Gate-agent set mismatch")
    audit: dict[str, Any] = {
        "assertion_id": assertion.assertion_id.value,
        "operands": _canonical(assertion.operands),
        "consumed_operand_names": sorted(operands),
        "consumed_exactly_once": True,
    }
    if assertion_name == "absolute_discount":
        _, _, terms = _expected_utility(first_contract)
        if len(terms) != 2 or terms[0]["start_tick"] == terms[1]["start_tick"]:
            raise AssertionError("absolute discount witness requires two distinct clocks")
        audit["absolute_discount_terms"] = terms
    if assertion_name == "commit_next":
        audit["commit_next_probe"] = _commit_next_probe(first_contract)
    del runs
    return audit


def _validate_absences(
    run: D0Run, result: Any, policy: _TickScriptedPolicy,
    public_trace: tuple[ExpectedPublicRecord, ...],
    private_trace: tuple[ExpectedPrivateRecord, ...],
) -> None:
    consumed: list[tuple[AbsenceKind, int]] = []
    for absence in run.expected_absences:
        step = run.script[absence.step_index]
        actions = tuple(
            action for action in result.actions
            if action.commit_tick == step.commit_tick
            and (step.agent_id is None or action.agent_id == step.agent_id)
            and (step.target_id is None or action.target_id == step.target_id)
            and (step.action is None or action.mode == step.action.value)
        )
        completions = tuple(
            event for event in result.private_audit_events
            if (step.agent_id is None or event.agent_id == step.agent_id)
            and (step.target_id is None or event.target_id == step.target_id)
            and (step.action is None or event.mode == step.action.value)
            and any(action.finish_tick == event.tick for action in actions)
        )
        if absence.kind in (AbsenceKind.NO_COMMIT, AbsenceKind.NO_COMPLETION, AbsenceKind.NO_COUNTER_READ):
            if actions or completions:
                raise AssertionError(f"absence violated: {absence}")
        if absence.kind is AbsenceKind.NO_COUNTER_READ and (step.counter_key is not None or step.expected_uniform is not None):
            raise AssertionError("rejected step unexpectedly registered a counter read")
        if absence.kind is AbsenceKind.NO_EARLY_TERMINATION:
            calls = tuple(call for call in policy.calls if call[0].tick == step.commit_tick)
            if len(calls) != 1:
                raise AssertionError("busy witness requires one planning call")
            finishes = tuple(
                agent.busy_action.finish_tick for agent in calls[0][0].agents
                if agent.busy_action is not None
            )
            terminal = quantize_tick(result.record.makespan, DynamicConfig().tick_size)
            if not finishes or terminal < min(finishes):
                raise AssertionError("busy event terminated early")
        if absence.kind not in AbsenceKind:
            raise AssertionError(f"unknown absence: {absence.kind}")
        consumed.append((absence.kind, absence.step_index))
    if len(consumed) != len(set(consumed)):
        raise AssertionError("canonical absences must be consumed exactly once")
    del public_trace, private_trace


def _registered_gates(run: D0Run) -> list[dict[str, Any]]:
    if run.expected_gate is None:
        return []
    return [{
        "gate": "scheduler",
        "tick": run.expected_gate.tick,
        "reason": run.expected_gate.gate.value.replace("-", "_"),
        "details": [],
    }]


def _actual_gates(result: Any) -> list[dict[str, Any]]:
    return [
        {"gate": gate.gate, "tick": gate.tick, "reason": gate.reason,
         "details": [list(item) for item in gate.details]}
        for gate in result.gate_failures
    ]


_TERMINALS = {
    "normal": "normal", "no_positive": "no_positive",
    "horizon": "horizon", "gate_failure": "allocation_stall",
}


def _execute_run(run: D0Run) -> dict[str, Any]:
    grids = tuple(sorted(
        {step.commit_tick for step in run.script if step.commit_tick > 0}
        | {event.tick for event in run.expected_public_trace
           if event.kind is PublicRecordKind.SCHEDULER and event.tick > 0}
    ))
    witness_name = run.scenario.scenario_id.removeprefix("D0-")
    real_policy = witness_name in {
        "defer_reactivated_by_event",
        "b1m_frozen_suffix_auto_next",
    }
    policy: Any = (
        _ObservedRealPolicy(run.method_id, run.scenario.t_max_tick)
        if real_policy
        else _TickScriptedPolicy(run)
    )
    result = run_episode(run.scenario, policy, method=run.method_id, planning_grid_ticks=grids)
    public_trace = _logical_public_trace(run, result, policy)
    private_trace = _logical_private_trace(run, result)
    if public_trace != run.expected_public_trace:
        raise AssertionError("public canonical trace mismatch")
    expected_private_trace = _expected_private_trace(run)
    if _canonical(private_trace) != expected_private_trace:
        raise AssertionError("private canonical trace mismatch")
    _validate_absences(run, result, policy, public_trace, private_trace)
    expected_utility, expected_resources, _ = _expected_utility(run)
    expected_gates = _registered_gates(run)
    actual_gates = _actual_gates(result)
    if actual_gates != expected_gates:
        raise AssertionError(f"registered Gate mismatch: {actual_gates!r} != {expected_gates!r}")
    expected_terminal = {
        "state": _TERMINALS[run.expected_terminal.value], "tick": run.terminal_tick
    }
    actual_terminal = {
        "state": result.record.termination,
        "tick": (
            run.scenario.t_max_tick
            if result.record.termination == "horizon"
            else quantize_tick(result.record.makespan, DynamicConfig().tick_size)
        ),
    }
    if actual_terminal != expected_terminal:
        raise AssertionError(f"termination mismatch: {actual_terminal!r} != {expected_terminal!r}")
    utility_fields = (
        "destroyed_value", "service_cost", "distance_cost", "ammo_cost",
        "realized_utility", "normalized_utility", "gross_scenario_value",
    )
    utility_record = {name: getattr(result.record, name) for name in utility_fields}
    resources = {
        "distance_consumed": result.record.distance_consumed,
        "ammo_consumed": result.record.ammo_consumed,
    }
    if any(
        not math.isclose(utility_record[name], value, rel_tol=0.0, abs_tol=1e-10)
        for name, value in expected_utility.items()
    ):
        raise AssertionError("actual utility differs from independent fixture oracle")
    if any(
        not math.isclose(resources[name], value, rel_tol=0.0, abs_tol=1e-10)
        for name, value in expected_resources.items()
    ):
        raise AssertionError("actual resources differ from independent fixture oracle")
    actual_counter_events = []
    for event in result.private_audit_events:
        source = _source_step_index(
            run, tick=event.tick, target_id=event.target_id,
            agent_id=event.agent_id, mode=event.mode,
        )
        expected_step = run.script[source]
        actual_counter_events.append({
            "source_step_index": source,
            "counter_key": _canonical(event.counter_key),
            "draw": event.draw,
            "physical_success": event.physical_success,
            "damage_after": event.damage_after,
            "key_matches_expected": event.counter_key == expected_step.counter_key,
            "draw_matches_expected": event.draw == float(expected_step.expected_uniform),
        })
    policy_diagnostics: Any = []
    if isinstance(policy, _ObservedRealPolicy):
        if run.method_id == "B1m":
            actions = result.actions
            policy_diagnostics = {
                "policy": type(policy.delegate).__name__,
                "planning_calls": policy.planning_calls,
                "auto_next_calls": policy.auto_next_calls,
                "auto_next_commit_count": policy.auto_next_commit_count,
                "tick0_frozen_suffix_count": policy.tick0_frozen_suffix_count,
                "completion_count": len(result.private_audit_events),
                "suffix_prevented_false_termination": result.record.termination != "no_positive",
                "active_leg_only_resources": (
                    len(actions) == 2
                    and actions[0].reserved_ammo == 1.0
                    and math.isclose(actions[0].reserved_distance, 2.0, abs_tol=1e-12)
                    and actions[1].commit_tick == actions[0].finish_tick
                ),
                "tick0_committed_action_count": sum(
                    action.commit_tick == 0 for action in actions
                ),
                "tick0_reserved_ammo": math.fsum(
                    action.reserved_ammo for action in actions if action.commit_tick == 0
                ),
                "termination": result.record.termination,
                "last_completion_tick": max(
                    event.tick for event in result.private_audit_events
                ),
                "termination_tick": quantize_tick(
                    result.record.makespan, DynamicConfig().tick_size
                ),
                "t_max_tick": run.scenario.t_max_tick,
                "planning": policy.decisions,
            }
        else:
            policy_diagnostics = policy.decisions
    return {
        "run_id": run.run_id,
        "method_id": run.method_id,
        "public_trace": _canonical(public_trace),
        "expected_public_trace": _canonical(run.expected_public_trace),
        "private_audit_trace": _canonical(private_trace),
        "expected_private_audit_trace": expected_private_trace,
        "digests": {
            "public_trace": _digest(public_trace),
            "private_audit_trace": _digest(private_trace),
            "initial_public": result.record.public_initial_digest,
            "initial_private_truth": result.record.initial_truth_digest,
            "final_private_truth": result.record.final_truth_digest,
        },
        "utility_decomposition": utility_record,
        "expected_utility_decomposition": expected_utility,
        "resources": resources,
        "expected_resources": expected_resources,
        "actual_counter_events": actual_counter_events,
        "policy_diagnostics": policy_diagnostics,
        "termination": {"expected": expected_terminal, "actual": actual_terminal},
        "gates": {"expected": expected_gates, "actual": actual_gates},
    }


def run_d0_witness(witness_id: str) -> dict[str, Any]:
    """Execute one named witness, including an independent replay of every run."""

    fixtures = d0_scenarios()
    by_name = {fixture.name: (order, fixture) for order, fixture in enumerate(fixtures, 1)}
    if witness_id not in by_name:
        raise ValueError(f"unknown D0 witness: {witness_id}")
    order, fixture = by_name[witness_id]
    runs: list[dict[str, Any]] = []
    try:
        for contract in fixture.runs:
            first = _execute_run(contract)
            replay = _execute_run(contract)
            equal = first == replay
            bytes_equal = canonical_d0_bytes(first) == canonical_d0_bytes(replay)
            first["same_process_replay"] = {
                "fieldwise_equal": equal,
                "canonical_bytes_equal": bytes_equal,
                "canonical_sha256": _digest(replay),
            }
            if not equal or not bytes_equal:
                raise AssertionError("fresh execution replay mismatch")
            runs.append(first)
        if witness_id == "attack_ack_hides_outcome":
            if runs[0]["public_trace"] != runs[1]["public_trace"]:
                raise AssertionError("Attack ACK leaked private outcome")
        if witness_id == "counter_replay_shared_initial_truth":
            if runs[0]["digests"]["initial_private_truth"] != runs[1]["digests"]["initial_private_truth"]:
                raise AssertionError("cross-method initial truth mismatch")
            if runs[0]["actual_counter_events"] != runs[1]["actual_counter_events"]:
                raise AssertionError("cross-method actual counter replay mismatch")
        focused_audit = _consume_focused_assertions(fixture, runs)
    except Exception as error:
        return {
            "witness_id": witness_id, "order": order, "status": "failed",
            "runs": runs, "error": {"type": type(error).__name__, "message": str(error)},
        }
    return {
        "witness_id": witness_id,
        "order": order,
        "status": "passed",
        "runs": runs,
        "focused_assertion_audit": focused_audit,
        "error": None,
    }


def run_all_d0() -> dict[str, Any]:
    """Execute all D0 fixtures in specification order and aggregate Gate status."""

    records = [run_d0_witness(fixture.name) for fixture in d0_scenarios()]
    failed = [record for record in records if record["status"] != "passed"]
    run_count = sum(len(record["runs"]) for record in records)
    unexpected_gates = sum(
        record["status"] == "passed"
        and run["gates"]["actual"] != run["gates"]["expected"]
        for record in records for run in record["runs"]
    )
    summary = {
        "status": "passed" if not failed and unexpected_gates == 0 else "failed",
        "witness_count": len(records), "run_count": run_count,
        "passed": len(records) - len(failed), "failed": len(failed),
        "gate_failures": unexpected_gates,
        "registered_gate_count": sum(
            len(run["gates"]["expected"])
            for record in records for run in record["runs"]
        ),
        "witness_order": [record["witness_id"] for record in records],
    }
    return {"records": records, "summary": summary}


def _fresh_process_aggregate() -> bytes:
    workspace = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(workspace / "src") + (
        os.pathsep + existing if existing else ""
    )
    command = (
        "from uav_lifecycle.dynamic_d0 import canonical_d0_bytes,run_all_d0;"
        "import sys;sys.stdout.buffer.write(canonical_d0_bytes(run_all_d0()))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=workspace,
        env=environment,
        check=True,
        capture_output=True,
    )
    if completed.stderr:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
    return completed.stdout


def write_d0_artifacts(output_root: str | Path) -> dict[str, Any]:
    """Atomically persist D0 records, summary, and Task7 checkpoint."""

    root = Path(output_root)
    aggregate = run_all_d0()
    local_bytes = canonical_d0_bytes(aggregate)
    fresh_bytes = _fresh_process_aggregate()
    fieldwise_equal = json.loads(fresh_bytes) == _canonical(aggregate)
    bytes_equal = fresh_bytes == local_bytes
    if not fieldwise_equal or not bytes_equal:
        raise AssertionError("fresh-process D0 replay mismatch")
    witnesses = root / "d0_witnesses"
    checkpoint_dir = root / "checkpoints"
    records_path = witnesses / "d0_records.json"
    summary_path = witnesses / "d0_summary.json"
    checkpoint_path = checkpoint_dir / "task07.json"
    write_json_atomic(records_path, aggregate["records"])
    replay = {
        "fieldwise_equal": fieldwise_equal,
        "canonical_bytes_equal": bytes_equal,
        "canonical_sha256": sha256(local_bytes).hexdigest(),
    }
    summary = {
        **aggregate["summary"],
        "spec_version": "dynamic_lifecycle_mainline_v2",
        "stage": "D0",
        "d1_executed": False,
        "d2_authorized": False,
        "contract_digest": d0_contract_digest(),
        "records_file": records_path.as_posix(),
        "records_file_sha256": sha256_file(records_path),
        "strict_json_no_nan": True,
        "fresh_process_replay": replay,
    }
    write_json_atomic(summary_path, summary)
    checkpoint = {
        "task": 7,
        "status": aggregate["summary"]["status"],
        "spec_version": "dynamic_lifecycle_mainline_v2",
        "no_git": True,
        "d2_authorized": False,
        "counts": {
            "witnesses": aggregate["summary"]["witness_count"],
            "runs": aggregate["summary"]["run_count"],
            "failed": aggregate["summary"]["failed"],
            "unexpected_gate_failures": aggregate["summary"]["gate_failures"],
            "registered_gates": aggregate["summary"]["registered_gate_count"],
        },
        "fresh_process_replay": replay,
        "files": [
            {"path": records_path.as_posix(), "sha256": sha256_file(records_path)},
            {"path": summary_path.as_posix(), "sha256": sha256_file(summary_path)},
        ],
    }
    write_json_atomic(checkpoint_path, checkpoint)
    return {
        "records_path": str(records_path),
        "summary_path": str(summary_path),
        "checkpoint_path": str(checkpoint_path),
        "fresh_process_replay": replay,
    }


__all__ = [
    "canonical_d0_bytes",
    "run_all_d0",
    "run_d0_witness",
    "write_d0_artifacts",
]
