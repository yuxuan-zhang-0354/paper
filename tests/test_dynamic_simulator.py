"""Scripted public-policy witnesses for the dynamic event simulator."""

from __future__ import annotations

from dataclasses import replace
import inspect
from math import exp

import pytest

from uav_lifecycle.attack import predict_attack
from uav_lifecycle.dynamic_planning import (
    PlannedPath,
    build_planning_problem,
    commit_ticks,
    plan_all_mode_exact,
)
from uav_lifecycle.dynamic_d0 import run_d0_witness
from uav_lifecycle.dynamic_rng import DrawKey
from uav_lifecycle.dynamic_rng import uniform01
from uav_lifecycle.dynamic_policies import ActionProposal, PolicyDecision
from uav_lifecycle.dynamic_scenarios import (
    AbsenceKind,
    D0Action,
    D0Fault,
    ExpectedPrivateRecord,
    ExpectedPublicRecord,
    PrivateRecordKind,
    PublicRecordKind,
    RejectionReason,
    d0_scenarios,
)
from uav_lifecycle.dynamic_simulator import (
    advance_public_belief,
    commit_batch,
    complete_event_batch,
    evaluate_real_utility,
    initialize_state,
    run_episode,
)
from uav_lifecycle.dynamic_types import (
    DynamicConfig,
    PlanningClock,
    GateFailure,
    PrivateTarget,
    PublicActionAck,
    PublicObservation,
    PublicSnapshot,
    ResourceLedger,
    quantize_tick,
)
from uav_lifecycle.mode_allocation import Mode as AllocationMode, evaluate_mode_path


class ScriptedPolicy:
    """Test double indexed only by keys derivable from public snapshots."""

    def __init__(self, script, positives=None, *, clock=PlanningClock.EVENT_DRIVEN):
        self.script = dict(script)
        self.positives = dict(positives or {})
        self.calls = []
        self.planning_clock = clock

    @staticmethod
    def key(snapshot: PublicSnapshot):
        return snapshot.tick, len(snapshot.observations), len(snapshot.acknowledgements)

    def decide(self, snapshot: PublicSnapshot):
        assert isinstance(snapshot, PublicSnapshot)
        self.calls.append(self.key(snapshot))
        return self.script.get(self.key(snapshot), ())

    def positive_pair_count(self, snapshot: PublicSnapshot):
        return self.positives.get(self.key(snapshot), 0)


class TickScriptedPolicy:
    """Canonical-fixture adapter whose runtime input remains public-only."""

    def __init__(self, run):
        self.decisions = {}
        self.planning_clock = PlanningClock.PERIODIC
        self.calls = []
        self.future_decision_ticks = tuple(
            sorted({step.commit_tick for step in run.script if step.committed})
        )
        self.force_stall = run.expected_gate is not None
        for step in run.script:
            if step.committed:
                self.decisions.setdefault(step.commit_tick, []).append(
                    (step.agent_id, step.target_id, step.action.value)
                )

    def decide(self, snapshot: PublicSnapshot):
        assert isinstance(snapshot, PublicSnapshot)
        decisions = tuple(self.decisions.get(snapshot.tick, ()))
        self.calls.append((snapshot, decisions, self.positive_pair_count(snapshot)))
        return decisions

    def positive_pair_count(self, snapshot: PublicSnapshot):
        if self.force_stall and snapshot.tick == 0:
            return 1
        return max(
            len(self.decisions.get(snapshot.tick, ())),
            int(any(tick > snapshot.tick for tick in self.future_decision_ticks)),
        )


def _fixture(name):
    return next(item for item in d0_scenarios() if item.name == name)


def _single_policy(mode="attack", *, followup=()):
    return ScriptedPolicy({(0, 0, 0): ((0, 0, mode),), **dict(followup)})


def test_commit_timeline_lock_reservation_ordinal_and_suffix_release():
    scenario = _fixture("commit_next_releases_suffix").scenario
    scenario = replace(scenario, agents=scenario.agents[:1])
    config = DynamicConfig()
    state = initialize_state(scenario)
    start, finish = commit_ticks(0, (0.0, 0.0), (2.0, 0.0), "attack", config)
    planned = plan_all_mode_exact(state.snapshot(), config, scenario.t_max_tick)
    assert len(planned.paths) == 1 and len(planned.paths[0].tasks) == 2

    outcome = commit_batch(state, planned.paths, config, scenario.t_max_tick)
    action = outcome.committed[0]
    assert (action.start_tick, action.finish_tick, action.ordinal) == (start, finish, 0)
    assert outcome.state.target_locks == ((0, 0),)
    assert outcome.state.agents[0].ammo == ResourceLedger(2.0, 1.0, 0.0)
    assert outcome.state.agents[0].distance == ResourceLedger(38.0, 2.0, 0.0)
    assert len(outcome.state.completion_events) == 1
    assert all(lock[0] != 1 for lock in outcome.state.target_locks)


def test_whole_commit_batch_is_atomic_on_duplicate_target():
    scenario = _fixture("simultaneous_completion_batch").scenario
    state = initialize_state(scenario)
    outcome = commit_batch(
        state, ((0, 0, "attack"), (1, 0, "recon")), DynamicConfig(), scenario.t_max_tick
    )
    assert not outcome.committed
    assert outcome.state == state
    assert outcome.gates == (GateFailure("commit_batch", 0, "duplicate_target"),)


def test_locked_target_and_busy_agent_rejections_are_atomic():
    scenario = _fixture("target_lock_prevents_overlap").scenario
    config = DynamicConfig()
    active = commit_batch(
        initialize_state(scenario), ((0, 0, "attack"),), config, scenario.t_max_tick
    ).state
    locked = commit_batch(active, ((1, 0, "attack"),), config, scenario.t_max_tick)
    busy = commit_batch(active, ((0, 1, "attack"),), config, scenario.t_max_tick)
    assert locked.state == busy.state == active
    assert locked.gates[0].reason == "target_locked"
    assert busy.gates[0].reason == "busy_agent"


def test_completion_consumes_reservation_once_and_unlocks():
    scenario = _fixture("initial_wreck_zero_reward").scenario
    config = DynamicConfig()
    committed = commit_batch(
        initialize_state(scenario), ((0, 0, "attack"),), config, scenario.t_max_tick
    ).state
    completed = complete_event_batch(
        committed, committed.completion_events, config
    )
    agent = completed.state.agents[0]
    assert agent.ammo == ResourceLedger(2.0, 0.0, 1.0)
    assert agent.distance == ResourceLedger(38.0, 0.0, 2.0)
    assert agent.position == (2.0, 0.0) and agent.active_action is None
    assert completed.state.target_locks == ()
    with pytest.raises(ValueError, match="every event due|not active"):
        complete_event_batch(completed.state, committed.completion_events, config)


def test_same_tick_completion_batch_cannot_be_partially_settled():
    scenario = _fixture("simultaneous_completion_batch").scenario
    config = DynamicConfig()
    state = commit_batch(
        initialize_state(scenario),
        ((0, 0, "attack"), (1, 1, "attack")),
        config,
        scenario.t_max_tick,
    ).state
    with pytest.raises(ValueError, match="every event due"):
        complete_event_batch(state, state.completion_events[:1], config)


def test_recon_fixture_uses_one_joint_draw_and_joint_bayes_update():
    fixture = _fixture("recon_joint_bayes")
    result = run_episode(fixture.scenario, _single_policy("recon"), method="P")
    expected = fixture.expected_public_trace[0]
    assert len(result.public_events) == len(result.private_audit_events) == 1
    assert isinstance(result.public_events[0], PublicObservation)
    assert result.public_events[0].observation == expected.observation
    assert result.record.final_beliefs[0] == pytest.approx(expected.posterior_belief)
    assert float(fixture.script[0].expected_uniform) == result.private_audit_events[0].draw
    assert result.private_audit_events[0].counter_key == fixture.script[0].counter_key


def test_bda_damage_observation_and_bayes_update_before_attack():
    fixture = _fixture("bda_before_first_attack")
    result = run_episode(fixture.scenario, _single_policy("bda"), method="P")
    event = result.public_events[0]
    assert isinstance(event, PublicObservation) and event.mode == "bda"
    assert event.observation in ("A", "D")
    assert result.record.bda_count == 1 and result.record.recon_count == 0


def test_attack_ack_and_public_prediction_do_not_leak_private_outcome():
    fixture = _fixture("attack_ack_hides_outcome")
    success = run_episode(fixture.runs[0].scenario, _single_policy(), method="P")
    failure = run_episode(fixture.runs[1].scenario, _single_policy(), method="P")
    assert success.public_events == failure.public_events
    assert isinstance(success.public_events[0], PublicActionAck)
    expected = predict_attack(
        fixture.scenario.targets[0].belief,
        DynamicConfig().attack_success_high,
        DynamicConfig().attack_success_low,
    )
    assert success.record.final_beliefs[0] == pytest.approx(expected)
    assert success.private_audit_events != failure.private_audit_events


def test_hidden_truth_change_cannot_change_next_public_policy_decision():
    base = _fixture("initial_wreck_zero_reward").scenario
    alternate = replace(
        base,
        private_targets=(PrivateTarget(0, "L", "A", False), base.private_targets[1]),
    )
    first = _single_policy("attack")
    second = _single_policy("attack")
    run_episode(base, first, method="P")
    run_episode(alternate, second, method="P")
    assert first.calls == second.calls


def test_semantic_crn_draw_replays_across_schedules():
    scenario = _fixture("counter_replay_shared_initial_truth").scenario
    event_policy = ScriptedPolicy({
        (0, 0, 0): ((0, 1, "bda"),),
        (quantize_tick(3.5, DynamicConfig().tick_size), 1, 0): ((0, 0, "recon"),),
    })
    event_result = run_episode(scenario, event_policy, method="P")
    first_finish = quantize_tick(3.5, DynamicConfig().tick_size)
    grid = quantize_tick(4.0, DynamicConfig().tick_size)
    periodic_policy = ScriptedPolicy(
        {
            (0, 0, 0): ((0, 1, "bda"),),
            (grid, 1, 0): ((0, 0, "recon"),),
        },
        {(first_finish, 1, 0): 1},
        clock=PlanningClock.PERIODIC,
    )
    periodic_result = run_episode(
        scenario, periodic_policy, method="periodic", planning_grid_ticks=(grid,)
    )
    event_recon = next(e for e in event_result.private_audit_events if e.mode == "recon")
    periodic_recon = next(e for e in periodic_result.private_audit_events if e.mode == "recon")
    assert event_recon.draw == periodic_recon.draw


def test_first_destruction_paid_once_and_audit_reconstructs_utility():
    fixture = _fixture("first_new_destroyed_paid_once")
    first_finish = fixture.script[0].finish_tick
    policy = ScriptedPolicy({
        (0, 0, 0): ((0, 0, "attack"),),
        (first_finish, 0, 1): ((0, 0, "attack"),),
    })
    result = run_episode(fixture.scenario, policy, method="P")
    assert [event.realized_reward for event in result.private_audit_events] == [100.0, 0.0]
    rebuilt = evaluate_real_utility(
        fixture.scenario, result.actions, result.private_audit_events, DynamicConfig()
    )
    assert rebuilt.realized_utility == pytest.approx(result.record.realized_utility)
    expected_reward = exp(-0.02 * 4.0) * 100.0
    expected_service = exp(-0.02 * 2.0) * 6.0 + exp(-0.02 * 4.0) * 6.0
    assert rebuilt.destroyed_value == pytest.approx(expected_reward)
    assert rebuilt.service_cost == pytest.approx(expected_service)
    assert rebuilt.distance_cost == pytest.approx(0.2)
    assert rebuilt.ammo_cost == pytest.approx(1.0)
    assert rebuilt.gross_scenario_value == 130.0


def test_same_tick_completions_settle_at_horizon_before_termination():
    fixture = _fixture("completion_at_horizon_settles_first")
    result = run_episode(fixture.scenario, _single_policy(), method="P")
    assert result.record.termination == "horizon"
    assert result.record.event_count == 1
    assert result.public_events[0].tick == fixture.scenario.t_max_tick
    assert result.record.destroyed_value > 0.0


def test_busy_queue_prevents_early_termination_and_tick_zero_can_end_normally():
    busy = _fixture("busy_event_prevents_early_termination")
    result = run_episode(busy.scenario, _single_policy(), method="P")
    assert result.record.event_count == 1 and result.record.makespan == 4.0

    empty = run_episode(busy.scenario, ScriptedPolicy({}), method="P")
    assert empty.record.termination == "no_positive" and empty.record.makespan == 0.0


def test_future_periodic_grid_does_not_prevent_tick_zero_no_positive_termination():
    scenario = _fixture("no_positive_normal_termination").scenario
    grid = quantize_tick(5.0, DynamicConfig().tick_size)
    policy = ScriptedPolicy({}, clock=PlanningClock.PERIODIC)
    result = run_episode(scenario, policy, method="periodic", planning_grid_ticks=(grid,))
    assert policy.calls == [(0, 0, 0)]
    assert result.record.termination == "no_positive"
    assert result.record.replan_count == 1


def test_method_clock_controls_off_grid_completion_replanning():
    scenario = _fixture("busy_event_prevents_early_termination").scenario
    finish = quantize_tick(4.0, DynamicConfig().tick_size)
    grid = quantize_tick(5.0, DynamicConfig().tick_size)
    event_policy = ScriptedPolicy({
        (0, 0, 0): ((0, 0, "attack"),),
        (finish, 0, 1): ((0, 1, "bda"),),
    })
    periodic_policy = ScriptedPolicy(
        {
            (0, 0, 0): ((0, 0, "attack"),),
            (grid, 0, 1): ((0, 1, "bda"),),
        },
        {(finish, 0, 1): 1},
        clock=PlanningClock.PERIODIC,
    )
    event_result = run_episode(scenario, event_policy, method="event")
    periodic_result = run_episode(
        scenario, periodic_policy, method="periodic", planning_grid_ticks=(grid,)
    )
    assert finish in tuple(call[0] for call in event_policy.calls)
    assert finish not in tuple(call[0] for call in periodic_policy.calls)
    assert grid in tuple(call[0] for call in periodic_policy.calls)
    assert event_result.record.action_count == periodic_result.record.action_count == 2


def test_event_guard_is_internal_and_replay_deterministic():
    assert "event_guard" not in inspect.signature(run_episode).parameters
    scenario = _fixture("simultaneous_completion_batch").scenario
    first = run_episode(scenario, _single_policy(), method="P")
    second = run_episode(scenario, _single_policy(), method="P")
    assert first == second
    assert all(gate.reason != "event_guard" for gate in first.gate_failures)


def test_positive_pair_zero_commit_is_allocation_stall_gate():
    fixture = _fixture("positive_task_zero_commit_stall_gate")
    policy = ScriptedPolicy({}, {(0, 0, 0): 1})
    result = run_episode(fixture.scenario, policy, method="P")
    assert result.record.status == "gate_failure"
    assert result.record.termination == "allocation_stall"
    assert result.gate_failures[0].reason == "allocation_stall"


def test_allocator_failure_does_not_discard_an_unrelated_inflight_action():
    scenario = _fixture("simultaneous_completion_batch").scenario
    scenario = replace(
        scenario,
        targets=(
            scenario.targets[0],
            replace(scenario.targets[1], position=(8.0, 0.0)),
        ),
    )

    class FailingAfterFirstCompletion:
        planning_clock = PlanningClock.EVENT_DRIVEN

        def __init__(self):
            self.calls = 0

        def decide(self, snapshot):
            self.calls += 1
            if self.calls == 1:
                return PolicyDecision(
                    (
                        ActionProposal(0, 0, "attack"),
                        ActionProposal(1, 1, "attack"),
                    ),
                    (),
                    b"",
                    (),
                )
            return PolicyDecision(
                (),
                (),
                b"",
                (),
                (GateFailure("johnson", snapshot.tick, "cycle"),),
            )

        def positive_pair_count(self, snapshot):
            return 2

    result = run_episode(scenario, FailingAfterFirstCompletion(), method="P")
    assert result.record.status == "gate_failure"
    assert result.record.termination == "cycle"
    assert result.record.action_count == result.record.event_count == 2
    assert len(result.actions) == len(result.private_audit_events) == 2


@pytest.mark.parametrize("fixture_name", ["horizon_range_ammo_rejection"])
def test_hard_limits_reject_whole_batch_without_overrun(fixture_name):
    scenario = _fixture(fixture_name).scenario
    config = DynamicConfig()
    state = initialize_state(scenario)
    late = quantize_tick(15.0, config.tick_size)
    horizon = commit_batch(state, ((0, 0, "attack"),), config, scenario.t_max_tick, tick=late)
    range_fail = commit_batch(state, ((1, 1, "attack"),), config, scenario.t_max_tick)
    ammo_fail = commit_batch(state, ((2, 2, "attack"),), config, scenario.t_max_tick)
    assert {horizon.gates[0].reason, range_fail.gates[0].reason, ammo_fail.gates[0].reason} == {
        "horizon", "range", "ammo"
    }
    assert horizon.state == range_fail.state == ammo_fail.state == state


def test_zero_evidence_becomes_gate_without_division(monkeypatch):
    import uav_lifecycle.dynamic_simulator as simulator

    def impossible(*args, **kwargs):
        raise ValueError("observation has zero predictive probability")

    monkeypatch.setattr(simulator, "bayes_update", impossible)
    belief, gate = advance_public_belief(
        (0.4, 0.1, 0.2, 0.3), "bda", "A", DynamicConfig(), tick=7
    )
    assert belief == (0.4, 0.1, 0.2, 0.3)
    assert gate == GateFailure("belief", 7, "zero_evidence")


_D0_RUNS = tuple(
    (fixture.name, run) for fixture in d0_scenarios() for run in fixture.runs
)


@pytest.mark.parametrize(
    "fixture_name,run",
    _D0_RUNS,
    ids=[f"{name}-{run.run_id}" for name, run in _D0_RUNS],
)
def test_all_canonical_d0_committed_physics_traces(fixture_name, run):
    """Consume all D0 committed-action witnesses; Task 7 owns terminal aggregation."""

    if fixture_name == "b1m_frozen_suffix_auto_next":
        assert run_d0_witness(fixture_name)["status"] == "passed"
        return

    grids = tuple(sorted(
        {step.commit_tick for step in run.script if step.commit_tick > 0}
        | {
            event.tick
            for event in run.expected_public_trace
            if event.kind is PublicRecordKind.SCHEDULER and event.tick > 0
        }
    ))
    result = run_episode(
        run.scenario,
        (policy := TickScriptedPolicy(run)),
        method=run.method_id,
        planning_grid_ticks=grids,
    )
    expected_public = tuple(
        event
        for event in run.expected_public_trace
        if event.kind in (PublicRecordKind.OBSERVATION, PublicRecordKind.ATTACK_ACK)
    )
    expected_private = tuple(
        event
        for event in run.expected_private_audit_trace
        if event.kind is PrivateRecordKind.COMPLETION
    )
    assert len(result.public_events) == len(expected_public)
    assert len(result.private_audit_events) == len(expected_private)
    assert result.record.distance_consumed == pytest.approx(
        sum(action.reserved_distance for action in result.actions)
    )
    assert result.record.ammo_consumed == pytest.approx(
        sum(action.reserved_ammo for action in result.actions)
    )
    for actual, expected in zip(result.public_events, expected_public, strict=True):
        assert isinstance(actual, PublicActionAck) == (expected.kind is PublicRecordKind.ATTACK_ACK)
        assert (actual.tick, actual.target_id, actual.agent_id, actual.mode) == (
            expected.tick,
            expected.target_id,
            expected.agent_id,
            expected.mode.value,
        )
        if isinstance(actual, PublicObservation):
            assert actual.observation == expected.observation
        prior = run.scenario.targets[expected.target_id].belief
        earlier = tuple(
            item for item in expected_public
            if item.target_id == expected.target_id and item.event_id < expected.event_id
        )
        for item in earlier:
            prior = item.posterior_belief
        assert prior is not None
        posterior, gate = advance_public_belief(
            prior, expected.mode.value, expected.observation, DynamicConfig(), tick=expected.tick
        )
        assert gate is None
        assert posterior == pytest.approx(expected.posterior_belief)
    for actual, expected in zip(result.private_audit_events, expected_private, strict=True):
        assert (actual.tick, actual.target_id, actual.agent_id, actual.mode) == (
            expected.tick,
            expected.target_id,
            expected.agent_id,
            expected.mode.value,
        )
        assert actual.draw == float(expected.draw)
        assert actual.true_category == expected.true_category
        assert actual.damage_before == expected.damage_before
        assert actual.damage_after == expected.damage_after
        assert actual.realized_reward == expected.realized_reward
        if expected.mode.value == "attack":
            assert actual.physical_success is expected.physical_success
        else:
            assert expected.physical_success is None and actual.physical_success is False
        assert actual.invalid_attack is (
            expected.mode.value == "attack" and expected.damage_before == "D"
        )
        assert actual.initial_wreck_attack is (
            expected.mode.value == "attack"
            and run.scenario.private_targets[expected.target_id].true_damage == "D"
        )
        step = run.script[expected.source_step_index]
        action = next(
            item
            for item in result.actions
            if item.finish_tick == expected.tick
            and item.target_id == expected.target_id
            and item.agent_id == expected.agent_id
            and item.mode == expected.mode.value
        )
        reconstructed_key = DrawKey(
            "sha256-u64-v1",
            "dynamic-lifecycle-mainline-v2",
            "d1-generator-v1",
            run.scenario.cell_id,
            run.scenario.seed,
            "target",
            action.target_id,
            action.mode,
            action.ordinal,
            0,
        )
        assert reconstructed_key == step.counter_key
    for absence in run.expected_absences:
        step = run.script[absence.step_index]
        assert not any(
            action.commit_tick == step.commit_tick
            and action.agent_id == step.agent_id
            and action.target_id == step.target_id
            and (step.action is None or action.mode == step.action.value)
            for action in result.actions
        )
    scheduler_expected = tuple(
        event for event in run.expected_public_trace if event.kind is PublicRecordKind.SCHEDULER
    )
    for expected in scheduler_expected:
        matches = tuple(call for call in policy.calls if call[0].tick == expected.tick)
        assert len(matches) == 1
        snapshot, decisions, positive = matches[0]
        assert sum(agent.busy_action is not None for agent in snapshot.agents) == expected.queue_depth
        assert positive == expected.idle_positive_tasks
        assert len(decisions) == expected.commits
    terminal_map = {
        "normal": "normal",
        "no_positive": "no_positive",
        "horizon": "horizon",
        "gate_failure": "allocation_stall",
    }
    assert result.record.termination == terminal_map[run.expected_terminal.value]
    assert bool(result.gate_failures) is (run.expected_gate is not None)
    if run.expected_gate is not None:
        assert result.gate_failures[-1].tick == run.expected_gate.tick
        assert result.gate_failures[-1].reason == "allocation_stall"
    assert quantize_tick(result.record.makespan, DynamicConfig().tick_size) == run.terminal_tick
    for target_id in range(len(run.scenario.targets)):
        expected_updates = tuple(
            event.posterior_belief
            for event in expected_public
            if event.target_id == target_id
        )
        expected_final = (
            expected_updates[-1]
            if expected_updates
            else run.scenario.targets[target_id].belief
        )
        assert result.record.final_beliefs[target_id] == pytest.approx(expected_final)


def test_canonical_harness_contains_all_22_fixtures_and_24_runs():
    assert len(d0_scenarios()) == 22
    assert len(_D0_RUNS) == 24


def _source_step_index(run, *, tick, target_id, agent_id, mode):
    matches = tuple(
        index
        for index, step in enumerate(run.script)
        if step.committed
        and step.finish_tick == tick
        and step.target_id == target_id
        and step.agent_id == agent_id
        and step.action.value == mode
    )
    assert len(matches) == 1
    return matches[0]


def _logical_public_trace(run, result, policy):
    beliefs = [target.belief for target in run.scenario.targets]
    records = []
    for event in result.public_events:
        source = _source_step_index(
            run,
            tick=event.tick,
            target_id=event.target_id,
            agent_id=event.agent_id,
            mode=event.mode,
        )
        observation = event.observation if isinstance(event, PublicObservation) else None
        posterior, gate = advance_public_belief(
            beliefs[event.target_id], event.mode, observation, DynamicConfig(), tick=event.tick
        )
        assert gate is None
        beliefs[event.target_id] = posterior
        records.append(ExpectedPublicRecord(
            PublicRecordKind.ATTACK_ACK
            if isinstance(event, PublicActionAck)
            else PublicRecordKind.OBSERVATION,
            0,
            event.tick,
            event.target_id,
            event.agent_id,
            D0Action(event.mode),
            observation,
            source,
            posterior_belief=posterior,
        ))
    expected_scheduler = tuple(
        item for item in run.expected_public_trace if item.kind is PublicRecordKind.SCHEDULER
    )
    for expected in expected_scheduler:
        calls = tuple(call for call in policy.calls if call[0].tick == expected.tick)
        assert len(calls) == 1
        snapshot, decisions, positive = calls[0]
        records.append(ExpectedPublicRecord(
            PublicRecordKind.SCHEDULER,
            0,
            snapshot.tick,
            0,
            0,
            None,
            None,
            expected.source_step_index,
            sum(agent.busy_action is not None for agent in snapshot.agents),
            positive,
            len(decisions),
        ))
    if any(item.kind is PublicRecordKind.TERMINATION for item in run.expected_public_trace):
        records.append(ExpectedPublicRecord(
            PublicRecordKind.TERMINATION,
            0,
            quantize_tick(result.record.makespan, DynamicConfig().tick_size),
            0,
            0,
            None,
            None,
            None,
            0,
            0,
            0,
        ))
    rank = {
        PublicRecordKind.OBSERVATION: 0,
        PublicRecordKind.ATTACK_ACK: 0,
        PublicRecordKind.SCHEDULER: 1,
        PublicRecordKind.TERMINATION: 2,
    }
    return tuple(
        replace(record, event_id=index)
        for index, record in enumerate(sorted(records, key=lambda item: (item.tick, rank[item.kind])))
    )


def _state_before_step(run, step_index):
    state = initialize_state(run.scenario)
    config = DynamicConfig()
    for index, step in enumerate(run.script[:step_index]):
        if not step.committed:
            continue
        due = tuple(event for event in state.completion_events if event.finish_tick <= step.commit_tick)
        if due:
            for tick in sorted({event.finish_tick for event in due}):
                batch = tuple(event for event in state.completion_events if event.finish_tick == tick)
                state = complete_event_batch(state, batch, config).state
        outcome = commit_batch(
            state,
            ((step.agent_id, step.target_id, step.action.value),),
            config,
            run.scenario.t_max_tick,
            tick=step.commit_tick,
        )
        assert not outcome.gates
        state = outcome.state
    return state


def _logical_rejection(run, expected):
    assert expected.source_step_index is not None
    step = run.script[expected.source_step_index]
    state = _state_before_step(run, expected.source_step_index)
    config = DynamicConfig()
    direct_faults = {
        D0Fault.REJECT_LOCKED,
        D0Fault.REJECT_HORIZON,
        D0Fault.REJECT_RANGE,
        D0Fault.REJECT_AMMO,
    }
    if step.fault is D0Fault.RELEASE_SUFFIX:
        first = next(item for item in run.script[:expected.source_step_index] if item.committed)
        initial = initialize_state(run.scenario)
        problem = build_planning_problem(initial.snapshot(), config, run.scenario.t_max_tick)
        local_agent = problem.global_agent_ids.index(first.agent_id)
        local_path = (
            (problem.global_target_ids.index(first.target_id), AllocationMode(first.action.value)),
            (problem.global_target_ids.index(step.target_id), AllocationMode(step.action.value)),
        )
        evaluation = evaluate_mode_path(problem.instance, local_agent, local_path)
        completion_ticks = tuple(
            int(evaluate_mode_path(problem.instance, local_agent, local_path[:index]).completion_tick)
            for index in range(1, len(local_path) + 1)
        )
        candidate = PlannedPath(
            first.agent_id,
            ((first.target_id, AllocationMode(first.action.value)),
             (step.target_id, AllocationMode(step.action.value))),
            evaluation.score,
            evaluation.start_ticks,
            completion_ticks,
        )
        outcome = commit_batch(
            initial, (candidate,), config, run.scenario.t_max_tick, tick=first.commit_tick
        )
        assert outcome.state == state
        assert len(outcome.committed) == 1
        assert outcome.committed[0].target_id == first.target_id
        assert step.target_id not in dict(outcome.state.target_locks)
        assert all(action.target_id != step.target_id for action in outcome.state.actions)
        candidates = ()
    elif step.fault is D0Fault.EXCLUDE_BUSY:
        unlocked = next(target.target_id for target in state.targets if target.target_id not in dict(state.target_locks))
        candidates = ((step.agent_id, unlocked, "attack"),)
    elif step.fault in direct_faults:
        assert step.action is not None
        candidates = ((step.agent_id, step.target_id, step.action.value),)
    else:
        candidates = ()
    outcome = commit_batch(
        state,
        candidates,
        config,
        run.scenario.t_max_tick,
        tick=step.commit_tick,
    )
    assert outcome.state == state
    assert outcome.committed == ()
    gate_reason = outcome.gates[0].reason if outcome.gates else None
    reason_by_gate = {
        "horizon": RejectionReason.HORIZON,
        "range": RejectionReason.RANGE,
        "ammo": RejectionReason.AMMO,
        "target_locked": RejectionReason.TARGET_LOCKED,
        "busy_agent": RejectionReason.BUSY_AGENT,
    }
    if gate_reason is not None:
        reason = reason_by_gate[gate_reason]
    else:
        reason = {
            D0Fault.RELEASE_SUFFIX: RejectionReason.SUFFIX_RELEASED,
            D0Fault.FORCE_DEFER: RejectionReason.DEFERRED,
            D0Fault.ZERO_COMMIT: RejectionReason.ALLOCATION_STALL,
        }[step.fault]
    target_id = 0 if step.target_id is None else step.target_id
    agent_id = 0 if step.agent_id is None else step.agent_id
    truth = state.private_targets[target_id]
    return ExpectedPrivateRecord(
        PrivateRecordKind.REJECTION,
        0,
        step.commit_tick,
        target_id,
        agent_id,
        step.action,
        truth.true_category,
        truth.true_damage,
        truth.true_damage,
        None,
        None,
        0.0,
        expected.source_step_index,
        reason,
    )


def _logical_private_trace(run, result):
    records = []
    for event in result.private_audit_events:
        source = _source_step_index(
            run,
            tick=event.tick,
            target_id=event.target_id,
            agent_id=event.agent_id,
            mode=event.mode,
        )
        step = run.script[source]
        records.append(ExpectedPrivateRecord(
            PrivateRecordKind.COMPLETION,
            0,
            event.tick,
            event.target_id,
            event.agent_id,
            D0Action(event.mode),
            event.true_category,
            event.damage_before,
            event.damage_after,
            uniform01(step.counter_key),
            event.physical_success if event.mode == "attack" else None,
            event.realized_reward,
            source,
            None,
            event.realized_reward > 0.0,
        ))
    for expected in run.expected_private_audit_trace:
        if expected.kind is PrivateRecordKind.REJECTION:
            records.append(_logical_rejection(run, expected))
    if any(item.kind is PrivateRecordKind.TERMINATION for item in run.expected_private_audit_trace):
        truth = run.scenario.private_targets[0]
        records.append(ExpectedPrivateRecord(
            PrivateRecordKind.TERMINATION,
            0,
            quantize_tick(result.record.makespan, DynamicConfig().tick_size),
            0,
            0,
            None,
            truth.true_category,
            truth.true_damage,
            truth.true_damage,
            None,
            None,
            0.0,
        ))
    rank = {
        PrivateRecordKind.COMPLETION: 0,
        PrivateRecordKind.REJECTION: 1,
        PrivateRecordKind.ALLOCATION: 1,
        PrivateRecordKind.SCHEDULER: 2,
        PrivateRecordKind.REPLAY: 2,
        PrivateRecordKind.TERMINATION: 3,
    }
    return tuple(
        replace(record, event_id=index)
        for index, record in enumerate(sorted(records, key=lambda item: (item.tick, rank[item.kind])))
    )


def _consume_absences_once(run, result, policy, logical_public, logical_private):
    consumed = []
    for absence in run.expected_absences:
        step = run.script[absence.step_index]
        probe_state = _state_before_step(run, absence.step_index)
        matching_active = tuple(
            agent.active_action
            for agent in probe_state.agents
            if agent.active_action is not None
            and agent.active_action.commit_tick == step.commit_tick
            and (step.agent_id is None or agent.active_action.agent_id == step.agent_id)
            and (step.target_id is None or agent.active_action.target_id == step.target_id)
            and (step.action is None or agent.active_action.mode == step.action.value)
        )
        matching_actions = tuple(
            action for action in result.actions
            if action.commit_tick == step.commit_tick
            and (step.agent_id is None or action.agent_id == step.agent_id)
            and (step.target_id is None or action.target_id == step.target_id)
            and (step.action is None or action.mode == step.action.value)
        )
        matching_completions = tuple(
            event for event in result.private_audit_events
            if (step.agent_id is None or event.agent_id == step.agent_id)
            and (step.target_id is None or event.target_id == step.target_id)
            and (step.action is None or event.mode == step.action.value)
            and any(
                action in matching_actions and action.finish_tick == event.tick
                for action in result.actions
            )
        )
        if absence.kind is AbsenceKind.NO_COMMIT:
            assert not matching_actions and not matching_active
            assert all(
                record.kind not in (PublicRecordKind.OBSERVATION, PublicRecordKind.ATTACK_ACK)
                or record.source_step_index != absence.step_index
                for record in logical_public
            )
        elif absence.kind is AbsenceKind.NO_COMPLETION:
            assert not matching_actions and not matching_active and not matching_completions
            assert all(
                record.kind is not PrivateRecordKind.COMPLETION
                or record.source_step_index != absence.step_index
                for record in logical_private
            )
            assert all(
                record.kind not in (PublicRecordKind.OBSERVATION, PublicRecordKind.ATTACK_ACK)
                or record.source_step_index != absence.step_index
                for record in logical_public
            )
        elif absence.kind is AbsenceKind.NO_COUNTER_READ:
            assert not matching_actions and not matching_completions
            assert step.counter_key is None and step.expected_uniform is None
            assert all(
                record.kind is not PrivateRecordKind.COMPLETION
                or record.source_step_index != absence.step_index
                for record in logical_private
            )
        elif absence.kind is AbsenceKind.NO_EARLY_TERMINATION:
            calls = tuple(call for call in policy.calls if call[0].tick == step.commit_tick)
            assert len(calls) == 1
            snapshot = calls[0][0]
            busy_finishes = tuple(
                agent.busy_action.finish_tick
                for agent in snapshot.agents
                if agent.busy_action is not None
            )
            assert busy_finishes
            terminal_tick = quantize_tick(result.record.makespan, DynamicConfig().tick_size)
            assert terminal_tick >= min(busy_finishes)
        else:
            raise AssertionError(f"unknown canonical absence kind: {absence.kind!r}")
        consumed.append((absence.kind, absence.step_index))
    assert len(consumed) == len(run.expected_absences)
    assert len(set(consumed)) == len(consumed)


@pytest.mark.parametrize(
    "fixture_name,run",
    _D0_RUNS,
    ids=[f"typed-{name}-{run.run_id}" for name, run in _D0_RUNS],
)
def test_all_canonical_typed_logical_records_and_absences(fixture_name, run):
    if fixture_name == "b1m_frozen_suffix_auto_next":
        assert run_d0_witness(fixture_name)["status"] == "passed"
        return
    grids = tuple(sorted(
        {step.commit_tick for step in run.script if step.commit_tick > 0}
        | {
            event.tick
            for event in run.expected_public_trace
            if event.kind is PublicRecordKind.SCHEDULER and event.tick > 0
        }
    ))
    policy = TickScriptedPolicy(run)
    result = run_episode(
        run.scenario, policy, method=run.method_id, planning_grid_ticks=grids
    )
    actual_public = _logical_public_trace(run, result, policy)
    actual_private = _logical_private_trace(run, result)
    assert actual_public == run.expected_public_trace
    assert actual_private == run.expected_private_audit_trace
    assert len(actual_public) == len(run.expected_public_trace)
    assert len(actual_private) == len(run.expected_private_audit_trace)
    _consume_absences_once(run, result, policy, actual_public, actual_private)
