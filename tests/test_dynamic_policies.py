"""Public-only lifecycle policy contracts and matched-baseline witnesses."""

from __future__ import annotations

from dataclasses import replace
import inspect
import json

import pytest

from uav_lifecycle.dynamic_planning import PlannedPath, PlanningResult
from uav_lifecycle.dynamic_scenarios import d0_scenarios
from uav_lifecycle.dynamic_simulator import (
    CommitBatchResult,
    commit_batch,
    complete_event_batch,
    initialize_state,
    run_episode,
)
from uav_lifecycle.dynamic_types import (
    DynamicConfig,
    DynamicScenario,
    GateFailure,
    InternalAgentState,
    PlanningClock,
    PrivateTarget,
    PublicActionAck,
    PublicAgent,
    PublicObservation,
    PublicSnapshot,
    PublicTarget,
    ResourceLedger,
    quantize_tick,
)
from uav_lifecycle.mode_allocation import Mode
from uav_lifecycle.dynamic_policies import (
    ActionProposal,
    AttackOnlyPolicy,
    DynamicPPolicy,
    DynamicVanillaPolicy,
    ExactMyopicPolicy,
    FixedOrderPolicy,
    NearestPositivePolicy,
    NoBDAPolicy,
    OneShotMatchedPolicy,
    PeriodicPolicy,
    PolicyDecision,
    StandardOneShotCBBAPolicy,
    make_policy,
)


_SHARED = (
    "public_api", "resources", "durations", "bayes_attack", "nonpreemption",
    "target_lock", "horizon", "crn", "evaluator", "completion_before_planning",
    "deterministic_ties", "one_target_one_commit",
)


@pytest.mark.parametrize(
    ("method", "kind", "modes", "generator", "planner", "clock", "execution"),
    [
        ("P", DynamicPPolicy, ("recon", "attack", "bda", "defer"), "gated_screening", "johnson_warped", PlanningClock.EVENT_DRIVEN, "commit_next"),
        ("B1m", OneShotMatchedPolicy, ("recon", "attack", "bda", "defer"), "gated_screening", "johnson_warped", PlanningClock.B1M_ONE_SHOT, "frozen_paths_auto_next"),
        ("B2", FixedOrderPolicy, ("fixed_phase", "defer"), "current_phase", "johnson_warped", PlanningClock.EVENT_DRIVEN, "commit_next"),
        ("B3", AttackOnlyPolicy, ("attack", "defer"), "attack_only", "johnson_warped", PlanningClock.EVENT_DRIVEN, "commit_next"),
        ("B4", NoBDAPolicy, ("recon", "attack", "defer"), "no_bda_screening", "johnson_warped", PlanningClock.EVENT_DRIVEN, "commit_next"),
        ("B5(2)", PeriodicPolicy, ("recon", "attack", "bda", "defer"), "gated_screening", "johnson_warped", PlanningClock.PERIODIC, "commit_next"),
        ("B5(4)", PeriodicPolicy, ("recon", "attack", "bda", "defer"), "gated_screening", "johnson_warped", PlanningClock.PERIODIC, "commit_next"),
        ("B5(8)", PeriodicPolicy, ("recon", "attack", "bda", "defer"), "gated_screening", "johnson_warped", PlanningClock.PERIODIC, "commit_next"),
        ("B6", NearestPositivePolicy, ("recon", "attack", "bda", "defer"), "gated_screening", "nearest_positive", PlanningClock.EVENT_DRIVEN, "one_task_per_agent"),
        ("CEX", ExactMyopicPolicy, ("recon", "attack", "bda", "defer"), "all_mode_exact", "centralized_exact", PlanningClock.EVENT_DRIVEN, "commit_next"),
        ("DVCBBA", DynamicVanillaPolicy, ("recon", "attack", "bda", "defer"), "gated_screening", "vanilla_raw", PlanningClock.EVENT_DRIVEN, "commit_next"),
        ("SCBBA", StandardOneShotCBBAPolicy, ("recon", "attack", "bda", "defer"), "t0_fixed_screened_tasks", "vanilla_raw", PlanningClock.B1M_ONE_SHOT, "frozen_paths_auto_next"),
    ],
)
def test_exact_method_matrix(method, kind, modes, generator, planner, clock, execution):
    policy = make_policy(method, DynamicConfig())
    assert isinstance(policy, kind)
    assert policy.contract.allowed_modes == modes
    assert policy.contract.task_generation == generator
    assert policy.contract.planner == planner
    assert policy.contract.planning_clock is clock
    assert policy.contract.execution_rule == execution
    assert policy.contract.shared == _SHARED


def _snapshot(*, tick=0, beliefs=None, events=()):
    beliefs = beliefs or ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))
    observations = tuple(event for event in events if isinstance(event, PublicObservation))
    acks = tuple(event for event in events if isinstance(event, PublicActionAck))
    return PublicSnapshot(
        tick,
        tuple(PublicTarget(i, (float(i + 1), 0.0), belief) for i, belief in enumerate(beliefs)),
        (PublicAgent(0, (0.0, 0.0), 4.0, 40.0, None),),
        (), observations, acks,
    )


def _bind(policy, seconds=30.0):
    policy.bind_horizon(quantize_tick(seconds, DynamicConfig().tick_size))
    return policy


def test_p_and_b1m_tick_zero_share_one_planning_function_and_identical_bytes(monkeypatch):
    import uav_lifecycle.dynamic_policies as policies

    calls = []
    real = policies.plan_johnson

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(policies, "plan_johnson", counted)
    snapshot = _snapshot()
    p = _bind(DynamicPPolicy(DynamicConfig()))
    b1m = _bind(OneShotMatchedPolicy(DynamicConfig()))
    p_decision = p.decide(snapshot)
    b_decision = b1m.decide(snapshot)
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert p_decision.planning_bytes == b_decision.planning_bytes
    assert p_decision.planned_paths == b_decision.planned_paths
    assert json.loads(p_decision.planning_bytes)["method"] == "johnson"


def test_policy_decision_is_public_proposal_not_action_commit():
    decision = _bind(DynamicPPolicy(DynamicConfig())).decide(_snapshot())
    assert isinstance(decision, PolicyDecision)
    assert all(isinstance(item, ActionProposal) for item in decision.proposals)
    assert all(type(item).__name__ != "ActionCommit" for item in decision.proposals)
    targets = [item.target_id for item in decision.proposals]
    assert len(targets) == len(set(targets))


def test_p_replans_after_public_completion_and_b1m_never_reads_later_belief(monkeypatch):
    import uav_lifecycle.dynamic_policies as policies

    seen = []
    real = policies.plan_johnson

    def recording(snapshot, *args, **kwargs):
        seen.append(snapshot)
        return real(snapshot, *args, **kwargs)

    monkeypatch.setattr(policies, "plan_johnson", recording)
    p = _bind(DynamicPPolicy(DynamicConfig()))
    b1m = _bind(OneShotMatchedPolicy(DynamicConfig()))
    first = _snapshot()
    p.decide(first)
    b1m.decide(first)
    changed = _snapshot(
        tick=1,
        beliefs=((0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        events=(PublicActionAck(0, 1, 0, 0, "attack"),),
    )
    p.decide(changed)
    with pytest.raises(RuntimeError, match="tick 0"):
        b1m.decide(changed)
    assert seen == [first, first, changed]


def test_b1m_suffix_auto_next_uses_no_planner_and_only_active_leg(monkeypatch):
    import uav_lifecycle.dynamic_policies as policies

    config = DynamicConfig()
    policy = _bind(OneShotMatchedPolicy(config), 20.0)
    path = PlannedPath(
        0,
        ((0, Mode.BDA), (1, Mode.ATTACK)),
        5.0,
        (quantize_tick(1.0, config.tick_size), quantize_tick(3.5, config.tick_size)),
        (quantize_tick(2.5, config.tick_size), quantize_tick(5.5, config.tick_size)),
    )
    monkeypatch.setattr(policies, "plan_johnson", lambda *args: PlanningResult("johnson", "valid", (path,), 2))
    first = policy.decide(_snapshot())
    assert first.proposals[0].target_id == 0
    assert policy.has_pending_suffix()
    monkeypatch.setattr(policies, "plan_johnson", lambda *args: pytest.fail("B1m replanned"))
    later = _snapshot(tick=path.completion_ticks[0], beliefs=((0, 1, 0, 0), (0, 0, 0, 1)))
    next_decision = policy.auto_next(later, (0,))
    assert tuple((p.agent_id, p.target_id, p.mode) for p in next_decision.proposals) == ((0, 1, "attack"),)
    assert not policy.has_pending_suffix()
    assert policy.positive_pair_count(later) == 0


def test_fixed_order_lifecycle_including_optional_terminal_attack(monkeypatch):
    import uav_lifecycle.dynamic_policies as policies

    policy = _bind(FixedOrderPolicy(DynamicConfig()))
    chosen = []

    def fake(snapshot, config, max_tick, *, allowed_modes=None):
        mode = allowed_modes[0][0]
        chosen.append(mode)
        path = PlannedPath(0, ((0, mode),), 1.0, (snapshot.tick,), (snapshot.tick + 1,))
        return PlanningResult("johnson", "valid", (path,), 1)

    monkeypatch.setattr(policies, "plan_johnson", fake)
    snap = _snapshot(beliefs=((1, 0, 0, 0),))
    policy.decide(snap)
    policy.decide(replace(snap, tick=1, observations=(PublicObservation(0, 1, 0, 0, "recon", ("H", "A")),)))
    policy.decide(replace(snap, tick=2, acknowledgements=(PublicActionAck(1, 2, 0, 0, "attack"),)))
    policy.decide(replace(snap, tick=3, observations=(PublicObservation(2, 3, 0, 0, "bda", "A"),)))
    policy.decide(replace(snap, tick=4, acknowledgements=(PublicActionAck(3, 4, 0, 0, "attack"),)))
    assert chosen == [Mode.RECON, Mode.ATTACK, Mode.BDA, Mode.ATTACK]
    assert policy.phase(0) == "done"


def test_fixed_order_terminal_stop_marks_done_when_attack_is_not_positive():
    policy = _bind(FixedOrderPolicy(DynamicConfig()))
    events = (
        PublicObservation(0, 1, 0, 0, "recon", ("H", "D")),
        PublicActionAck(1, 2, 0, 0, "attack"),
        PublicObservation(2, 3, 0, 0, "bda", "D"),
    )
    snapshot = _snapshot(
        tick=3, beliefs=((0.0, 1.0, 0.0, 0.0),), events=events
    )
    decision = policy.decide(snapshot)
    assert decision.proposals == ()
    assert policy.phase(0) == "done"


@pytest.mark.parametrize(
    ("policy_type", "expected"),
    [(AttackOnlyPolicy, {Mode.ATTACK}), (NoBDAPolicy, {Mode.RECON, Mode.ATTACK})],
)
def test_mode_ablation_filters_are_applied_inside_frozen_planning(monkeypatch, policy_type, expected):
    import uav_lifecycle.dynamic_policies as policies

    captured = []

    def fake(snapshot, config, max_tick, *, allowed_modes=None):
        captured.append(allowed_modes)
        return PlanningResult("johnson", "valid", (), 0)

    monkeypatch.setattr(policies, "plan_johnson", fake)
    _bind(policy_type(DynamicConfig())).decide(_snapshot())
    assert all(set(modes) == expected for modes in captured[0].values())


def test_periodic_grids_are_absolute_only():
    config = DynamicConfig()
    policy = _bind(make_policy("B5(4)", config), 20.0)
    delta = quantize_tick(4.0, config.tick_size)
    assert policy.planning_grid_ticks() == tuple(range(0, quantize_tick(20.0, config.tick_size) + 1, delta))
    off_grid = replace(_snapshot(), tick=quantize_tick(3.0, config.tick_size))
    assert policy.decide(off_grid).proposals == ()


def test_periodic_plans_unconditionally_at_each_absolute_grid(monkeypatch):
    import uav_lifecycle.dynamic_policies as policies

    config = DynamicConfig()
    policy = _bind(PeriodicPolicy(config, 4), 20.0)
    calls = []
    real = policies.plan_johnson

    def counted(snapshot, *args, **kwargs):
        calls.append(snapshot.tick)
        return real(snapshot, *args, **kwargs)

    monkeypatch.setattr(policies, "plan_johnson", counted)
    at_zero = policy.decide(_snapshot())
    at_four = policy.decide(
        replace(_snapshot(), tick=quantize_tick(4.0, config.tick_size))
    )
    off_grid = policy.decide(
        replace(_snapshot(), tick=quantize_tick(5.0, config.tick_size))
    )
    assert calls == [0, quantize_tick(4.0, config.tick_size)]
    assert at_zero.planning_bytes and at_four.planning_bytes
    assert at_zero.planning_bytes != at_four.planning_bytes
    assert off_grid.planning_bytes == b""


@pytest.mark.parametrize(
    ("belief", "mode"),
    [
        ((0.0, 0.0, 0.3, 0.7), "recon"),
        ((0.1, 0.0, 0.2, 0.7), "bda"),
        ((0.0, 0.0, 0.0, 1.0), None),
    ],
)
def test_p_generates_optional_sensing_or_defer_from_public_positive_tasks(belief, mode):
    snapshot = _snapshot(beliefs=(belief,))
    decision = _bind(DynamicPPolicy(DynamicConfig())).decide(snapshot)
    assert (decision.proposals[0].mode if decision.proposals else None) == mode


def test_b1m_simulator_auto_starts_suffix_at_completion_without_false_termination(monkeypatch):
    import uav_lifecycle.dynamic_policies as policies

    fixture = next(item for item in d0_scenarios() if item.name == "commit_next_releases_suffix")
    scenario = replace(fixture.scenario, agents=fixture.scenario.agents[:1])
    config = DynamicConfig()
    real = policies.plan_johnson(
        initialize_state(scenario).snapshot(), config, scenario.t_max_tick
    )
    multi = next(path for path in real.paths if len(path.tasks) > 1)
    monkeypatch.setattr(policies, "plan_johnson", lambda *args, **kwargs: PlanningResult("johnson", "valid", (multi,), 1))
    result = run_episode(scenario, OneShotMatchedPolicy(config), config=config, method="B1m")
    assert len(result.actions) == len(multi.tasks)
    assert tuple(action.commit_tick for action in result.actions[1:]) == tuple(action.finish_tick for action in result.actions[:-1])
    assert result.record.termination != "no_positive"


def test_b1m_suffix_has_no_early_lock_reservation_ordinal_or_event(monkeypatch):
    import uav_lifecycle.dynamic_policies as policies

    fixture = next(item for item in d0_scenarios() if item.name == "commit_next_releases_suffix")
    scenario = replace(fixture.scenario, agents=fixture.scenario.agents[:1])
    config = DynamicConfig()
    initial = initialize_state(scenario)
    real = policies.plan_johnson(initial.snapshot(), config, scenario.t_max_tick)
    path = next(item for item in real.paths if len(item.tasks) > 1)
    monkeypatch.setattr(
        policies,
        "plan_johnson",
        lambda *args, **kwargs: PlanningResult("johnson", "valid", (path,), 1),
    )
    policy = OneShotMatchedPolicy(config)
    policy.bind_horizon(scenario.t_max_tick)
    decision = policy.decide(initial.snapshot())
    first = commit_batch(
        initial,
        tuple(proposal.commit_candidate() for proposal in decision.proposals),
        config,
        scenario.t_max_tick,
    )
    first_target, first_mode = path.tasks[0]
    suffix_target, suffix_mode = path.tasks[1]
    assert first.state.target_locks == ((first_target, path.agent_id),)
    assert len(first.state.completion_events) == len(first.state.actions) == 1
    assert first.state.ordinals == ((first_target, first_mode.value, 1),)
    assert suffix_target not in dict(first.state.target_locks)
    assert all(action.target_id != suffix_target for action in first.state.actions)
    assert first.state.agents[path.agent_id].active_action.target_id == first_target
    assert first.state.agents[path.agent_id].ammo.reserved == (1.0 if first_mode is Mode.ATTACK else 0.0)
    assert (
        first.state.agents[path.agent_id].distance.reserved
        == first.committed[0].reserved_distance
    )

    completed = complete_event_batch(
        first.state, first.state.completion_events, config
    ).state
    next_decision = policy.auto_next(completed.snapshot(), (path.agent_id,))
    second = commit_batch(
        completed,
        tuple(proposal.commit_candidate() for proposal in next_decision.proposals),
        config,
        scenario.t_max_tick,
    )
    assert second.gates == ()
    assert second.state.target_locks == ((suffix_target, path.agent_id),)
    assert len(second.state.completion_events) == 1
    assert (suffix_target, suffix_mode.value, 1) in second.state.ordinals
    assert second.state.agents[path.agent_id].active_action.target_id == suffix_target
    assert second.state.agents[path.agent_id].ammo.reserved == (1.0 if suffix_mode is Mode.ATTACK else 0.0)
    assert (
        second.state.agents[path.agent_id].distance.reserved
        == second.committed[0].reserved_distance
    )


def test_b1m_suffix_commit_rejection_is_one_gate_without_fallback_or_partial_mutation(
    monkeypatch,
):
    import uav_lifecycle.dynamic_policies as policies
    import uav_lifecycle.dynamic_simulator as simulator

    fixture = next(item for item in d0_scenarios() if item.name == "commit_next_releases_suffix")
    scenario = replace(fixture.scenario, agents=fixture.scenario.agents[:1])
    config = DynamicConfig()
    planned = policies.plan_johnson(
        initialize_state(scenario).snapshot(), config, scenario.t_max_tick
    )
    path = next(item for item in planned.paths if len(item.tasks) > 1)
    planner_calls = 0

    def one_plan(*args, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        return PlanningResult("johnson", "valid", (path,), 1)

    monkeypatch.setattr(policies, "plan_johnson", one_plan)
    real_commit = simulator.commit_batch
    commit_calls = 0

    def reject_second(state, candidates, cfg, max_tick, **kwargs):
        nonlocal commit_calls
        commit_calls += 1
        if commit_calls == 2:
            return CommitBatchResult(
                state,
                (),
                (GateFailure("commit_batch", state.tick, "range"),),
            )
        return real_commit(state, candidates, cfg, max_tick, **kwargs)

    monkeypatch.setattr(simulator, "commit_batch", reject_second)
    result = run_episode(scenario, OneShotMatchedPolicy(config), config=config, method="B1m")
    assert planner_calls == 1
    assert commit_calls == 2
    assert result.gate_failures == (
        GateFailure(
            "b1m_suffix",
            path.completion_ticks[0],
            "infeasible",
            (("commit_reason", "range"),),
        ),
    )
    assert len(result.actions) == 1
    assert result.actions[0].target_id == path.tasks[0][0]


def test_p_supports_consecutive_attacks_and_cross_uav_sequential_handoff():
    config = DynamicConfig()

    def agent(agent_id, ammo):
        return InternalAgentState(
            agent_id,
            (0.0, 0.0),
            ResourceLedger(ammo, 0.0, 0.0),
            ResourceLedger(20.0, 0.0, 0.0),
            ammo,
            20.0,
            None,
        )

    scenario = DynamicScenario(
        "policy-handoff",
        "policy-handoff",
        1,
        "policy-handoff",
        (PublicTarget(0, (0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),),
        (PrivateTarget(0, "H", "A", False),),
        (agent(0, 1.0), agent(1, 3.0)),
        quantize_tick(15.0, config.tick_size),
    )
    result = run_episode(scenario, DynamicPPolicy(config), config=config, method="P")
    attacks = tuple(action for action in result.actions if action.mode == "attack")
    assert tuple(action.agent_id for action in attacks[:2]) == (0, 1)
    assert attacks[1].commit_tick == attacks[0].finish_tick
    assert any(
        left.agent_id == right.agent_id and left.finish_tick == right.commit_tick
        for left, right in zip(attacks[1:], attacks[2:])
    )
    assert result.record.handoff_count >= 1


def test_public_only_construction_and_calls_reject_private_or_rng_inputs():
    signature = inspect.signature(DynamicPPolicy)
    assert tuple(signature.parameters) == ("config",)
    with pytest.raises(TypeError):
        DynamicPPolicy(DynamicConfig(), private_truth=(PrivateTarget(0, "H", "A", False),))
    policy = _bind(DynamicPPolicy(DynamicConfig()))
    with pytest.raises(TypeError, match="PublicSnapshot"):
        policy.decide(object())


def test_unknown_method_and_unregistered_period_are_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        make_policy("B7", DynamicConfig())
    with pytest.raises(ValueError, match="2, 4, or 8"):
        make_policy("B5(3)", DynamicConfig())


def test_b1m_one_shot_clock_is_rejected_for_other_policies():
    class Impostor:
        method = "P"
        planning_clock = PlanningClock.B1M_ONE_SHOT

        def decide(self, snapshot):
            return ()

    scenario = next(
        item.scenario for item in d0_scenarios() if item.name == "no_positive_normal_termination"
    )
    with pytest.raises(TypeError, match="reserved for registered one-shot baselines"):
        run_episode(scenario, Impostor(), method="P")
