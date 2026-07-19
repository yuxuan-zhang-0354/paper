from __future__ import annotations

from random import Random
from time import perf_counter

from .belief import bda_kernel, recon_kernel
from .mode_allocation import Mode, ModeAgent, ModeInstance, ModeTask, mode_utilities
from .mode_cbba import ModeMethod, run_mode_cbba, screen_modes, validate_mode_result
from .mode_exact import solve_all_mode_exact, solve_fixed_mode_exact
from .mode_fallback import run_ranked_fallback
from .scenarios import validation_parameter_sets


_STRATIFIED_CONFIG_ID = "recon_damage_plus_010_r2_a6_b3"
_STRATIFIED_BELIEFS = (
    ("recon", (0.0, 0.42, 0.24, 0.34)),
    ("attack", (1.0, 0.0, 0.0, 0.0)),
    ("bda", (0.26, 0.66, 0.0, 0.08)),
    ("defer", (0.0, 0.08, 0.06, 0.86)),
)


def _task(target: int, mode: Mode, x: float, utility: float, ammo: int = 0, duration: float = 0) -> ModeTask:
    return ModeTask(target, mode, (x, 0.0), duration, ammo, utility)


def _instance(
    name: str,
    agents: tuple[ModeAgent, ...],
    groups: tuple[tuple[ModeTask, ...], ...],
    *,
    distance_cost: float = 0.0,
    ammo_cost: float = 0.0,
) -> ModeInstance:
    return ModeInstance(agents, groups, 0.0, distance_cost, ammo_cost, name)


def tier0_mode_instances() -> tuple[ModeInstance, ...]:
    roomy = (ModeAgent(0, (0.0, 0.0), 20.0, 20.0, 2),)
    return (
        _instance("recon_selected", roomy, ((_task(0, Mode.RECON, 1, 10), _task(0, Mode.ATTACK, 1, 5, 1), _task(0, Mode.BDA, 1, 3)),)),
        _instance("attack_selected", roomy, ((_task(0, Mode.RECON, 1, 4), _task(0, Mode.ATTACK, 1, 12, 1), _task(0, Mode.BDA, 1, 6)),)),
        _instance("bda_selected", roomy, ((_task(0, Mode.RECON, 1, 5), _task(0, Mode.ATTACK, 1, 4, 1), _task(0, Mode.BDA, 1, 11)),)),
        _instance("defer", roomy, ((_task(0, Mode.RECON, 1, -1), _task(0, Mode.ATTACK, 1, -2, 1), _task(0, Mode.BDA, 1, 0)),)),
        _instance(
            "shared_witness_ammo",
            (ModeAgent(0, (0, 0), 10, 10, 1),),
            ((_task(0, Mode.ATTACK, 0, 10, 1),), (_task(1, Mode.ATTACK, 0, 9, 1),)),
        ),
        _instance(
            "horizon_conflict",
            (ModeAgent(0, (0, 0), 6, 10, 0),),
            ((_task(0, Mode.RECON, 0, 10, duration=6),), (_task(1, Mode.BDA, 0, 9, duration=6),)),
        ),
        _instance(
            "range_conflict",
            (ModeAgent(0, (0, 0), 30, 5, 0),),
            ((_task(0, Mode.RECON, 5, 10),), (_task(1, Mode.BDA, -5, 9),)),
        ),
        _instance(
            "mode_substitution",
            (ModeAgent(0, (0, 0), 10, 10, 1),),
            (
                (_task(0, Mode.RECON, 0, 8), _task(0, Mode.ATTACK, 0, 10, 1), _task(0, Mode.BDA, 0, 1)),
                (_task(1, Mode.RECON, 0, -1), _task(1, Mode.ATTACK, 0, 100, 1), _task(1, Mode.BDA, 0, -1)),
            ),
        ),
    )


def random_mode_instance(
    n_agents: int,
    n_targets: int,
    seed: int,
    ammo_tightness: str,
    horizon_tightness: str,
    continuation: str,
    belief_profile: str = "uniform",
) -> ModeInstance:
    if n_agents not in (2, 3, 4) or n_targets not in (3, 4, 5):
        raise ValueError("random mode cells require N in {2,3,4}, M in {3,4,5}")
    levels = {"tight": 0, "medium": 1, "loose": 2}
    if ammo_tightness not in levels or horizon_tightness not in levels:
        raise ValueError("tightness must be tight, medium, or loose")
    if continuation not in {"optimistic", "no_continuation", "ammo_reachability_gate"}:
        raise ValueError("unknown continuation")
    if belief_profile not in {"uniform", "stratified"}:
        raise ValueError("belief profile must be uniform or stratified")
    variant_rank = {"optimistic": 0, "no_continuation": 1, "ammo_reachability_gate": 2}[continuation]
    rng = Random(
        n_agents * 10**8
        + n_targets * 10**6
        + seed * 100
        + levels[ammo_tightness] * 10
        + levels[horizon_tightness] * 3
        + variant_rank
    )
    ammo_by_level = {
        "tight": max(0, n_targets // (2 * n_agents)),
        "medium": max(1, n_targets // n_agents),
        "loose": n_targets,
    }
    horizon_by_level = {"tight": 10.0, "medium": 20.0, "loose": 35.0}
    agents = tuple(
        ModeAgent(
            i,
            (float(rng.randint(-6, 6)), float(rng.randint(-6, 6))),
            horizon_by_level[horizon_tightness],
            {"tight": 12.0, "medium": 24.0, "loose": 40.0}[horizon_tightness],
            ammo_by_level[ammo_tightness],
        )
        for i in range(n_agents)
    )
    configs = validation_parameter_sets()
    if belief_profile == "stratified":
        config = next(config for config in configs if config.config_id == _STRATIFIED_CONFIG_ID)
        archetypes = list(_STRATIFIED_BELIEFS)
        rng.shuffle(archetypes)
    else:
        config = configs[(seed + 17 * variant_rank) % len(configs)]
        archetypes = []
    zr = recon_kernel(config.recon_class_matrix, config.recon_damage_matrix)
    zb = bda_kernel(config.bda_damage_matrix)
    groups: list[tuple[ModeTask, ...]] = []
    for target_id in range(n_targets):
        position = (float(rng.randint(-6, 6)), float(rng.randint(-6, 6)))
        if belief_profile == "stratified":
            _, belief = archetypes[target_id % len(archetypes)]
        else:
            raw = [rng.random() + 0.05 for _ in range(4)]
            total = sum(raw)
            belief = tuple(value / total for value in raw)
        gates = {}
        for mode, duration, ammo in (
            (Mode.RECON, config.params.duration_r, 0),
            (Mode.ATTACK, config.params.duration_a, 1),
            (Mode.BDA, config.params.duration_b, 0),
        ):
            gates[mode] = any(
                agent.ammo >= ammo + 1
                and ((agent.origin[0] - position[0]) ** 2 + (agent.origin[1] - position[1]) ** 2) ** 0.5 <= agent.distance_budget
                and ((agent.origin[0] - position[0]) ** 2 + (agent.origin[1] - position[1]) ** 2) ** 0.5 + duration + config.params.duration_a <= agent.horizon
                for agent in agents
            )
        utilities = mode_utilities(belief, zr, zb, config.params, continuation, gates)
        groups.append(
            tuple(
                ModeTask(
                    target_id,
                    mode,
                    position,
                    {Mode.RECON: config.params.duration_r, Mode.ATTACK: config.params.duration_a, Mode.BDA: config.params.duration_b}[mode],
                    int(mode is Mode.ATTACK),
                    utilities[mode],
                )
                for mode in Mode
            )
        )
    return ModeInstance(
        agents,
        tuple(groups),
        config.params.beta,
        0.10,
        0.50,
        f"R-N{n_agents}-M{n_targets}-A{ammo_tightness}-H{horizon_tightness}-{continuation}-P{belief_profile}-S{seed}",
        continuation,
    )


def evaluate_mode_instance(instance: ModeInstance) -> dict[str, object]:
    screened = screen_modes(instance)
    screened_modes = {item.target_id: item.mode for item in screened if item is not None}
    all_mode = solve_all_mode_exact(instance)
    fixed_mode = solve_fixed_mode_exact(instance, screened_modes)
    full_raw = run_mode_cbba(instance, screened, ModeMethod.FULL_REBUILD_RAW)
    johnson = run_mode_cbba(instance, screened, ModeMethod.JOHNSON_WARPED)
    full_report = validate_mode_result(instance, screened, full_raw)
    johnson_report = validate_mode_result(instance, screened, johnson)
    decomposition_valid = (
        full_raw.status == "converged"
        and johnson.status == "converged"
        and sum(full_report.values()) == 0
        and sum(johnson_report.values()) == 0
    )
    substitutions = sum(
        item is not None
        and all_mode.target_modes[target_id] is not None
        and item.mode is not all_mode.target_modes[target_id]
        for target_id, item in enumerate(screened)
    )
    return {
        "instance_id": instance.instance_id,
        "continuation": instance.continuation,
        "screened_task_count": sum(item is not None for item in screened),
        "screened_modes": [None if item is None else item.mode.value for item in screened],
        "screened_witnesses": [None if item is None else item.witness_agent for item in screened],
        "central_modes": [None if mode is None else mode.value for mode in all_mode.target_modes],
        "all_mode_score": all_mode.score,
        "fixed_mode_score": fixed_mode.score,
        "full_raw_score": full_raw.true_score,
        "johnson_score": johnson.true_score,
        "screening_loss": all_mode.score - fixed_mode.score,
        "allocation_loss": fixed_mode.score - full_raw.true_score,
        "warping_loss": full_raw.true_score - johnson.true_score,
        "decomposition_valid": decomposition_valid,
        "full_raw_ratio": full_raw.true_score / all_mode.score if all_mode.score > 0 else 1.0,
        "johnson_ratio": johnson.true_score / all_mode.score if all_mode.score > 0 else 1.0,
        "mode_substitutions": substitutions,
        "orphan_count": len(johnson.orphan_targets),
        "orphan_targets": list(johnson.orphan_targets),
        "full_raw_status": full_raw.status,
        "johnson_status": johnson.status,
        "full_raw_rounds": full_raw.rounds,
        "johnson_rounds": johnson.rounds,
        "full_raw_gate_failures": sum(full_report.values()),
        "johnson_gate_failures": sum(johnson_report.values()),
        **{f"full_raw_{key}": value for key, value in full_report.items()},
        **{f"johnson_{key}": value for key, value in johnson_report.items()},
    }


def evaluate_fallback_instance(instance: ModeInstance) -> dict[str, object]:
    """Evaluate ranked fallback as a paired extension of the frozen base record."""
    record = evaluate_mode_instance(instance)
    started = perf_counter()
    result = run_ranked_fallback(instance)
    elapsed = perf_counter() - started

    gate_telemetry = fallback_gate_telemetry(result)
    attempted_iterations = (
        result.iterations + result.base_gate_failures + result.late_gate_failures
    )
    base_target_count = len(result.base_targets) if result.valid else 0
    base_orphan_count = len(result.base_orphans) if result.valid else 0
    unresolved_count = len(result.fallback_unresolved) if result.valid else 0
    resolved_count = len(result.resolved_targets) if result.valid else 0
    newly_unassigned_count = len(result.newly_unassigned) if result.valid else 0
    selected = next(
        (
            iteration
            for iteration in result.iterations
            if iteration.index == result.selected_iteration
        ),
        None,
    )
    fallback_score = selected.result.true_score if selected is not None else None
    base_score = float(record["johnson_score"])

    record.update(
        {
            "fallback_valid": result.valid,
            "fallback_score": fallback_score,
            "fallback_gain": (
                fallback_score - base_score if fallback_score is not None else None
            ),
            "fallback_ratio": (
                fallback_score / float(record["all_mode_score"])
                if fallback_score is not None and float(record["all_mode_score"]) > 0
                else (1.0 if fallback_score is not None else None)
            ),
            "fallback_regret": (
                float(record["all_mode_score"]) - fallback_score
                if fallback_score is not None
                else None
            ),
            "fallback_base_target_count": base_target_count,
            "fallback_iteration_count": len(result.iterations),
            "fallback_total_johnson_calls": result.total_johnson_calls,
            "fallback_johnson_rounds": sum(
                iteration.result.rounds for iteration in attempted_iterations
            ),
            **gate_telemetry,
            "fallback_unresolved_count": unresolved_count,
            "fallback_unresolved_rate": (
                unresolved_count / base_target_count if base_target_count else None
            ),
            "fallback_unresolved_targets": list(result.fallback_unresolved),
            "fallback_wall_clock_seconds": elapsed,
            "base_orphan_count": base_orphan_count,
            "base_orphan_rate": (
                base_orphan_count / base_target_count if base_target_count else None
            ),
            "base_orphan_targets": list(result.base_orphans),
            "resolved_count": resolved_count,
            "resolved_rate": (
                resolved_count / base_orphan_count if base_orphan_count else None
            ),
            "resolved_targets": list(result.resolved_targets),
            "newly_unassigned_count": newly_unassigned_count,
            "newly_unassigned_rate": (
                newly_unassigned_count / base_target_count
                if base_target_count
                else None
            ),
            "newly_unassigned_targets": list(result.newly_unassigned),
            "selected_iteration": result.selected_iteration,
            "selected_modes": [
                None if mode is None else mode.value
                for mode in result.selected_assigned_modes
            ],
            "selected_mode_switches": len(result.selected_switches),
            "selected_switched_targets": list(result.selected_switches),
            "selected_defer_count": len(result.selected_defers),
            "selected_defer_targets": list(result.selected_defers),
            "search_advances": result.search_advances,
            "search_exhausted_count": len(result.search_exhausted_targets),
            "search_exhausted_targets": list(result.search_exhausted_targets),
        }
    )
    return record


def fallback_gate_telemetry(result) -> dict[str, int]:
    """Return aggregate and per-validation-key counts for attempted runs."""

    attempted = result.iterations + result.base_gate_failures + result.late_gate_failures
    base = result.base_gate_failures or result.iterations[:1]
    late = result.iterations[1:] + result.late_gate_failures
    keys = tuple(key for key, _ in attempted[0].gate_report)

    def totals(iterations):
        return {
            key: sum(dict(iteration.gate_report)[key] for iteration in iterations)
            for key in keys
        }

    all_counts = totals(attempted)
    base_counts = totals(base)
    late_counts = totals(late)
    return {
        "fallback_gate_failures": sum(all_counts.values()),
        "fallback_base_gate_failures": sum(base_counts.values()),
        "fallback_late_gate_failures": sum(late_counts.values()),
        **{f"fallback_{key}": value for key, value in all_counts.items()},
        **{f"fallback_base_{key}": value for key, value in base_counts.items()},
        **{f"fallback_late_{key}": value for key, value in late_counts.items()},
    }
