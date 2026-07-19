from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import log
from random import Random

from .allocation_model import StaticAgent, StaticInstance, StaticTask
from .cbba_static import Method, build_bundle, run_static_cbba, validate_result
from .exact_allocation import solve_exact


@dataclass(frozen=True)
class CalibrationBase:
    base_id: str
    instance: StaticInstance
    target_task: int
    raw: float
    warped: float
    gap: float
    relative_gap: float


@dataclass(frozen=True)
class StressCase:
    case_id: str
    base: CalibrationBase
    rho: float
    slack: float
    competitor_bid: float
    competitor_distance: float
    instance: StaticInstance


@dataclass(frozen=True)
class SelectedStratum:
    key: tuple[str, str, str, str]
    count: int
    allocation_change_rate: float
    decisive_rate: float
    base_count: int


def detect_rises(instance: StaticInstance) -> tuple[CalibrationBase, ...]:
    if len(instance.agents) != 1:
        raise ValueError("rise detection requires exactly one primary agent")
    state = build_bundle(
        instance,
        agent_id=0,
        method=Method.JOHNSON_WARPED,
        winners=(None,) * len(instance.tasks),
    )
    rises: list[CalibrationBase] = []
    for entry in state.entries:
        gap = entry.raw_score - entry.external_bid
        if gap > 1e-9:
            rises.append(
                CalibrationBase(
                    base_id=f"{instance.instance_id}-T{entry.task_id}",
                    instance=instance,
                    target_task=entry.task_id,
                    raw=entry.raw_score,
                    warped=entry.external_bid,
                    gap=gap,
                    relative_gap=gap / entry.raw_score,
                )
            )
    return tuple(rises)


def construct_competition(
    base: CalibrationBase,
    rho: float,
    slack: float,
    instance_id: str,
) -> StressCase | None:
    if not 0.0 < rho < 1.0 or slack < 0.0:
        raise ValueError("rho must lie in (0,1) and slack must be nonnegative")
    primary = base.instance
    if primary.discount >= 1.0:
        return None
    target = primary.tasks[base.target_task]
    desired_bid = base.warped + rho * base.gap
    ratio = desired_bid / target.value
    if not 0.0 < ratio <= 1.0:
        return None
    distance = log(ratio) / log(primary.discount)
    competitor = StaticAgent(
        1,
        (target.position[0], target.position[1] + distance),
        distance + slack,
        1,
    )
    instance = StaticInstance(
        agents=(primary.agents[0], competitor),
        tasks=primary.tasks,
        discount=primary.discount,
        instance_id=instance_id,
    )
    competitor_state = build_bundle(
        instance,
        agent_id=1,
        method=Method.FULL_REBUILD_RAW,
        winners=(None,) * len(instance.tasks),
    )
    if competitor_state.bundle != (base.target_task,):
        return None
    return StressCase(instance_id, base, rho, slack, desired_bid, distance, instance)


def calibration_bases(limit: int | None = None):
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    yielded = 0
    grid = product(
        (-1.0, 0.0, 1.0),
        (-1.0, 0.0, 1.0),
        (0.96, 0.97, 0.98, 0.99),
        (40.0, 60.0, 80.0),
        (80.0, 100.0, 120.0),
        (40.0, 60.0, 80.0),
        (0.0, 1.0, 2.0),
        (24.0, 27.0, 30.0, 33.0, 36.0),
    )
    for index, (origin_dx, target_dy, q, v0, v1, v2, service1, deadline) in enumerate(grid):
        instance = StaticInstance(
            agents=(StaticAgent(0, (5.0 + origin_dx, 3.0), deadline, 3),),
            tasks=(
                StaticTask(0, (3.0, -1.0), 0.0, v0),
                StaticTask(1, (-10.0, 5.0), service1, v1),
                StaticTask(2, (7.0, 1.0 + target_dy), 0.0, v2),
            ),
            discount=q,
            instance_id=f"B-{index:05d}",
        )
        for rise in detect_rises(instance):
            yield rise
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def calibration_cases(limit: int | None = None):
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    yielded = 0
    for base in calibration_bases():
        for rho in (0.1, 0.3, 0.5, 0.7, 0.9):
            for slack in (0.05, 0.25, 0.75):
                case = construct_competition(
                    base,
                    rho,
                    slack,
                    f"CAL-{base.base_id}-R{int(rho * 10)}-S{int(slack * 100):02d}",
                )
                if case is None:
                    continue
                yield case
                yielded += 1
                if limit is not None and yielded >= limit:
                    return


def _winner_ids(result) -> tuple[int | None, ...]:
    return tuple(None if winner is None else winner.agent_id for winner in result.winners)


def evaluate_stress_case(case: StressCase) -> dict:
    exact = solve_exact(case.instance)
    full = run_static_cbba(case.instance, Method.FULL_REBUILD_RAW)
    johnson = run_static_cbba(case.instance, Method.JOHNSON_WARPED)
    report = validate_result(case.instance, johnson)
    gate_failures = sum(report.values()) + (johnson.status != "converged")
    full_winners = _winner_ids(full)
    johnson_winners = _winner_ids(johnson)
    full_paths = tuple(state.path for state in full.states)
    johnson_paths = tuple(state.path for state in johnson.states)
    target = case.base.target_task
    target_changed = full_winners[target] != johnson_winners[target]
    in_interval = case.base.warped < case.competitor_bid < case.base.raw
    delta_j = johnson.true_score - full.true_score
    predicted_delta = case.competitor_bid - case.base.raw
    return {
        "case_id": case.case_id,
        "base_id": case.base.base_id,
        "target_task": target,
        "discount": case.instance.discount,
        "primary_deadline": case.instance.agents[0].deadline,
        "values": tuple(task.value for task in case.instance.tasks),
        "service1": case.instance.tasks[1].service,
        "raw": case.base.raw,
        "warped": case.base.warped,
        "gap": case.base.gap,
        "relative_gap": case.base.relative_gap,
        "rho": case.rho,
        "slack": case.slack,
        "competitor_bid": case.competitor_bid,
        "competitor_distance": case.competitor_distance,
        "full_winners": full_winners,
        "johnson_winners": johnson_winners,
        "full_paths": full_paths,
        "johnson_paths": johnson_paths,
        "full_score": full.true_score,
        "johnson_score": johnson.true_score,
        "exact_score": exact.score,
        "full_ratio": full.true_score / exact.score,
        "johnson_ratio": johnson.true_score / exact.score,
        "delta_j": delta_j,
        "predicted_delta_j": predicted_delta,
        "delta_identity_error": delta_j - predicted_delta,
        "allocation_changed": full_winners != johnson_winners or full_paths != johnson_paths,
        "target_winner_changed": target_changed,
        "warping_decisive": in_interval and target_changed,
        "full_status": full.status,
        "johnson_status": johnson.status,
        "full_rounds": full.rounds,
        "johnson_rounds": johnson.rounds,
        "johnson_gate_e_failures": int(gate_failures),
        **report,
    }


def stratum_key(record: dict) -> tuple[str, str, str, str]:
    q = float(record["discount"])
    deadline = float(record["primary_deadline"])
    gap = float(record["relative_gap"])
    slack = float(record["slack"])
    q_band = "low" if q <= 0.97 else "high"
    deadline_band = "tight" if deadline <= 27 else "medium" if deadline <= 30 else "loose"
    gap_band = "small" if gap <= 0.02 else "medium" if gap <= 0.05 else "large"
    slack_band = "small" if slack <= 0.25 else "large"
    return q_band, deadline_band, gap_band, slack_band


def select_strata(records: list[dict], top_k: int = 12, minimum_count: int = 30) -> tuple[SelectedStratum, ...]:
    if top_k < 1 or minimum_count < 1:
        raise ValueError("top_k and minimum_count must be positive")
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for record in records:
        groups.setdefault(stratum_key(record), []).append(record)
    selected: list[SelectedStratum] = []
    for key, rows in groups.items():
        if len(rows) < minimum_count or any(row["johnson_gate_e_failures"] for row in rows):
            continue
        selected.append(
            SelectedStratum(
                key=key,
                count=len(rows),
                allocation_change_rate=sum(bool(row["allocation_changed"]) for row in rows) / len(rows),
                decisive_rate=sum(bool(row["warping_decisive"]) for row in rows) / len(rows),
                base_count=len({row["base_id"] for row in rows}),
            )
        )
    selected.sort(
        key=lambda item: (
            -item.allocation_change_rate,
            -item.decisive_rate,
            -item.base_count,
            item.key,
        )
    )
    return tuple(selected[:top_k])


def confirmation_cases(
    selected: tuple[SelectedStratum, ...],
    per_stratum: int = 100,
    seed: int = 7132026,
) -> tuple[StressCase, ...]:
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    cases: list[StressCase] = []
    for selection in selected:
        q_band, deadline_band, _gap_band, slack_band = selection.key
        q = 0.965 if q_band == "low" else 0.985
        deadline = {"tight": 25.5, "medium": 30.0, "loose": 34.5}[deadline_band]
        slack = 0.10 if slack_band == "small" else 0.50
        parameters = list(
            product(
                (50.0, 70.0),
                (90.0, 110.0),
                (50.0, 70.0),
                (0.5, 1.5),
                (-0.15, 0.15),
                (-0.15, 0.15),
                (0.2, 0.4, 0.6, 0.8),
            )
        )
        stable_offset = sum((index + 1) * sum(map(ord, value)) for index, value in enumerate(selection.key))
        Random(seed + stable_offset).shuffle(parameters)
        accepted = 0
        for index, (v0, v1, v2, service1, origin_jitter, target_jitter, rho) in enumerate(parameters):
            instance = StaticInstance(
                agents=(StaticAgent(0, (5.0 + origin_jitter, 3.0 - origin_jitter), deadline, 3),),
                tasks=(
                    StaticTask(0, (3.0, -1.0), 0.0, v0),
                    StaticTask(1, (-10.0, 5.0), service1, v1),
                    StaticTask(2, (7.0 - target_jitter, 1.0 + target_jitter), 0.0, v2),
                ),
                discount=q,
                instance_id=f"CONFBASE-{'-'.join(selection.key)}-{index:04d}",
            )
            for rise in detect_rises(instance):
                stub = {
                    "discount": q,
                    "primary_deadline": deadline,
                    "relative_gap": rise.relative_gap,
                    "slack": slack,
                }
                if stratum_key(stub) != selection.key:
                    continue
                case = construct_competition(
                    rise,
                    rho=rho,
                    slack=slack,
                    instance_id=f"CONF-{'-'.join(selection.key)}-{accepted:03d}",
                )
                if case is None:
                    continue
                cases.append(case)
                accepted += 1
                if accepted >= per_stratum:
                    break
            if accepted >= per_stratum:
                break
    return tuple(cases)


def confirmation_coverage(
    selected: tuple[SelectedStratum, ...],
    cases: tuple[StressCase, ...],
    per_stratum: int = 100,
) -> dict:
    counts = {selection.key: 0 for selection in selected}
    for case in cases:
        key = stratum_key(
            {
                "discount": case.instance.discount,
                "primary_deadline": case.instance.agents[0].deadline,
                "relative_gap": case.base.relative_gap,
                "slack": case.slack,
            }
        )
        if key in counts:
            counts[key] += 1
    encoded = {"|".join(key): count for key, count in counts.items()}
    shortages = {key: per_stratum - count for key, count in encoded.items() if count < per_stratum}
    return {
        "requested_count": len(selected) * per_stratum,
        "generated_count": len(cases),
        "covered_strata": sum(count > 0 for count in counts.values()),
        "selected_strata": len(selected),
        "counts_by_stratum": encoded,
        "shortages": shortages,
    }
