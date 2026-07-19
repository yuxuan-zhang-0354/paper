from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import dist

from .allocation_model import TOL, StaticInstance, best_true_insertion, evaluate_agent_path


class Method(str, Enum):
    STANDARD_RAW = "standard_raw"
    FULL_REBUILD_RAW = "full_rebuild_raw"
    SURROGATE = "surrogate"
    JOHNSON_WARPED = "johnson_warped"


@dataclass(frozen=True)
class Winner:
    agent_id: int
    external_bid: float


@dataclass(frozen=True)
class BundleEntry:
    task_id: int
    raw_score: float
    external_bid: float


@dataclass(frozen=True)
class AgentBundle:
    agent_id: int
    bundle: tuple[int, ...]
    path: tuple[int, ...]
    entries: tuple[BundleEntry, ...]


@dataclass(frozen=True)
class CBBAResult:
    method: Method
    status: str
    rounds: int
    winners: tuple[Winner | None, ...]
    states: tuple[AgentBundle, ...]
    true_score: float
    cycle_period: int | None = None


def _threshold(winner: Winner | None, agent_id: int, full_rebuild: bool) -> float:
    if winner is None or (full_rebuild and winner.agent_id == agent_id):
        return 0.0
    return winner.external_bid


def _surrogate_candidate(
    instance: StaticInstance,
    agent_id: int,
    path: tuple[int, ...],
    bundle: tuple[int, ...],
    task_id: int,
):
    agent = instance.agents[agent_id]
    task = instance.tasks[task_id]
    current_weight = sum(
        2.0 * dist(agent.origin, instance.tasks[j].position) + instance.tasks[j].service
        for j in bundle
    )
    weight = 2.0 * dist(agent.origin, task.position) + task.service
    if current_weight + weight > agent.deadline + TOL:
        return None
    insertion = best_true_insertion(instance, agent_id, path, task_id)
    if insertion is None:
        return None
    raw = task.value * instance.discount ** dist(agent.origin, task.position)
    return raw, insertion.path


def build_bundle(
    instance: StaticInstance,
    agent_id: int,
    method: Method,
    winners: tuple[Winner | None, ...],
    previous_state: AgentBundle | None = None,
) -> AgentBundle:
    full_rebuild = method in (Method.FULL_REBUILD_RAW, Method.JOHNSON_WARPED)
    if full_rebuild or previous_state is None:
        bundle: tuple[int, ...] = ()
        path: tuple[int, ...] = ()
        entries: tuple[BundleEntry, ...] = ()
    else:
        bundle, path, entries = previous_state.bundle, previous_state.path, previous_state.entries

    capacity = instance.agents[agent_id].capacity
    while len(bundle) < capacity:
        candidates: list[tuple[float, int, float, tuple[int, ...]]] = []
        for task_id in range(len(instance.tasks)):
            if task_id in bundle:
                continue
            if method is Method.SURROGATE:
                candidate = _surrogate_candidate(instance, agent_id, path, bundle, task_id)
                if candidate is None:
                    continue
                raw, candidate_path = candidate
            else:
                insertion = best_true_insertion(instance, agent_id, path, task_id)
                if insertion is None:
                    continue
                raw, candidate_path = insertion.marginal, insertion.path
            if raw <= TOL:
                continue
            external = min(raw, entries[-1].external_bid) if method is Method.JOHNSON_WARPED and entries else raw
            threshold = _threshold(winners[task_id], agent_id, full_rebuild)
            if external > threshold + TOL:
                candidates.append((raw, task_id, external, candidate_path))
        if not candidates:
            break
        raw, task_id, external, path = min(candidates, key=lambda x: (-x[0], x[1]))
        bundle += (task_id,)
        entries += (BundleEntry(task_id, raw, external),)
    return AgentBundle(agent_id, bundle, path, entries)


def reduce_winners(states: tuple[AgentBundle, ...], task_count: int) -> tuple[Winner | None, ...]:
    winners: list[Winner | None] = [None] * task_count
    for state in states:
        for entry in state.entries:
            candidate = Winner(state.agent_id, entry.external_bid)
            current = winners[entry.task_id]
            if current is None or candidate.external_bid > current.external_bid + TOL or (
                abs(candidate.external_bid - current.external_bid) <= TOL
                and candidate.agent_id < current.agent_id
            ):
                winners[entry.task_id] = candidate
    return tuple(winners)


def _release_suffix(state: AgentBundle, winners: tuple[Winner | None, ...]) -> AgentBundle:
    cut = len(state.bundle)
    for index, task_id in enumerate(state.bundle):
        winner = winners[task_id]
        if winner is None or winner.agent_id != state.agent_id:
            cut = index
            break
    removed = set(state.bundle[cut:])
    return AgentBundle(
        state.agent_id,
        state.bundle[:cut],
        tuple(task_id for task_id in state.path if task_id not in removed),
        state.entries[:cut],
    )


def _snapshot(winners: tuple[Winner | None, ...], states: tuple[AgentBundle, ...]):
    winner_key = tuple(
        None if winner is None else (winner.agent_id, round(winner.external_bid, 12))
        for winner in winners
    )
    state_key = tuple(
        (
            state.bundle,
            state.path,
            tuple((e.task_id, round(e.raw_score, 12), round(e.external_bid, 12)) for e in state.entries),
        )
        for state in states
    )
    return winner_key, state_key


def run_static_cbba(instance: StaticInstance, method: Method) -> CBBAResult:
    winners: tuple[Winner | None, ...] = (None,) * len(instance.tasks)
    states = tuple(AgentBundle(i, (), (), ()) for i in range(len(instance.agents)))
    seen: dict[tuple, int] = {}
    previous_snapshot = None
    stable_rounds = 0
    max_rounds = max(100, 10 * len(instance.agents) * len(instance.tasks))
    status = "timeout"
    cycle_period = None
    for round_number in range(1, max_rounds + 1):
        built = tuple(
            build_bundle(instance, i, method, winners, states[i])
            for i in range(len(instance.agents))
        )
        new_winners = reduce_winners(built, len(instance.tasks))
        if method in (Method.STANDARD_RAW, Method.SURROGATE):
            built = tuple(_release_suffix(state, new_winners) for state in built)
        snapshot = _snapshot(new_winners, built)
        if snapshot == previous_snapshot:
            stable_rounds += 1
            if stable_rounds >= 3:
                status = "converged"
                winners, states = new_winners, built
                break
        else:
            stable_rounds = 0
            if snapshot in seen:
                status = "cycle"
                cycle_period = round_number - seen[snapshot]
                winners, states = new_winners, built
                break
        seen.setdefault(snapshot, round_number)
        previous_snapshot = snapshot
        winners, states = new_winners, built
    true_score = sum(evaluate_agent_path(instance, s.agent_id, s.path).score for s in states)
    return CBBAResult(method, status, round_number, winners, states, true_score, cycle_period)


def validate_result(instance: StaticInstance, result: CBBAResult) -> dict[str, int]:
    held: dict[int, list[int]] = {}
    infeasible = 0
    mismatches = 0
    monotonicity = 0
    for state in result.states:
        for task_id in state.bundle:
            held.setdefault(task_id, []).append(state.agent_id)
        if not evaluate_agent_path(instance, state.agent_id, state.path).feasible:
            infeasible += 1
        if set(state.bundle) != set(state.path):
            mismatches += 1
        if result.method is Method.JOHNSON_WARPED:
            bids = [entry.external_bid for entry in state.entries]
            monotonicity += sum(right > left + TOL for left, right in zip(bids, bids[1:]))
    conflicts = sum(len(agents) != 1 for task_id, agents in held.items() if result.winners[task_id] is not None)
    conflicts += sum(
        1
        for task_id, winner in enumerate(result.winners)
        if winner is not None and held.get(task_id, []) != [winner.agent_id]
    )
    replay = 0
    if result.method is Method.JOHNSON_WARPED:
        for agent_id in range(len(instance.agents)):
            first = build_bundle(instance, agent_id, result.method, result.winners)
            second = build_bundle(instance, agent_id, result.method, result.winners)
            replay += first != second
    return {
        "winner_conflicts": conflicts,
        "infeasible_paths": infeasible,
        "bundle_path_mismatches": mismatches,
        "warped_monotonicity_violations": monotonicity,
        "replay_mismatches": replay,
    }
