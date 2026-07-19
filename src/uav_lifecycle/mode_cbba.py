from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .mode_allocation import TOL, Mode, ModeInstance, ModeKey, best_mode_insertion, evaluate_mode_path


_MODE_RANK = {Mode.RECON: 0, Mode.ATTACK: 1, Mode.BDA: 2}


class ModeMethod(str, Enum):
    STANDARD_RAW = "standard_raw"
    FULL_REBUILD_RAW = "full_rebuild_raw"
    WARPED_RETAIN = "warped_retain"
    JOHNSON_WARPED = "johnson_warped"


@dataclass(frozen=True)
class ScreenedTask:
    target_id: int
    mode: Mode
    witness_agent: int
    witness_value: float

    @property
    def key(self) -> ModeKey:
        return self.target_id, self.mode


@dataclass(frozen=True)
class ModeWinner:
    agent_id: int
    external_bid: float


@dataclass(frozen=True)
class ModeBundleEntry:
    key: ModeKey
    raw_score: float
    external_bid: float


@dataclass(frozen=True)
class ModeAgentBundle:
    agent_id: int
    bundle: tuple[int, ...]
    path: tuple[ModeKey, ...]
    entries: tuple[ModeBundleEntry, ...]


@dataclass(frozen=True)
class ModeCBBAResult:
    method: ModeMethod
    status: str
    rounds: int
    winners: tuple[ModeWinner | None, ...]
    states: tuple[ModeAgentBundle, ...]
    true_score: float
    orphan_targets: tuple[int, ...]
    cycle_period: int | None = None
    message_packets: int = 0
    message_scalars: int = 0
    warping_activations: int = 0
    raw_prefix_increases: int = 0


def screen_modes(instance: ModeInstance) -> tuple[ScreenedTask | None, ...]:
    selected: list[ScreenedTask | None] = []
    for target_id, tasks in enumerate(instance.tasks_by_target):
        candidates: list[ScreenedTask] = []
        for task in tasks:
            witnesses: list[tuple[float, int]] = []
            for agent_id in range(len(instance.agents)):
                evaluation = evaluate_mode_path(instance, agent_id, ((target_id, task.mode),))
                if evaluation.feasible:
                    witnesses.append((evaluation.score, agent_id))
            if witnesses:
                value, agent_id = min(witnesses, key=lambda item: (-item[0], item[1]))
                candidates.append(ScreenedTask(target_id, task.mode, agent_id, value))
        if not candidates:
            selected.append(None)
            continue
        best = min(candidates, key=lambda item: (-item.witness_value, _MODE_RANK[item.mode]))
        selected.append(best if best.witness_value > TOL else None)
    return tuple(selected)


def _threshold(winner: ModeWinner | None, agent_id: int, full_rebuild: bool) -> float:
    if winner is None or (full_rebuild and winner.agent_id == agent_id):
        return 0.0
    return winner.external_bid


def _build_bundle(
    instance: ModeInstance,
    screened: tuple[ScreenedTask | None, ...],
    agent_id: int,
    method: ModeMethod,
    winners: tuple[ModeWinner | None, ...],
    previous_state: ModeAgentBundle | None = None,
) -> ModeAgentBundle:
    full_rebuild = method in (ModeMethod.FULL_REBUILD_RAW, ModeMethod.JOHNSON_WARPED)
    if full_rebuild or previous_state is None:
        bundle: tuple[int, ...] = ()
        path: tuple[ModeKey, ...] = ()
        entries: tuple[ModeBundleEntry, ...] = ()
    else:
        bundle, path, entries = previous_state.bundle, previous_state.path, previous_state.entries
    while True:
        candidates: list[tuple[float, int, float, tuple[ModeKey, ...], ModeKey]] = []
        for target_id, screened_task in enumerate(screened):
            if screened_task is None or target_id in bundle:
                continue
            insertion = best_mode_insertion(instance, agent_id, path, screened_task.key)
            if insertion is None or insertion.marginal <= TOL:
                continue
            raw = insertion.marginal
            external = (
                min(raw, entries[-1].external_bid)
                if method in (ModeMethod.WARPED_RETAIN, ModeMethod.JOHNSON_WARPED) and entries
                else raw
            )
            if external > _threshold(winners[target_id], agent_id, full_rebuild) + TOL:
                candidates.append((raw, target_id, external, insertion.path, screened_task.key))
        if not candidates:
            break
        raw, target_id, external, path, key = min(candidates, key=lambda item: (-item[0], item[1]))
        bundle += (target_id,)
        entries += (ModeBundleEntry(key, raw, external),)
    return ModeAgentBundle(agent_id, bundle, path, entries)


def _release_suffix(
    state: ModeAgentBundle, winners: tuple[ModeWinner | None, ...],
) -> ModeAgentBundle:
    cut = len(state.bundle)
    for index, target_id in enumerate(state.bundle):
        winner = winners[target_id]
        if winner is None or winner.agent_id != state.agent_id:
            cut = index
            break
    removed = set(state.bundle[cut:])
    return ModeAgentBundle(
        state.agent_id,
        state.bundle[:cut],
        tuple(key for key in state.path if key[0] not in removed),
        state.entries[:cut],
    )


def _reduce_winners(
    states: tuple[ModeAgentBundle, ...], target_count: int
) -> tuple[ModeWinner | None, ...]:
    winners: list[ModeWinner | None] = [None] * target_count
    for state in states:
        for entry in state.entries:
            target_id = entry.key[0]
            candidate = ModeWinner(state.agent_id, entry.external_bid)
            current = winners[target_id]
            if current is None or candidate.external_bid > current.external_bid + TOL or (
                abs(candidate.external_bid - current.external_bid) <= TOL
                and candidate.agent_id < current.agent_id
            ):
                winners[target_id] = candidate
    return tuple(winners)


def _snapshot(
    winners: tuple[ModeWinner | None, ...], states: tuple[ModeAgentBundle, ...]
) -> tuple:
    winner_key = tuple(
        None if winner is None else (winner.agent_id, round(winner.external_bid, 12))
        for winner in winners
    )
    state_key = tuple(
        (
            state.bundle,
            state.path,
            tuple(
                (entry.key, round(entry.raw_score, 12), round(entry.external_bid, 12))
                for entry in state.entries
            ),
        )
        for state in states
    )
    return winner_key, state_key


def run_mode_cbba(
    instance: ModeInstance,
    screened: tuple[ScreenedTask | None, ...],
    method: ModeMethod,
) -> ModeCBBAResult:
    if len(screened) != len(instance.tasks_by_target):
        raise ValueError("screening must contain one entry per target")
    winners: tuple[ModeWinner | None, ...] = (None,) * len(screened)
    states = tuple(ModeAgentBundle(i, (), (), ()) for i in range(len(instance.agents)))
    previous_snapshot = None
    seen: dict[tuple, int] = {}
    stable_rounds = 0
    status = "timeout"
    cycle_period = None
    max_rounds = max(100, 10 * len(instance.agents) * max(1, len(screened)))
    for round_number in range(1, max_rounds + 1):
        built = tuple(
            _build_bundle(instance, screened, agent_id, method, winners, states[agent_id])
            for agent_id in range(len(instance.agents))
        )
        new_winners = _reduce_winners(built, len(screened))
        if method in (ModeMethod.STANDARD_RAW, ModeMethod.WARPED_RETAIN):
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

    true_score = sum(
        evaluate_mode_path(instance, state.agent_id, state.path).score for state in states
    )
    orphan_targets = tuple(
        target_id
        for target_id, screened_task in enumerate(screened)
        if screened_task is not None and winners[target_id] is None
    )
    warping_activations = sum(
        entry.raw_score > entry.external_bid + TOL
        for state in states for entry in state.entries
    )
    raw_prefix_increases = sum(
        right.raw_score > left.raw_score + TOL
        for state in states
        for left, right in zip(state.entries, state.entries[1:])
    )
    return ModeCBBAResult(
        method,
        status,
        round_number,
        winners,
        states,
        true_score,
        orphan_targets,
        cycle_period,
        round_number * len(instance.agents) * max(0, len(instance.agents) - 1),
        round_number * len(instance.agents) * max(0, len(instance.agents) - 1) * len(screened),
        warping_activations,
        raw_prefix_increases,
    )


def validate_mode_result(
    instance: ModeInstance,
    screened: tuple[ScreenedTask | None, ...],
    result: ModeCBBAResult,
) -> dict[str, int]:
    held: dict[int, list[int]] = {}
    infeasible = 0
    mismatches = 0
    monotonicity = 0
    for state in result.states:
        for target_id in state.bundle:
            held.setdefault(target_id, []).append(state.agent_id)
        if not evaluate_mode_path(instance, state.agent_id, state.path).feasible:
            infeasible += 1
        if set(state.bundle) != {key[0] for key in state.path}:
            mismatches += 1
        if result.method in (ModeMethod.WARPED_RETAIN, ModeMethod.JOHNSON_WARPED):
            bids = [entry.external_bid for entry in state.entries]
            monotonicity += sum(right > left + TOL for left, right in zip(bids, bids[1:]))
    conflicts = sum(
        held.get(target_id, []) != [winner.agent_id]
        for target_id, winner in enumerate(result.winners)
        if winner is not None
    )
    replay = 0
    if result.method in (ModeMethod.WARPED_RETAIN, ModeMethod.JOHNSON_WARPED):
        for agent_id in range(len(instance.agents)):
            first = _build_bundle(instance, screened, agent_id, result.method, result.winners)
            second = _build_bundle(instance, screened, agent_id, result.method, result.winners)
            replay += first != second
    return {
        "winner_conflicts": conflicts,
        "infeasible_paths": infeasible,
        "bundle_path_mismatches": mismatches,
        "warped_monotonicity_violations": monotonicity,
        "replay_mismatches": replay,
        "cycle_or_timeout": int(result.status != "converged"),
    }
