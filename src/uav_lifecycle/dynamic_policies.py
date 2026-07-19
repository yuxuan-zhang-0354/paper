"""Public-only dynamic lifecycle policies for P and matched baselines."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

from .dynamic_planning import (
    PlannedPath,
    PlanningResult,
    plan_all_mode_exact,
    plan_factorial_variant,
    plan_johnson,
    plan_vanilla,
    plan_nearest_positive,
    positive_single_task_targets,
)
from .dynamic_types import (
    DynamicConfig,
    GateFailure,
    PlanningClock,
    PublicSnapshot,
    quantize_tick,
)
from .mode_allocation import Mode
from .mode_cbba import ModeMethod


_PUBLIC_MODES = ("recon", "attack", "bda", "defer")
_SHARED_CONTRACT = (
    "public_api",
    "resources",
    "durations",
    "bayes_attack",
    "nonpreemption",
    "target_lock",
    "horizon",
    "crn",
    "evaluator",
    "completion_before_planning",
    "deterministic_ties",
    "one_target_one_commit",
)


@dataclass(frozen=True, slots=True)
class MethodContract:
    allowed_modes: tuple[str, ...]
    task_generation: str
    planner: str
    planning_clock: PlanningClock
    execution_rule: str
    shared: tuple[str, ...] = _SHARED_CONTRACT


@dataclass(frozen=True, slots=True)
class ActionProposal:
    """A public controller proposal; it deliberately is not a physics commit."""

    agent_id: int
    target_id: int
    mode: str
    planned_path: PlannedPath | None = None

    def commit_candidate(self) -> PlannedPath | tuple[int, int, str]:
        return (
            self.planned_path
            if self.planned_path is not None
            else (self.agent_id, self.target_id, self.mode)
        )


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    proposals: tuple[ActionProposal, ...]
    pending_suffixes: tuple[tuple[int, tuple[tuple[int, str], ...]], ...]
    planning_bytes: bytes
    planned_paths: tuple[PlannedPath, ...]
    gates: tuple[GateFailure, ...] = ()


def _validate_snapshot(snapshot: PublicSnapshot) -> None:
    if not isinstance(snapshot, PublicSnapshot):
        raise TypeError("policy input must be PublicSnapshot")


def _mode_value(mode: object) -> str:
    return str(getattr(mode, "value", mode))


def _planning_bytes(result: PlanningResult) -> bytes:
    payload = {
        "gates": [
            {
                "details": list(gate.details),
                "gate": gate.gate,
                "reason": gate.reason,
                "tick": gate.tick,
            }
            for gate in result.gates
        ],
        "method": result.method,
        "paths": [
            {
                "agent_id": path.agent_id,
                "completion_ticks": [int(tick) for tick in path.completion_ticks],
                "score": float(path.score),
                "start_ticks": [int(tick) for tick in path.start_ticks],
                "tasks": [(int(target), _mode_value(mode)) for target, mode in path.tasks],
            }
            for path in result.paths
        ],
        "positive_pair_count": int(result.positive_pair_count),
        "rounds": int(result.rounds),
        "search_bound": None if result.search_bound is None else int(result.search_bound),
        "search_count": None if result.search_count is None else int(result.search_count),
        "status": result.status,
        "allocator_status": result.allocator_status,
        "message_packets": int(result.message_packets),
        "message_scalars": int(result.message_scalars),
        "allocation_objective": float(result.allocation_objective),
        "audit_report": [[str(name), int(value)] for name, value in result.audit_report],
        "eligible_pair_count": int(result.eligible_pair_count),
        "screened_positive_pair_count": int(
            result.positive_pair_count
            if result.screened_positive_pair_count is None
            else result.screened_positive_pair_count
        ),
        "warping_activations": int(result.warping_activations),
        "raw_prefix_increases": int(result.raw_prefix_increases),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _proposals(paths: tuple[PlannedPath, ...]) -> tuple[ActionProposal, ...]:
    proposals = tuple(
        ActionProposal(path.agent_id, path.tasks[0][0], _mode_value(path.tasks[0][1]), path)
        for path in paths
        if path.tasks
    )
    targets = tuple(item.target_id for item in proposals)
    if len(targets) != len(set(targets)):
        raise RuntimeError("planner violated one-target-one-commit")
    return proposals


def _decision(result: PlanningResult) -> PolicyDecision:
    return PolicyDecision(
        _proposals(result.paths) if not result.gates else (),
        (),
        _planning_bytes(result),
        result.paths,
        result.gates,
    )


class _PolicyBase:
    method = ""
    contract: MethodContract
    planning_clock = PlanningClock.EVENT_DRIVEN

    def __init__(self, config: DynamicConfig):
        if not isinstance(config, DynamicConfig):
            raise TypeError("config must be DynamicConfig")
        self.config = config
        self._max_tick: int | None = None
        self._last_positive = 0
        self._last_fingerprint: tuple[object, ...] | None = None
        self.allocator_audits: list[dict[str, object]] = []

    def bind_horizon(self, max_tick: int) -> None:
        if isinstance(max_tick, bool) or not isinstance(max_tick, int) or max_tick <= 0:
            raise ValueError("max_tick must be a positive integer")
        if self._max_tick is not None and self._max_tick != max_tick:
            raise ValueError("policy horizon cannot change during an episode")
        self._max_tick = max_tick

    def _horizon(self) -> int:
        if self._max_tick is None:
            raise RuntimeError("bind_horizon must be called before planning")
        return self._max_tick

    def positive_pair_count(self, snapshot: PublicSnapshot) -> int:
        _validate_snapshot(snapshot)
        return self._last_positive

    @staticmethod
    def _fingerprint(snapshot: PublicSnapshot) -> tuple[object, ...]:
        return (
            snapshot.target_locks,
            snapshot.observations,
            snapshot.acknowledgements,
            tuple(
                (
                    agent.agent_id,
                    agent.position,
                    agent.available_ammo,
                    agent.available_distance,
                    agent.busy_action,
                )
                for agent in snapshot.agents
            ),
        )

    def _run_johnson(
        self,
        snapshot: PublicSnapshot,
        *,
        allowed_modes: dict[int, tuple[Mode, ...]] | None = None,
    ) -> PolicyDecision:
        result = plan_johnson(
            snapshot,
            self.config,
            self._horizon(),
            **({} if allowed_modes is None else {"allowed_modes": allowed_modes}),
        )
        self._last_positive = int(result.positive_pair_count)
        self._last_fingerprint = self._fingerprint(snapshot)
        self.allocator_audits.append(json.loads(_planning_bytes(result)))
        return _decision(result)

    def _run_vanilla(self, snapshot: PublicSnapshot) -> PolicyDecision:
        result = plan_vanilla(snapshot, self.config, self._horizon())
        self._last_positive = int(result.positive_pair_count)
        self._last_fingerprint = self._fingerprint(snapshot)
        self.allocator_audits.append(json.loads(_planning_bytes(result)))
        return _decision(result)

    def _run_factorial(
        self, snapshot: PublicSnapshot, allocator: ModeMethod,
    ) -> PolicyDecision:
        result = plan_factorial_variant(
            snapshot, self.config, self._horizon(), allocator,
        )
        self._last_positive = int(result.positive_pair_count)
        self._last_fingerprint = self._fingerprint(snapshot)
        self.allocator_audits.append(json.loads(_planning_bytes(result)))
        return _decision(result)


class DynamicPPolicy(_PolicyBase):
    method = "P"
    contract = MethodContract(
        _PUBLIC_MODES,
        "gated_screening",
        "johnson_warped",
        PlanningClock.EVENT_DRIVEN,
        "commit_next",
    )
    planning_clock = PlanningClock.EVENT_DRIVEN

    def __init__(self, config: DynamicConfig):
        super().__init__(config)

    def decide(self, snapshot: PublicSnapshot) -> PolicyDecision:
        _validate_snapshot(snapshot)
        fingerprint = self._fingerprint(snapshot)
        if self._last_fingerprint == fingerprint:
            return PolicyDecision((), (), b"", (), ())
        return self._run_johnson(snapshot)


class DynamicVanillaPolicy(_PolicyBase):
    method = "DVCBBA"
    contract = MethodContract(
        _PUBLIC_MODES, "gated_screening", "vanilla_raw",
        PlanningClock.EVENT_DRIVEN, "commit_next",
    )
    planning_clock = PlanningClock.EVENT_DRIVEN

    def decide(self, snapshot: PublicSnapshot) -> PolicyDecision:
        _validate_snapshot(snapshot)
        fingerprint = self._fingerprint(snapshot)
        if self._last_fingerprint == fingerprint:
            return PolicyDecision((), (), b"", (), ())
        return self._run_vanilla(snapshot)


class DynamicFactorialPolicy(_PolicyBase):
    planning_clock = PlanningClock.EVENT_DRIVEN

    def __init__(self, config: DynamicConfig, label: str, allocator: ModeMethod):
        super().__init__(config)
        self.method = label
        self.allocator = allocator
        self.contract = MethodContract(
            _PUBLIC_MODES, "gated_screening", allocator.value,
            PlanningClock.EVENT_DRIVEN, "commit_next",
        )

    def decide(self, snapshot: PublicSnapshot) -> PolicyDecision:
        _validate_snapshot(snapshot)
        fingerprint = self._fingerprint(snapshot)
        if self._last_fingerprint == fingerprint:
            return PolicyDecision((), (), b"", (), ())
        return self._run_factorial(snapshot, self.allocator)


class OneShotMatchedPolicy(_PolicyBase):
    method = "B1m"
    contract = MethodContract(
        _PUBLIC_MODES,
        "gated_screening",
        "johnson_warped",
        PlanningClock.B1M_ONE_SHOT,
        "frozen_paths_auto_next",
    )
    planning_clock = PlanningClock.B1M_ONE_SHOT

    def __init__(self, config: DynamicConfig):
        super().__init__(config)
        self._planned = False
        self._suffixes: dict[int, tuple[tuple[int, str], ...]] = {}

    def decide(self, snapshot: PublicSnapshot) -> PolicyDecision:
        _validate_snapshot(snapshot)
        if self._planned or snapshot.tick != 0:
            raise RuntimeError("B1m may plan only at tick 0")
        result = self._plan(snapshot)
        self._planned = True
        self._last_positive = int(result.positive_pair_count)
        self.allocator_audits.append(json.loads(_planning_bytes(result)))
        assigned_targets = tuple(
            target for path in result.paths for target, _ in path.tasks
        )
        if len(assigned_targets) != len(set(assigned_targets)):
            gate = GateFailure("b1m_frozen", snapshot.tick, "duplicate_target")
            return PolicyDecision(
                (), (), _planning_bytes(result), result.paths, (gate,)
            )
        self._suffixes = {
            path.agent_id: tuple((target, _mode_value(mode)) for target, mode in path.tasks[1:])
            for path in result.paths
            if len(path.tasks) > 1
        }
        base = _decision(result)
        return PolicyDecision(
            base.proposals,
            tuple(sorted(self._suffixes.items())),
            base.planning_bytes,
            base.planned_paths,
            base.gates,
        )

    def _plan(self, snapshot: PublicSnapshot) -> PlanningResult:
        return plan_johnson(snapshot, self.config, self._horizon())

    def has_pending_suffix(self) -> bool:
        return any(self._suffixes.values())

    def auto_next(
        self, snapshot: PublicSnapshot, completed_agent_ids: tuple[int, ...]
    ) -> PolicyDecision:
        _validate_snapshot(snapshot)
        proposals: list[ActionProposal] = []
        for agent_id in sorted(set(completed_agent_ids)):
            suffix = self._suffixes.get(agent_id, ())
            if not suffix:
                continue
            target_id, mode = suffix[0]
            proposals.append(ActionProposal(agent_id, target_id, mode))
            remainder = suffix[1:]
            if remainder:
                self._suffixes[agent_id] = remainder
            else:
                self._suffixes.pop(agent_id, None)
        targets = tuple(item.target_id for item in proposals)
        if len(targets) != len(set(targets)):
            gate = GateFailure("b1m_suffix", snapshot.tick, "duplicate_target")
            return PolicyDecision((), tuple(sorted(self._suffixes.items())), b"", (), (gate,))
        if not self.has_pending_suffix():
            self._last_positive = 0
        return PolicyDecision(tuple(proposals), tuple(sorted(self._suffixes.items())), b"", ())


class StandardOneShotCBBAPolicy(OneShotMatchedPolicy):
    method = "SCBBA"
    contract = MethodContract(
        _PUBLIC_MODES, "t0_fixed_screened_tasks", "vanilla_raw",
        PlanningClock.B1M_ONE_SHOT, "frozen_paths_auto_next",
    )

    def _plan(self, snapshot: PublicSnapshot) -> PlanningResult:
        return plan_vanilla(snapshot, self.config, self._horizon())

class FixedOrderPolicy(_PolicyBase):
    method = "B2"
    contract = MethodContract(
        ("fixed_phase", "defer"),
        "current_phase",
        "johnson_warped",
        PlanningClock.EVENT_DRIVEN,
        "commit_next",
    )
    planning_clock = PlanningClock.EVENT_DRIVEN
    _MODE_BY_PHASE = {
        "recon": Mode.RECON,
        "attack": Mode.ATTACK,
        "bda": Mode.BDA,
        "terminal_attack": Mode.ATTACK,
    }

    def __init__(self, config: DynamicConfig):
        super().__init__(config)
        self._phases: dict[int, str] = {}
        self._seen_events: set[int] = set()

    def phase(self, target_id: int) -> str:
        return self._phases.get(target_id, "recon")

    def _consume_completions(self, snapshot: PublicSnapshot) -> None:
        events = sorted(
            (*snapshot.observations, *snapshot.acknowledgements),
            key=lambda event: event.event_id,
        )
        next_phase = {
            "recon": "attack",
            "attack": "bda",
            "bda": "terminal_attack",
            "terminal_attack": "done",
        }
        for event in events:
            if event.event_id in self._seen_events:
                continue
            current = self.phase(event.target_id)
            expected = self._MODE_BY_PHASE.get(current)
            if expected is not None and event.mode == expected.value:
                self._phases[event.target_id] = next_phase[current]
            self._seen_events.add(event.event_id)

    def decide(self, snapshot: PublicSnapshot) -> PolicyDecision:
        _validate_snapshot(snapshot)
        self._consume_completions(snapshot)
        allowed = {
            target.target_id: (() if self.phase(target.target_id) == "done" else (self._MODE_BY_PHASE[self.phase(target.target_id)],))
            for target in snapshot.targets
        }
        if not any(allowed.values()):
            self._last_positive = 0
            result = PlanningResult("johnson", "valid", (), 0)
            return _decision(result)
        decision = self._run_johnson(snapshot, allowed_modes=allowed)
        terminal = tuple(
            target_id
            for target_id in allowed
            if self.phase(target_id) == "terminal_attack"
        )
        if terminal:
            positive = positive_single_task_targets(
                snapshot,
                self.config,
                self._horizon(),
                allowed_modes=allowed,
            )
            for target_id in terminal:
                if target_id not in positive:
                    self._phases[target_id] = "done"
        return decision


class AttackOnlyPolicy(_PolicyBase):
    method = "B3"
    contract = MethodContract(
        ("attack", "defer"),
        "attack_only",
        "johnson_warped",
        PlanningClock.EVENT_DRIVEN,
        "commit_next",
    )
    planning_clock = PlanningClock.EVENT_DRIVEN

    def __init__(self, config: DynamicConfig):
        super().__init__(config)

    def decide(self, snapshot: PublicSnapshot) -> PolicyDecision:
        _validate_snapshot(snapshot)
        allowed = {target.target_id: (Mode.ATTACK,) for target in snapshot.targets}
        return self._run_johnson(snapshot, allowed_modes=allowed)


class NoBDAPolicy(_PolicyBase):
    method = "B4"
    contract = MethodContract(
        ("recon", "attack", "defer"),
        "no_bda_screening",
        "johnson_warped",
        PlanningClock.EVENT_DRIVEN,
        "commit_next",
    )
    planning_clock = PlanningClock.EVENT_DRIVEN

    def __init__(self, config: DynamicConfig):
        super().__init__(config)

    def decide(self, snapshot: PublicSnapshot) -> PolicyDecision:
        _validate_snapshot(snapshot)
        allowed = {
            target.target_id: (Mode.RECON, Mode.ATTACK) for target in snapshot.targets
        }
        return self._run_johnson(snapshot, allowed_modes=allowed)


class PeriodicPolicy(DynamicPPolicy):
    contract = MethodContract(
        _PUBLIC_MODES,
        "gated_screening",
        "johnson_warped",
        PlanningClock.PERIODIC,
        "commit_next",
    )
    planning_clock = PlanningClock.PERIODIC

    def __init__(self, config: DynamicConfig, period: int = 4):
        if period not in (2, 4, 8):
            raise ValueError("B5 period must be 2, 4, or 8")
        super().__init__(config)
        self.period = period
        self.method = f"B5({period})"

    def planning_grid_ticks(self) -> tuple[int, ...]:
        horizon = self._horizon()
        step = quantize_tick(float(self.period), self.config.tick_size)
        if step <= 0:
            raise RuntimeError("period quantized to zero")
        return tuple(range(0, horizon + 1, step))

    def decide(self, snapshot: PublicSnapshot) -> PolicyDecision:
        _validate_snapshot(snapshot)
        if snapshot.tick not in set(self.planning_grid_ticks()):
            return PolicyDecision((), (), b"", ())
        return self._run_johnson(snapshot)


class NearestPositivePolicy(_PolicyBase):
    method = "B6"
    contract = MethodContract(
        _PUBLIC_MODES,
        "gated_screening",
        "nearest_positive",
        PlanningClock.EVENT_DRIVEN,
        "one_task_per_agent",
    )
    planning_clock = PlanningClock.EVENT_DRIVEN

    def __init__(self, config: DynamicConfig):
        super().__init__(config)

    def decide(self, snapshot: PublicSnapshot) -> PolicyDecision:
        _validate_snapshot(snapshot)
        result = plan_nearest_positive(snapshot, self.config, self._horizon())
        self._last_positive = int(result.positive_pair_count)
        self.allocator_audits.append(json.loads(_planning_bytes(result)))
        return _decision(result)


class ExactMyopicPolicy(_PolicyBase):
    method = "CEX"
    contract = MethodContract(
        _PUBLIC_MODES,
        "all_mode_exact",
        "centralized_exact",
        PlanningClock.EVENT_DRIVEN,
        "commit_next",
    )
    planning_clock = PlanningClock.EVENT_DRIVEN

    def __init__(self, config: DynamicConfig):
        super().__init__(config)

    def decide(self, snapshot: PublicSnapshot) -> PolicyDecision:
        _validate_snapshot(snapshot)
        result = plan_all_mode_exact(snapshot, self.config, self._horizon())
        self._last_positive = int(result.positive_pair_count)
        self.allocator_audits.append(json.loads(_planning_bytes(result)))
        return _decision(result)


def make_policy(method: str, config: DynamicConfig) -> _PolicyBase:
    factorial = {
        "V00": ModeMethod.STANDARD_RAW,
        "V01": ModeMethod.FULL_REBUILD_RAW,
        "V10": ModeMethod.WARPED_RETAIN,
        "V11": ModeMethod.JOHNSON_WARPED,
    }
    if method in factorial:
        return DynamicFactorialPolicy(config, method, factorial[method])
    factories = {
        "P": DynamicPPolicy,
        "B1m": OneShotMatchedPolicy,
        "B2": FixedOrderPolicy,
        "B3": AttackOnlyPolicy,
        "B4": NoBDAPolicy,
        "B6": NearestPositivePolicy,
        "CEX": ExactMyopicPolicy,
        "DVCBBA": DynamicVanillaPolicy,
        "SCBBA": StandardOneShotCBBAPolicy,
    }
    if method in factories:
        return factories[method](config)
    match = re.fullmatch(r"B5\((\d+)\)", method)
    if match:
        return PeriodicPolicy(config, int(match.group(1)))
    raise ValueError(f"unknown method: {method}")


__all__ = [
    "ActionProposal",
    "AttackOnlyPolicy",
    "DynamicPPolicy",
    "DynamicFactorialPolicy",
    "DynamicVanillaPolicy",
    "ExactMyopicPolicy",
    "FixedOrderPolicy",
    "MethodContract",
    "NearestPositivePolicy",
    "NoBDAPolicy",
    "OneShotMatchedPolicy",
    "StandardOneShotCBBAPolicy",
    "PeriodicPolicy",
    "PolicyDecision",
    "make_policy",
]
