"""Targeted P-versus-raw-CBBA validation under matched failure recovery."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_dynamic_d3 import _scenario as d3_scenario
from experiments.run_dynamic_d3 import load_frozen_design
from experiments.run_dynamic_d4 import load_design
from experiments.run_dynamic_mainline import _csv_text, _text_atomic
from uav_lifecycle.artifacts import sha256_file, write_json_atomic
from uav_lifecycle.dynamic_d4 import generate_battlefield_structure, generate_reachability
from uav_lifecycle.dynamic_policies import PolicyDecision, make_policy
from uav_lifecycle.dynamic_simulator import run_episode
from uav_lifecycle.dynamic_types import (
    DynamicConfig,
    DynamicScenario,
    EnvironmentModel,
    PlanningClock,
    PublicSnapshot,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/experiments/2026-07-19-failure-policy-validation-plan.md"
D3_MANIFEST = ROOT / "results/dynamic_mainline/d3_design/d3_manifest.json"
D3_AUTH = ROOT / "results/dynamic_mainline/d3_design/execution_authorization.json"
D4_MANIFEST = ROOT / "results/dynamic_mainline/d4_design/d4_manifest.json"
D4_AUTH = ROOT / "results/dynamic_mainline/d4_design/execution_authorization.json"
DEFAULT_OUTPUT = ROOT / "results/dynamic_mainline/d6_failure_policy_validation"
SUITES = ("scale", "allocation_pressure", "battlefield_structure", "reachability")
PRIMARIES = ("P", "DVCBBA")
RECOVERIES = ("none", "b6")
BOOTSTRAPS = 10_000
TOL = 1e-12


@dataclass(frozen=True, slots=True)
class ScenarioCase:
    suite: str
    condition: str
    scenario: DynamicScenario
    config: DynamicConfig
    environment: EnvironmentModel


@dataclass(frozen=True, slots=True)
class Work:
    case: ScenarioCase
    primary: str
    recovery: str


class MatchedRecoveryPolicy:
    """Run one primary allocator and optionally use B6 on failed epochs."""

    planning_clock = PlanningClock.EVENT_DRIVEN

    def __init__(self, primary: str, recovery: str, config: DynamicConfig):
        self.method = f"{primary}_{recovery}"
        self.primary = make_policy(primary, config)
        self.fallback = make_policy("B6", config) if recovery == "b6" else None
        self.primary_failures: list[dict[str, Any]] = []
        self._last_positive = 0

    def bind_horizon(self, max_tick: int) -> None:
        self.primary.bind_horizon(max_tick)
        if self.fallback is not None:
            self.fallback.bind_horizon(max_tick)

    def decide(self, snapshot: PublicSnapshot) -> PolicyDecision:
        decision = self.primary.decide(snapshot)
        self._last_positive = self.primary.positive_pair_count(snapshot)
        if not decision.gates:
            return decision
        audit = self.primary.allocator_audits[-1]
        self.primary_failures.append(
            {
                "tick": snapshot.tick,
                "reasons": tuple(gate.reason for gate in decision.gates),
                "allocator_status": audit["allocator_status"],
            }
        )
        if self.fallback is None:
            return decision
        recovered = self.fallback.decide(snapshot)
        self._last_positive = self.fallback.positive_pair_count(snapshot)
        return recovered

    def positive_pair_count(self, snapshot: PublicSnapshot) -> int:
        return self._last_positive


def _cases() -> tuple[ScenarioCase, ...]:
    cases: list[ScenarioCase] = []
    d3 = load_frozen_design(D3_MANIFEST, D3_AUTH)
    for scenario_id in d3["scenario_ids"]:
        meta = d3["scenario_metadata"][scenario_id]
        if meta["suite"] not in SUITES:
            continue
        scenario, config, environment = d3_scenario(meta)
        cases.append(
            ScenarioCase(meta["suite"], meta["condition"], scenario, config, environment)
        )

    d4 = load_design(D4_MANIFEST, D4_AUTH)
    config = DynamicConfig()
    environment = EnvironmentModel.from_config(config)
    for scenario_id in d4["scenario_ids"]:
        meta = d4["scenario_metadata"][scenario_id]
        scenario = (
            generate_battlefield_structure(
                meta["structure"], float(meta["wreck_rate"]), int(meta["index"]), config
            )
            if meta["suite"] == "battlefield_structure"
            else generate_reachability(
                float(meta["map_scale"]), float(meta["time_scale"]), int(meta["index"]), config
            )
        )
        if scenario.scenario_id != scenario_id:
            raise RuntimeError("D4 scenario does not match its frozen manifest")
        cases.append(
            ScenarioCase(meta["suite"], meta["condition"], scenario, config, environment)
        )

    counts = {suite: sum(case.suite == suite for case in cases) for suite in SUITES}
    expected = {
        "scale": 672,
        "allocation_pressure": 384,
        "battlefield_structure": 1024,
        "reachability": 576,
    }
    if counts != expected:
        raise RuntimeError(f"unexpected source scenario counts: {counts}")
    return tuple(cases)


def _failure_summary(policy: MatchedRecoveryPolicy) -> dict[str, Any]:
    audits = policy.primary.allocator_audits
    failures = policy.primary_failures
    return {
        "primary_call_count": len(audits),
        "primary_failure_count": len(failures),
        "initial_failure": int(bool(failures) and failures[0]["tick"] == 0),
        "cycle_count": sum(item["allocator_status"] == "cycle" for item in failures),
        "round_cap_count": sum(item["allocator_status"] == "timeout" for item in failures),
        "primary_rounds": sum(int(audit["rounds"]) for audit in audits),
        "primary_packets": sum(int(audit["message_packets"]) for audit in audits),
        "primary_target_entries": sum(int(audit["message_scalars"]) for audit in audits),
        "fallback_count": 0 if policy.fallback is None else len(policy.fallback.allocator_audits),
        "failure_reasons": ";".join(
            "+".join(item["reasons"]) for item in failures
        ),
    }


def _worker(work: Work) -> dict[str, Any]:
    label = f"{work.primary}_{work.recovery}"
    try:
        policy = MatchedRecoveryPolicy(work.primary, work.recovery, work.case.config)
        result = run_episode(
            work.case.scenario,
            policy,
            config=work.case.config,
            environment_model=work.case.environment,
            method=label,
        )
        record = asdict(result.record)
        keep = {
            name: record[name]
            for name in (
                "scenario_id", "cell_id", "termination", "status", "event_count",
                "action_count", "replan_count", "destroyed_value", "service_cost",
                "distance_cost", "ammo_cost", "realized_utility", "normalized_utility",
                "distance_consumed", "ammo_consumed", "makespan",
            )
        }
        return {
            "kind": "result",
            "suite": work.case.suite,
            "condition": work.case.condition,
            "primary": work.primary,
            "recovery": work.recovery,
            **keep,
            **_failure_summary(policy),
        }
    except Exception as error:
        return {
            "kind": "failure",
            "suite": work.case.suite,
            "condition": work.case.condition,
            "primary": work.primary,
            "recovery": work.recovery,
            "scenario_id": work.case.scenario.scenario_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }


def _seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _stratum(row: dict[str, Any]) -> str:
    return row["cell_id"] if row["suite"] == "scale" else row["condition"]


def _contrast(rows: list[dict[str, Any]], suite: str, recovery: str) -> dict[str, Any]:
    selected = [
        row for row in rows if row["suite"] == suite and row["recovery"] == recovery
    ]
    nested: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        nested[row["scenario_id"]][row["primary"]] = row
    strata: dict[str, list[float]] = defaultdict(list)
    method_strata: dict[str, dict[str, list[float]]] = {
        method: defaultdict(list) for method in PRIMARIES
    }
    for methods in nested.values():
        if set(methods) != set(PRIMARIES):
            raise RuntimeError("incomplete paired rectangle")
        stratum = _stratum(methods["P"])
        left = float(methods["P"]["normalized_utility"])
        right = float(methods["DVCBBA"]["normalized_utility"])
        strata[stratum].append(left - right)
        method_strata["P"][stratum].append(left)
        method_strata["DVCBBA"][stratum].append(right)

    bootstrap_means = []
    for name, values in sorted(strata.items()):
        array = np.asarray(values)
        rng = np.random.default_rng(_seed("d6", suite, recovery, name))
        draws = array[rng.integers(0, len(array), size=(BOOTSTRAPS, len(array)))]
        bootstrap_means.append(draws.mean(axis=1))
    distribution = np.mean(np.vstack(bootstrap_means), axis=0)
    flat = np.concatenate([np.asarray(values) for values in strata.values()])
    means = {
        method: float(np.mean([
            np.mean(values) for values in method_strata[method].values()
        ]))
        for method in PRIMARIES
    }
    difference = float(np.mean([np.mean(values) for values in strata.values()]))
    return {
        "suite": suite,
        "recovery": recovery,
        "scenario_count": len(flat),
        "stratum_count": len(strata),
        "mean_p": means["P"],
        "mean_dvcbba": means["DVCBBA"],
        "paired_difference": difference,
        "ci_low": float(np.quantile(distribution, 0.025)),
        "ci_high": float(np.quantile(distribution, 0.975)),
        "relative_gain_percent": 100.0 * difference / means["DVCBBA"],
        "win": int(np.sum(flat > TOL)),
        "tie": int(np.sum(np.abs(flat) <= TOL)),
        "loss": int(np.sum(flat < -TOL)),
    }


def _diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for suite in SUITES:
        for recovery in RECOVERIES:
            for primary in PRIMARIES:
                selected = [
                    row for row in rows
                    if row["suite"] == suite
                    and row["recovery"] == recovery
                    and row["primary"] == primary
                ]
                output.append({
                    "suite": suite,
                    "recovery": recovery,
                    "primary": primary,
                    "episodes": len(selected),
                    "episodes_with_failure": sum(row["primary_failure_count"] > 0 for row in selected),
                    "initial_failure_episodes": sum(row["initial_failure"] for row in selected),
                    "cycle_calls": sum(row["cycle_count"] for row in selected),
                    "round_cap_calls": sum(row["round_cap_count"] for row in selected),
                    "fallback_calls": sum(row["fallback_count"] for row in selected),
                    "zero_action_episodes": sum(row["action_count"] == 0 for row in selected),
                    "unsettled_episode_count": sum(row["action_count"] != row["event_count"] for row in selected),
                })
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0])
    columns.extend(sorted({key for row in rows for key in row} - set(columns)))
    _text_atomic(path, _csv_text(rows, tuple(columns)))


def run(output: Path, workers: int) -> dict[str, Any]:
    cases = _cases()
    works = tuple(
        Work(case, primary, recovery)
        for case in cases
        for recovery in RECOVERIES
        for primary in PRIMARIES
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        payloads = list(pool.map(_worker, works, chunksize=4))
    failures = [row for row in payloads if row["kind"] != "result"]
    rows = [row for row in payloads if row["kind"] == "result"]
    rows.sort(key=lambda row: (row["scenario_id"], row["recovery"], row["primary"]))
    contrasts = [
        _contrast(rows, suite, recovery)
        for suite in SUITES
        for recovery in RECOVERIES
    ] if not failures else []
    diagnostics = _diagnostics(rows) if not failures else []

    output.mkdir(parents=True, exist_ok=True)
    if rows:
        _write_csv(output / "records.csv", rows)
    if contrasts:
        _write_csv(output / "contrasts.csv", contrasts)
        _write_csv(output / "allocator_diagnostics.csv", diagnostics)
    if failures:
        _write_csv(output / "failures.csv", failures)

    manifest = {
        "plan_sha256": sha256_file(PLAN),
        "d3_manifest_sha256": sha256_file(D3_MANIFEST),
        "d4_manifest_sha256": sha256_file(D4_MANIFEST),
        "simulator_sha256": sha256_file(ROOT / "src/uav_lifecycle/dynamic_simulator.py"),
        "planning_sha256": sha256_file(ROOT / "src/uav_lifecycle/dynamic_planning.py"),
        "scenario_count": len(cases),
        "expected_record_count": len(works),
        "suites": list(SUITES),
        "primaries": list(PRIMARIES),
        "recoveries": list(RECOVERIES),
        "bootstrap_resamples": BOOTSTRAPS,
    }
    write_json_atomic(output / "manifest.json", manifest)
    summary = {
        "status": "COMPLETE" if len(rows) == len(works) and not failures else "FAILED",
        "record_count": len(rows),
        "expected_record_count": len(works),
        "failure_count": len(failures),
        "unsettled_episode_count": sum(row["action_count"] != row["event_count"] for row in rows),
        "workers": workers,
        "contrasts": contrasts,
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=max(1, min(22, os.cpu_count() or 2)))
    args = parser.parse_args()
    summary = run(args.output, args.workers)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

