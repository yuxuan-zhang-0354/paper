"""Post-hoc D2 diagnostic for standard one-shot and dynamic vanilla CBBA."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from time import process_time_ns
from typing import Any

import numpy as np

from experiments.run_dynamic_mainline import _csv_text, _text_atomic
from uav_lifecycle.artifacts import write_json_atomic
from uav_lifecycle.dynamic_planning import build_planning_problem
from uav_lifecycle.dynamic_policies import make_policy
from uav_lifecycle.dynamic_scenarios import D1_CELLS, generate_d2_scenario
from uav_lifecycle.dynamic_simulator import run_episode
from uav_lifecycle.dynamic_types import DynamicConfig
from uav_lifecycle.mode_cbba import screen_modes
from uav_lifecycle.mode_exact import solve_all_mode_exact, solve_fixed_mode_exact


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/dynamic_mainline/cbba_baselines"
CONFIG_ID = "recon_damage_plus_010_r2_a6_b3"


@dataclass(frozen=True, slots=True)
class Work:
    scenario: Any
    method: str
    exact_audit: bool = False


class ExactGapPolicy:
    def __init__(self, inner: object):
        self.inner = inner
        self.max_tick: int | None = None
        self.exact_rows: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def bind_horizon(self, max_tick: int) -> None:
        self.max_tick = max_tick
        self.inner.bind_horizon(max_tick)

    def decide(self, snapshot):
        if self.max_tick is None:
            raise RuntimeError("horizon not bound")
        config = self.inner.config
        problem = build_planning_problem(snapshot, config, self.max_tick)
        screened = screen_modes(problem.instance)
        fixed_modes = {
            target_id: task.mode for target_id, task in enumerate(screened) if task is not None
        }
        fixed = solve_fixed_mode_exact(problem.instance, fixed_modes)
        all_mode = solve_all_mode_exact(problem.instance)
        decision = self.inner.decide(snapshot)
        audit = self.inner.allocator_audits[-1]
        objective = float(audit["allocation_objective"])
        self.exact_rows.append({
            "tick": snapshot.tick,
            "allocator_status": audit["allocator_status"],
            "allocator_objective": objective,
            "fixed_mode_exact_objective": fixed.score,
            "all_mode_exact_objective": all_mode.score,
            "allocation_gap": fixed.score - objective,
            "screening_gap": all_mode.score - fixed.score,
            "cex_gap": all_mode.score - objective,
        })
        return decision


def _worker(work: Work) -> dict[str, Any]:
    config = DynamicConfig()
    inner = make_policy(work.method, config)
    policy = ExactGapPolicy(inner) if work.exact_audit else inner
    start = process_time_ns()
    try:
        result = run_episode(work.scenario, policy, config=config, method=work.method)
        audits = inner.allocator_audits
        return {
            "kind": "result", "scenario_id": work.scenario.scenario_id,
            "cell_id": work.scenario.cell_id, "method": work.method,
            "record": result.record, "audits": audits,
            "exact_rows": [] if not work.exact_audit else policy.exact_rows,
            "process_time_ns": process_time_ns() - start,
        }
    except Exception as error:
        return {
            "kind": "failure", "scenario_id": work.scenario.scenario_id,
            "cell_id": work.scenario.cell_id, "method": work.method,
            "error": f"{type(error).__name__}: {error}",
        }


def _scenarios() -> tuple[Any, ...]:
    return tuple(
        generate_d2_scenario(cell, index, CONFIG_ID)
        for cell in sorted(D1_CELLS, key=lambda item: item.cell_id)
        for index in range(1000, 1512)
    )


def _run(works: tuple[Work, ...], workers: int = 22) -> list[dict[str, Any]]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_worker, works))


def _audit_summary(payloads: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for payload in payloads:
        if payload["kind"] != "result":
            continue
        audits = payload["audits"]
        report = defaultdict(int)
        for audit in audits:
            for name, value in audit["audit_report"]:
                report[name] += int(value)
        statuses = Counter(audit["allocator_status"] for audit in audits)
        record = payload["record"]
        rows.append({
            "scenario_id": payload["scenario_id"], "cell_id": payload["cell_id"],
            "method": payload["method"], "episode_status": record.status,
            "termination": record.termination, "normalized_utility": record.normalized_utility,
            "realized_utility": record.realized_utility,
            "allocator_calls": len(audits), "converged_calls": statuses["converged"],
            "cycle_calls": statuses["cycle"], "timeout_calls": statuses["timeout"],
            "winner_conflicts": report["winner_conflicts"],
            "infeasible_paths": report["infeasible_paths"],
            "bundle_path_mismatches": report["bundle_path_mismatches"],
            "rounds": sum(int(audit["rounds"]) for audit in audits),
            "message_packets": sum(int(audit["message_packets"]) for audit in audits),
            "message_scalars": sum(int(audit["message_scalars"]) for audit in audits),
            "allocation_objective_sum": sum(float(audit["allocation_objective"]) for audit in audits),
            "process_time_ns": payload["process_time_ns"],
        })
    summary = {}
    for method in ("SCBBA", "DVCBBA"):
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            "episodes": len(selected),
            "episodes_with_allocator_nonconvergence": sum(
                row["cycle_calls"] + row["timeout_calls"] > 0 for row in selected
            ),
            "allocator_calls": sum(row["allocator_calls"] for row in selected),
            "cycle_calls": sum(row["cycle_calls"] for row in selected),
            "timeout_calls": sum(row["timeout_calls"] for row in selected),
            "winner_conflicts": sum(row["winner_conflicts"] for row in selected),
            "mean_rounds": float(np.mean([row["rounds"] for row in selected])),
            "mean_message_packets": float(np.mean([row["message_packets"] for row in selected])),
            "mean_message_scalars": float(np.mean([row["message_scalars"] for row in selected])),
            "mean_normalized_utility": float(np.mean([row["normalized_utility"] for row in selected])),
        }
    return rows, summary


def run() -> dict[str, Any]:
    scenarios = _scenarios()
    works = tuple(Work(scenario, method) for scenario in scenarios for method in ("SCBBA", "DVCBBA"))
    payloads = _run(works)
    failures = [payload for payload in payloads if payload["kind"] == "failure"]
    if failures:
        raise RuntimeError(f"baseline worker failure: {failures[:3]}")
    rows, summary = _audit_summary(payloads)
    _text_atomic(OUTPUT / "cbba_baseline_records.csv", _csv_text(rows, tuple(rows[0])))

    exact_scenarios = tuple(
        scenario for scenario in scenarios
        if int(scenario.scenario_id.rsplit("S", 1)[1]) < 1016
    )
    exact_works = tuple(
        Work(scenario, method, True)
        for scenario in exact_scenarios for method in ("P", "SCBBA", "DVCBBA")
    )
    exact_payloads = _run(exact_works)
    exact_rows = []
    for payload in exact_payloads:
        if payload["kind"] != "result":
            raise RuntimeError(f"exact audit worker failure: {payload}")
        for call_index, row in enumerate(payload["exact_rows"]):
            exact_rows.append({
                "scenario_id": payload["scenario_id"], "cell_id": payload["cell_id"],
                "method": payload["method"], "call_index": call_index, **row,
            })
    _text_atomic(OUTPUT / "cbba_exact_gap.csv", _csv_text(exact_rows, tuple(exact_rows[0])))
    exact_summary = {}
    for method in ("P", "SCBBA", "DVCBBA"):
        selected = [row for row in exact_rows if row["method"] == method]
        exact_summary[method] = {
            "allocator_epochs": len(selected),
            "convergence_rate": sum(row["allocator_status"] == "converged" for row in selected) / len(selected),
            "mean_allocation_gap": float(np.mean([row["allocation_gap"] for row in selected])),
            "mean_cex_gap": float(np.mean([row["cex_gap"] for row in selected])),
        }
    result = {
        "status": "COMPLETE", "analysis_label": "post_hoc_baseline_diagnostic",
        "worker_failures": 0, "methods": summary, "exact_subset": exact_summary,
        "formal_d3_claim": False,
    }
    write_json_atomic(OUTPUT / "cbba_baseline_summary.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
