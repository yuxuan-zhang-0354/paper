"""Independent-seed pilot for the frozen D3 allocation-pressure design."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from time import process_time_ns
from typing import Any

import numpy as np

from experiments.run_dynamic_mainline import _csv_text, _text_atomic
from uav_lifecycle.artifacts import write_json_atomic
from uav_lifecycle.dynamic_d3 import (
    ALLOCATION_PRESSURE_CONDITIONS,
    generate_allocation_pressure,
)
from uav_lifecycle.dynamic_policies import make_policy
from uav_lifecycle.dynamic_simulator import run_episode
from uav_lifecycle.dynamic_types import DynamicConfig


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/dynamic_mainline/allocation_pressure_pilot"
METHODS = ("P", "SCBBA", "DVCBBA", "B6")


@dataclass(frozen=True, slots=True)
class Work:
    condition: str
    index: int
    method: str


def _worker(work: Work) -> dict[str, Any]:
    config = DynamicConfig()
    scenario = generate_allocation_pressure(work.condition, work.index, config, pilot=True)
    policy = make_policy(work.method, config)
    start = process_time_ns()
    try:
        result = run_episode(scenario, policy, config=config, method=work.method)
        audits = policy.allocator_audits
        statuses = Counter(str(audit["allocator_status"]) for audit in audits)
        report = defaultdict(int)
        for audit in audits:
            for name, value in audit["audit_report"]:
                report[str(name)] += int(value)
        positive = sum(int(audit["screened_positive_pair_count"]) for audit in audits)
        eligible = sum(int(audit["eligible_pair_count"]) for audit in audits)
        objectives = [float(audit["allocation_objective"]) for audit in audits]
        record = result.record
        return {
            "condition": work.condition, "index": work.index,
            "scenario_id": scenario.scenario_id, "method": work.method,
            "status": record.status, "termination": record.termination,
            "normalized_utility": record.normalized_utility,
            "realized_utility": record.realized_utility,
            "destroyed_value": record.destroyed_value,
            "service_cost": record.service_cost,
            "distance_cost": record.distance_cost,
            "ammo_cost": record.ammo_cost,
            "makespan": record.makespan,
            "allocator_calls": len(audits),
            "converged_calls": statuses["converged"],
            "cycle_calls": statuses["cycle"],
            "round_cap_calls": statuses["timeout"],
            "winner_conflicts": report["winner_conflicts"],
            "rounds": sum(int(audit["rounds"]) for audit in audits),
            "message_packets": sum(int(audit["message_packets"]) for audit in audits),
            "message_scalars": sum(int(audit["message_scalars"]) for audit in audits),
            "positive_pairs": positive, "eligible_pairs": eligible,
            "positive_pair_density": positive / eligible if eligible else 0.0,
            "mean_allocation_objective": float(np.mean(objectives)) if objectives else 0.0,
            "process_time_ns": process_time_ns() - start,
            "gate_count": len(record.allocator_gates),
        }
    except Exception as error:
        return {
            "condition": work.condition, "index": work.index, "method": work.method,
            "status": "failed", "error": f"{type(error).__name__}: {error}",
            "process_time_ns": process_time_ns() - start,
        }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for condition in ALLOCATION_PRESSURE_CONDITIONS:
        cells[condition] = {}
        for method in METHODS:
            selected = [row for row in rows if row["condition"] == condition and row["method"] == method]
            cells[condition][method] = {
                "episodes": len(selected),
                "mean_normalized_utility": float(np.mean([row["normalized_utility"] for row in selected])),
                "mean_positive_pair_density": float(np.mean([row["positive_pair_density"] for row in selected])),
                "episodes_with_nonconvergence": sum(
                    row["cycle_calls"] + row["round_cap_calls"] > 0 for row in selected
                ),
                "cycle_calls": sum(row["cycle_calls"] for row in selected),
                "round_cap_calls": sum(row["round_cap_calls"] for row in selected),
                "winner_conflicts": sum(row["winner_conflicts"] for row in selected),
                "mean_rounds": float(np.mean([row["rounds"] for row in selected])),
                "mean_message_packets": float(np.mean([row["message_packets"] for row in selected])),
                "mean_allocation_objective": float(np.mean([row["mean_allocation_objective"] for row in selected])),
            }
        by_method = {
            method: {row["index"]: row for row in rows if row["condition"] == condition and row["method"] == method}
            for method in METHODS
        }
        cells[condition]["paired_gaps"] = {}
        for baseline in ("SCBBA", "DVCBBA", "B6"):
            differences = [
                by_method["P"][index]["normalized_utility"]
                - by_method[baseline][index]["normalized_utility"]
                for index in sorted(by_method["P"])
            ]
            cells[condition]["paired_gaps"][f"P-{baseline}"] = {
                "mean": float(np.mean(differences)),
                "median": float(np.median(differences)),
                "win_tie_loss": [
                    sum(value > 1e-12 for value in differences),
                    sum(abs(value) <= 1e-12 for value in differences),
                    sum(value < -1e-12 for value in differences),
                ],
            }
    return cells


def run(workers: int = 22) -> dict[str, Any]:
    works = tuple(
        Work(condition, index, method)
        for condition in ALLOCATION_PRESSURE_CONDITIONS
        for index in range(6500, 6564)
        for method in METHODS
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_worker, works))
    failures = [row for row in rows if row["status"] == "failed"]
    gates = [row for row in rows if int(row.get("gate_count", 0))]
    if failures or gates:
        result = {
            "status": "FAILED_INCOMPLETE", "records": len(rows),
            "failures": failures[:10], "gate_rows": gates[:10],
        }
    else:
        result = {
            "status": "COMPLETE", "label": "design_pilot_not_confirmatory",
            "formal_d3_scenarios_reused": False, "records": len(rows),
            "scenario_count": len(works) // len(METHODS),
            "methods": list(METHODS), "conditions": list(ALLOCATION_PRESSURE_CONDITIONS),
            "cells": _summarize(rows),
        }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _text_atomic(OUTPUT / "records.csv", _csv_text(rows, tuple(rows[0])))
    write_json_atomic(OUTPUT / "summary.json", result)
    return result


if __name__ == "__main__":
    summary = run()
    print(json.dumps({key: value for key, value in summary.items() if key != "cells"}, indent=2))
    raise SystemExit(0 if summary["status"] == "COMPLETE" else 1)
