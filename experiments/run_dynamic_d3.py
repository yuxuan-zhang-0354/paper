"""Manifest-driven D3 runner. Formal execution requires a matching authorization."""

from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from time import process_time_ns
from typing import Any

from experiments.build_d3_manifests import canonical_digest
from experiments.run_dynamic_mainline import (
    PRIVATE_COLUMNS, PUBLIC_COLUMNS, _csv_text, _private_rows, _public_rows,
    _record_row, _text_atomic, _write_records,
)
from uav_lifecycle.artifacts import sha256_file, write_json_atomic
from uav_lifecycle.dynamic_d3 import (
    D3_SCALE_CELLS, environment_model, generate_allocation_pressure,
    generate_cbba_isolation, generate_continuous, generate_mismatch,
    generate_scale, generate_weight, utility_config,
)
from uav_lifecycle.dynamic_planning import build_planning_problem
from uav_lifecycle.dynamic_policies import make_policy
from uav_lifecycle.dynamic_scenarios import D1_CELLS
from uav_lifecycle.dynamic_simulator import run_episode
from uav_lifecycle.dynamic_types import DynamicConfig, EnvironmentModel
from uav_lifecycle.mode_cbba import screen_modes
from uav_lifecycle.mode_exact import solve_all_mode_exact, solve_fixed_mode_exact


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results/dynamic_mainline/d3_design/d3_manifest.json"
DEFAULT_AUTHORIZATION = ROOT / "results/dynamic_mainline/d3_design/execution_authorization.json"
DEFAULT_OUTPUT = ROOT / "results/dynamic_mainline/d3_external_validation"


@dataclass(frozen=True, slots=True)
class Work:
    suite: str
    scenario: Any
    config: Any
    environment: EnvironmentModel
    method: str


class ExactAuditPolicy:
    def __init__(self, inner: Any):
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
        problem = build_planning_problem(snapshot, self.inner.config, self.max_tick)
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
            "allocator_objective": objective,
            "fixed_mode_exact_objective": fixed.score,
            "all_mode_exact_objective": all_mode.score,
            "fixed_screened_task_exact_gap": fixed.score - objective,
            "all_mode_cex_gap": all_mode.score - objective,
        })
        return decision


class TimedPolicy:
    def __init__(self, inner: Any):
        self.inner = inner
        self.planning_process_time_ns: list[int] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def decide(self, snapshot):
        start = process_time_ns()
        decision = self.inner.decide(snapshot)
        elapsed = process_time_ns() - start
        if decision.planning_bytes:
            self.planning_process_time_ns.append(elapsed)
        return decision


def load_frozen_design(manifest_path: Path, authorization_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if canonical_digest(unsigned) != manifest.get("manifest_digest"):
        raise RuntimeError("D3 manifest digest mismatch")
    if not authorization_path.is_file():
        raise RuntimeError("formal D3 execution is not authorized")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if not authorization.get("authorized") or authorization.get("manifest_digest") != manifest["manifest_digest"]:
        raise RuntimeError("formal D3 execution is not authorized for this manifest")
    return manifest


def _scenario(meta: dict[str, Any]) -> tuple[Any, Any, EnvironmentModel]:
    nominal = DynamicConfig()
    base_cells = {cell.cell_id: cell for cell in D1_CELLS}
    scale_cells = {cell.cell_id: cell for cell in D3_SCALE_CELLS}
    suite, condition, cell_id, index = (
        meta["suite"], meta["condition"], meta["cell_id"], int(meta["index"]),
    )
    if suite == "scale":
        config = nominal
        scenario = generate_scale(scale_cells[cell_id], index, config)
        environment = EnvironmentModel.from_config(config)
    elif suite == "continuous_belief":
        config = nominal
        scenario = generate_continuous(base_cells[cell_id], index, config)
        environment = EnvironmentModel.from_config(config)
    elif suite == "model_mismatch":
        config = nominal
        scenario = generate_mismatch(base_cells[cell_id], condition, index, config)
        environment = environment_model(condition, config)
    elif suite == "utility_profile":
        config = utility_config(condition)
        scenario = generate_weight(base_cells[cell_id], condition, index, config)
        environment = EnvironmentModel.from_config(config)
    elif suite == "cbba_isolation":
        config = nominal
        scenario = generate_cbba_isolation(base_cells[cell_id], index, config)
        environment = EnvironmentModel.from_config(config)
    elif suite == "allocation_pressure":
        config = nominal
        scenario = generate_allocation_pressure(condition, index, config)
        environment = EnvironmentModel.from_config(config)
    else:
        raise ValueError(f"unknown D3 suite: {suite}")
    return scenario, config, environment


def build_works(manifest: dict[str, Any]) -> tuple[Work, ...]:
    generated = {
        scenario_id: _scenario(manifest["scenario_metadata"][scenario_id])
        for scenario_id in manifest["scenario_ids"]
    }
    if [generated[item][0].scenario_id for item in manifest["scenario_ids"]] != manifest["scenario_ids"]:
        raise RuntimeError("generated D3 scenario IDs do not match manifest")
    works = tuple(
        Work(
            manifest["scenario_metadata"][scenario_id]["suite"],
            generated[scenario_id][0], generated[scenario_id][1], generated[scenario_id][2], method,
        )
        for scenario_id, method in manifest["expected_rectangle"]
    )
    if [[work.scenario.scenario_id, work.method] for work in works] != manifest["expected_rectangle"]:
        raise RuntimeError("generated D3 rectangle does not match manifest")
    return works


def _worker(work: Work) -> dict[str, Any]:
    inner = make_policy(work.method, work.config)
    timed = TimedPolicy(inner)
    exact = work.suite == "cbba_isolation" and work.method in {"P", "SCBBA", "DVCBBA"}
    policy = ExactAuditPolicy(timed) if exact else timed
    start = process_time_ns()
    try:
        result = run_episode(
            work.scenario, policy, config=work.config,
            environment_model=work.environment, method=work.method,
        )
        payload = {"kind": "result", "key": (work.scenario.scenario_id, work.method), "result": result}
        record = _record_row(payload, {work.scenario.scenario_id: work.scenario})
        runtime = []
        exact_rows = policy.exact_rows if exact else []
        for call_index, audit in enumerate(inner.allocator_audits):
            report = dict(audit["audit_report"])
            row = {
                "scenario_id": work.scenario.scenario_id, "cell_id": work.scenario.cell_id,
                "suite": work.suite, "method": work.method, "call_index": call_index,
                "allocator_status": audit["allocator_status"], "rounds": audit["rounds"],
                "message_packets": audit["message_packets"], "message_scalars": audit["message_scalars"],
                "allocation_objective": audit["allocation_objective"],
                "positive_pairs": audit["screened_positive_pair_count"],
                "eligible_pairs": audit["eligible_pair_count"],
                "winner_conflicts": report.get("winner_conflicts", 0),
                "warping_activations": audit.get("warping_activations", 0),
                "raw_prefix_increases": audit.get("raw_prefix_increases", 0),
                "planning_process_time_ns": timed.planning_process_time_ns[call_index],
            }
            if call_index < len(exact_rows):
                row.update(exact_rows[call_index])
            runtime.append(row)
        return {
            "kind": "result", "key": payload["key"], "record": record,
            "public": _public_rows(result), "private": _private_rows(result),
            "runtime": runtime, "episode_process_time_ns": process_time_ns() - start,
        }
    except Exception as error:
        return {
            "kind": "failure", "key": (work.scenario.scenario_id, work.method),
            "error_type": type(error).__name__, "error_message": str(error),
            "episode_process_time_ns": process_time_ns() - start,
        }


def _execute(works: tuple[Work, ...], workers: int) -> list[dict[str, Any]]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_worker, works))


def _health(payloads: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, int]:
    keys = [tuple(payload["key"]) for payload in payloads]
    expected = {tuple(item) for item in manifest["expected_rectangle"]}
    records = [payload["record"] for payload in payloads if payload["kind"] == "result"]
    numeric = ("normalized_utility", "realized_utility", "gross_scenario_value")
    return {
        "record_count": len(records), "missing_count": len(expected - set(keys)),
        "extra_count": len(set(keys) - expected), "duplicate_count": len(keys) - len(set(keys)),
        "failure_count": sum(payload["kind"] != "result" for payload in payloads),
        "nonterminal_count": sum(row.get("terminal") is not True for row in records),
        "gate_count": sum(len(row.get("allocator_gates", ())) for row in records),
        "nan_count": sum(
            not math.isfinite(float(row[name])) for row in records for name in numeric
        ),
    }


def _write(
    destination: Path, payloads: list[dict[str, Any]], method_order: list[str],
    prefix: str = "d3",
) -> None:
    records, public, private, runtime = _projection(payloads, method_order)
    _write_records(destination / f"{prefix}_records.csv", records)
    _text_atomic(destination / f"{prefix}_public_events.csv", _csv_text(public, PUBLIC_COLUMNS))
    _text_atomic(destination / f"{prefix}_private_audit_events.csv", _csv_text(private, PRIVATE_COLUMNS))
    if runtime:
        base = list(runtime[0])
        extra = sorted({name for row in runtime for name in row} - set(base))
        _text_atomic(destination / f"{prefix}_runtime.csv", _csv_text(runtime, tuple(base + extra)))


def _projection(
    payloads: list[dict[str, Any]], method_order: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    results = [payload for payload in payloads if payload["kind"] == "result"]
    records = [payload["record"] for payload in results]
    public = [row for payload in results for row in payload["public"]]
    private = [row for payload in results for row in payload["private"]]
    runtime = [row for payload in results for row in payload["runtime"]]
    order = {method: index for index, method in enumerate(method_order)}
    records.sort(key=lambda row: (row["scenario_id"], order[row["method"]]))
    def event_key(row: dict[str, Any]) -> tuple[str, int, int]:
        return row["scenario_id"], order[row["method"]], row["event_id"]
    public.sort(key=event_key)
    private.sort(key=event_key)
    runtime.sort(key=lambda row: (row["scenario_id"], order[row["method"]], row["call_index"]))
    return records, public, private, runtime


def run(manifest_path: Path, authorization_path: Path, output: Path, workers: int) -> dict[str, Any]:
    manifest = load_frozen_design(manifest_path, authorization_path)
    works = build_works(manifest)
    payloads = _execute(works, workers)
    health = _health(payloads, manifest)
    _write(output / "canonical", payloads, manifest["methods"])
    expected = manifest["expected_record_count"]
    complete = health == {
        "record_count": expected, "missing_count": 0, "extra_count": 0,
        "duplicate_count": 0, "failure_count": 0, "nonterminal_count": 0,
        "gate_count": 0, "nan_count": 0,
    }
    replay: dict[str, Any] = {"status": "NOT_RUN"}
    if complete:
        replay_ids = set(manifest["replay_audit"]["scenario_ids"])
        replay_works = tuple(work for work in works if work.scenario.scenario_id in replay_ids)
        canonical = _projection(
            [payload for payload in payloads if payload["key"][0] in replay_ids], manifest["methods"],
        )[:3]
        mismatches = {}
        for replay_workers in manifest["replay_audit"]["workers"]:
            replay_payloads = _execute(replay_works, replay_workers)
            replay_health = _health(replay_payloads, {
                **manifest,
                "expected_rectangle": [[work.scenario.scenario_id, work.method] for work in replay_works],
            })
            actual = _projection(replay_payloads, manifest["methods"])[:3]
            mismatches[str(replay_workers)] = actual != canonical or replay_health["failure_count"] > 0
        replay = {"status": "PASS" if not any(mismatches.values()) else "FAILED", "mismatches": mismatches}
    complete = complete and replay["status"] == "PASS"
    summary = {
        "status": "D3_COMPLETE" if complete else "D3_FAILED_INCOMPLETE",
        "manifest_digest": manifest["manifest_digest"], "health": health,
        "replay": replay,
        "canonical_workers": workers,
        "baseline_nonconvergence_is_measured_outcome": True,
    }
    write_json_atomic(output / "summary.json", summary)
    inventory = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "artifact_inventory.json"
    }
    write_json_atomic(output / "artifact_inventory.json", inventory)
    return summary


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=max(1, min(22, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()
    result = run(args.manifest, args.authorization, args.output, args.workers)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "D3_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
