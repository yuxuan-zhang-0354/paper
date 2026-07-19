"""Build explanatory D2 tables and exact-match decision traces."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
import json
from math import log
from pathlib import Path
from time import process_time_ns
from typing import Any

import numpy as np

from experiments.run_dynamic_mainline import (
    PRIVATE_COLUMNS,
    PUBLIC_COLUMNS,
    WorkerInput,
    _canonical_bytes,
    _csv_text,
    _digest,
    _private_rows,
    _public_rows,
    _record_row,
    _text_atomic,
)
from uav_lifecycle.artifacts import sha256_file, write_json_atomic
from uav_lifecycle.dynamic_policies import make_policy
from uav_lifecycle.dynamic_scenarios import D1_CELLS, dynamic_config_registry, generate_d2_scenario
from uav_lifecycle.dynamic_simulator import run_episode


ROOT = Path(__file__).resolve().parents[1]
D2 = ROOT / "results/dynamic_mainline/d2_confirmation"
DESIGN = ROOT / "results/dynamic_mainline/d2_diagnostics/design_manifest.json"
OUTPUT = ROOT / "results/dynamic_mainline/d2_diagnostics"
TRACE_METHODS = ("P", "B4", "B5(2)", "B5(4)", "B5(8)")
CONTRASTS = ("B1m", "B2", "B3", "B4", "B5(2)", "B5(4)", "B5(8)", "B6", "CEX")
METRICS = (
    "normalized_utility", "destroyed_value", "service_cost", "distance_cost",
    "ammo_cost", "makespan", "final_joint_brier_score", "recon_count", "bda_count",
    "continuous_attack_count", "handoff_count", "replan_count", "cbba_round_count",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _quantiles(values: list[float]) -> dict[str, float]:
    return {
        name: float(np.quantile(values, q))
        for name, q in (("p05", .05), ("p25", .25), ("median", .5), ("p75", .75), ("p95", .95))
    }


def artifact_tables(records: list[dict[str, str]]) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, float]]]:
    methods = sorted({row["method"] for row in records})
    method_rows = []
    for method in methods:
        selected = [row for row in records if row["method"] == method]
        output: dict[str, Any] = {"method": method, "scenario_count": len(selected)}
        for metric in METRICS:
            values = [float(row[metric]) for row in selected]
            output[f"{metric}_mean"] = float(np.mean(values))
            output.update({f"{metric}_{key}": value for key, value in _quantiles(values).items()})
        method_rows.append(output)
    _write_dict_csv(OUTPUT / "diagnostic_method_summary.csv", method_rows)

    by_scenario: dict[str, dict[str, dict[str, str]]] = {}
    for row in records:
        by_scenario.setdefault(row["scenario_id"], {})[row["method"]] = row
    paired_rows = []
    paired_map: dict[tuple[str, str], dict[str, float]] = {}
    summaries = {}
    for comparator in CONTRASTS:
        contrast = f"P-{comparator}"
        values = []
        for scenario_id, methods_by_name in by_scenario.items():
            left, right = methods_by_name["P"], methods_by_name[comparator]
            row: dict[str, Any] = {
                "scenario_id": scenario_id, "cell_id": left["cell_id"], "contrast": contrast,
            }
            for metric in METRICS:
                row[f"delta_{metric}"] = float(left[metric]) - float(right[metric])
            paired_rows.append(row)
            paired_map[(scenario_id, contrast)] = {
                key: float(value) for key, value in row.items() if key.startswith("delta_")
            }
            values.append(row["delta_normalized_utility"])
        summaries[contrast] = {
            "mean": float(np.mean(values)), **_quantiles(values),
            "win": sum(value > 1e-12 for value in values),
            "tie": sum(abs(value) <= 1e-12 for value in values),
            "loss": sum(value < -1e-12 for value in values),
        }
    _write_dict_csv(OUTPUT / "diagnostic_paired_decomposition.csv", paired_rows)
    return summaries, paired_map


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = tuple(rows[0]) if rows else ()
    _text_atomic(path, _csv_text(rows, columns))


class TracePolicy:
    def __init__(self, inner: object):
        self.inner = inner
        self.trace: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def decide(self, snapshot):
        start = process_time_ns()
        decision = self.inner.decide(snapshot)
        elapsed = process_time_ns() - start
        payload = json.loads(decision.planning_bytes or b"{}")
        self.trace.append({
            "call_index": len(self.trace), "trigger": "decide", "tick": snapshot.tick,
            "positive_pair_count": payload.get("positive_pair_count", 0),
            "rounds": payload.get("rounds", 0),
            "proposals": [
                [item.agent_id, item.target_id, str(getattr(item.mode, "value", item.mode))]
                for item in decision.proposals
            ],
            "paths": payload.get("paths", []),
            "path_score_sum": sum(float(item.get("score", 0.0)) for item in payload.get("paths", [])),
            "no_commit": not decision.proposals,
            "target_beliefs": [list(target.belief) for target in snapshot.targets],
            "process_time_ns": elapsed,
        })
        return decision


@dataclass(frozen=True, slots=True)
class TraceWork:
    scenario: Any
    method: str


def _trace_worker(work: TraceWork) -> dict[str, Any]:
    config = dynamic_config_registry()["recon_damage_plus_010_r2_a6_b3"]
    policy = TracePolicy(make_policy(work.method, config))
    try:
        result = run_episode(work.scenario, policy, config=config, method=work.method)
        return {"kind": "result", "result": result, "trace": policy.trace}
    except Exception as error:
        return {
            "kind": "failure", "scenario_id": work.scenario.scenario_id,
            "method": work.method, "error": f"{type(error).__name__}: {error}",
        }


def _normalized(row: dict[str, Any], columns: tuple[str, ...]) -> tuple[str, ...]:
    values = []
    for name in columns:
        value = row.get(name, "")
        if isinstance(value, (dict, tuple, list)):
            value = _canonical_bytes(value).decode("ascii")
        values.append(str(value))
    return tuple(values)


def trace_replay(workers: int = 22) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config_id = "recon_damage_plus_010_r2_a6_b3"
    scenarios = tuple(
        generate_d2_scenario(cell, index, config_id)
        for cell in sorted(D1_CELLS, key=lambda item: item.cell_id)
        for index in range(1000, 1512)
    )
    works = tuple(TraceWork(scenario, method) for scenario in scenarios for method in TRACE_METHODS)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        payloads = list(pool.map(_trace_worker, works))
    failures = [payload for payload in payloads if payload["kind"] != "result"]
    if failures:
        raise RuntimeError(f"trace replay worker failures: {failures[:3]}")

    canonical_records = {
        (row["scenario_id"], row["method"]): row["record_digest"]
        for row in _read_csv(D2 / "canonical/records.csv")
        if row["method"] in TRACE_METHODS
    }
    canonical_public: dict[tuple[str, str], list[tuple[str, ...]]] = {}
    for row in _read_csv(D2 / "canonical/public_events.csv"):
        if row["method"] in TRACE_METHODS:
            canonical_public.setdefault((row["scenario_id"], row["method"]), []).append(
                tuple(row[name] for name in PUBLIC_COLUMNS)
            )
    canonical_private: dict[tuple[str, str], list[tuple[str, ...]]] = {}
    for row in _read_csv(D2 / "canonical/private_audit_events.csv"):
        if row["method"] in TRACE_METHODS:
            canonical_private.setdefault((row["scenario_id"], row["method"]), []).append(
                tuple(row[name] for name in PRIVATE_COLUMNS)
            )

    scenario_map = {scenario.scenario_id: scenario for scenario in scenarios}
    mismatches = []
    trace_rows = []
    for work, payload in zip(works, payloads, strict=True):
        result = payload["result"]
        key = (work.scenario.scenario_id, work.method)
        record = _record_row({"kind": "result", "key": key, "result": result}, scenario_map)
        public = [_normalized(row, PUBLIC_COLUMNS) for row in _public_rows(result)]
        private = [_normalized(row, PRIVATE_COLUMNS) for row in _private_rows(result)]
        if (
            record["record_digest"] != canonical_records.get(key)
            or public != canonical_public.get(key, [])
            or private != canonical_private.get(key, [])
        ):
            mismatches.append(key)
        for item in payload["trace"]:
            trace_rows.append({
                "scenario_id": key[0], "cell_id": result.record.cell_id,
                "method": key[1], **item,
            })
    if mismatches:
        raise RuntimeError(f"trace replay differs from canonical D2: {mismatches[:3]}")
    _write_dict_csv(OUTPUT / "diagnostic_planning_trace.csv", trace_rows)
    return trace_rows, {
        "status": "PASS", "episode_count": len(works), "mismatch_count": 0,
        "trace_row_count": len(trace_rows),
    }


def bda_table(
    trace_rows: list[dict[str, Any]], paired: dict[tuple[str, str], dict[str, float]],
) -> list[dict[str, Any]]:
    traces: dict[str, list[dict[str, Any]]] = {}
    for row in trace_rows:
        if row["method"] == "P":
            traces.setdefault(row["scenario_id"], []).append(row)
    public = [
        row for row in _read_csv(D2 / "canonical/public_events.csv")
        if row["method"] == "P"
    ]
    by_scenario: dict[str, list[dict[str, str]]] = {}
    for row in public:
        by_scenario.setdefault(row["scenario_id"], []).append(row)
    output = []
    for scenario_id, events in by_scenario.items():
        history = sorted(events, key=lambda row: int(row["event_id"]))
        for event in history:
            if event["mode"] != "bda":
                continue
            target = int(event["target_id"])
            tick = int(event["tick"])
            rows = traces[scenario_id]
            prior = [
                row for row in rows if int(row["tick"]) <= tick
                and any(item[1] == target and item[2] == "bda" for item in row["proposals"])
            ][-1]
            after_candidates = [
                row for row in rows
                if int(row["tick"]) == tick and int(row["call_index"]) > int(prior["call_index"])
            ]
            after = after_candidates[0] if after_candidates else None
            before_belief = prior["target_beliefs"][target]
            after_belief = before_belief if after is None else after["target_beliefs"][target]
            later = [
                row for row in rows if int(row["call_index"]) > int(prior["call_index"])
                and any(item[1] == target for item in row["proposals"])
            ]
            next_mode = "none" if not later else next(
                item[2] for item in later[0]["proposals"] if item[1] == target
            )
            entropy = lambda belief: -sum(value * log(value) for value in belief if value > 0)
            attacks = sum(
                other["mode"] == "attack" and int(other["target_id"]) == target
                and int(other["event_id"]) < int(event["event_id"])
                for other in history
            )
            delta = paired[(scenario_id, "P-B4")]
            output.append({
                "scenario_id": scenario_id, "cell_id": event["cell_id"],
                "event_id": int(event["event_id"]), "tick": tick, "target_id": target,
                "observation": event["observation"],
                "belief_before": before_belief, "belief_after": after_belief,
                "probability_alive_before": before_belief[0] + before_belief[2],
                "probability_alive_after": after_belief[0] + after_belief[2],
                "delta_probability_alive": after_belief[0] + after_belief[2] - before_belief[0] - before_belief[2],
                "delta_entropy": entropy(after_belief) - entropy(before_belief),
                "prior_attacks_same_target": attacks,
                "next_same_target_action": next_mode,
                **delta,
            })
    _write_dict_csv(OUTPUT / "diagnostic_bda_events.csv", output)
    return output


def periodic_table(trace_rows: list[dict[str, Any]], records: list[dict[str, str]]) -> list[dict[str, Any]]:
    record_map = {(row["scenario_id"], row["method"]): row for row in records}
    traces: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in trace_rows:
        if row["method"].startswith("B5("):
            traces.setdefault((row["scenario_id"], row["method"]), []).append(row)
    output = []
    for event in _read_csv(D2 / "canonical/public_events.csv"):
        method = event["method"]
        if method not in ("B5(2)", "B5(4)", "B5(8)"):
            continue
        period = int(method[3])
        tick = int(event["tick"])
        period_tick = int(period / 1e-10)
        wait_tick = (-tick) % period_tick
        later = [row for row in traces[(event["scenario_id"], method)] if int(row["tick"]) >= tick]
        next_trace = later[0] if later else None
        record = record_map[(event["scenario_id"], method)]
        output.append({
            "scenario_id": event["scenario_id"], "cell_id": event["cell_id"],
            "method": method, "event_id": event["event_id"], "completion_tick": tick,
            "completed_mode": event["mode"], "wait_to_next_grid": wait_tick * 1e-10,
            "next_grid_path_score": "" if next_trace is None else next_trace["path_score_sum"],
            "replan_count": record["replan_count"], "cbba_round_count": record["cbba_round_count"],
            "ammo_consumed": record["ammo_consumed"], "distance_consumed": record["distance_consumed"],
        })
    _write_dict_csv(OUTPUT / "diagnostic_periodic_trace.csv", output)
    return output


def run() -> dict[str, Any]:
    manifest = json.loads(DESIGN.read_text(encoding="utf-8"))
    for source in manifest["sources"].values():
        if sha256_file(ROOT / source["path"]) != source["sha256"]:
            raise RuntimeError("frozen D2 diagnostic source hash mismatch")
    records = _read_csv(D2 / "canonical/records.csv")
    contrast_summary, paired = artifact_tables(records)
    trace_rows, replay = trace_replay()
    bda = bda_table(trace_rows, paired)
    periodic = periodic_table(trace_rows, records)
    summary = {
        "status": "COMPLETE", "analysis_label": "post_hoc_explanatory_not_confirmatory",
        "source_manifest_digest": manifest["source_manifest_digest"],
        "trace_replay": replay, "contrast_distributions": contrast_summary,
        "bda_event_count": len(bda), "periodic_event_count": len(periodic),
        "algorithm_or_parameter_changed": False,
    }
    write_json_atomic(OUTPUT / "diagnostic_summary.json", summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "status": result["status"], "trace_replay": result["trace_replay"],
        "bda_event_count": result["bda_event_count"],
        "periodic_event_count": result["periodic_event_count"],
    }, indent=2))
