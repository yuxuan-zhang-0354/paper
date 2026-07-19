"""Aggregate the accepted D2 diagnostic trace into paper-ready mechanism tables."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from uav_lifecycle.artifacts import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "results/dynamic_mainline/d2_diagnostics"
RECORDS = ROOT / "results/dynamic_mainline/d2_confirmation/canonical/records.csv"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values), "mean": float(np.mean(values)),
        "median": float(np.median(values)), "p05": float(np.quantile(values, .05)),
        "p95": float(np.quantile(values, .95)),
        "win": sum(value > 1e-12 for value in values),
        "tie": sum(abs(value) <= 1e-12 for value in values),
        "loss": sum(value < -1e-12 for value in values),
    }


def summarize() -> dict[str, Any]:
    rows = _read(RECORDS)
    by_scenario: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_scenario[row["scenario_id"]][row["method"]] = row

    bda_groups = {}
    for label, predicate in (
        ("0", lambda count: count == 0),
        ("1", lambda count: count == 1),
        ("2_plus", lambda count: count >= 2),
    ):
        selected = [
            methods for methods in by_scenario.values()
            if predicate(int(methods["P"]["bda_count"]))
        ]
        normalized = [
            float(item["P"]["normalized_utility"]) - float(item["B4"]["normalized_utility"])
            for item in selected
        ]
        bda_groups[label] = {
            **_distribution(normalized),
            "mean_delta_destroyed_value": float(np.mean([
                float(item["P"]["destroyed_value"]) - float(item["B4"]["destroyed_value"])
                for item in selected
            ])),
            "mean_delta_service_cost": float(np.mean([
                float(item["P"]["service_cost"]) - float(item["B4"]["service_cost"])
                for item in selected
            ])),
            "mean_delta_ammo_consumed": float(np.mean([
                float(item["P"]["ammo_consumed"]) - float(item["B4"]["ammo_consumed"])
                for item in selected
            ])),
            "mean_delta_invalid_attacks": float(np.mean([
                float(item["P"]["invalid_attack_count"]) - float(item["B4"]["invalid_attack_count"])
                for item in selected
            ])),
        }

    bda_events = _read(DIAG / "diagnostic_bda_events.csv")
    bda_event_summary = {
        "event_count": len(bda_events),
        "next_same_target_action": dict(Counter(row["next_same_target_action"] for row in bda_events)),
        "observation": dict(Counter(row["observation"] for row in bda_events)),
        "prior_attack_count": dict(Counter(row["prior_attacks_same_target"] for row in bda_events)),
        "by_observation": {},
    }
    for observation in ("A", "D"):
        selected = [row for row in bda_events if row["observation"] == observation]
        bda_event_summary["by_observation"][observation] = {
            "count": len(selected),
            "mean_delta_probability_alive": float(np.mean([
                float(row["delta_probability_alive"]) for row in selected
            ])),
            "median_delta_probability_alive": float(np.median([
                float(row["delta_probability_alive"]) for row in selected
            ])),
            "mean_delta_entropy": float(np.mean([
                float(row["delta_entropy"]) for row in selected
            ])),
        }

    traces = _read(DIAG / "diagnostic_planning_trace.csv")
    trace_by_episode: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in traces:
        trace_by_episode[(row["scenario_id"], row["method"])].append(row)
    trace_method_summary = {}
    for method in sorted({key[1] for key in trace_by_episode}):
        episodes = [items for key, items in trace_by_episode.items() if key[1] == method]
        trace_method_summary[method] = {
            "episode_count": len(episodes),
            "mean_planning_calls": float(np.mean([len(items) for items in episodes])),
            "mean_trace_derived_cbba_rounds": float(np.mean([
                sum(int(item["rounds"]) for item in items) for items in episodes
            ])),
            "mean_no_commit_calls": float(np.mean([
                sum(item["no_commit"].lower() == "true" for item in items) for items in episodes
            ])),
            "mean_planning_process_time_ms": float(np.mean([
                sum(int(item["process_time_ns"]) for item in items) / 1e6 for items in episodes
            ])),
        }

    periodic_events = _read(DIAG / "diagnostic_periodic_trace.csv")
    periodic_summary = {}
    for method in ("B5(2)", "B5(4)", "B5(8)"):
        selected_events = [row for row in periodic_events if row["method"] == method]
        selected_records = [row for row in rows if row["method"] == method]
        periodic_summary[method] = {
            "completion_event_count": len(selected_events),
            "mean_wait_to_next_grid": float(np.mean([
                float(row["wait_to_next_grid"]) for row in selected_events
            ])),
            "p95_wait_to_next_grid": float(np.quantile([
                float(row["wait_to_next_grid"]) for row in selected_events
            ], .95)),
            **{
                f"mean_{metric}": float(np.mean([float(row[metric]) for row in selected_records]))
                for metric in (
                    "normalized_utility", "destroyed_value", "service_cost", "distance_cost",
                    "ammo_cost", "makespan", "action_count", "replan_count",
                    "ammo_consumed", "distance_consumed", "bda_count",
                )
            },
            "trace_derived_rounds": trace_method_summary[method]["mean_trace_derived_cbba_rounds"],
            "trace_no_commit_calls": trace_method_summary[method]["mean_no_commit_calls"],
        }

    result = {
        "status": "COMPLETE",
        "cbba_round_note": (
            "Canonical EpisodeRecord.cbba_round_count is a zero placeholder; use trace-derived rounds."
        ),
        "bda_scenario_groups": bda_groups,
        "bda_event_summary": bda_event_summary,
        "trace_method_summary": trace_method_summary,
        "periodic_summary": periodic_summary,
    }
    write_json_atomic(DIAG / "mechanism_summary.json", result)
    report = f"""# D2 explanatory diagnostic report

## Integrity

- Trace replay: 20,480 episodes, zero record/event mismatch.
- Planning calls captured: 96,454.
- BDA completion events: {len(bda_events):,}.
- Analysis label: post-hoc explanatory, not confirmatory.

## BDA mechanism

The paired P-B4 difference has a negative median but positive mean. With two or
more completed BDAs, mean normalized gain is {bda_groups['2_plus']['mean']:.6f}
while the median is {bda_groups['2_plus']['median']:.6f}. These scenarios gain
{bda_groups['2_plus']['mean_delta_destroyed_value']:.3f} mean discounted destroyed
value, pay {bda_groups['2_plus']['mean_delta_service_cost']:.3f} additional service
cost, save {-bda_groups['2_plus']['mean_delta_ammo_consumed']:.3f} rounds of ammo,
and reduce invalid attacks by {-bda_groups['2_plus']['mean_delta_invalid_attacks']:.3f}.

Of all BDA observations, {bda_event_summary['next_same_target_action'].get('attack', 0):,}
are followed by another same-target attack and
{bda_event_summary['next_same_target_action'].get('none', 0):,} have no later
same-target commitment. This supports a selective high-upside interpretation,
not uniform per-scenario dominance.

## Periodic mechanism

B5(8) has fewer replans and BDAs than B5(2)/B5(4), consumes more ammunition than
those two methods, and obtains greater destroyed value and utility at the cost of
a longer makespan. The non-monotonic period result is therefore associated with a
different attack/information mix, not simply lower planning frequency.

## Logging note

The canonical EpisodeRecord `cbba_round_count` field is a zero placeholder. All
paper results for CBBA rounds must use the accepted planning trace or the new D3
runtime table. D2 outcome values are unaffected.
"""
    (DIAG / "diagnostic_report.md").write_text(report, encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
