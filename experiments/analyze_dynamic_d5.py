"""Registered D5 factorial summaries and paired scenario bootstrap intervals."""

from __future__ import annotations

from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from experiments.run_dynamic_mainline import _csv_text, _text_atomic
from uav_lifecycle.artifacts import sha256_file, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/dynamic_mainline/d5_factorial_ablation"
SOURCE = BASE / "formal/canonical"
MANIFEST = BASE / "design/d5_manifest.json"
OUTPUT = BASE / "analysis"
METHODS = ("V00", "V01", "V10", "V11")
BOOTSTRAPS = 10_000
TOL = 1e-12


def _seed(*parts: object) -> int:
    return int.from_bytes(sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def _ci(values: np.ndarray, token: str) -> tuple[float, float, float]:
    rng = np.random.default_rng(_seed(token))
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))]
    means = draws.mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _write(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    fields += sorted({key for row in rows for key in row} - set(fields))
    _text_atomic(path, _csv_text(rows, tuple(fields)))


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    metadata = manifest["scenario_metadata"]
    records: dict[tuple[str, str], dict] = {}
    with (SOURCE / "d5_records.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            meta = metadata[row["scenario_id"]]
            row.update(suite=meta["suite"], condition=meta["condition"])
            for name in ("normalized_utility", "action_count", "replan_count"):
                row[name] = float(row[name])
            records[row["scenario_id"], row["method"]] = row

    runtime: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with (SOURCE / "d5_runtime.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            state = runtime[row["scenario_id"], row["method"]]
            positive = int(row["positive_pairs"]) > 0
            legal = (
                row["allocator_status"] == "converged"
                and int(row["winner_conflicts"]) == 0
                and float(row["allocation_objective"]) > TOL
            )
            state["planning_calls"] += 1
            state["positive_epochs"] += positive
            state["legal_commit_epochs"] += positive and legal
            state["allocation_stalls"] += positive and not legal
            state["cycles"] += row["allocator_status"] == "cycle"
            state["round_caps"] += row["allocator_status"] == "timeout"
            state["rounds"] += int(row["rounds"])
            state["message_packets"] += int(row["message_packets"])
            state["winner_conflicts"] += int(row["winner_conflicts"])
            state["planning_process_time_ns"] += int(row["planning_process_time_ns"])
            state["warping_activations"] += int(row["warping_activations"])
            state["raw_prefix_increases"] += int(row["raw_prefix_increases"])

    episode_rows = []
    for key, record in records.items():
        state = runtime[key]
        commit_epochs = state["legal_commit_epochs"]
        commits = record["action_count"]
        positive = state["positive_epochs"]
        episode_rows.append({
            "scenario_id": key[0], "suite": record["suite"],
            "stratum": record["condition"] if record["suite"] == "pressure" else record["cell_id"],
            "method": key[1], "normalized_utility": record["normalized_utility"],
            "action_count": record["action_count"], "planning_calls": state["planning_calls"],
            "positive_epochs": positive, "legal_commit_epochs": commit_epochs,
            "legal_commit_epoch_rate": commit_epochs / positive if positive else 1.0,
            "allocation_stalls": state["allocation_stalls"],
            "allocation_stall_rate": state["allocation_stalls"] / positive if positive else 0.0,
            "cycles": state["cycles"], "round_caps": state["round_caps"],
            "nonconverged_episode": int(state["cycles"] + state["round_caps"] > 0),
            "winner_conflicts": state["winner_conflicts"],
            "rounds_per_commit": state["rounds"] / commits if commits else 0.0,
            "messages_per_commit": state["message_packets"] / commits if commits else 0.0,
            "planning_ms_per_commit": state["planning_process_time_ns"] / 1e6 / commits if commits else 0.0,
            "warping_activations": state["warping_activations"],
            "raw_prefix_increases": state["raw_prefix_increases"],
        })
    _write(OUTPUT / "episode_diagnostics.csv", episode_rows)

    summaries = []
    for suite in ("pressure", "scale"):
        for method in METHODS:
            selected = [row for row in episode_rows if row["suite"] == suite and row["method"] == method]
            summaries.append({
                "suite": suite, "method": method, "episodes": len(selected),
                **{
                    f"mean_{name}": float(np.mean([row[name] for row in selected]))
                    for name in (
                        "normalized_utility", "legal_commit_epoch_rate", "allocation_stall_rate",
                        "nonconverged_episode", "cycles", "round_caps", "winner_conflicts",
                        "rounds_per_commit", "messages_per_commit", "planning_ms_per_commit",
                        "warping_activations", "raw_prefix_increases",
                    )
                },
                "total_cycles": int(sum(row["cycles"] for row in selected)),
                "total_round_caps": int(sum(row["round_caps"] for row in selected)),
                "total_stalls": int(sum(row["allocation_stalls"] for row in selected)),
            })
    _write(OUTPUT / "method_summary.csv", summaries)

    nested = defaultdict(dict)
    for row in episode_rows:
        nested[row["scenario_id"]][row["method"]] = row
    contrasts = []
    effects = []
    for suite in ("pressure", "scale"):
        ids = sorted({row["scenario_id"] for row in episode_rows if row["suite"] == suite})
        strata = sorted({nested[sid]["V11"]["stratum"] for sid in ids})
        for stratum in ("ALL", *strata):
            chosen = ids if stratum == "ALL" else [sid for sid in ids if nested[sid]["V11"]["stratum"] == stratum]
            for baseline in ("V00", "V01", "V10"):
                values = np.asarray([
                    nested[sid]["V11"]["normalized_utility"] - nested[sid][baseline]["normalized_utility"]
                    for sid in chosen
                ])
                mean, low, high = _ci(values, f"{suite}|{stratum}|V11-{baseline}")
                contrasts.append({
                    "suite": suite, "stratum": stratum, "contrast": f"V11-{baseline}",
                    "scenarios": len(values), "mean_difference": mean,
                    "ci_low": low, "ci_high": high,
                    "win": int(np.sum(values > TOL)), "tie": int(np.sum(abs(values) <= TOL)),
                    "loss": int(np.sum(values < -TOL)),
                })
            per_scenario = {
                "warping": np.asarray([
                    ((nested[sid]["V10"]["normalized_utility"] - nested[sid]["V00"]["normalized_utility"])
                    + (nested[sid]["V11"]["normalized_utility"] - nested[sid]["V01"]["normalized_utility"])) / 2
                    for sid in chosen
                ]),
                "full_reconstruction": np.asarray([
                    ((nested[sid]["V01"]["normalized_utility"] - nested[sid]["V00"]["normalized_utility"])
                    + (nested[sid]["V11"]["normalized_utility"] - nested[sid]["V10"]["normalized_utility"])) / 2
                    for sid in chosen
                ]),
                "interaction": np.asarray([
                    nested[sid]["V11"]["normalized_utility"] - nested[sid]["V10"]["normalized_utility"]
                    - nested[sid]["V01"]["normalized_utility"] + nested[sid]["V00"]["normalized_utility"]
                    for sid in chosen
                ]),
            }
            for name, values in per_scenario.items():
                mean, low, high = _ci(values, f"{suite}|{stratum}|{name}")
                effects.append({
                    "suite": suite, "stratum": stratum, "effect": name,
                    "scenarios": len(values), "mean": mean, "ci_low": low, "ci_high": high,
                })
    _write(OUTPUT / "paired_contrasts.csv", contrasts)
    _write(OUTPUT / "factorial_effects.csv", effects)

    summary = {
        "status": "D5_ANALYSIS_COMPLETE", "bootstrap_iterations": BOOTSTRAPS,
        "records": len(records), "episodes": len(episode_rows),
        "commit_epoch_count_never_exceeds_action_count": all(
            row["legal_commit_epochs"] <= row["action_count"] + TOL for row in episode_rows
        ),
        "source_summary_sha256": sha256_file(BASE / "formal/summary.json"),
    }
    write_json_atomic(OUTPUT / "summary.json", summary)
    inventory = {
        str(path.relative_to(OUTPUT)): sha256_file(path)
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name != "artifact_inventory.json"
    }
    write_json_atomic(OUTPUT / "artifact_inventory.json", inventory)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
