"""Diagnostic decomposition for the frozen D6 failure-policy validation."""

from __future__ import annotations

import csv
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_dynamic_mainline import _csv_text, _text_atomic
from uav_lifecycle.artifacts import sha256_file, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/dynamic_mainline/d6_failure_policy_validation"
SUITES = ("scale", "allocation_pressure", "battlefield_structure", "reachability")
BOOTSTRAPS = 10_000
NUMERIC = (
    "normalized_utility", "primary_failure_count", "initial_failure", "cycle_count",
    "fallback_count", "primary_rounds", "primary_packets", "primary_target_entries",
    "action_count", "event_count",
)


def _rows() -> list[dict[str, Any]]:
    with (SOURCE / "records.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        for name in NUMERIC:
            row[name] = float(row[name])
    return rows


def _metadata() -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for path in (
        ROOT / "results/dynamic_mainline/d3_design/d3_manifest.json",
        ROOT / "results/dynamic_mainline/d4_design/d4_manifest.json",
    ):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        metadata.update(manifest["scenario_metadata"])
    return metadata


def _seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _blocked_contrasts(
    rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    nested = {
        (row["scenario_id"], row["primary"], row["recovery"]): row for row in rows
    }
    output = []
    for suite in SUITES:
        scenario_ids = sorted({row["scenario_id"] for row in rows if row["suite"] == suite})
        for recovery in ("none", "b6"):
            differences: list[float] = []
            blocks: dict[int, list[float]] = defaultdict(list)
            p_blocks: dict[int, list[float]] = defaultdict(list)
            raw_blocks: dict[int, list[float]] = defaultdict(list)
            for scenario_id in scenario_ids:
                p = float(nested[(scenario_id, "P", recovery)]["normalized_utility"])
                raw = float(
                    nested[(scenario_id, "DVCBBA", recovery)]["normalized_utility"]
                )
                index = int(metadata[scenario_id]["index"])
                differences.append(p - raw)
                blocks[index].append(p - raw)
                p_blocks[index].append(p)
                raw_blocks[index].append(raw)
            sizes = {len(values) for values in blocks.values()}
            if len(sizes) != 1:
                raise RuntimeError(f"incomplete shared-seed blocks for {suite}")
            block_values = np.asarray([
                np.mean(blocks[index]) for index in sorted(blocks)
            ])
            rng = np.random.default_rng(_seed("d6-block", suite, recovery))
            draws = block_values[
                rng.integers(0, len(block_values), size=(BOOTSTRAPS, len(block_values)))
            ].mean(axis=1)
            mean_p = float(np.mean([
                np.mean(p_blocks[index]) for index in sorted(p_blocks)
            ]))
            mean_raw = float(np.mean([
                np.mean(raw_blocks[index]) for index in sorted(raw_blocks)
            ]))
            difference = float(block_values.mean())
            episode = np.asarray(differences)
            output.append({
                "suite": suite,
                "recovery": recovery,
                "scenario_count": len(episode),
                "seed_block_count": len(block_values),
                "conditions_per_block": sizes.pop(),
                "mean_p": mean_p,
                "mean_dvcbba": mean_raw,
                "paired_difference": difference,
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "relative_gain_percent": 100.0 * difference / mean_raw,
                "win": int(np.sum(episode > 1e-12)),
                "tie": int(np.sum(np.abs(episode) <= 1e-12)),
                "loss": int(np.sum(episode < -1e-12)),
                "bootstrap_unit": "shared_seed_block",
            })
    return output


def _legacy_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    legacy: dict[tuple[str, str], float] = {}
    for path in (
        ROOT / "results/dynamic_mainline/d3_external_validation/canonical/d3_records.csv",
        ROOT / "results/dynamic_mainline/d4_sensitivity/canonical/d4_records.csv",
    ):
        with path.open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                if row["method"] in {"P", "DVCBBA"}:
                    legacy[(row["scenario_id"], row["method"])] = float(
                        row["normalized_utility"]
                    )
    differences = [
        abs(float(row["normalized_utility"]) - legacy[(row["scenario_id"], row["primary"])])
        for row in rows
        if row["recovery"] == "none"
    ]
    return {
        "rows_checked": len(differences),
        "mismatch_count": sum(value > 1e-12 for value in differences),
        "max_absolute_difference": max(differences),
    }


def _subset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nested = {
        (row["scenario_id"], row["primary"], row["recovery"]): row for row in rows
    }
    output = []
    for suite in SUITES:
        scenario_ids = sorted({row["scenario_id"] for row in rows if row["suite"] == suite})
        failed = {
            scenario_id for scenario_id in scenario_ids
            if nested[(scenario_id, "DVCBBA", "none")]["primary_failure_count"] > 0
        }
        initial = {
            scenario_id for scenario_id in scenario_ids
            if nested[(scenario_id, "DVCBBA", "none")]["initial_failure"] > 0
        }
        subsets = {
            "all": scenario_ids,
            "failure": [item for item in scenario_ids if item in failed],
            "initial_failure": [item for item in scenario_ids if item in initial],
            "no_failure": [item for item in scenario_ids if item not in failed],
        }
        for name, selected in subsets.items():
            p = [nested[(item, "P", "none")]["normalized_utility"] for item in selected]
            raw_none = [
                nested[(item, "DVCBBA", "none")]["normalized_utility"] for item in selected
            ]
            raw_b6 = [
                nested[(item, "DVCBBA", "b6")]["normalized_utility"] for item in selected
            ]
            output.append({
                "suite": suite,
                "subset": name,
                "scenario_count": len(selected),
                "mean_p": sum(p) / len(p),
                "mean_raw_none": sum(raw_none) / len(raw_none),
                "mean_raw_b6": sum(raw_b6) / len(raw_b6),
                "mean_p_minus_raw_b6": sum(a - b for a, b in zip(p, raw_b6)) / len(p),
                "mean_raw_b6_minus_none": (
                    sum(a - b for a, b in zip(raw_b6, raw_none)) / len(p)
                ),
            })
    return output


def _workload_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for suite in SUITES:
        for recovery in ("none", "b6"):
            for primary in ("P", "DVCBBA"):
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
                    "mean_primary_rounds": sum(row["primary_rounds"] for row in selected) / len(selected),
                    "mean_primary_packets": sum(row["primary_packets"] for row in selected) / len(selected),
                    "mean_primary_target_entries": (
                        sum(row["primary_target_entries"] for row in selected) / len(selected)
                    ),
                    "failure_episode_rate": (
                        sum(row["primary_failure_count"] > 0 for row in selected) / len(selected)
                    ),
                    "fallback_calls": int(sum(row["fallback_count"] for row in selected)),
                })
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _text_atomic(path, _csv_text(rows, tuple(rows[0])))


def main() -> int:
    rows = _rows()
    metadata = _metadata()
    legacy = _legacy_check(rows)
    contrasts = _blocked_contrasts(rows, metadata)
    subsets = _subset_rows(rows)
    workload = _workload_rows(rows)
    _write_csv(SOURCE / "blocked_contrasts.csv", contrasts)
    _write_csv(SOURCE / "failure_subset_contrasts.csv", subsets)
    _write_csv(SOURCE / "workload_diagnostics.csv", workload)
    summary = {
        "status": "VERIFIED" if legacy["mismatch_count"] == 0 else "MISMATCH",
        "records_sha256": sha256_file(SOURCE / "records.csv"),
        "legacy_no_fallback_consistency": legacy,
        "unsettled_episode_count": sum(row["action_count"] != row["event_count"] for row in rows),
        "fallback_failure_rates": {
            row["suite"]: row["failure_episode_rate"]
            for row in workload
            if row["recovery"] == "b6" and row["primary"] == "DVCBBA"
        },
        "blocked_contrasts": contrasts,
    }
    write_json_atomic(SOURCE / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
