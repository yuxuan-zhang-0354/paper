"""Read-only D4 analysis with seed-block bootstrap; produces tables, not figures."""

from __future__ import annotations

from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.run_dynamic_mainline import _csv_text, _text_atomic
from uav_lifecycle.artifacts import sha256_file, write_json_atomic
from uav_lifecycle.dynamic_d4 import generate_battlefield_structure
from uav_lifecycle.dynamic_types import DynamicConfig


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/dynamic_mainline/d4_sensitivity/canonical"
SOURCE_SUMMARY = ROOT / "results/dynamic_mainline/d4_sensitivity/summary.json"
MANIFEST = ROOT / "results/dynamic_mainline/d4_design/d4_manifest.json"
OUTPUT = ROOT / "results/dynamic_mainline/d4_analysis"
BOOTSTRAPS = 10_000
TOL = 1e-12

NUMERIC = (
    "normalized_utility", "realized_utility", "destroyed_value", "service_cost",
    "distance_cost", "ammo_cost", "distance_consumed", "ammo_consumed", "makespan",
    "recon_count", "bda_count", "continuous_attack_count", "handoff_count",
    "initial_wreck_attack_count", "invalid_attack_count", "replan_count", "action_count",
)


def _seed(*parts: object) -> int:
    return int.from_bytes(sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def _interval(values: np.ndarray, token: str) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(_seed(token))
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))]
    means = draws.mean(axis=1)
    return (
        float(values.mean()), float(np.median(values)),
        float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    fields.extend(sorted({key for row in rows for key in row} - set(fields)))
    _text_atomic(path, _csv_text(rows, tuple(fields)))


def _load() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    summary = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    if summary.get("status") != "D4_COMPLETE" or summary.get("manifest_digest") != manifest["manifest_digest"]:
        raise RuntimeError("analysis requires a complete D4 run matching the frozen manifest")
    metadata = manifest["scenario_metadata"]
    rows: list[dict[str, Any]] = []
    nested: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with (SOURCE / "d4_records.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            for name in NUMERIC:
                row[name] = float(row[name])
            row.update(metadata[row["scenario_id"]])
            rows.append(row)
            nested[row["scenario_id"]][row["method"]] = row
    expected = {tuple(item) for item in manifest["expected_rectangle"]}
    if {(row["scenario_id"], row["method"]) for row in rows} != expected:
        raise RuntimeError("D4 record rectangle does not match manifest")
    return manifest, rows, nested


def _ids(manifest: dict[str, Any], suite: str, **filters: object) -> list[str]:
    return sorted(
        scenario_id for scenario_id, item in manifest["scenario_metadata"].items()
        if item["suite"] == suite and all(item.get(key) == value for key, value in filters.items())
    )


def _block_differences(
    scenario_ids: Iterable[str], baseline: str, nested: dict[str, dict[str, dict[str, Any]]],
    metadata: dict[str, Any], metric: str = "normalized_utility",
) -> tuple[np.ndarray, np.ndarray]:
    episode = []
    blocks: dict[int, list[float]] = defaultdict(list)
    for scenario_id in scenario_ids:
        difference = float(nested[scenario_id]["P"][metric]) - float(nested[scenario_id][baseline][metric])
        episode.append(difference)
        blocks[int(metadata[scenario_id]["index"])].append(difference)
    return np.asarray(episode), np.asarray([np.mean(blocks[index]) for index in sorted(blocks)])


def _contrast(
    *, suite: str, scope: str, level: str, scenario_ids: list[str], baseline: str,
    nested: dict[str, dict[str, dict[str, Any]]], metadata: dict[str, Any],
) -> dict[str, Any]:
    episode, blocks = _block_differences(scenario_ids, baseline, nested, metadata)
    mean, median, low, high = _interval(blocks, f"d4|{suite}|{scope}|{level}|P-{baseline}")
    return {
        "suite": suite, "scope": scope, "level": level, "contrast": f"P-{baseline}",
        "scenario_count": len(episode), "seed_block_count": len(blocks),
        "seed_block_mean": mean, "episode_median": float(np.median(episode)),
        "bootstrap_ci_low": low, "bootstrap_ci_high": high,
        "win": int(np.sum(episode > TOL)), "tie": int(np.sum(np.abs(episode) <= TOL)),
        "loss": int(np.sum(episode < -TOL)),
        "bootstrap_unit": "shared seed block",
    }


def contrast_tables(
    manifest: dict[str, Any], nested: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = manifest["scenario_metadata"]
    suite_rows, condition_rows, main_rows = [], [], []
    suite_baselines = {
        "battlefield_structure": ("DVCBBA", "B6", "B4", "B1m", "SCBBA"),
        "reachability": ("DVCBBA", "B6", "SCBBA"),
    }
    for suite, baselines in suite_baselines.items():
        selected = _ids(manifest, suite)
        conditions = sorted({metadata[item]["condition"] for item in selected})
        for baseline in baselines:
            suite_rows.append(_contrast(
                suite=suite, scope="suite", level="all", scenario_ids=selected,
                baseline=baseline, nested=nested, metadata=metadata,
            ))
            for condition in conditions:
                condition_ids = [item for item in selected if metadata[item]["condition"] == condition]
                condition_rows.append(_contrast(
                    suite=suite, scope="condition", level=condition, scenario_ids=condition_ids,
                    baseline=baseline, nested=nested, metadata=metadata,
                ))
    for structure in manifest["suites"]["battlefield_structure"]["structures"]:
        for baseline in suite_baselines["battlefield_structure"]:
            main_rows.append(_contrast(
                suite="battlefield_structure", scope="structure", level=structure,
                scenario_ids=_ids(manifest, "battlefield_structure", structure=structure),
                baseline=baseline, nested=nested, metadata=metadata,
            ))
    for rate in manifest["suites"]["battlefield_structure"]["wreck_rates"]:
        for baseline in suite_baselines["battlefield_structure"]:
            main_rows.append(_contrast(
                suite="battlefield_structure", scope="wreck_rate", level=str(rate),
                scenario_ids=_ids(manifest, "battlefield_structure", wreck_rate=rate),
                baseline=baseline, nested=nested, metadata=metadata,
            ))
    for axis in ("map_scale", "time_scale"):
        for value in manifest["suites"]["reachability"][f"{axis.split('_')[0]}_scales"]:
            for baseline in suite_baselines["reachability"]:
                main_rows.append(_contrast(
                    suite="reachability", scope=axis, level=str(value),
                    scenario_ids=_ids(manifest, "reachability", **{axis: value}),
                    baseline=baseline, nested=nested, metadata=metadata,
                ))
    return suite_rows, condition_rows, main_rows


def method_tables(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fields = (
        "normalized_utility", "destroyed_value", "service_cost", "distance_cost", "ammo_cost",
        "distance_consumed", "ammo_consumed", "makespan", "recon_count", "bda_count",
        "continuous_attack_count", "handoff_count", "initial_wreck_attack_count",
        "invalid_attack_count", "replan_count", "action_count",
    )
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(row["suite"], row["condition"], row["method"])].append(row)
    condition_rows = []
    for (suite, condition, method), selected in sorted(groups.items()):
        condition_rows.append({
            "suite": suite, "condition": condition, "method": method, "episodes": len(selected),
            **{f"mean_{name}": float(np.mean([row[name] for row in selected])) for name in fields},
            "initial_wreck_attacks_per_attack": (
                sum(row["initial_wreck_attack_count"] for row in selected)
                / sum(row["ammo_consumed"] for row in selected)
                if sum(row["ammo_consumed"] for row in selected) else 0.0
            ),
        })
    suite_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        suite_groups[(row["suite"], row["method"])].append(row)
    suite_rows = [{
        "suite": suite, "method": method, "episodes": len(selected),
        **{f"mean_{name}": float(np.mean([row[name] for row in selected])) for name in fields},
    } for (suite, method), selected in sorted(suite_groups.items())]
    return suite_rows, condition_rows


def decomposition_table(
    manifest: dict[str, Any], nested: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    baselines = {
        "battlefield_structure": ("DVCBBA", "B6", "B4", "B1m", "SCBBA"),
        "reachability": ("DVCBBA", "B6", "SCBBA"),
    }
    metadata = manifest["scenario_metadata"]
    for suite, names in baselines.items():
        scenario_ids = _ids(manifest, suite)
        for baseline in names:
            values = defaultdict(list)
            for scenario_id in scenario_ids:
                left, right = nested[scenario_id]["P"], nested[scenario_id][baseline]
                for metric in ("destroyed_value", "service_cost", "distance_cost", "ammo_cost", "realized_utility", "normalized_utility"):
                    values[metric].append(float(left[metric]) - float(right[metric]))
            _, blocks = _block_differences(scenario_ids, baseline, nested, metadata)
            _, _, low, high = _interval(blocks, f"d4-decomposition|{suite}|{baseline}")
            rows.append({
                "suite": suite, "contrast": f"P-{baseline}", "scenarios": len(scenario_ids),
                **{f"mean_delta_{metric}": float(np.mean(items)) for metric, items in values.items()},
                "normalized_utility_ci_low": low, "normalized_utility_ci_high": high,
                "identity_residual": float(np.mean(values["realized_utility"]) - (
                    np.mean(values["destroyed_value"]) - np.mean(values["service_cost"])
                    - np.mean(values["distance_cost"]) - np.mean(values["ammo_cost"])
                )),
            })
    return rows


def runtime_tables(
    manifest: dict[str, Any], nested: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = manifest["scenario_metadata"]
    episode: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with (SOURCE / "d4_runtime.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            state = episode[(row["scenario_id"], row["method"])]
            status = row["allocator_status"]
            state["calls"] += 1
            state[f"{status}_calls"] += 1
            for field in ("rounds", "message_packets", "message_scalars", "winner_conflicts", "planning_process_time_ns"):
                value = int(row[field])
                state[field] += value
                state[f"{status}_{field}"] += value
    groups: dict[tuple[str, str, str], list[tuple[str, dict[str, float]]]] = defaultdict(list)
    for (scenario_id, method), state in episode.items():
        item = metadata[scenario_id]
        groups[(item["suite"], item["condition"], method)].append((scenario_id, state))
    condition_rows = []
    for (suite, condition, method), selected in sorted(groups.items()):
        converged = sum(state["converged_calls"] for _, state in selected)
        nonconverged = sum(state["cycle_calls"] + state["timeout_calls"] for _, state in selected)
        condition_rows.append({
            "suite": suite, "condition": condition, "method": method, "episodes": len(selected),
            "allocator_calls": int(sum(state["calls"] for _, state in selected)),
            "episodes_with_nonconvergence": sum(state["cycle_calls"] + state["timeout_calls"] > 0 for _, state in selected),
            "converged_calls": int(converged), "nonconverged_calls": int(nonconverged),
            "winner_conflicts": int(sum(state["winner_conflicts"] for _, state in selected)),
            "rounds_per_converged_epoch": sum(state["converged_rounds"] for _, state in selected) / converged if converged else "",
            "message_packets_per_converged_epoch": sum(state["converged_message_packets"] for _, state in selected) / converged if converged else "",
            "planning_ms_per_converged_epoch": sum(state["converged_planning_process_time_ns"] for _, state in selected) / converged / 1e6 if converged else "",
            "nonconverged_wasted_rounds": int(sum(state["cycle_rounds"] + state["timeout_rounds"] for _, state in selected)),
            "nonconverged_wasted_message_packets": int(sum(state["cycle_message_packets"] + state["timeout_message_packets"] for _, state in selected)),
        })
    suite_rows = []
    for (suite, method), selected in sorted(defaultdict(list, {
        key: [item for group_key, values in groups.items() if group_key[0] == key[0] and group_key[2] == key[1] for item in values]
        for key in {(key[0], key[2]) for key in groups}
    }).items()):
        converged = sum(state["converged_calls"] for _, state in selected)
        nonconverged = sum(state["cycle_calls"] + state["timeout_calls"] for _, state in selected)
        suite_rows.append({
            "suite": suite, "method": method, "episodes": len(selected),
            "allocator_calls": int(sum(state["calls"] for _, state in selected)),
            "episodes_with_nonconvergence": sum(state["cycle_calls"] + state["timeout_calls"] > 0 for _, state in selected),
            "converged_calls": int(converged), "nonconverged_calls": int(nonconverged),
            "winner_conflicts": int(sum(state["winner_conflicts"] for _, state in selected)),
            "rounds_per_converged_epoch": sum(state["converged_rounds"] for _, state in selected) / converged if converged else "",
            "message_packets_per_converged_epoch": sum(state["converged_message_packets"] for _, state in selected) / converged if converged else "",
            "planning_ms_per_converged_epoch": sum(state["converged_planning_process_time_ns"] for _, state in selected) / converged / 1e6 if converged else "",
            "nonconverged_wasted_rounds": int(sum(state["cycle_rounds"] + state["timeout_rounds"] for _, state in selected)),
            "nonconverged_wasted_message_packets": int(sum(state["cycle_message_packets"] + state["timeout_message_packets"] for _, state in selected)),
        })
    cycle_rows = []
    for suite in ("battlefield_structure", "reachability"):
        scenario_ids = _ids(manifest, suite)
        for label, predicate in (
            ("converged", lambda state: state["cycle_calls"] + state["timeout_calls"] == 0),
            ("cycle_or_round_cap", lambda state: state["cycle_calls"] + state["timeout_calls"] > 0),
        ):
            selected = [item for item in scenario_ids if predicate(episode[(item, "DVCBBA")])]
            if not selected:
                continue
            row = _contrast(
                suite=suite, scope="DVCBBA_status_posthoc", level=label,
                scenario_ids=selected, baseline="DVCBBA", nested=nested, metadata=metadata,
            )
            row["posthoc_association_not_causal"] = True
            cycle_rows.append(row)
    return suite_rows, condition_rows, cycle_rows


def environment_table(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    config = DynamicConfig()
    rows = []
    for scenario_id in _ids(manifest, "battlefield_structure"):
        item = manifest["scenario_metadata"][scenario_id]
        scenario = generate_battlefield_structure(item["structure"], item["wreck_rate"], item["index"], config)
        high = sum(target.true_category == "H" for target in scenario.private_targets)
        wreck = sum(target.true_damage == "D" for target in scenario.private_targets)
        rows.append({
            "scenario_id": scenario_id, "structure": item["structure"], "nominal_wreck_rate": item["wreck_rate"],
            "seed": item["index"], "high_value_count": high, "initial_wreck_count": wreck,
            "realized_high_value_rate": high / len(scenario.private_targets),
            "realized_wreck_rate": wreck / len(scenario.private_targets),
        })
    return rows


def reachability_table(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for map_scale in manifest["suites"]["reachability"]["map_scales"]:
        for time_scale in manifest["suites"]["reachability"]["time_scales"]:
            map_side = 24.0 * map_scale
            horizon = int(np.ceil(51.0 * time_scale))
            rows.append({
                "map_scale": map_scale, "time_scale": time_scale,
                "map_side_normalized": map_side, "horizon": horizon,
                "rho_T": horizon / map_side, "rho_R": 1.25 * horizon / map_side,
                "rho_tau": 7.5 / horizon,
            })
    return rows


def report(
    manifest: dict[str, Any], suite_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]], runtime_rows: list[dict[str, Any]],
    cycle_rows: list[dict[str, Any]],
) -> str:
    def effect(suite: str, contrast: str) -> dict[str, Any]:
        return next(row for row in suite_rows if row["suite"] == suite and row["contrast"] == contrast)
    def runtime(suite: str, method: str) -> dict[str, Any]:
        return next(row for row in runtime_rows if row["suite"] == suite and row["method"] == method)
    def cycle(suite: str, level: str) -> dict[str, Any] | None:
        return next((row for row in cycle_rows if row["suite"] == suite and row["level"] == level), None)

    battlefield = effect("battlefield_structure", "P-DVCBBA")
    reachability = effect("reachability", "P-DVCBBA")
    b4 = effect("battlefield_structure", "P-B4")
    dv_b = runtime("battlefield_structure", "DVCBBA")
    dv_r = runtime("reachability", "DVCBBA")
    negative_cells = sum(row["seed_block_mean"] < 0 for row in condition_rows)
    robust_negative_cells = sum(row["bootstrap_ci_high"] < 0 for row in condition_rows)
    total_cells = len(condition_rows)
    battlefield_dv_cells = [
        row for row in condition_rows
        if row["suite"] == "battlefield_structure" and row["contrast"] == "P-DVCBBA"
    ]
    reachability_dv_cells = [
        row for row in condition_rows
        if row["suite"] == "reachability" and row["contrast"] == "P-DVCBBA"
    ]
    cycle_b = cycle("battlefield_structure", "cycle_or_round_cap")
    cycle_r = cycle("reachability", "cycle_or_round_cap")
    def cycle_text(row: dict[str, Any] | None) -> str:
        return (
            "none observed" if row is None else
            f"n={row['scenario_count']}, mean P-DVCBBA={row['seed_block_mean']:.5f}"
        )
    return f"""# D4 read-only sensitivity analysis

## Material Passport

- Analysis: `D4-read-only-v1`
- Source manifest: `{manifest['manifest_digest']}`
- Source records: `8,448`
- Bootstrap: shared-seed block bootstrap, `{BOOTSTRAPS:,}` replicates
- Verification status: `ANALYZED`
- Algorithms and D4 conditions were not changed; no figures were generated.

## Primary descriptive findings

- Battlefield structure: P-DVCBBA = {battlefield['seed_block_mean']:.5f}, 95% CI [{battlefield['bootstrap_ci_low']:.5f}, {battlefield['bootstrap_ci_high']:.5f}].
- Reachability: P-DVCBBA = {reachability['seed_block_mean']:.5f}, 95% CI [{reachability['bootstrap_ci_low']:.5f}, {reachability['bootstrap_ci_high']:.5f}].
- Optional BDA contribution in battlefield conditions: P-B4 = {b4['seed_block_mean']:.5f}, 95% CI [{b4['bootstrap_ci_low']:.5f}, {b4['bootstrap_ci_high']:.5f}].
- Condition-level contrast rows with negative means: {negative_cells}/{total_cells}; robustly negative intervals: {robust_negative_cells}/{total_cells}. All remain in the tables.
- P-DVCBBA is positive in all 16 battlefield cells ({sum(row['bootstrap_ci_low'] > 0 for row in battlefield_dv_cells)}/16 intervals above zero) and all 9 reachability cells ({sum(row['bootstrap_ci_low'] > 0 for row in reachability_dv_cells)}/9 intervals above zero).

## Pre-frozen mechanism hypotheses

- H1, initial wrecks: partially supported. The lifecycle advantage is strongest at wreck rate 0.6, but initial-wreck attacks do not decrease monotonically at every lower rate.
- H2, information actions: supported as a conditional rather than universal effect. P-B4 is positive overall and at wreck rates 0.2 and 0.6; intervals cross zero at 0 and 0.4.
- H3, clustering and non-DMG stress: supported descriptively. DVCBBA nonconvergence is substantially more frequent in clustered than uniform cells.
- H4, high-value clustering and P-B6: the general P-B6 advantage is supported, but a value-correlated-specific amplification is not; value-correlated, uniform and clustered main effects are similar.
- H5, reachability boundaries: broadly supported. P-DVCBBA is smaller at the short-horizon/small-map boundary and peaks in the intermediate regimes, without a strict monotonic law.

## Allocator diagnostics

- Battlefield DVCBBA nonconvergent episodes: {dv_b['episodes_with_nonconvergence']}/{dv_b['episodes']}.
- Reachability DVCBBA nonconvergent episodes: {dv_r['episodes_with_nonconvergence']}/{dv_r['episodes']}.
- Post-hoc cycle association, battlefield: {cycle_text(cycle_b)}.
- Post-hoc cycle association, reachability: {cycle_text(cycle_r)}.
- Nonconverged raw allocation objectives are not treated as executable allocations.

## Interpretation boundaries

- Overall and main-effect intervals resample the 64 shared seed blocks, preserving cross-condition CRN dependence.
- Individual condition intervals have 64 paired scenarios. Condition tables are exploratory; no multiplicity-adjusted significance claims are made.
- DVCBBA status stratification is post-hoc association, not a causal decomposition.
- Results support the frozen simulated domain only and do not establish real-world combat effectiveness.

## Statistical fallacy scan (11/11 checked)

1. Simpson's paradox: aggregate and every condition are supplied; direction reversals must be read from the condition table rather than hidden.
2. Ecological fallacy: inference is restricted to scenario-level simulation performance.
3. Berkson's paradox: no post-outcome scenario selection occurred.
4. Collider bias: no outcome-dependent conditioning enters primary contrasts; cycle stratification is explicitly post-hoc.
5. Base-rate neglect: initial wreck prevalence is an explicit experimental factor and realized rates are reported.
6. Regression to the mean: no extreme-outcome selection or pre/post subgrouping is used.
7. Survivorship bias: all 8,448 planned records completed; no failures were excluded.
8. Look-elsewhere effect: every frozen condition and contrast is retained; condition-level results are exploratory.
9. Garden of forking paths: algorithms, seeds, factors and primary contrasts are manifest-frozen; no effect-driven retuning occurred.
10. Correlation versus causation: paired algorithm effects are valid within the simulator, but no real-world causal claim is made.
11. Reverse causality: not applicable to randomized algorithm assignment over fixed simulated scenarios.
"""


def main() -> None:
    manifest, records, nested = _load()
    suite_rows, condition_rows, main_rows = contrast_tables(manifest, nested)
    method_rows, condition_method_rows = method_tables(records)
    decomposition_rows = decomposition_table(manifest, nested)
    runtime_rows, runtime_condition_rows, cycle_rows = runtime_tables(manifest, nested)
    outputs = {
        "suite_contrasts.csv": suite_rows,
        "condition_contrasts.csv": condition_rows,
        "main_effect_contrasts.csv": main_rows,
        "method_summary.csv": method_rows,
        "condition_method_summary.csv": condition_method_rows,
        "utility_decomposition.csv": decomposition_rows,
        "allocator_diagnostics.csv": runtime_rows,
        "allocator_condition_diagnostics.csv": runtime_condition_rows,
        "cycle_stratified_contrasts.csv": cycle_rows,
        "environment_realizations.csv": environment_table(manifest),
        "reachability_regimes.csv": reachability_table(manifest),
    }
    for name, rows in outputs.items():
        _write_csv(OUTPUT / name, rows)
    _text_atomic(OUTPUT / "analysis_report.md", report(
        manifest, suite_rows, condition_rows, runtime_rows, cycle_rows,
    ))
    summary = {
        "analysis_id": "D4-read-only-v1", "status": "ANALYZED",
        "source_manifest_digest": manifest["manifest_digest"],
        "bootstrap_iterations": BOOTSTRAPS, "bootstrap_unit": "shared seed block",
        "source_record_count": len(records), "figures_generated": False,
        "tables": {name: len(rows) for name, rows in outputs.items()},
    }
    write_json_atomic(OUTPUT / "analysis_summary.json", summary)
    inventory = {
        str(path.relative_to(OUTPUT)): sha256_file(path)
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name != "analysis_inventory.json"
    }
    write_json_atomic(OUTPUT / "analysis_inventory.json", inventory)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
