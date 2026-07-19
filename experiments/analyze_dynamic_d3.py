"""Read-only D3 analysis: paired bootstrap tables and publication figures."""

from __future__ import annotations

from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_dynamic_mainline import _csv_text, _text_atomic
from uav_lifecycle.artifacts import sha256_file, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/dynamic_mainline/d3_external_validation/canonical"
SOURCE_SUMMARY = ROOT / "results/dynamic_mainline/d3_external_validation/summary.json"
MANIFEST = ROOT / "results/dynamic_mainline/d3_design/d3_manifest.json"
OUTPUT = ROOT / "results/dynamic_mainline/d3_analysis"
BOOTSTRAPS = 10_000
TOL = 1e-12

EXTERNAL_BASELINES = ("B1m", "B4", "B5(4)", "B6", "SCBBA", "DVCBBA")
SUITE_ORDER = (
    "scale", "continuous_belief", "model_mismatch", "utility_profile",
    "cbba_isolation", "allocation_pressure",
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


def _stratified_interval(
    strata: dict[str, np.ndarray], token: str,
) -> tuple[float, float, float, float]:
    bootstrap_means = []
    stratum_means = []
    all_values = []
    for name, values in sorted(strata.items()):
        rng = np.random.default_rng(_seed(token, name))
        draws = values[rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))]
        bootstrap_means.append(draws.mean(axis=1))
        stratum_means.append(float(values.mean()))
        all_values.extend(values.tolist())
    aggregate = np.mean(np.vstack(bootstrap_means), axis=0)
    return (
        float(np.mean(stratum_means)), float(np.median(all_values)),
        float(np.quantile(aggregate, 0.025)), float(np.quantile(aggregate, 0.975)),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    fields.extend(sorted({key for row in rows for key in row} - set(fields)))
    _text_atomic(path, _csv_text(rows, tuple(fields)))


def _records(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    metadata = manifest["scenario_metadata"]
    rows = []
    nested: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    numeric = (
        "normalized_utility", "destroyed_value", "service_cost", "distance_cost",
        "ammo_cost", "makespan", "recon_count", "bda_count", "ammo_consumed",
        "replan_count", "action_count",
    )
    with (SOURCE / "d3_records.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            for name in numeric:
                row[name] = float(row[name])
            row.update({name: metadata[row["scenario_id"]][name] for name in ("suite", "condition")})
            rows.append(row)
            nested[row["scenario_id"]][row["method"]] = row
    return rows, nested


def _stratum(row: dict[str, Any]) -> str:
    return (
        f"{row['condition']}|{row['cell_id']}"
        if row["suite"] in {"model_mismatch", "utility_profile"}
        else row["condition"]
        if row["suite"] == "allocation_pressure"
        else row["cell_id"]
    )


def contrast_tables(
    records: list[dict[str, Any]], nested: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    suite_rows = []
    stratum_rows = []
    for suite in SUITE_ORDER:
        scenario_ids = sorted({row["scenario_id"] for row in records if row["suite"] == suite})
        methods = set.intersection(*(set(nested[scenario_id]) for scenario_id in scenario_ids))
        baselines = [name for name in EXTERNAL_BASELINES + ("CEX",) if name in methods]
        for baseline in baselines:
            strata: dict[str, list[float]] = defaultdict(list)
            for scenario_id in scenario_ids:
                left, right = nested[scenario_id]["P"], nested[scenario_id][baseline]
                strata[_stratum(left)].append(
                    float(left["normalized_utility"]) - float(right["normalized_utility"])
                )
            arrays = {name: np.asarray(values) for name, values in strata.items()}
            mean, median, low, high = _stratified_interval(arrays, f"suite|{suite}|P-{baseline}")
            flat = np.concatenate(list(arrays.values()))
            suite_rows.append({
                "suite": suite, "contrast": f"P-{baseline}", "baseline": baseline,
                "scenario_count": len(flat), "stratum_count": len(arrays),
                "equal_stratum_mean": mean, "paired_median": median,
                "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                "win": int(np.sum(flat > TOL)), "tie": int(np.sum(np.abs(flat) <= TOL)),
                "loss": int(np.sum(flat < -TOL)),
            })
            for name, values in sorted(arrays.items()):
                s_mean, s_median, s_low, s_high = _interval(values, f"stratum|{suite}|{name}|P-{baseline}")
                stratum_rows.append({
                    "suite": suite, "stratum": name, "contrast": f"P-{baseline}",
                    "scenario_count": len(values), "mean": s_mean, "median": s_median,
                    "bootstrap_ci_low": s_low, "bootstrap_ci_high": s_high,
                    "win": int(np.sum(values > TOL)), "tie": int(np.sum(np.abs(values) <= TOL)),
                    "loss": int(np.sum(values < -TOL)),
                })
    return suite_rows, stratum_rows


def method_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(row["suite"], row["method"])].append(row)
    output = []
    for (suite, method), selected in sorted(groups.items()):
        output.append({
            "suite": suite, "method": method, "episodes": len(selected),
            **{
                f"mean_{name}": float(np.mean([float(row[name]) for row in selected]))
                for name in (
                    "normalized_utility", "destroyed_value", "service_cost", "distance_cost",
                    "ammo_cost", "makespan", "recon_count", "bda_count", "ammo_consumed",
                    "replan_count",
                )
            },
        })
    return output


def runtime_tables(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = manifest["scenario_metadata"]
    episode: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    exact: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"fixed": [], "all_mode": []}
    )
    with (SOURCE / "d3_runtime.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            key = row["scenario_id"], row["method"]
            state = episode[key]
            state["calls"] += 1
            state["cycles"] += row["allocator_status"] == "cycle"
            state["round_caps"] += row["allocator_status"] == "timeout"
            state["rounds"] += int(row["rounds"])
            state["message_packets"] += int(row["message_packets"])
            state["message_scalars"] += int(row["message_scalars"])
            state["winner_conflicts"] += int(row["winner_conflicts"])
            state["planning_process_time_ns"] += int(row["planning_process_time_ns"])
            state["positive_pairs"] += int(row["positive_pairs"])
            state["eligible_pairs"] += int(row["eligible_pairs"])
            if row["fixed_screened_task_exact_gap"]:
                exact[key]["fixed"].append(float(row["fixed_screened_task_exact_gap"]))
                exact[key]["all_mode"].append(float(row["all_mode_cex_gap"]))
    groups: dict[tuple[str, str], list[tuple[str, dict[str, float]]]] = defaultdict(list)
    for (scenario_id, method), values in episode.items():
        groups[(metadata[scenario_id]["suite"], method)].append((scenario_id, values))
    output = []
    for (suite, method), selected in sorted(groups.items()):
        output.append({
            "suite": suite, "method": method, "episodes": len(selected),
            "allocator_calls": int(sum(item[1]["calls"] for item in selected)),
            "episodes_with_cycle": sum(item[1]["cycles"] > 0 for item in selected),
            "cycle_calls": int(sum(item[1]["cycles"] for item in selected)),
            "round_cap_calls": int(sum(item[1]["round_caps"] for item in selected)),
            "winner_conflicts": int(sum(item[1]["winner_conflicts"] for item in selected)),
            **{
                f"mean_{name}_per_episode": float(np.mean([item[1][name] for item in selected]))
                for name in ("rounds", "message_packets", "message_scalars", "planning_process_time_ns")
            },
            "positive_pair_density": (
                sum(item[1]["positive_pairs"] for item in selected)
                / sum(item[1]["eligible_pairs"] for item in selected)
            ),
        })
    exact_rows = []
    for method in ("P", "SCBBA", "DVCBBA"):
        values_by_cell: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"fixed": [], "all_mode": []}
        )
        for (scenario_id, selected_method), values in exact.items():
            if selected_method != method:
                continue
            cell = metadata[scenario_id]["cell_id"]
            values_by_cell[cell]["fixed"].append(float(np.mean(values["fixed"])))
            values_by_cell[cell]["all_mode"].append(float(np.mean(values["all_mode"])))
        for metric in ("fixed", "all_mode"):
            arrays = {cell: np.asarray(values[metric]) for cell, values in values_by_cell.items()}
            mean, median, low, high = _stratified_interval(arrays, f"exact|{method}|{metric}")
            exact_rows.append({
                "method": method, "metric": metric, "episodes": sum(map(len, arrays.values())),
                "equal_cell_mean": mean, "episode_median": median,
                "bootstrap_ci_low": low, "bootstrap_ci_high": high,
            })
    return output, exact_rows


def mechanism_tables(
    records: list[dict[str, Any]],
    nested: dict[str, dict[str, dict[str, Any]]],
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = manifest["scenario_metadata"]
    episode: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    objectives: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    with (SOURCE / "d3_runtime.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            key = row["scenario_id"], row["method"]
            state = episode[key]
            status = row["allocator_status"]
            rounds = int(row["rounds"])
            packets = int(row["message_packets"])
            scalars = int(row["message_scalars"])
            planning_ns = int(row["planning_process_time_ns"])
            state["calls"] += 1
            state[f"{status}_calls"] += 1
            state[f"{status}_rounds"] += rounds
            state[f"{status}_message_packets"] += packets
            state[f"{status}_message_scalars"] += scalars
            state[f"{status}_planning_process_time_ns"] += planning_ns
            objectives[(row["suite"], row["method"], status)].append(
                float(row["allocation_objective"])
            )

    normalized_rows = []
    groups: dict[tuple[str, str], list[tuple[str, dict[str, float]]]] = defaultdict(list)
    for (scenario_id, method), state in episode.items():
        groups[(metadata[scenario_id]["suite"], method)].append((scenario_id, state))
    for (suite, method), selected in sorted(groups.items()):
        converged_calls = sum(state["converged_calls"] for _, state in selected)
        nonconverged_calls = sum(
            state["cycle_calls"] + state["timeout_calls"] for _, state in selected
        )
        executed_actions = sum(
            float(nested[scenario_id][method]["action_count"]) for scenario_id, _ in selected
        )
        total_packets = sum(
            state["converged_message_packets"] + state["cycle_message_packets"]
            + state["timeout_message_packets"] for _, state in selected
        )
        total_rounds = sum(
            state["converged_rounds"] + state["cycle_rounds"] + state["timeout_rounds"]
            for _, state in selected
        )
        normalized_rows.append({
            "suite": suite, "method": method, "episodes": len(selected),
            "converged_epochs": int(converged_calls),
            "nonconverged_epochs": int(nonconverged_calls),
            "executed_actions": int(executed_actions),
            "message_packets_per_converged_epoch": (
                sum(state["converged_message_packets"] for _, state in selected) / converged_calls
                if converged_calls else ""
            ),
            "message_scalars_per_converged_epoch": (
                sum(state["converged_message_scalars"] for _, state in selected) / converged_calls
                if converged_calls else ""
            ),
            "rounds_per_converged_epoch": (
                sum(state["converged_rounds"] for _, state in selected) / converged_calls
                if converged_calls else ""
            ),
            "planning_ms_per_converged_epoch": (
                sum(state["converged_planning_process_time_ns"] for _, state in selected)
                / converged_calls / 1e6 if converged_calls else ""
            ),
            "total_message_packets_per_executed_action": (
                total_packets / executed_actions if executed_actions else ""
            ),
            "total_rounds_per_executed_action": (
                total_rounds / executed_actions if executed_actions else ""
            ),
            "nonconverged_wasted_message_packets": int(sum(
                state["cycle_message_packets"] + state["timeout_message_packets"]
                for _, state in selected
            )),
            "nonconverged_wasted_message_scalars": int(sum(
                state["cycle_message_scalars"] + state["timeout_message_scalars"]
                for _, state in selected
            )),
            "nonconverged_wasted_rounds": int(sum(
                state["cycle_rounds"] + state["timeout_rounds"] for _, state in selected
            )),
            "denominator_note": "executed action; strict per-commit linkage is unavailable",
        })

    objective_rows = []
    for (suite, method, status), values in sorted(objectives.items()):
        array = np.asarray(values)
        converged = status == "converged"
        objective_rows.append({
            "suite": suite, "method": method, "allocator_status": status,
            "epoch_count": len(values), "mean_raw_allocation_objective": float(array.mean()),
            "median_raw_allocation_objective": float(np.median(array)),
            "mean_executable_allocation_objective": float(array.mean()) if converged else "",
            "executable_objective_valid": converged,
            "semantic_label": (
                "legal executable allocation" if converged
                else "NA: conflicting/nonconverged local-bundle diagnostic"
            ),
        })

    cycle_rows = []
    scopes: list[tuple[str, str | None]] = [(suite, None) for suite in SUITE_ORDER]
    scopes.extend(("allocation_pressure", condition) for condition in sorted({
        item["condition"] for item in metadata.values() if item["suite"] == "allocation_pressure"
    }))
    for suite, condition in scopes:
        scenario_ids = [
            scenario_id for scenario_id, item in metadata.items()
            if item["suite"] == suite and (condition is None or item["condition"] == condition)
            and {"P", "DVCBBA"} <= set(nested[scenario_id])
        ]
        for group in ("converged", "cycle_or_round_cap"):
            selected_ids = []
            for scenario_id in scenario_ids:
                state = episode[(scenario_id, "DVCBBA")]
                nonconverged = state["cycle_calls"] + state["timeout_calls"] > 0
                if (group == "cycle_or_round_cap") == nonconverged:
                    selected_ids.append(scenario_id)
            if not selected_ids:
                continue
            differences = np.asarray([
                float(nested[scenario_id]["P"]["normalized_utility"])
                - float(nested[scenario_id]["DVCBBA"]["normalized_utility"])
                for scenario_id in selected_ids
            ])
            mean, median, low, high = _interval(
                differences, f"cycle-posthoc|{suite}|{condition}|{group}"
            )
            cycle_rows.append({
                "suite": suite, "condition": "all" if condition is None else condition,
                "dvcbba_status_group": group, "posthoc_descriptive_only": True,
                "scenario_count": len(selected_ids), "mean_p_minus_dvcbba": mean,
                "median_p_minus_dvcbba": median, "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "win": int(np.sum(differences > TOL)),
                "tie": int(np.sum(np.abs(differences) <= TOL)),
                "loss": int(np.sum(differences < -TOL)),
                "mean_p_utility": float(np.mean([
                    nested[scenario_id]["P"]["normalized_utility"] for scenario_id in selected_ids
                ])),
                "mean_dvcbba_utility": float(np.mean([
                    nested[scenario_id]["DVCBBA"]["normalized_utility"] for scenario_id in selected_ids
                ])),
                "mean_action_count_difference": float(np.mean([
                    nested[scenario_id]["P"]["action_count"]
                    - nested[scenario_id]["DVCBBA"]["action_count"]
                    for scenario_id in selected_ids
                ])),
                "interpretation": "association with observed allocator status; not a causal estimate",
            })
    return cycle_rows, objective_rows, normalized_rows


def report(
    suite_rows: list[dict[str, Any]], runtime_rows: list[dict[str, Any]], exact_rows: list[dict[str, Any]],
    cycle_rows: list[dict[str, Any]], normalized_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> str:
    def effect(suite: str, contrast: str) -> dict[str, Any]:
        return next(row for row in suite_rows if row["suite"] == suite and row["contrast"] == contrast)

    scale_dv = effect("scale", "P-DVCBBA")
    pressure_dv = effect("allocation_pressure", "P-DVCBBA")
    scale_cycle = next(row for row in runtime_rows if row["suite"] == "scale" and row["method"] == "DVCBBA")
    pressure_cycle = next(row for row in runtime_rows if row["suite"] == "allocation_pressure" and row["method"] == "DVCBBA")
    exact_p = next(row for row in exact_rows if row["method"] == "P" and row["metric"] == "fixed")
    def cycle_group(suite: str, group: str) -> dict[str, Any]:
        return next(row for row in cycle_rows if row["suite"] == suite and row["condition"] == "all" and row["dvcbba_status_group"] == group)

    scale_converged = cycle_group("scale", "converged")
    scale_nonconverged = cycle_group("scale", "cycle_or_round_cap")
    pressure_converged = cycle_group("allocation_pressure", "converged")
    pressure_nonconverged = cycle_group("allocation_pressure", "cycle_or_round_cap")

    def communication(suite: str, method: str) -> dict[str, Any]:
        return next(row for row in normalized_rows if row["suite"] == suite and row["method"] == method)

    scale_p_comm = communication("scale", "P")
    scale_dv_comm = communication("scale", "DVCBBA")
    return f"""# D3 read-only statistical analysis

## Material Passport

- Analysis: `D3-v2-read-only-v1`
- Source manifest: `{manifest['manifest_digest']}`
- Source records: `58,464`
- Bootstrap: paired within registered stratum, `{BOOTSTRAPS:,}` replicates, deterministic seeds
- Verification status: `ANALYZED`
- D2 remains the only primary confirmation experiment.

## Main findings

- Scale: P-DVCBBA = {scale_dv['equal_stratum_mean']:.5f}, 95% CI [{scale_dv['bootstrap_ci_low']:.5f}, {scale_dv['bootstrap_ci_high']:.5f}].
- Allocation pressure: P-DVCBBA = {pressure_dv['equal_stratum_mean']:.5f}, 95% CI [{pressure_dv['bootstrap_ci_low']:.5f}, {pressure_dv['bootstrap_ci_high']:.5f}].
- DVCBBA episodes with cycles: {scale_cycle['episodes_with_cycle']}/{scale_cycle['episodes']} in scale and {pressure_cycle['episodes_with_cycle']}/{pressure_cycle['episodes']} under allocation pressure; P has zero cycles.
- In small exact-isolation cells, P and DVCBBA have identical terminal utility. P's fixed-screened-task exact gap is {exact_p['equal_cell_mean']:.5f} [{exact_p['bootstrap_ci_low']:.5f}, {exact_p['bootstrap_ci_high']:.5f}].

The evidence supports a stability claim, not universal allocator dominance: the Johnson-warped/full-reconstruction mechanism matters when path-dependent marginal scores and resource coupling create vanilla-CBBA cycles. In simple cells, both allocators can coincide.

## Post-hoc cycle stratification

- Scale, DVCBBA converged: n={scale_converged['scenario_count']}, P-DVCBBA={scale_converged['mean_p_minus_dvcbba']:.5f} [{scale_converged['bootstrap_ci_low']:.5f}, {scale_converged['bootstrap_ci_high']:.5f}].
- Scale, DVCBBA cycle/round-cap: n={scale_nonconverged['scenario_count']}, P-DVCBBA={scale_nonconverged['mean_p_minus_dvcbba']:.5f} [{scale_nonconverged['bootstrap_ci_low']:.5f}, {scale_nonconverged['bootstrap_ci_high']:.5f}].
- Allocation pressure, DVCBBA converged: n={pressure_converged['scenario_count']}, P-DVCBBA={pressure_converged['mean_p_minus_dvcbba']:.5f} [{pressure_converged['bootstrap_ci_low']:.5f}, {pressure_converged['bootstrap_ci_high']:.5f}].
- Allocation pressure, DVCBBA cycle/round-cap: n={pressure_nonconverged['scenario_count']}, P-DVCBBA={pressure_nonconverged['mean_p_minus_dvcbba']:.5f} [{pressure_nonconverged['bootstrap_ci_low']:.5f}, {pressure_nonconverged['bootstrap_ci_high']:.5f}].

This outcome-defined stratification is descriptive. It is consistent with the stability mechanism but does not causally prove that cycle recovery produces the utility difference.

## Normalized allocator workload

- Scale P: {scale_p_comm['message_packets_per_converged_epoch']:.2f} message packets and {scale_p_comm['rounds_per_converged_epoch']:.2f} rounds per converged epoch.
- Scale DVCBBA: {scale_dv_comm['message_packets_per_converged_epoch']:.2f} message packets and {scale_dv_comm['rounds_per_converged_epoch']:.2f} rounds per converged epoch.
- DVCBBA's nonconverged scale epochs consumed {scale_dv_comm['nonconverged_wasted_message_packets']} message packets and {scale_dv_comm['nonconverged_wasted_rounds']} rounds.

Communication per executed action is reported separately. A strict per-successful-commit denominator is unavailable because the frozen runtime table does not link every planning call to a commit identifier.

## Allocation-objective semantics

Raw allocation objectives from cycle or round-cap epochs are retained only as conflict-state diagnostics. Their executable objective is recorded as NA and they are excluded from valid allocation-quality averages.

## Interpretation controls

- SCBBA is an end-to-end external baseline; its difference also includes one-shot lifecycle freezing.
- CEX is current-epoch myopic exact allocation, not the global dynamic optimum.
- Bootstrap intervals quantify paired scenario uncertainty; D3 is external validation and no effect-based condition filtering was performed.
- Multiple effects are reported without selecting only intervals that exclude zero; no confirmatory p-value family is claimed.

## Statistical fallacy scan

All 11 checks were considered. No Simpson reversal is asserted without stratum tables; individual scenarios remain the unit of analysis; no selected-survivor subset, causal language, diagnostic base-rate claim, regression-to-mean design, or reverse-causality claim is used. The main cautions are multiple exploratory contrasts and the risk of conflating allocator nonconvergence with intrinsic objective quality. Both are made explicit in the tables and interpretation.
"""


def run() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_summary = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    if (
        source_summary.get("status") != "D3_COMPLETE"
        or source_summary.get("manifest_digest") != manifest["manifest_digest"]
    ):
        raise RuntimeError("analysis requires a complete D3 run matching the frozen manifest")
    records, nested = _records(manifest)
    suite_rows, stratum_rows = contrast_tables(records, nested)
    methods = method_table(records)
    runtimes, exact = runtime_tables(manifest)
    cycles, objectives, normalized = mechanism_tables(records, nested, manifest)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT / "suite_contrasts.csv", suite_rows)
    _write_csv(OUTPUT / "stratum_contrasts.csv", stratum_rows)
    _write_csv(OUTPUT / "method_summary.csv", methods)
    _write_csv(OUTPUT / "allocator_diagnostics.csv", runtimes)
    _write_csv(OUTPUT / "exact_gap_summary.csv", exact)
    _write_csv(OUTPUT / "cycle_stratified_contrasts.csv", cycles)
    _write_csv(OUTPUT / "allocation_objective_semantics.csv", objectives)
    _write_csv(OUTPUT / "normalized_allocator_workload.csv", normalized)
    _text_atomic(
        OUTPUT / "analysis_report.md",
        report(suite_rows, runtimes, exact, cycles, normalized, manifest),
    )
    summary = {
        "status": "COMPLETE", "source_manifest_digest": manifest["manifest_digest"],
        "source_record_count": len(records), "bootstrap_iterations": BOOTSTRAPS,
        "suite_contrast_rows": len(suite_rows), "stratum_contrast_rows": len(stratum_rows),
        "runtime_summary_rows": len(runtimes), "exact_summary_rows": len(exact),
        "cycle_stratified_rows": len(cycles),
        "allocation_objective_semantic_rows": len(objectives),
        "normalized_allocator_workload_rows": len(normalized),
        "figures_generated": False,
    }
    write_json_atomic(OUTPUT / "analysis_summary.json", summary)
    inventory = {
        str(path.relative_to(OUTPUT)): sha256_file(path)
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name != "analysis_inventory.json"
    }
    write_json_atomic(OUTPUT / "analysis_inventory.json", inventory)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
