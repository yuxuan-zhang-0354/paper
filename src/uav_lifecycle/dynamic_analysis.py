"""Effect-blind calibration and exploratory paired analysis for D1."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .dynamic_scenarios import D1_CELLS


REGIONS = ("R", "A", "B", "Defer")
TIERS = ("tight", "loose")
PRIMARY = ("P", "B1m")
SECONDARY = ("B2", "B3", "B4", "B5(4)", "B6")
SENSITIVITY = ("B5(2)", "B5(8)")


@dataclass(frozen=True, slots=True)
class PublicTick0Coverage:
    scenario_id: str
    resource_tier: str
    initial_action_regions: tuple[str, ...]
    positive_single_task: bool
    resource_blocked: bool


@dataclass(frozen=True, slots=True)
class MethodBlindDiagnostics:
    gate_count: int
    numerical_failure_count: int
    runtime_seconds: float


@dataclass(frozen=True, slots=True)
class CalibrationCoverageRecord:
    scenario_id: str
    resource_tier: str
    initial_action_regions: tuple[str, ...]
    positive_single_task: bool
    resource_blocked: bool
    gate_count: int
    numerical_failure_count: int
    runtime_seconds: float


@dataclass(frozen=True, slots=True)
class CalibrationBatch:
    calibration_run_id: str
    registered_config_id: str
    expected_scenario_ids: tuple[str, ...]
    expected_target_counts: Mapping[str, int]
    records: tuple[CalibrationCoverageRecord, ...]
    d0_passed: bool


@dataclass(frozen=True, slots=True)
class CalibrationSelection:
    status: str
    selected_run_id: str | None
    selected_config_id: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class MatrixValidation:
    status: str
    reasons: tuple[str, ...]
    expected_count: int
    record_count: int
    gate_count: int


def _decoded(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _gates(row: Mapping[str, Any]) -> list[Any]:
    value = _decoded(row.get("allocator_gates", []))
    return value if isinstance(value, list) else []


def project_calibration_coverage(
    coverage: tuple[PublicTick0Coverage, ...],
    diagnostics: Mapping[str, MethodBlindDiagnostics],
    expected_scenario_ids: tuple[str, ...],
    expected_target_counts: Mapping[str, int],
) -> tuple[CalibrationCoverageRecord, ...]:
    """Join typed public screening with typed method-blind diagnostics."""

    if any(not isinstance(row, PublicTick0Coverage) for row in coverage):
        raise TypeError("coverage must contain only PublicTick0Coverage")
    if any(not isinstance(value, MethodBlindDiagnostics) for value in diagnostics.values()):
        raise TypeError("diagnostics must contain only MethodBlindDiagnostics")
    expected, ids = set(expected_scenario_ids), [row.scenario_id for row in coverage]
    if len(expected) != len(expected_scenario_ids) or len(ids) != len(set(ids)):
        raise ValueError("duplicate expected or coverage scenario")
    if set(ids) != expected or set(diagnostics) != expected:
        raise ValueError("missing or extra coverage scenario")
    records = tuple(
        CalibrationCoverageRecord(
            row.scenario_id, row.resource_tier, row.initial_action_regions,
            row.positive_single_task, row.resource_blocked,
            diagnostics[row.scenario_id].gate_count,
            diagnostics[row.scenario_id].numerical_failure_count,
            diagnostics[row.scenario_id].runtime_seconds,
        ) for row in coverage
    )
    summarize_effect_blind(records, expected_scenario_ids, expected_target_counts)
    return records


def summarize_effect_blind(
    records: tuple[CalibrationCoverageRecord, ...],
    expected_scenario_ids: tuple[str, ...],
    expected_target_counts: Mapping[str, int],
) -> dict[str, Any]:
    if not records or any(not isinstance(row, CalibrationCoverageRecord) for row in records):
        raise TypeError("summary requires nonempty tuple[CalibrationCoverageRecord, ...]")
    expected, ids = set(expected_scenario_ids), [row.scenario_id for row in records]
    if len(expected) != len(expected_scenario_ids) or len(ids) != len(set(ids)):
        raise ValueError("duplicate expected or coverage scenario")
    if set(ids) != expected:
        raise ValueError("missing or extra coverage scenario")
    if set(expected_target_counts) != expected:
        raise ValueError("missing or extra expected target count")
    counts = {region: 0 for region in REGIONS}
    tier_counts = {tier: {"positive": 0, "blocked": 0} for tier in TIERS}
    for row in records:
        if row.resource_tier not in TIERS or not row.initial_action_regions:
            raise ValueError("unknown tier or empty action regions")
        if len(row.initial_action_regions) != expected_target_counts[row.scenario_id]:
            raise ValueError("action region count disagrees with scenario target count")
        if any(region not in REGIONS for region in row.initial_action_regions):
            raise ValueError("unknown action region")
        if row.gate_count < 0 or row.numerical_failure_count < 0 or not math.isfinite(row.runtime_seconds) or row.runtime_seconds < 0:
            raise ValueError("invalid method-blind diagnostics")
        for region in row.initial_action_regions:
            counts[region] += 1
        tier_counts[row.resource_tier]["positive"] += int(row.positive_single_task)
        tier_counts[row.resource_tier]["blocked"] += int(row.resource_blocked)
    total = sum(counts.values())
    shares = {region: counts[region] / total for region in REGIONS}
    passes = (
        all(share >= 0.01 for share in shares.values())
        and all(tier_counts[tier][kind] >= 1 for tier in TIERS for kind in ("positive", "blocked"))
        and sum(row.gate_count + row.numerical_failure_count for row in records) == 0
    )
    return {
        "public_scenario_count": len(records), "public_target_count": total,
        "action_region_shares": shares,
        "resource_tiers": {
            tier: {
                "positive_feasible_scenarios": values["positive"],
                "resource_blocked_scenarios": values["blocked"],
            } for tier, values in tier_counts.items()
        },
        "gate_count": sum(row.gate_count for row in records),
        "numerical_failure_count": sum(row.numerical_failure_count for row in records),
        "runtime_seconds": sum(row.runtime_seconds for row in records),
        "passes_frozen_rule": passes,
    }


def select_first_passing_calibration(
    batches: tuple[CalibrationBatch, ...],
) -> CalibrationSelection:
    if not isinstance(batches, tuple) or any(not isinstance(item, CalibrationBatch) for item in batches):
        raise TypeError("selector accepts only tuple[CalibrationBatch, ...]")
    if len(batches) > 3:
        raise ValueError("at most three pre-registered calibration versions are allowed")
    for batch in batches:
        summary = summarize_effect_blind(
            batch.records, batch.expected_scenario_ids, batch.expected_target_counts,
        )
        if batch.d0_passed and summary["passes_frozen_rule"]:
            return CalibrationSelection(
                "LOCK_FIRST_PASS", batch.calibration_run_id,
                batch.registered_config_id, "first version satisfying frozen public rule",
            )
    return CalibrationSelection(
        "NO_PASS_REQUIRES_USER_AUTHORIZATION", None, None,
        "no pre-registered version passed; do not auto-create a configuration",
    )


def validate_method_matrix(
    records: Iterable[Mapping[str, Any]],
    expected_rectangle: Iterable[tuple[str, str]],
    *,
    expected_cells: Mapping[str, str],
    required_cells: Iterable[str] | None = None,
    replay_verified: bool | None = None,
) -> MatrixValidation:
    rows = tuple(records)
    expected = tuple(tuple(key) for key in expected_rectangle)
    expected_set = set(expected)
    keys = tuple((str(row.get("scenario_id")), str(row.get("method"))) for row in rows)
    reasons: list[str] = []
    expected_scenarios = {scenario for scenario, _ in expected}
    required = (
        {cell.cell_id for cell in D1_CELLS}
        if required_cells is None else set(required_cells)
    )
    if set(expected_cells) != expected_scenarios:
        reasons.append("expected scenario-to-cell mapping mismatch")
    if set(expected_cells.values()) != required:
        reasons.append("required cells missing or extra")
    if len(expected_set) != len(expected):
        reasons.append("duplicate expected key")
    if len(set(keys)) != len(keys):
        reasons.append("duplicate record")
    if expected_set - set(keys):
        reasons.append("missing record")
    if set(keys) - expected_set:
        reasons.append("extra record")
    gate_count = sum(len(_gates(row)) for row in rows)
    if gate_count:
        reasons.append("Gate failure")
    if replay_verified is False:
        reasons.append("independent replay verification failed")
    by_scenario: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        scenario_id = str(row.get("scenario_id"))
        by_scenario.setdefault(scenario_id, []).append(row)
        if row.get("cell_id") != expected_cells.get(scenario_id):
            reasons.append(f"scenario {scenario_id} disagrees with expected cell")
        status = str(row.get("status", ""))
        if status != "complete":
            reasons.append(f"terminal status {status or 'missing'}")
        error_type = str(row.get("error_type", ""))
        if "timeout" in error_type.lower():
            reasons.append("timeout")
        values: dict[str, float] = {}
        for name in ("normalized_utility", "realized_utility", "gross_scenario_value"):
            try:
                value = float(row[name])
            except (KeyError, TypeError, ValueError):
                reasons.append(f"invalid {name}")
                continue
            if not math.isfinite(value):
                reasons.append(f"nonfinite {name}")
            else:
                values[name] = value
        gross = values.get("gross_scenario_value")
        if gross is not None and gross <= 0:
            reasons.append("nonpositive gross_scenario_value")
        if gross and {"normalized_utility", "realized_utility"} <= values.keys() and not math.isclose(
            values["normalized_utility"], values["realized_utility"] / gross,
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            reasons.append("normalized utility mismatch")
        audit = str(row.get("replay_audit", "")).lower()
        if audit in {"mismatch", "failed"} or (
            replay_verified is not True and audit not in {"match", "verified"}
        ):
            reasons.append("replay mismatch or not verified")
    for scenario, scenario_rows in by_scenario.items():
        for name in (
            "cell_id", "initial_truth_digest", "public_initial_digest",
            "gross_scenario_value",
        ):
            values = {str(row.get(name, "")) for row in scenario_rows}
            if len(values) != 1 or "" in values:
                reasons.append(f"scenario {scenario} disagrees on {name}")
    unique_reasons = tuple(dict.fromkeys(reasons))
    return MatrixValidation(
        "COMPLETE" if not unique_reasons else "FAILED/INCOMPLETE",
        unique_reasons, len(expected), len(rows), gate_count,
    )


def _draw_index(
    manifest: str, contrast: str, cell: str, replicate: int, draw: int, size: int,
) -> int:
    token = f"dynamic-analysis-v1|{manifest}|{contrast}|{cell}|{replicate}|{draw}"
    return int.from_bytes(sha256(token.encode()).digest()[:8], "big") % size


def _contrast_summary(
    records: tuple[Mapping[str, Any], ...],
    comparator: str,
    iterations: int,
    manifest: str,
) -> dict[str, Any]:
    by_scenario: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in records:
        by_scenario.setdefault(str(row["scenario_id"]), {})[str(row["method"])] = row
    by_cell: dict[str, list[float]] = {}
    for methods in by_scenario.values():
        if "P" not in methods or comparator not in methods:
            continue
        left, right = methods["P"], methods[comparator]
        if left["cell_id"] != right["cell_id"]:
            raise ValueError("paired methods disagree on cell_id")
        difference = float(left["normalized_utility"]) - float(right["normalized_utility"])
        by_cell.setdefault(str(left["cell_id"]), []).append(difference)
    if not by_cell:
        raise ValueError(f"no complete P-{comparator} pairs")
    cell_means = [float(np.mean(values)) for values in by_cell.values()]
    differences = [value for values in by_cell.values() for value in values]
    contrast = f"P-{comparator}"
    bootstrap = []
    for replicate in range(iterations):
        means = [
            float(np.mean([
                values[_draw_index(manifest, contrast, cell, replicate, draw, len(values))]
                for draw in range(len(values))
            ]))
            for cell, values in sorted(by_cell.items())
        ]
        bootstrap.append(float(np.mean(means)))
    tolerance = 1e-12
    delta_quantiles = {
        name: float(np.quantile(differences, probability))
        for name, probability in (("median", 0.5), ("p05", 0.05), ("p95", 0.95))
    }
    bootstrap_quantiles = {
        name: float(np.quantile(bootstrap, probability))
        for name, probability in (("p025", 0.025), ("median", 0.5), ("p975", 0.975))
    }
    return {
        "label": "exploratory",
        "mean": float(np.mean(cell_means)),
        "episode_weighted_mean": float(np.mean(differences)),
        "delta_quantiles": delta_quantiles,
        "bootstrap_quantiles": bootstrap_quantiles,
        "win_tie_loss": {
            "win": sum(value > tolerance for value in differences),
            "tie": sum(abs(value) <= tolerance for value in differences),
            "loss": sum(value < -tolerance for value in differences),
        },
        "cell_count": len(by_cell), "pair_count": len(differences),
    }


def paired_d1_summary(
    records: Iterable[Mapping[str, Any]],
    expected_rectangle: Iterable[tuple[str, str]],
    *,
    expected_cells: Mapping[str, str],
    required_cells: Iterable[str] | None = None,
    bootstrap_iterations: int = 10_000,
    manifest_digest: str = "unregistered-manifest",
    replay_verified: bool | None = None,
) -> dict[str, Any]:
    rows = tuple(records)
    expected = tuple(expected_rectangle)
    validation = validate_method_matrix(
        rows, expected, expected_cells=expected_cells,
        required_cells=required_cells, replay_verified=replay_verified,
    )
    if validation.status != "COMPLETE":
        return {
            "status": "FAILED/INCOMPLETE", "reasons": list(validation.reasons),
            "principal_output_available": False, "analysis_label": "exploratory",
            "d1_confirmatory_success": False, "d2_authorized": False,
        }
    methods = {str(row["method"]) for row in rows}
    if not set(PRIMARY) <= methods:
        return {
            "status": "FAILED/INCOMPLETE", "reasons": ["primary P-B1m pair missing"],
            "principal_output_available": False, "analysis_label": "exploratory",
            "d1_confirmatory_success": False, "d2_authorized": False,
        }
    contrasts: dict[str, Any] = {}
    for comparator in (PRIMARY[1], *SECONDARY, *SENSITIVITY, "CEX"):
        if comparator in methods:
            name = f"P-{comparator}"
            contrasts[name] = _contrast_summary(
                rows, comparator, bootstrap_iterations, manifest_digest,
            )
            contrasts[name]["family"] = (
                "primary" if comparator == "B1m"
                else "secondary" if comparator in SECONDARY
                else "sensitivity" if comparator in SENSITIVITY
                else "cex_separate"
            )
    return {
        "status": "COMPLETE", "principal_output_available": True,
        "analysis_label": "exploratory",
        "contrasts": contrasts,
        "manifest_digest": manifest_digest,
        "quantile_convention": "linear_type7", "bootstrap_namespace": "dynamic-analysis-v1",
        "bootstrap_key_template": "dynamic-analysis-v1|{manifest}|{contrast}|{cell}|{replicate}|{draw}",
        "replay_evidence": (
            "external_worker_replay" if replay_verified is True
            else "per_episode_terminal_record"
        ),
        "bootstrap_iterations": bootstrap_iterations,
        "future_d2_rule": {
            "effect_at_least": 0.01, "ci_lower_above": 0.0,
            "complete_matrix": True, "zero_gates": True,
        },
        "d1_confirmatory_success": False,
        "d2_authorized": False,
    }


def write_dynamic_verdict(path: str | Path, summary: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dynamic D1 exploratory verdict", "",
        f"- Status: `{summary.get('status', 'UNKNOWN')}`",
        "- Analysis label: `exploratory`",
        "- D1 confirmatory success: `false`",
        "- This result does not authorize D2.", "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "CalibrationBatch", "CalibrationCoverageRecord", "CalibrationSelection",
    "MatrixValidation", "MethodBlindDiagnostics", "PublicTick0Coverage",
    "paired_d1_summary", "project_calibration_coverage",
    "select_first_passing_calibration", "summarize_effect_blind",
    "validate_method_matrix", "write_dynamic_verdict",
]
