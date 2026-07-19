"""Execute Gates A-D and write an auditable first-batch manifest."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from uav_lifecycle.artifacts import jsonable, sha256_file, write_json_atomic
from uav_lifecycle.scenarios import (
    PREREGISTRATION_SHA256,
    preregistration_path,
    validation_parameter_sets,
)

from .reproduce_dmg_counterexample import (
    AUDITED_VALUES,
    AUDIT_TOLERANCE,
    write_counterexample_artifacts,
)
from .run_belief_sweep import run_belief_sweep
from .run_properties import write_property_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_METADATA_KEYS = frozenset(
    {"timestamp", "started_at", "finished_at", "duration_seconds"}
)


def _without_runtime_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_runtime_metadata(item)
            for key, item in value.items()
            if key not in RUNTIME_METADATA_KEYS
        }
    if isinstance(value, list):
        return [_without_runtime_metadata(item) for item in value]
    return value


def semantic_json_equal(first: Any, second: Any) -> bool:
    """Compare JSON data while ignoring explicitly registered runtime fields."""

    return _without_runtime_metadata(jsonable(first)) == _without_runtime_metadata(
        jsonable(second)
    )


def extract_passed_test_count(output: str) -> int:
    """Extract pytest's final passed count or reject an unrecognized report."""

    matches = re.findall(r"(?m)(\d+)\s+passed(?:[,\s]|$)", output)
    if not matches:
        raise ValueError("pytest output does not contain a passed-test count")
    return int(matches[-1])


def hash_files(
    paths: tuple[Path, ...] | list[Path], root: str | Path
) -> dict[str, str]:
    """Hash a frozen list of source files with stable root-relative keys."""

    root_path = Path(root).resolve()
    return {
        path.resolve().relative_to(root_path).as_posix(): sha256_file(path)
        for path in sorted((Path(path) for path in paths), key=lambda item: str(item))
    }


def _source_paths() -> tuple[Path, ...]:
    paths: list[Path] = []
    for directory in ("src", "experiments", "tests"):
        paths.extend((PROJECT_ROOT / directory).rglob("*.py"))
    paths.extend(
        (
            PROJECT_ROOT / "pyproject.toml",
            PROJECT_ROOT
            / "docs/superpowers/specs/2026-07-13-multi-uav-validation-harness-design.md",
            PROJECT_ROOT
            / "docs/superpowers/plans/2026-07-13-first-batch-validation-implementation.md",
            preregistration_path(),
        )
    )
    return tuple(sorted(paths, key=lambda path: path.as_posix()))


def write_running_manifest(
    output: str | Path,
    run_id: str,
    started_at: str,
    source_hashes: dict[str, str],
) -> None:
    """Atomically invalidate any stale PASS before a new run starts."""

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        destination / "first_batch_manifest.json",
        {
            "status": "RUNNING",
            "run_id": run_id,
            "started_at": started_at,
            "source_hashes_start": source_hashes,
        },
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_first_batch(
    output: str | Path = "results", workers: int | None = None
) -> dict[str, Any]:
    """Run the formal validation batch, preserving failures in the manifest."""

    destination = Path(output)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    source_paths = _source_paths()
    source_hashes_start = hash_files(source_paths, PROJECT_ROOT)
    write_running_manifest(
        destination,
        run_id=run_id,
        started_at=started_at,
        source_hashes=source_hashes_start,
    )
    step_records: list[dict[str, Any]] = []
    unexplained_exceptions: list[str] = []

    def timed_step(
        name: str, command: str, action: Callable[[], Any]
    ) -> Any | None:
        start = perf_counter()
        try:
            result = action()
        except Exception as exc:  # retained as auditable FAIL evidence
            step_records.append(
                {
                    "name": name,
                    "command": command,
                    "execution_status": "FAIL",
                    "duration_seconds": perf_counter() - start,
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )
            unexplained_exceptions.append(f"{name}: {type(exc).__name__}: {exc}")
            return None
        step_records.append(
            {
                "name": name,
                "command": command,
                "execution_status": "PASS",
                "duration_seconds": perf_counter() - start,
            }
        )
        return result

    def run_tests(index: int) -> int:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        log = completed.stdout
        if completed.stderr:
            log += "\n[stderr]\n" + completed.stderr
        (destination / f"test_run_{index}.log").write_text(log, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"pytest run {index} exited {completed.returncode}")
        return extract_passed_test_count(log)

    test_count_1 = timed_step(
        "deterministic_tests_1", "python -m pytest -q", lambda: run_tests(1)
    )
    test_count_2 = timed_step(
        "deterministic_tests_2", "python -m pytest -q", lambda: run_tests(2)
    )
    properties = timed_step(
        "gate_a_b_properties",
        "python -m experiments.run_properties --output results/properties",
        lambda: write_property_artifacts(destination / "properties"),
    )
    sweep = timed_step(
        "gate_c_belief_sweep",
        "python -m experiments.run_belief_sweep --step 0.02 --output results/belief_sweep",
        lambda: run_belief_sweep(
            0.02, destination / "belief_sweep", workers=workers
        ),
    )
    counterexample = timed_step(
        "gate_d_counterexample",
        "python -m experiments.reproduce_dmg_counterexample --output results/dmg_counterexample",
        lambda: write_counterexample_artifacts(destination / "dmg_counterexample"),
    )
    repro_properties = timed_step(
        "reproduce_properties",
        "python -m experiments.run_properties --output results/repro_check/properties",
        lambda: write_property_artifacts(destination / "repro_check/properties"),
    )
    repro_counterexample = timed_step(
        "reproduce_counterexample",
        "python -m experiments.reproduce_dmg_counterexample --output results/repro_check/dmg_counterexample",
        lambda: write_counterexample_artifacts(
            destination / "repro_check/dmg_counterexample"
        ),
    )

    property_reproducible = bool(
        properties is not None
        and repro_properties is not None
        and semantic_json_equal(properties, repro_properties)
        and semantic_json_equal(
            _read_json(destination / "properties/config.json"),
            _read_json(destination / "repro_check/properties/config.json"),
        )
    )
    counterexample_reproducible = bool(
        counterexample is not None
        and repro_counterexample is not None
        and semantic_json_equal(counterexample, repro_counterexample)
        and semantic_json_equal(
            _read_json(destination / "dmg_counterexample/config.json"),
            _read_json(destination / "repro_check/dmg_counterexample/config.json"),
        )
    )

    preregistration_digest = sha256_file(preregistration_path())
    registration_snapshot_valid = (
        preregistration_digest == PREREGISTRATION_SHA256
    )
    expected_configs: Any = None
    registration_unchanged = False
    try:
        expected_configs = jsonable(validation_parameter_sets())
        sweep_config = _read_json(destination / "belief_sweep/config.json")
        property_config = _read_json(destination / "properties/config.json")
        registration_unchanged = bool(
            registration_snapshot_valid
            and sweep_config["configurations"] == expected_configs
            and property_config["baseline_config"] == expected_configs[0]
        )
    except (
        OSError,
        KeyError,
        ValueError,
        TypeError,
        RuntimeError,
        json.JSONDecodeError,
    ):
        registration_unchanged = False

    gate_a_pass = bool(properties and properties.get("gate_a_pass"))
    gate_b_pass = bool(properties and properties.get("gate_b_pass"))
    gate_c_pass = bool(sweep and sweep.get("gate_c_pass"))
    gate_d_pass = bool(
        counterexample
        and counterexample.get("raw_dmg_holds")
        and counterexample.get("constrained_dmg_violated")
        and all(
            abs(float(counterexample[name]) - expected) <= AUDIT_TOLERANCE
            for name, expected in AUDITED_VALUES.items()
        )
    )
    tests_pass = bool(
        test_count_1 is not None
        and test_count_2 is not None
        and test_count_1 == test_count_2
    )

    required_relative_paths = (
        "test_run_1.log",
        "test_run_2.log",
        "properties/config.json",
        "properties/summary.json",
        "properties/run.log",
        "belief_sweep/config.json",
        "belief_sweep/decision_regions.csv",
        "belief_sweep/action_counts.json",
        "belief_sweep/summary.json",
        "belief_sweep/run.log",
        "dmg_counterexample/config.json",
        "dmg_counterexample/reproduction.json",
        "dmg_counterexample/run.log",
        "repro_check/properties/config.json",
        "repro_check/properties/summary.json",
        "repro_check/dmg_counterexample/config.json",
        "repro_check/dmg_counterexample/reproduction.json",
    )
    artifact_presence = {
        relative: (destination / relative).is_file()
        for relative in required_relative_paths
    }
    artifacts_complete = all(artifact_presence.values())
    artifact_hashes = {
        relative: sha256_file(destination / relative)
        for relative, present in artifact_presence.items()
        if present
    }
    source_hashes_end = hash_files(source_paths, PROJECT_ROOT)
    sources_unchanged_during_run = source_hashes_start == source_hashes_end
    acceptance_by_step = {
        "deterministic_tests_1": tests_pass,
        "deterministic_tests_2": tests_pass,
        "gate_a_b_properties": gate_a_pass and gate_b_pass,
        "gate_c_belief_sweep": gate_c_pass,
        "gate_d_counterexample": gate_d_pass,
        "reproduce_properties": property_reproducible,
        "reproduce_counterexample": counterexample_reproducible,
    }
    for record in step_records:
        record["acceptance_status"] = (
            "PASS" if acceptance_by_step.get(record["name"], False) else "FAIL"
        )
    status = (
        "PASS"
        if all(
            (
                tests_pass,
                gate_a_pass,
                gate_b_pass,
                gate_c_pass,
                gate_d_pass,
                property_reproducible,
                counterexample_reproducible,
                registration_unchanged,
                registration_snapshot_valid,
                sources_unchanged_during_run,
                artifacts_complete,
                not unexplained_exceptions,
            )
        )
        else "FAIL"
    )
    finished_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "status": status,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "test_counts": [test_count_1, test_count_2],
        "gates": {
            "A": gate_a_pass,
            "B": gate_b_pass,
            "C": gate_c_pass,
            "D": gate_d_pass,
        },
        "reproducibility": {
            "properties": property_reproducible,
            "counterexample": counterexample_reproducible,
        },
        "registration_unchanged": registration_unchanged,
        "registration_snapshot": {
            "path": preregistration_path().relative_to(PROJECT_ROOT).as_posix(),
            "expected_sha256": PREREGISTRATION_SHA256,
            "actual_sha256": preregistration_digest,
            "digest_valid": registration_snapshot_valid,
        },
        "sources_unchanged_during_run": sources_unchanged_during_run,
        "source_hashes_start": source_hashes_start,
        "source_hashes_end": source_hashes_end,
        "artifacts_complete": artifacts_complete,
        "artifact_presence": artifact_presence,
        "artifact_hashes_sha256": artifact_hashes,
        "steps": step_records,
        "unexplained_exceptions": unexplained_exceptions,
    }
    write_json_atomic(destination / "first_batch_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    manifest = run_first_batch(args.output, args.workers)
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
