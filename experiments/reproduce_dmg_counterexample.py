"""Persist the audited deadline-induced DMG counterexample."""

from argparse import ArgumentParser
from pathlib import Path

from uav_lifecycle.artifacts import write_json_atomic
from uav_lifecycle.path_score import reproduce_deadline_counterexample


AUDITED_VALUES = {
    "raw_small": 23.318506084536523,
    "raw_large": 22.519091091023625,
    "feasible_small": 9.877664575357215,
    "feasible_large": 22.519091091023625,
}
AUDIT_TOLERANCE = 1e-9


def write_counterexample_artifacts(output: str | Path) -> dict[str, object]:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    result = reproduce_deadline_counterexample()
    write_json_atomic(
        destination / "config.json",
        {
            "experiment": "deadline_constrained_dmg_counterexample",
            "origin": [0.0, 0.0],
            "discount_base": 0.98,
            "deadline": 29.0,
            "small_path": ["A"],
            "large_path": ["K", "A"],
            "inserted_task": "J",
            "tasks": {
                "A": {
                    "location": [-7.0, -7.0],
                    "service_time": 1.0,
                    "value": 100.0,
                },
                "K": {
                    "location": [5.0, -2.0],
                    "service_time": 0.0,
                    "value": 50.0,
                },
                "J": {
                    "location": [8.0, -2.0],
                    "service_time": 3.0,
                    "value": 40.0,
                },
            },
            "audited_values": AUDITED_VALUES,
            "tolerance": AUDIT_TOLERANCE,
        },
    )
    write_json_atomic(destination / "reproduction.json", result)
    (destination / "run.log").write_text(
        (
            "counterexample: "
            f"raw_dmg_holds={result['raw_dmg_holds']}, "
            "constrained_dmg_violated="
            f"{result['constrained_dmg_violated']}\n"
        ),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/dmg_counterexample")
    )
    args = parser.parse_args()
    result = write_counterexample_artifacts(args.output)
    values_match = all(
        abs(float(result[name]) - expected) <= AUDIT_TOLERANCE
        for name, expected in AUDITED_VALUES.items()
    )
    return (
        0
        if values_match
        and result["raw_dmg_holds"]
        and result["constrained_dmg_violated"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
