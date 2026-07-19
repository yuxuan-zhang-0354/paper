"""Minimal pre-execution D2 multiprocessing determinism smoke."""

from experiments.run_dynamic_d2 import (
    DEFAULT_AUTHORIZATION,
    DEFAULT_MANIFEST,
    _execute,
    build_scenarios,
    load_frozen_design,
)


def main() -> int:
    manifest = load_frozen_design(DEFAULT_MANIFEST, DEFAULT_AUTHORIZATION)
    scenario = build_scenarios(manifest)[:1]
    methods = ("P", "B1m")
    serial = _execute(scenario, methods, 1)
    parallel = _execute(scenario, methods, 2)
    if serial != parallel:
        raise RuntimeError("D2 smoke replay mismatch")
    print("D2 smoke PASS: 1 scenario x 2 methods, workers 1 == 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
