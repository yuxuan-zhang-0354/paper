# BDTL-CBBA: Code and Experimental Summary Data

This repository contains the Python implementation and experimental data bundle for **BDTL-CBBA**, an improved dynamic consensus-based bundle algorithm for multi-UAV reconnaissance--strike task allocation under uncertain target class and damage state.

BDTL-CBBA combines a belief-driven fixed-mode task lifecycle with completion-event replanning, prefix-warped bidding, full bundle reconstruction, and next-task commitment. The implementation includes the proposed method, comparison policies, centralized exact references for small planning epochs, scenario generators, experiment runners, and statistical-analysis scripts.

## Repository Contents

- `src/uav_lifecycle/`: algorithm, belief update, task lifecycle, CBBA, exact allocation, scenario, and simulation modules.
- `experiments/`: experiment-design, execution, analysis, and manuscript-data preparation scripts.
- `tests/`: unit and integration tests.
- `preregistration/` and `docs/superpowers/`: frozen scenario registration and experiment-design specifications required by the validation code.
- `results/dynamic_mainline/`: frozen D2--D5 experiment manifests and their matching execution metadata.
- `results/manuscript_data/`: compact CSV/JSON data used for the reported tables, statistical contrasts, and figure values.
- `pyproject.toml`: package metadata and dependencies.

The manuscript, submission PDFs, source figures, review files, and intermediate writing artifacts are intentionally excluded.

## Installation

Python 3.13 or later is recommended.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Validation

Run the test suite from the repository root:

```bash
python -m pytest
```

For the lightweight package, the expected result is **493 passed and 1 skipped**. The skipped test verifies source pinning against the omitted full D2 raw archive; all algorithmic, simulation, manifest, and analysis tests execute normally.

The compact manuscript-data package can be checked through:

```bash
python -c "import json; print(json.load(open('results/manuscript_data/validation_report.json'))['status'])"
```

The expected status is `PASS`.

## Running the Main Experiment Families

The runners write newly generated records to user-selected output directories. The examples below keep generated data under `results/generated/`, which is ignored by Git.

```bash
# D2 core evaluation
python -m experiments.run_dynamic_d2 \
  --manifest results/dynamic_mainline/d2_design/d2_manifest.json \
  --authorization results/dynamic_mainline/d2_design/execution_authorization.json \
  --output results/generated/d2

# D3 external validation and allocation pressure
python -m experiments.run_dynamic_d3 \
  --manifest results/dynamic_mainline/d3_design/d3_manifest.json \
  --authorization results/dynamic_mainline/d3_design/execution_authorization.json \
  --output results/generated/d3 --workers 22

# D4 battlefield and reachability sensitivity
python -m experiments.run_dynamic_d4 \
  --manifest results/dynamic_mainline/d4_design/d4_manifest.json \
  --authorization results/dynamic_mainline/d4_design/execution_authorization.json \
  --output results/generated/d4 --workers 22

# D5 bidding-mechanism factorial ablation
python -m experiments.run_dynamic_d5 \
  --manifest results/dynamic_mainline/d5_factorial_ablation/design/d5_manifest.json \
  --authorization results/dynamic_mainline/d5_factorial_ablation/design/execution_authorization.json \
  --output results/generated/d5 --workers 22
```

Worker counts may be reduced without changing the counter-based random outcomes. Full experiment families can be computationally intensive; the checked summary data are already available under `results/manuscript_data/`.

## Experimental Data

The complete raw run archive is approximately 1.1 GB and is not included in this lightweight repository. Instead, the repository provides:

1. frozen scenario manifests and execution metadata;
2. all manuscript-level aggregate contrasts and confidence intervals;
3. allocator stability, workload, BDA, scalability, robustness, reachability, exact-reference, and communication summaries;
4. an inventory and automated validation report.

See [DATA.md](DATA.md) and `results/manuscript_data/README.md` for details.
