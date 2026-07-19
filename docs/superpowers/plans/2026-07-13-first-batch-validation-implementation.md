# First-Batch Multi-UAV Validation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the first validation batch for the joint-belief lifecycle model: probability properties, attack-induced dependence, common-baseline action values, belief-simplex decision regions, and the audited deadline/DMG counterexample.

**Architecture:** A small pure-Python package exposes deterministic numerical functions with no global state. Pytest verifies each mathematical property before three experiment entry points persist machine-readable artifacts. The belief sweep parallelizes across pre-registered parameter sets, while a single parent process writes the shared CSV and summaries.

**Tech Stack:** Python 3.13, NumPy 2.5.1, pytest 9.1.1, standard-library `csv`, `json`, `logging`, and `concurrent.futures`. No pandas, Hypothesis, PyArrow, GPU, or network dependency.

## Global Constraints

- State order is exactly `(HA, HD, LA, LD)`.
- Confusion matrices use observation rows and truth columns; every column sums to one.
- Each Attack consumes one munition and uses `pi_h < pi_l` in the principal scenario.
- `Q` values are target-local resource-optimistic planning surrogates, not fleet residual values.
- First batch does not implement CBBA, ranked fallback, simultaneous attack, or a POMDP solver.
- Random tests use explicit deterministic seeds.
- Default parallelism is `min(22, max(1, os.cpu_count() - 2))` with no nested pools.
- Workers never append to a shared output file; the parent process performs aggregation.
- Do not initialize Git without user authorization. This workspace is not currently a Git repository, so commit steps are recorded as skipped checkpoints.

---

## File Map

- `pyproject.toml`: package metadata and pytest configuration.
- `src/uav_lifecycle/model.py`: state indices and probability validation.
- `src/uav_lifecycle/belief.py`: observation kernels, prediction probabilities, and Bayes updates.
- `src/uav_lifecycle/attack.py`: attack transition, expected reward, physical-state transition, and joint statistics.
- `src/uav_lifecycle/rollout.py`: exponential discount, terminal surrogate, and four action values.
- `src/uav_lifecycle/simplex.py`: exact rational simplex grid and action ranking.
- `src/uav_lifecycle/path_score.py`: deterministic route timing, scoring, and insertion evaluation.
- `src/uav_lifecycle/scenarios.py`: pre-registered matrices, parameter sets, and audited DMG scenario.
- `src/uav_lifecycle/artifacts.py`: atomic JSON writing and run metadata.
- `experiments/run_properties.py`: persistent Gate A/B summary.
- `experiments/run_belief_sweep.py`: parallel deterministic decision-region sweep.
- `experiments/reproduce_dmg_counterexample.py`: Gate D reproduction artifact.
- `tests/`: unit and property tests matching each module.

### Task 1: Package Skeleton and Probability Validation

**Files:**
- Create: `pyproject.toml`
- Create: `src/uav_lifecycle/__init__.py`
- Create: `src/uav_lifecycle/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `StateIndex`, `as_belief(values, atol=1e-12)`, `as_column_stochastic(matrix, atol=1e-12)`.
- Consumes: NumPy arrays or array-like values.

- [ ] **Step 1: Write failing model tests**

```python
import numpy as np
import pytest

from uav_lifecycle.model import StateIndex, as_belief, as_column_stochastic


def test_state_order_is_frozen():
    assert [member.name for member in StateIndex] == ["HA", "HD", "LA", "LD"]
    assert [member.value for member in StateIndex] == [0, 1, 2, 3]


def test_as_belief_accepts_simplex_vector():
    actual = as_belief([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(actual, [0.1, 0.2, 0.3, 0.4])


@pytest.mark.parametrize("bad", [[-0.1, 0.2, 0.4, 0.5], [0.2, 0.2, 0.2, 0.2], [1.0, 0.0]])
def test_as_belief_rejects_invalid_vectors(bad):
    with pytest.raises(ValueError):
        as_belief(bad)


def test_column_stochastic_convention():
    matrix = as_column_stochastic([[0.65, 0.15], [0.35, 0.85]])
    np.testing.assert_allclose(matrix.sum(axis=0), [1.0, 1.0])
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -m pytest tests/test_model.py -v
```

Expected: collection fails because `uav_lifecycle.model` does not exist.

- [ ] **Step 3: Add package configuration**

```toml
[build-system]
requires = ["setuptools>=80"]
build-backend = "setuptools.build_meta"

[project]
name = "uav-lifecycle-validation"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = ["numpy>=2.5"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

Create an empty `src/uav_lifecycle/__init__.py`, then run:

```powershell
python -m pip install -e .
```

Expected: editable package installation succeeds without downloading new dependencies.

- [ ] **Step 4: Implement strict validators**

```python
from enum import IntEnum
import numpy as np
from numpy.typing import ArrayLike, NDArray


class StateIndex(IntEnum):
    HA = 0
    HD = 1
    LA = 2
    LD = 3


def as_belief(values: ArrayLike, atol: float = 1e-12) -> NDArray[np.float64]:
    belief = np.asarray(values, dtype=np.float64)
    if belief.shape != (4,):
        raise ValueError(f"belief must have shape (4,), got {belief.shape}")
    if np.any(belief < -atol):
        raise ValueError("belief contains a negative probability")
    if not np.isclose(float(belief.sum()), 1.0, atol=atol, rtol=0.0):
        raise ValueError("belief probabilities must sum to one")
    belief = np.maximum(belief, 0.0)
    return belief / belief.sum()


def as_column_stochastic(values: ArrayLike, atol: float = 1e-12) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("matrix must be two-dimensional with at least one truth column")
    if np.any(matrix < -atol):
        raise ValueError("matrix contains a negative probability")
    if not np.allclose(matrix.sum(axis=0), 1.0, atol=atol, rtol=0.0):
        raise ValueError("each truth column must sum to one")
    return np.maximum(matrix, 0.0)
```

- [ ] **Step 5: Run tests and record checkpoint**

Run: `python -m pytest tests/test_model.py -v`  
Expected: 6 tests pass.  
Checkpoint: no Git commit because the workspace is not a repository.

### Task 2: Observation Kernels and Bayes Updates

**Files:**
- Create: `src/uav_lifecycle/belief.py`
- Test: `tests/test_belief.py`

**Interfaces:**
- Consumes: validated `(4,)` belief vectors and observation-row/truth-column matrices.
- Produces: `recon_kernel`, `bda_kernel`, `observation_probabilities`, `bayes_update`, `expected_posterior`.

- [ ] **Step 1: Write failing Bayes tests**

```python
import numpy as np
import pytest

from uav_lifecycle.belief import (
    bayes_update,
    bda_kernel,
    expected_posterior,
    observation_probabilities,
    recon_kernel,
)


CLASS = np.array([[0.65, 0.15], [0.35, 0.85]])
DAMAGE_R = np.array([[0.75, 0.25], [0.25, 0.75]])
DAMAGE_B = np.array([[0.92, 0.06], [0.08, 0.94]])
B = np.array([0.4, 0.1, 0.3, 0.2])


def test_recon_kernel_shape_and_columns():
    kernel = recon_kernel(CLASS, DAMAGE_R)
    assert kernel.shape == (4, 4)
    np.testing.assert_allclose(kernel.sum(axis=0), np.ones(4))


def test_bda_has_no_direct_class_channel():
    kernel = bda_kernel(DAMAGE_B)
    np.testing.assert_allclose(kernel[:, 0], kernel[:, 2])
    np.testing.assert_allclose(kernel[:, 1], kernel[:, 3])


def test_predictive_probabilities_and_posterior_are_normalized():
    kernel = recon_kernel(CLASS, DAMAGE_R)
    predictive = observation_probabilities(B, kernel)
    np.testing.assert_allclose(predictive.sum(), 1.0)
    posterior = bayes_update(B, kernel, 0)
    np.testing.assert_allclose(posterior.sum(), 1.0)
    assert np.all(posterior >= 0.0)


def test_bayes_martingale_identity():
    kernel = recon_kernel(CLASS, DAMAGE_R)
    np.testing.assert_allclose(expected_posterior(B, kernel), B, atol=1e-12)


def test_zero_probability_observation_is_rejected():
    kernel = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="zero predictive probability"):
        bayes_update(B, kernel, 1)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/test_belief.py -v`  
Expected: import failure for `uav_lifecycle.belief`.

- [ ] **Step 3: Implement kernel construction and Bayes update**

```python
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import as_belief, as_column_stochastic


def recon_kernel(class_matrix: ArrayLike, damage_matrix: ArrayLike) -> NDArray[np.float64]:
    mc = as_column_stochastic(class_matrix)
    ms = as_column_stochastic(damage_matrix)
    if mc.shape != (2, 2) or ms.shape != (2, 2):
        raise ValueError("binary Recon matrices must both have shape (2, 2)")
    kernel = np.empty((4, 4), dtype=np.float64)
    for oc in range(2):
        for os in range(2):
            row = 2 * oc + os
            for c in range(2):
                for s in range(2):
                    col = 2 * c + s
                    kernel[row, col] = mc[oc, c] * ms[os, s]
    return kernel


def bda_kernel(damage_matrix: ArrayLike) -> NDArray[np.float64]:
    ms = as_column_stochastic(damage_matrix)
    if ms.shape != (2, 2):
        raise ValueError("binary BDA damage matrix must have shape (2, 2)")
    return np.column_stack((ms[:, 0], ms[:, 1], ms[:, 0], ms[:, 1]))


def observation_probabilities(belief: ArrayLike, kernel: ArrayLike) -> NDArray[np.float64]:
    b = as_belief(belief)
    z = as_column_stochastic(kernel)
    if z.shape[1] != 4:
        raise ValueError("kernel must have four hidden-state columns")
    return z @ b


def bayes_update(belief: ArrayLike, kernel: ArrayLike, observation: int) -> NDArray[np.float64]:
    b = as_belief(belief)
    z = as_column_stochastic(kernel)
    if observation < 0 or observation >= z.shape[0]:
        raise IndexError("observation index out of range")
    numerator = z[observation] * b
    denominator = float(numerator.sum())
    if denominator <= 0.0:
        raise ValueError("observation has zero predictive probability")
    return numerator / denominator


def expected_posterior(belief: ArrayLike, kernel: ArrayLike) -> NDArray[np.float64]:
    b = as_belief(belief)
    probabilities = observation_probabilities(b, kernel)
    result = np.zeros(4, dtype=np.float64)
    for observation, probability in enumerate(probabilities):
        if probability > 0.0:
            result += probability * bayes_update(b, kernel, observation)
    return result
```

- [ ] **Step 4: Run tests and full regression**

Run: `python -m pytest tests/test_belief.py tests/test_model.py -v`  
Expected: all tests pass.  
Checkpoint: record test output; no Git commit in the non-repository workspace.

### Task 3: Attack Dynamics, Reward, and Joint-Dependence Evidence

**Files:**
- Create: `src/uav_lifecycle/attack.py`
- Test: `tests/test_attack.py`
- Test: `tests/test_joint_dependence.py`

**Interfaces:**
- Produces: `attack_matrix`, `predict_attack`, `expected_attack_reward`, `apply_physical_attack`, `marginals`, `survival_given_class`, `class_survival_covariance`.
- Consumes: validated beliefs, `pi_h`, `pi_l`, `value_h`, `value_l`.

- [ ] **Step 1: Write failing attack tests**

```python
import numpy as np

from uav_lifecycle.attack import (
    apply_physical_attack,
    attack_matrix,
    expected_attack_reward,
    predict_attack,
)


def test_attack_prediction_closes_simplex_and_preserves_class_marginal():
    belief = np.array([0.4, 0.1, 0.3, 0.2])
    predicted = predict_attack(belief, pi_h=0.4, pi_l=0.75)
    np.testing.assert_allclose(predicted.sum(), 1.0)
    np.testing.assert_allclose(predicted[[0, 1]].sum(), belief[[0, 1]].sum())
    np.testing.assert_allclose(predicted[[2, 3]].sum(), belief[[2, 3]].sum())
    assert predicted[[0, 2]].sum() <= belief[[0, 2]].sum()


def test_attack_matrix_is_column_stochastic():
    np.testing.assert_allclose(attack_matrix(0.4, 0.75).sum(axis=0), np.ones(4))


def test_two_attack_expected_reward_matches_at_least_one_success():
    belief = np.array([0.5, 0.0, 0.5, 0.0])
    first = expected_attack_reward(belief, 0.4, 0.75, 100.0, 30.0)
    after = predict_attack(belief, 0.4, 0.75)
    second = expected_attack_reward(after, 0.4, 0.75, 100.0, 30.0)
    expected = 0.5 * 100.0 * (1.0 - 0.6**2) + 0.5 * 30.0 * (1.0 - 0.25**2)
    np.testing.assert_allclose(first + second, expected)


def test_destroyed_state_is_absorbing_and_never_repays_reward():
    next_state, reward = apply_physical_attack(state=1, pi_h=0.4, pi_l=0.75, value_h=100.0, value_l=30.0, uniform=0.0)
    assert next_state == 1
    assert reward == 0.0
```

- [ ] **Step 2: Write failing joint-dependence tests**

```python
import numpy as np

from uav_lifecycle.attack import expected_attack_reward, predict_attack, survival_given_class
from uav_lifecycle.belief import bayes_update, bda_kernel


def test_general_post_attack_independence_condition():
    belief = np.array([0.4, 0.1, 0.1, 0.4])
    predicted = predict_attack(belief, pi_h=0.2, pi_l=0.8)
    alive_h, alive_l = survival_given_class(predicted)
    assert not np.isclose(alive_h, alive_l)


def test_bda_destroyed_observation_reduces_high_class_probability_when_high_targets_are_harder():
    prior = np.array([0.5, 0.0, 0.5, 0.0])
    attacked = predict_attack(prior, pi_h=0.2, pi_l=0.8)
    kernel = bda_kernel([[0.9, 0.1], [0.1, 0.9]])
    posterior = bayes_update(attacked, kernel, observation=1)
    assert posterior[[0, 1]].sum() < 0.5


def test_equal_marginals_do_not_determine_attack_value():
    independent = np.array([0.25, 0.25, 0.25, 0.25])
    correlated = np.array([0.5, 0.0, 0.0, 0.5])
    value_1 = expected_attack_reward(independent, 0.4, 0.75, 100.0, 30.0)
    value_2 = expected_attack_reward(correlated, 0.4, 0.75, 100.0, 30.0)
    assert not np.isclose(value_1, value_2)
```

- [ ] **Step 3: Run focused tests and verify RED**

Run: `python -m pytest tests/test_attack.py tests/test_joint_dependence.py -v`  
Expected: import failure for `uav_lifecycle.attack`.

- [ ] **Step 4: Implement minimal dynamics and statistics**

Implement the fixed transition matrix for `(HA, HD, LA, LD)`, validate every probability is in `[0,1]`, and map physical states `0/1` to high class and `2/3` to low class. `apply_physical_attack` must leave states 1 and 3 unchanged with zero reward; for states 0 and 2 it destroys exactly when `uniform < pi_c`.

```python
def survival_given_class(belief):
    b = as_belief(belief)
    p_h = b[0] + b[1]
    p_l = b[2] + b[3]
    return b[0] / p_h, b[2] / p_l
```

Raise `ValueError` if a requested conditional class has zero probability rather than silently dividing by zero.

- [ ] **Step 5: Run tests and regression**

Run: `python -m pytest tests/test_attack.py tests/test_joint_dependence.py tests/test_belief.py -v`  
Expected: all tests pass.  
Checkpoint: record test output; no Git commit.

### Task 4: Common-Baseline Rollout Values

**Files:**
- Create: `src/uav_lifecycle/rollout.py`
- Test: `tests/test_rollout.py`

**Interfaces:**
- Produces: immutable `RolloutParameters`, `discount`, `terminal_attack_value`, `terminal_surrogate`, `action_values`, `costless_information_value`.
- Consumes: one belief, Recon/BDA kernels, and fixed model parameters.

- [ ] **Step 1: Write failing rollout tests**

```python
import numpy as np

from uav_lifecycle.belief import bda_kernel, recon_kernel
from uav_lifecycle.rollout import RolloutParameters, action_values, costless_information_value, discount


PARAMS = RolloutParameters(
    value_h=100.0, value_l=30.0, pi_h=0.4, pi_l=0.75,
    duration_r=4.0, duration_a=2.0, duration_b=1.5,
    cost_r=2.0, cost_a=6.0, cost_b=1.0, beta=0.02,
)
RC = np.array([[0.65, 0.15], [0.35, 0.85]])
RS = np.array([[0.75, 0.25], [0.25, 0.75]])
BS = np.array([[0.92, 0.06], [0.08, 0.94]])


def test_discount_has_semigroup_property():
    assert np.isclose(discount(3.0, 0.02) * discount(4.0, 0.02), discount(7.0, 0.02))


def test_action_values_include_defer_zero_and_are_repeatable():
    belief = np.array([0.4, 0.1, 0.3, 0.2])
    zr = recon_kernel(RC, RS)
    zb = bda_kernel(BS)
    first = action_values(belief, zr, zb, PARAMS)
    second = action_values(belief, zr, zb, PARAMS)
    assert first == second
    assert first["defer"] == 0.0


def test_costless_information_value_is_nonnegative():
    belief = np.array([0.4, 0.1, 0.3, 0.2])
    zr = recon_kernel(RC, RS)
    assert costless_information_value(belief, zr, PARAMS) >= -1e-12
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/test_rollout.py -v`  
Expected: import failure for `uav_lifecycle.rollout`.

- [ ] **Step 3: Implement rollout exactly as specified**

Use a frozen dataclass. Validate nonnegative values, durations, costs, and beta; validate probabilities in `[0,1]`. Compute expected posterior terminal values by enumerating all positive-probability observations. Return a dictionary with keys exactly `recon`, `attack`, `bda`, and `defer`.

`costless_information_value` is only:

```python
expected_terminal = sum(
    probability * terminal_surrogate(bayes_update(belief, kernel, observation), params)
    for observation, probability in enumerate(observation_probabilities(belief, kernel))
    if probability > 0.0
)
return expected_terminal - terminal_surrogate(belief, params)
```

It must not be reported as net Recon/BDA utility because it excludes sensing cost and delay.

- [ ] **Step 4: Run all tests**

Run: `python -m pytest -q`  
Expected: all current tests pass.  
Checkpoint: record test output; no Git commit.

### Task 5: Pre-Registered Scenarios, Simplex Sweep, and Artifacts

**Files:**
- Create: `src/uav_lifecycle/scenarios.py`
- Create: `src/uav_lifecycle/simplex.py`
- Create: `src/uav_lifecycle/artifacts.py`
- Create: `experiments/__init__.py`
- Create: `experiments/run_properties.py`
- Create: `experiments/run_belief_sweep.py`
- Test: `tests/test_simplex.py`
- Test: `tests/test_scenarios.py`

**Interfaces:**
- Produces: `validation_parameter_sets()`, `simplex_grid(step)`, `rank_actions(values)`, atomic JSON artifacts, and deterministic CSV records.
- Consumes: rollout functions from Task 4.

- [ ] **Step 1: Write failing grid and scenario tests**

```python
from uav_lifecycle.scenarios import validation_parameter_sets
from uav_lifecycle.simplex import simplex_grid, rank_actions


def test_simplex_step_half_has_ten_points():
    grid = simplex_grid(0.5)
    assert grid.shape == (10, 4)
    assert (grid.sum(axis=1) == 1.0).all()


def test_rank_actions_uses_deterministic_order_for_ties():
    best, second, margin = rank_actions({"recon": 1.0, "attack": 1.0, "bda": 0.0, "defer": 0.0})
    assert (best, second, margin) == ("recon", "attack", 0.0)


def test_pre_registered_parameter_count_and_story_direction():
    configs = validation_parameter_sets()
    assert len(configs) == 108
    assert all(config.params.pi_h < config.params.pi_l for config in configs)
    assert len({config.config_id for config in configs}) == 108
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/test_simplex.py tests/test_scenarios.py -v`  
Expected: import failures for the new modules.

- [ ] **Step 3: Implement exact integer simplex enumeration**

Convert `step` to `denominator = round(1 / step)` and reject it unless `denominator * step` is within `1e-12` of one. Enumerate nonnegative integers `(a,b,c,d)` summing to `denominator`, then divide once at the end. For step `0.02`, assert the generated count is `comb(53, 3) = 23426`.

Tie order in `rank_actions` is fixed as `recon`, `attack`, `bda`, `defer` solely for determinism; it is not a scientific preference.

- [ ] **Step 4: Implement the 108 pre-registered configurations**

Create 27 cost triples from `{2,5,8} x {6,12,20} x {1,3,6}` and four sensor variants: baseline, Recon-class-minus-0.10, Recon-damage-plus-0.10, and BDA-damage-minus-0.10. Use immutable dataclasses and stable IDs such as `baseline_r2_a6_b1`.

- [ ] **Step 5: Implement property artifact runner**

`run_properties.py` runs deterministic seeded checks over 10,000 Dirichlet beliefs using NumPy RNG seed `7132026`, writes `results/properties/config.json`, `summary.json`, and `run.log`, and exits nonzero if any invariant failure count is positive.

- [ ] **Step 6: Implement parallel belief sweep**

Each worker receives one immutable config and the common `(23426,4)` grid, returns a list of tuples containing config ID, four belief components, four Q values, best action, second action, margin, `P(H)`, and `P(A)`. The parent writes `decision_regions.csv` incrementally as futures complete and aggregates `action_counts.json`.

Set these environment variables before creating the pool:

```python
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
```

Use `ProcessPoolExecutor(max_workers=min(22, max(1, (os.cpu_count() or 1) - 2)))` and protect the entry point with `if __name__ == "__main__":` for Windows spawn safety.

- [ ] **Step 7: Run tests and a small pilot**

Run:

```powershell
python -m pytest tests/test_simplex.py tests/test_scenarios.py -v
python -m experiments.run_belief_sweep --step 0.5 --workers 2 --output results/belief_sweep_pilot
```

Expected: tests pass; pilot writes 1080 CSV records plus JSON configuration/count artifacts without worker write collisions.

### Task 6: Path Scoring and Audited DMG Counterexample

**Files:**
- Create: `src/uav_lifecycle/path_score.py`
- Create: `experiments/reproduce_dmg_counterexample.py`
- Test: `tests/test_dmg_counterexample.py`

**Interfaces:**
- Produces: immutable `Task`, `evaluate_path`, `all_insertions`, `best_insertion`, and `reproduce_deadline_counterexample`.
- Consumes: Euclidean coordinates, nonnegative service times/values, discount base, and optional completion deadline.

- [ ] **Step 1: Write the failing audited-number test**

```python
import numpy as np

from uav_lifecycle.path_score import reproduce_deadline_counterexample


def test_deadline_counterexample_matches_audited_values():
    result = reproduce_deadline_counterexample()
    np.testing.assert_allclose(result["raw_small"], 23.318506084536523, atol=1e-9)
    np.testing.assert_allclose(result["raw_large"], 22.519091091023625, atol=1e-9)
    np.testing.assert_allclose(result["feasible_small"], 9.877664575357215, atol=1e-9)
    np.testing.assert_allclose(result["feasible_large"], 22.519091091023625, atol=1e-9)
    assert result["raw_dmg_holds"] is True
    assert result["constrained_dmg_violated"] is True
```

- [ ] **Step 2: Run focused test and verify RED**

Run: `python -m pytest tests/test_dmg_counterexample.py -v`  
Expected: import failure for `uav_lifecycle.path_score`.

- [ ] **Step 3: Implement route timing and scoring**

`evaluate_path` starts at `(0,0)` and time zero. For each task, add Euclidean travel time at unit speed, record start time, add `discount_base ** start_time * value`, then add service time. Completion time is the final service completion. `best_insertion` enumerates every order-preserving position and optionally rejects paths with completion time above the deadline.

Hard-code the audited scenario only in `reproduce_deadline_counterexample`; keep generic functions scenario-independent.

- [ ] **Step 4: Run focused and full test suites**

Run:

```powershell
python -m pytest tests/test_dmg_counterexample.py -v
python -m pytest -q
```

Expected: audited values match within `1e-9`; the entire suite passes.

- [ ] **Step 5: Write persistent counterexample artifacts**

Run:

```powershell
python -m experiments.reproduce_dmg_counterexample --output results/dmg_counterexample
```

Expected files: `config.json`, `reproduction.json`, and `run.log`.

### Task 7: Gate A-D Execution and Reproducibility Check

**Files:**
- Modify only if a test exposes a defect: files created in Tasks 1-6.
- Create: `results/first_batch_manifest.json` through the experiment runner, not by hand.

**Interfaces:**
- Consumes: all first-batch tests and experiment entry points.
- Produces: an auditable manifest containing hashes, commands, status, durations, and artifact paths.

- [ ] **Step 1: Run the complete deterministic test suite twice**

```powershell
python -m pytest -q
python -m pytest -q
```

Expected: both runs produce the same passing test count.

- [ ] **Step 2: Execute Gate A/B properties**

```powershell
python -m experiments.run_properties --output results/properties
```

Expected: zero invariant failures and process exit code zero.

- [ ] **Step 3: Execute the full 0.02 belief sweep**

```powershell
python -m experiments.run_belief_sweep --step 0.02 --output results/belief_sweep
```

Expected: `108 * 23426 = 2,530,008` records; the runner computes and stores the exact expected count rather than relying on a handwritten total. Every pre-registered configuration appears once in `action_counts.json`.

- [ ] **Step 4: Execute Gate D reproduction**

```powershell
python -m experiments.reproduce_dmg_counterexample --output results/dmg_counterexample
```

Expected: raw DMG true, constrained DMG violation true, and audited values within tolerance.

- [ ] **Step 5: Verify artifact determinism**

Rerun the property and counterexample experiments into `results/repro_check/`, compare semantic JSON content while ignoring timestamps, and require exact equality for all numerical fields.

- [ ] **Step 6: Record first-batch verdict**

Write `results/first_batch_manifest.json` programmatically with one of `PASS` or `FAIL`. `PASS` requires Gates A-D, complete artifact presence, zero unexplained exceptions, and no post-registration parameter mutation. Do not start CBBA implementation on `FAIL`.

---

## Plan Self-Review

- Spec coverage: Tasks 1-7 cover all first-batch files, properties, pre-registered parameters, artifacts, parallelism, and Gates A-D. Later CBBA/MILP/Monte Carlo work remains intentionally outside this plan.
- Placeholder scan: no TBD/TODO or unspecified error handling remains.
- Type consistency: all modules consume NumPy-compatible array-like values and return validated float arrays or immutable records; action keys are fixed across rollout, ranking, CSV, and summaries.
- Numerical correction: the full sweep count is deliberately computed in code because handwritten multiplication is error-prone.
