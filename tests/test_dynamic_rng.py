from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from hashlib import sha256
import math

import pytest

import uav_lifecycle.dynamic_rng as dynamic_rng
from uav_lifecycle.dynamic_rng import DrawKey, canonical_key_bytes, categorical, uniform01


def key(**changes: object) -> DrawKey:
    fields: dict[str, object] = {
        "rng_version": "sha256-u64-v1",
        "experiment_id": "dynamic-lifecycle-mainline-v2",
        "generator_version": "d1-generator-v1",
        "cell_id": "N2-M3-Rtight",
        "within_cell_seed": 7,
        "entity_namespace": "target",
        "entity_id": 3,
        "event_type": "attack",
        "occurrence_index": 0,
        "subdraw_index": 0,
    }
    fields.update(changes)
    return DrawKey(**fields)  # type: ignore[arg-type]


def test_canonical_key_is_exact_flat_ten_field_json_without_newline() -> None:
    actual = canonical_key_bytes(key())
    assert actual == (
        b'["sha256-u64-v1","dynamic-lifecycle-mainline-v2","d1-generator-v1",'
        b'"N2-M3-Rtight",7,"target",3,"attack",0,0]'
    )
    assert actual.isascii()
    assert not actual.endswith(b"\n")
    assert sha256(actual).hexdigest() == "6b9f898389ea253d50bb1ef8904c79a4db17dde773fde953286c914293779f0c"


def test_canonical_key_uses_json_ascii_escaping() -> None:
    actual = canonical_key_bytes(key(cell_id="quote\"\\tab\t"))
    assert b'"quote\\\"\\\\tab\\t"' in actual
    assert actual.isascii()


def test_namespaces_and_ordinals_separate_semantic_draws() -> None:
    target = key(entity_namespace="target")
    agent = key(entity_namespace="agent")
    assert sha256(canonical_key_bytes(agent)).hexdigest() == (
        "baad633028ecf557221dbf561661b8557b185525632a896e020ee8b446de646e"
    )
    assert uniform01(target) != uniform01(agent)
    assert uniform01(target) != uniform01(key(occurrence_index=1))
    assert uniform01(target) != uniform01(key(subdraw_index=1))


@pytest.mark.parametrize("namespace", ["", "method", "Target", "uav"])
def test_key_rejects_unknown_entity_namespace(namespace: str) -> None:
    with pytest.raises(ValueError):
        key(entity_namespace=namespace)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cell_id", "N2-μ"),
        ("cell_id", 17),
        ("event_type", None),
        ("within_cell_seed", -1),
        ("entity_id", -1),
        ("occurrence_index", -1),
        ("subdraw_index", -1),
        ("within_cell_seed", True),
    ],
)
def test_key_rejects_noncanonical_fields(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        key(**{field: value})


class _Digest:
    def __init__(self, first_u64: int) -> None:
        self._digest = first_u64.to_bytes(8, "big") + bytes(24)

    def digest(self) -> bytes:
        return self._digest


@pytest.mark.parametrize("n", [0, 2**64 - 1])
def test_uniform_uses_exact_open_interval_midpoint_mapping(
    monkeypatch: pytest.MonkeyPatch, n: int
) -> None:
    monkeypatch.setattr(dynamic_rng, "sha256", lambda _: _Digest(n))
    actual = uniform01(key())
    assert actual == Fraction(2 * n + 1, 2**65)
    assert Fraction(0) < actual < Fraction(1)


def test_uniform_is_pure_and_invariant_to_evaluation_and_worker_order() -> None:
    keys = [key(occurrence_index=i // 2, subdraw_index=i % 2) for i in range(16)]
    forward = {item: uniform01(item) for item in keys}
    reverse = {item: uniform01(item) for item in reversed(keys)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = dict(zip(keys, pool.map(uniform01, keys), strict=True))
    assert forward == reverse == parallel
    assert all(uniform01(item) == value for item, value in forward.items())


def test_categorical_compares_exact_binary64_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probabilities = (0.1, 0.2, 0.7)
    first = Fraction.from_float(probabilities[0])
    total = sum(map(Fraction.from_float, probabilities), Fraction())
    monkeypatch.setattr(dynamic_rng, "uniform01", lambda _: first / total)
    assert categorical(key(), probabilities) == 1


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ((0.15, 0.35, 0.25, 0.25), 3),  # Recon joint: HA, HD, LA, LD
        ((0.9, 0.1), 1),  # BDA: A, D
        ((0.2, 0.0, 0.8, 0.0), 2),
    ],
)
def test_categorical_maximum_draw_closes_on_final_positive_category(
    monkeypatch: pytest.MonkeyPatch,
    probabilities: tuple[float, ...],
    expected: int,
) -> None:
    monkeypatch.setattr(dynamic_rng, "uniform01", lambda _: Fraction(2**65 - 1, 2**65))
    assert categorical(key(), probabilities) == expected


@pytest.mark.parametrize(
    "probabilities",
    [(), (0.0, 0.0), (-0.1, 1.1), (math.inf, 1.0), (math.nan, 1.0)],
)
def test_categorical_rejects_invalid_weights(probabilities: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        categorical(key(), probabilities)
