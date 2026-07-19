"""Counter-based common-random-number primitives for dynamic experiments."""

from dataclasses import dataclass, fields
from fractions import Fraction
from hashlib import sha256
import json
import math


_ENTITY_NAMESPACES = frozenset({"scenario", "target", "agent"})
_STRING_FIELDS = (
    "rng_version",
    "experiment_id",
    "generator_version",
    "cell_id",
    "entity_namespace",
    "event_type",
)
_INTEGER_FIELDS = (
    "within_cell_seed",
    "entity_id",
    "occurrence_index",
    "subdraw_index",
)


@dataclass(frozen=True, slots=True)
class DrawKey:
    """The complete semantic address of one deterministic random draw."""

    rng_version: str
    experiment_id: str
    generator_version: str
    cell_id: str
    within_cell_seed: int
    entity_namespace: str
    entity_id: int
    event_type: str
    occurrence_index: int
    subdraw_index: int

    def __post_init__(self) -> None:
        for name in _STRING_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            try:
                value.encode("ascii")
            except UnicodeEncodeError as error:
                raise ValueError(f"{name} must contain only ASCII characters") from error
        for name in _INTEGER_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.entity_namespace not in _ENTITY_NAMESPACES:
            raise ValueError("entity_namespace must be scenario, target, or agent")


def canonical_key_bytes(key: DrawKey) -> bytes:
    """Serialize a draw key to the frozen flat canonical JSON representation."""

    values = [getattr(key, field.name) for field in fields(key)]
    return json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def uniform01(key: DrawKey) -> Fraction:
    """Return the exact open-interval uniform associated with ``key``."""

    digest = sha256(canonical_key_bytes(key)).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return Fraction(2 * value + 1, 2**65)


def categorical(key: DrawKey, probabilities: tuple[float, ...]) -> int:
    """Draw an index using the exact represented values of binary64 weights."""

    if not probabilities:
        raise ValueError("probabilities must not be empty")
    weights: list[Fraction] = []
    for probability in probabilities:
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError("probabilities must be finite and nonnegative")
        weights.append(Fraction.from_float(probability))
    total = sum(weights, Fraction())
    if total <= 0:
        raise ValueError("at least one probability must be positive")

    threshold = uniform01(key) * total
    cumulative = Fraction()
    final_positive = 0
    for index, weight in enumerate(weights):
        if weight > 0:
            final_positive = index
        cumulative += weight
        if threshold < cumulative:
            return index
    return final_positive
