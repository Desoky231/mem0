from __future__ import annotations

import math
import random
from collections.abc import Iterable


def wilson_interval(
    successes: int, total: int, *, z: float = 1.959963984540054
) -> list[float] | None:
    """Two-sided Wilson score interval for a Bernoulli proportion."""
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def bootstrap_mean_interval(
    values: Iterable[float],
    *,
    seed: int = 42,
    samples: int = 5000,
) -> list[float] | None:
    """Question-level percentile bootstrap CI for an arithmetic mean."""
    observed = list(values)
    if not observed:
        return None
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(observed) for _ in observed) / len(observed)
        for _ in range(samples)
    )
    return [
        round(means[int(0.025 * (samples - 1))], 4),
        round(means[int(0.975 * (samples - 1))], 4),
    ]


def paired_difference_interval(
    first: Iterable[float],
    second: Iterable[float],
    *,
    seed: int = 42,
    samples: int = 5000,
) -> list[float] | None:
    differences = [a - b for a, b in zip(first, second, strict=True)]
    return bootstrap_mean_interval(differences, seed=seed, samples=samples)
