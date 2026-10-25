import logging
import os
from collections.abc import Sequence
from itertools import chain, combinations
from typing import TypeVar

import fastrand  # type: ignore

logger = logging.getLogger("hoax")

PRG_DEFAULT_SEED = int.from_bytes(os.urandom(4))

PRG_BOUNDED = fastrand.pcg32bounded
PRG_UNIFORM = fastrand.pcg32_uniform
if fastrand.SIXTYFOUR:
    PRG_BOUNDED = fastrand.xorshift128plusbounded
    PRG_UNIFORM = fastrand.xorshift128plus_uniform


def PRG_SEED(seed: int) -> None:
    if fastrand.SIXTYFOUR:
        fastrand.xorshift128plus_seed1(seed)
        fastrand.xorshift128plus_seed2(seed)
    else:
        fastrand.pcg32_seed(seed)
    # Warm up the generator
    for _ in range(20):
        PRG_UNIFORM()


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


Model = tuple[tuple[str, bool], ...]


def dict2tuple(d: dict) -> Model:
    return tuple(sorted(d.items(), key=lambda x: (str(x[0]), x[1])))


def tuple2dict(t: Model) -> dict:
    return {sym: val for sym, val in t}


def prob(model: Model, pr: dict[str, float]) -> float:
    """Compute the probability of a model."""
    p = 1.0
    for ap, val in model:
        p *= pr[str(ap)] if val else (1 - pr[str(ap)])
    assert 0 <= p <= 1, f"Invalid probability {p} for model {model}"
    return p


T = TypeVar("T")


def pick(pop: Sequence[tuple[float, T]]) -> T:
    if len(pop) == 1:
        return pop[0][1]
    r = PRG_UNIFORM()
    for cumulative_prob, value in pop:
        if r < cumulative_prob:
            return value
    return pop[-1][1]  # Fallback
