from bisect import bisect
import logging
import os
from collections.abc import Sequence
from itertools import chain, combinations
from typing import TYPE_CHECKING, TypeVar

import fastrand  # type: ignore
from sympy.logic.inference import satisfiable  # type: ignore


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


logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logging.root.handlers.clear()
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
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def allsat(expr):
    for m in satisfiable(expr, algorithm="pycosat", all_models=True):
        if m is False:  # UNSAT
            return
        if TYPE_CHECKING:
            assert type(m) is dict
        yield {} if True in m else m


def includes(small: dict, big: dict) -> bool:
    for k, v in small.items():
        if k not in big or big[k] != v:
            return False
    return True


T = TypeVar('T')


def pick(pop: Sequence[tuple[float, T]]) -> T:
    r = PRG_UNIFORM()
    idx = bisect(pop, r, key=lambda x: x[0])
    return pop[idx][1]
