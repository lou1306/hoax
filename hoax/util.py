import logging
import os
from collections.abc import Sequence
from itertools import chain, combinations
from typing import TYPE_CHECKING, TypeVar

import fastrand  # type: ignore
from sympy.logic.inference import satisfiable  # type: ignore


Model = tuple[tuple[str, bool], ...]


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
    if len(pop) == 1:
        return pop[0][1]
    r = PRG_UNIFORM()
    # TODO TODO TODO
    # low, m, hi = 0, len(pop)// 2, len(pop)
    # while low < hi:
    #     if r < pop[m][0]:
    #         hi = m
    #     else:
    #         low = m + 1
    #     m = (low + hi) // 2
    for cumulative_prob, value in pop:
        if r < cumulative_prob:
            return value
    return pop[-1][1]  # Fallback
