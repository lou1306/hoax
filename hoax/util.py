import logging
import time
from collections.abc import Sequence
from itertools import chain, combinations
from typing import TYPE_CHECKING, TypeVar

import fastrand  # type: ignore
from sympy.logic.inference import satisfiable  # type: ignore

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logging.root.handlers.clear()
logger = logging.getLogger("hoax")


PRG_BOUNDED = fastrand.pcg32bounded
PRG_UNIFORM = fastrand.pcg32_uniform
PRG_DEFAULT_SEED = time.time_ns() & 0xFFFFFFFF

if fastrand.SIXTYFOUR:
    PRG_BOUNDED = fastrand.xorshift128plusbounded
    PRG_UNIFORM = fastrand.xorshift128plus_uniform


def PRG_SEED(seed: int) -> None:
    if fastrand.SIXTYFOUR:
        fastrand.xorshift128plus_seed1(seed)
        fastrand.xorshift128plus_seed2(seed)
    else:
        fastrand.pcg32_seed(seed)


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
    for cumulative_prob, value in pop:
        if r < cumulative_prob:
            return value
    return pop[-1][1]  # Fallback
