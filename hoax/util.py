import time
import fastrand  # type: ignore
from itertools import chain, combinations


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
