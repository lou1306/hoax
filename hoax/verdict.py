"""Three-valued verdicts for prefix monitoring.

``PrefixType`` is shared by the HOA acceptance checker and the BNet
acceptance monitor.  It lives in this small module so the BNet side does not
have to import ``hoax.runners`` (and with it networkit and the HOA parser).

A prefix is

* ``GOOD`` when every infinite extension satisfies the condition,
* ``BAD`` when every infinite extension violates it,
* ``UGLY`` when both an accepting and a rejecting extension exist.

``None`` is not a verdict: it means the monitor has not decided yet.
"""

from enum import Enum


class PrefixType(Enum):
    GOOD = 1
    BAD = 2
    UGLY = 3

    def invert(self) -> "PrefixType":
        if self == PrefixType.GOOD:
            return PrefixType.BAD
        if self == PrefixType.BAD:
            return PrefixType.GOOD
        return self
