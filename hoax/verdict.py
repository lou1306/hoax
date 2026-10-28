"""Three-valued verdicts for prefix monitoring.

``PrefixType`` is shared by the HOA acceptance checker and the BNet
acceptance monitor.  It lives in this small module so the BNet side does not
have to import ``hoax.runners`` (and with it networkit and the HOA parser).

A prefix is

* ``GOOD`` when every infinite extension satisfies the condition,
* ``BAD`` when every infinite extension violates it,
* ``UGLY`` when both an accepting and a rejecting extension exist *and every
  reachable bottom SCC is itself mixed* (the walk must eventually settle in a
  closed mixed class, so the outcome can never be resolved).

``None`` is not a verdict: it means the monitor has not decided yet.

``Undecided`` is likewise not a verdict: it reports *why* an exact symbolic
check stopped without a decision.  ``Undecided.INCONCLUSIVE`` means the
condition is structurally undecidable from the prefix; ``Undecided.TIMEOUT``
means the analysis hit its budget; ``Undecided.INDETERMINATE`` means both an
accepting and a rejecting extension still exist but some reachable bottom SCC
is fully good or fully bad, so a continuation could resolve it to GOOD or BAD.
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


class Undecided(Enum):
    """Why an exact symbolic check ended without a three-valued verdict.

    ``TIMEOUT`` and ``INCONCLUSIVE`` are treated identically by the monitor
    (both are cached and never recomputed), but ``TIMEOUT`` is tracked so a
    definitive-but-inconclusive trace can warn that it might be GOOD or BAD.
    ``INDETERMINATE`` is also cached and never recomputed, and is distinct from
    a timeout: it is a genuine (transient) non-decision, not a budget failure.
    """
    INCONCLUSIVE = 0  # structurally undecidable from the prefix
    TIMEOUT = 1       # budget exhausted (transient in principle)
    INDETERMINATE = 2  # both extensions exist, but some bottom SCC is clean
