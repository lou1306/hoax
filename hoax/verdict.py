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

``Undecided`` is the non-verdict side of the result type.  It records *why* an
exact symbolic check stopped without a decision:

* ``INCONCLUSIVE`` -- structurally undecidable from the prefix;
* ``INDETERMINATE`` -- both accepting and rejecting extensions still exist, but
  some reachable bottom SCC is fully good or fully bad, so a continuation could
  resolve it;
* ``RESOURCE_LIMIT`` -- the analysis hit a resource budget (time, BDD nodes,
  BDD memory, SCCs, or iterations);
* ``BACKEND_UNAVAILABLE`` -- the symbolic backend is not installed.

The concrete resource is carried in ``Undecided.resource`` and the human-readable
reason in ``Undecided.detail``.  There is deliberately no single
``Undecided.RESOURCE_LIMIT`` constant: construct the object with the resource
that was exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum


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


class UndecidedReason(StrEnum):
    INCONCLUSIVE = "inconclusive"
    INDETERMINATE = "indeterminate"
    RESOURCE_LIMIT = "resource-limit"
    BACKEND_UNAVAILABLE = "backend-unavailable"


class Resource(StrEnum):
    TIME = "time"
    BDD_NODES = "bdd-nodes"
    BDD_MEMORY = "bdd-memory"
    SCCS = "sccs"
    ITERATIONS = "iterations"


@dataclass(frozen=True, slots=True)
class Undecided:
    """A non-verdict outcome, with an optional resource and detail."""

    reason: UndecidedReason
    resource: Resource | None = None
    detail: str | None = None

    def __str__(self) -> str:
        if self.resource is not None:
            return f"{self.reason.value} ({self.resource.value}): {self.detail}"
        return f"{self.reason.value}: {self.detail}"


Verdict = PrefixType | Undecided
