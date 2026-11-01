"""Three-valued acceptance monitor for the BNet random walk.

The monitor keeps the symbolic BDD engine out of the sampling hot path: it is
imported lazily, only when an acceptance condition is actually being watched.
It first tries the cheap, sound interaction-graph rules from
:mod:`hoax.bnet_acceptance`; when those are inconclusive it falls back to the
exact BDD analysis in :mod:`hoax.bnet_symbolic`.

``check`` always returns a :data:`~hoax.verdict.Verdict`: either a
:class:`~hoax.verdict.PrefixType` or an :class:`~hoax.verdict.Undecided` with a
reason, and, for resource limits, the specific ``Resource`` and detail message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from cachetools import LRUCache

from .bnet_acceptance import Condition
from .verdict import (
    PrefixType,
    Undecided,
    UndecidedReason,
    Verdict,
)

if TYPE_CHECKING:
    from .bnet_runner import BnetRunner


class BnetMonitor:
    """Monitor one ``Condition`` along a :class:`BnetRunner` walk."""

    def __init__(self, condition: Condition, runner: "BnetRunner",
                 cache_size: int = 100_000) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be at least 1")
        self.condition = condition
        self.runner = runner
        self._state_verdict_cache: LRUCache[int, Verdict] = (
            LRUCache(maxsize=cache_size))

    def _is_terminal(self, state: int) -> bool:
        """Whether no positive-probability transition leaves ``state``."""
        return not self.runner.enabled_changes(state)

    def is_terminal(self, state: int) -> bool:
        """Public accessor for :meth:`_is_terminal`."""
        return self._is_terminal(state)

    def terminal(self, state: int) -> PrefixType:
        """Exact verdict when the run cannot leave ``state``."""
        return self.condition.terminal(state)

    def check(self, state: Optional[int] = None) -> Verdict:
        """Return a decided verdict or an ``Undecided`` reason.

        A terminal state is classified exactly from the condition's terminal
        rule.  Otherwise the cheap interaction-graph rules run first; an exact
        BDD analysis is only attempted when they are inconclusive.  The result
        is cached per visited state, including non-verdicts.
        """
        if state is None:
            state = self.runner.state
        assert state is not None

        cached = self._state_verdict_cache.get(state)
        if cached is not None:
            return cached

        if self.is_terminal(state):
            result: Verdict = self.terminal(state)
        else:
            local = self.condition.check(self.runner)
            if local is not None:
                result = local
            else:
                try:
                    from .bnet_symbolic import symbolic_verdict
                except ModuleNotFoundError:
                    result = Undecided(
                        UndecidedReason.BACKEND_UNAVAILABLE,
                        detail="symbolic backend (oxidd) is not installed")
                else:
                    result = symbolic_verdict(
                        self.condition, self.runner, state=state)

        self._state_verdict_cache[state] = result
        return result
