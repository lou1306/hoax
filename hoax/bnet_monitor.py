"""Three-valued acceptance monitor for the BNet random walk.

The monitor keeps the symbolic BDD engine out of the sampling hot path: it is
imported lazily, only when an acceptance condition is actually being watched.
It first tries the cheap, sound interaction-graph rules from
:mod:`hoax.bnet_acceptance`; when those are inconclusive it falls back to the
exact BDD analysis in :mod:`hoax.bnet_symbolic`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from cachetools import LRUCache

from .bnet_acceptance import Condition
from .verdict import PrefixType, Undecided

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
        self._state_verdict_cache: LRUCache[int, PrefixType | Undecided] = (
            LRUCache(maxsize=cache_size))
        self._timeout_occurred = False

    def _is_terminal(self, state: int) -> bool:
        """Whether no positive-probability transition leaves ``state``."""
        return not self.runner.enabled_changes(state)

    def is_terminal(self, state: int) -> bool:
        """Public accessor for :meth:`_is_terminal`."""
        return self._is_terminal(state)

    def terminal(self, state: int) -> PrefixType:
        """Exact verdict when the run cannot leave ``state``."""
        return self.condition.terminal(state)

    @property
    def timeout_occurred(self) -> bool:
        """Whether any exact check reported ``Undecided.TIMEOUT`` in this run."""
        return self._timeout_occurred

    def check(self, state: Optional[int] = None) -> Optional[PrefixType]:
        """Return a verdict for the current prefix, or ``None``.

        A terminal state is classified exactly from the condition's
        terminal rule.  Otherwise the cheap interaction-graph rules run first;
        an exact BDD analysis is only attempted when they are inconclusive.
        """
        if state is None:
            state = self.runner.state
        assert state is not None

        cached = self._state_verdict_cache.get(state)
        if cached is not None:
            # cached is a PrefixType or an Undecided marker (never None).
            return None if isinstance(cached, Undecided) else cached

        if self._is_terminal(state):
            return self.terminal(state)

        local = self.condition.check(self.runner)
        if local is not None:
            return local

        try:
            from .bnet_symbolic import symbolic_verdict
        except ModuleNotFoundError:
            # oxidd is not available: stay conservative and report no verdict.
            return None
        result = symbolic_verdict(self.condition, self.runner, state=state)
        if isinstance(result, Undecided):
            if result is Undecided.TIMEOUT:
                self._timeout_occurred = True
            self._state_verdict_cache[state] = result
            return None
        self._state_verdict_cache[state] = result
        return result
