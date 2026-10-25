"""Three-valued acceptance monitor for the BNet random walk.

The monitor keeps the symbolic BDD engine out of the sampling hot path: it is
imported lazily, only when an acceptance condition is actually being watched.
It first tries the cheap, sound interaction-graph rules from
:mod:`hoax.bnet_acceptance`; when those are inconclusive it falls back to the
exact BDD analysis in :mod:`hoax.bnet_symbolic`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .bnet_acceptance import Condition
from .verdict import PrefixType

if TYPE_CHECKING:
    from .bnet_runner import BnetRunner


class BnetMonitor:
    """Monitor one ``Condition`` along a :class:`BnetRunner` walk."""

    def __init__(self, condition: Condition, runner: "BnetRunner") -> None:
        self.condition = condition
        self.runner = runner
        self._exact_cache: dict[int, Optional[PrefixType]] = {}

    def _is_terminal(self, state: int) -> bool:
        """Whether no positive-probability transition leaves ``state``."""
        return not self.runner.enabled_changes(state)

    def is_terminal(self, state: int) -> bool:
        """Public accessor for :meth:`_is_terminal`."""
        return self._is_terminal(state)

    def terminal(self, state: int) -> PrefixType:
        """Exact verdict when the run cannot leave ``state``."""
        return self.condition.terminal(state)

    def check(self, state: Optional[int] = None) -> Optional[PrefixType]:
        """Return a verdict for the current prefix, or ``None``.

        A terminal state is classified exactly from the condition's
        terminal rule.  Otherwise the cheap interaction-graph rules run first;
        an exact BDD analysis is only attempted when they are inconclusive.
        """
        if state is None:
            state = self.runner.state
        assert state is not None

        cached = self._exact_cache.get(state)
        if cached is not None:
            return cached

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
        verdict = symbolic_verdict(self.condition, self.runner, state=state)
        self._exact_cache[state] = verdict
        return verdict
