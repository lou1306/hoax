"""On-the-fly random-walk runner for Boolean networks, without an HOA.

The runner reuses hoax's drivers, probability primitives and hook machinery,
but samples the next state straight from the network's update functions
(:class:`hoax.bnet.BNet`) instead of enumerating the successor distribution.
It caches only the set of variables that may fire in each visited state, not
the successor tree itself.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .bnet import BNet, parse_bnet_file
from .drivers import Driver, RandomDriver
from .runners import Action, Hook, Runner
from .util import PRG_SEED, PRG_UNIFORM


class BnetRunner(Runner):
    """A random-walk runner whose transition relation is built on the fly."""

    def __init__(self, net: BNet, drv: Driver) -> None:
        self.net = net
        self.driver = drv
        self.aps: list[str] = list(net.names)
        self.state: Optional[int] = None
        self._count = 0
        self._changes: dict[int, list[int]] = {}
        self.deadlock_actions: list[Action] = []
        self.nondet_actions: list[Action] = []
        self.transition_hooks: list[Hook] = []
        self.candidates: list[tuple[int, str]] = []
        self.pr = self.get_weights()

    def get_weights(self) -> dict[str, float]:
        """Read the firing probability of every variable from the driver."""
        pr = {}
        for ap in self.aps:
            drv = self.driver.find_driver(ap) if self.driver is not None else None
            pr[ap] = drv.cum_weights[0] if isinstance(drv, RandomDriver) else 0.5
        return pr

    @property
    def count(self) -> int:
        return self._count

    @count.setter
    def count(self, value: int) -> None:
        self._count = value

    def init(self, state: Optional[int] = None) -> None:
        if state is None:
            state = 0
            for i in range(self.net.n):
                if PRG_UNIFORM() < 0.5:
                    state |= 1 << i
        self.state = state
        self.count = 0

    def add_transition_hook(self, hook: Hook) -> None:
        self.transition_hooks.append(hook)

    def add_nondet_action(self, action: Action) -> None:
        self.nondet_actions.append(action)

    def add_deadlock_action(self, action: Action) -> None:
        self.deadlock_actions.append(action)

    def potential_changes(self, state: int) -> list[int]:
        """Indices of the variables that may fire in ``state``, cached."""
        changes = self._changes.get(state)
        if changes is None:
            changes = self.net.potential_changes(state)
            self._changes[state] = changes
        return changes

    def is_steady(self, state: int) -> bool:
        """Whether ``state`` has no successor other than itself."""
        return not self.potential_changes(state)

    def step(self, inputs: Optional[set] = None) -> Iterable[tuple]:
        assert self.state is not None
        changes = self.potential_changes(self.state)
        # Steady states consume no draw, matching the reference pick().
        r = PRG_UNIFORM() if changes else 0.0
        model, m_str, next_state = self.net.sample(self.state, self.pr, r, changes)
        old_state, self.state = self.state, next_state
        self.count += 1
        for hook in self.transition_hooks:
            hook.run(self)  # type: ignore[arg-type]
        return ((old_state, model, m_str, self.state),)


def format_state(net: BNet, state: int, order: Optional[Sequence[str]] = None) -> str:
    """Render a state as a bit string, by default in sorted-variable order."""
    names = sorted(net.names) if order is None else order
    return "".join("1" if (state >> net.index[name]) & 1 else "0" for name in names)


def main(argv: Optional[list[str]] = None) -> int:
    """Walk a ``.bnet`` network, printing one state per line (harness format)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("Usage: python -m hoax.bnet_runner <network.bnet> <bound>")
        return 1
    fname, bound = argv[0], int(argv[1])
    seed = os.environ.get("HOAX_SEED")
    if seed is not None:
        PRG_SEED(int(seed))

    tic = datetime.now()
    net = parse_bnet_file(fname)
    runner = BnetRunner(net, RandomDriver(list(net.names)))
    runner.init()
    assert runner.state is not None

    x = 0
    for x in range(1, bound + 1):
        sys.stdout.write(format_state(net, runner.state) + "\n")
        if runner.is_steady(runner.state):
            print("Reached a steady state.", file=sys.stderr)
            break
        if not runner.step():
            print("Reached a deadlock state.", file=sys.stderr)
            break
    else:
        print("Bound reached.", file=sys.stderr)
    toc = datetime.now()
    print(f"hoax,{Path(fname).stem},{x},{toc - tic}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
