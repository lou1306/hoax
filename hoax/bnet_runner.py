"""On-the-fly random-walk runner for Boolean networks, without an HOA.

The runner reuses hoax's drivers, probability primitives and hook machinery,
but samples the next state straight from the network's update functions
(:class:`hoax.bnet.BNet`) instead of enumerating the successor distribution.
It caches the update target and the set of variables that may fire in each
visited state, not the successor tree itself.
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Sequence

from .bnet import BNet, parse_bnet_file, vars_of
from .drivers import Driver, RandomDriver
from .util import PRG_SEED, PRG_UNIFORM, logger

if TYPE_CHECKING:
    # Import for type checking only: hoax.runners pulls in networkit and the
    # HOA machinery, which BnetRunner does not use.
    from .runners import Action, Hook


class SamplingMode(str, Enum):
    """How ``BnetRunner`` chooses the next transition."""

    STUTTER_FREE = "stutter-free"
    STUTTERING = "stuttering"
    ASYNC_UNIFORM = "async-uniform"
    ASYNC_WEIGHTED = "async-weighted"


def parse_weight_specs(specs: Iterable[str],
                       names: Optional[Iterable[str]] = None) -> dict[str, float]:
    """Parse repeated ``KEY=VALUE`` weight overrides from the command line."""
    weights: dict[str, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"expected KEY=VALUE, got {spec!r}")
        key, raw = spec.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("weight variable name must not be empty")
        try:
            weight = float(raw)
        except ValueError as e:
            raise ValueError(f"invalid weight for {key}: {raw!r}") from e
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError(
                f"weight for {key} must be a finite number in [0, 1]")
        weights[key] = weight

    if names is not None:
        unknown = sorted(set(weights) - set(names))
        if unknown:
            raise ValueError(
                f"unknown variable(s) in --weight: {', '.join(unknown)}")
    return weights


class BnetRunner:
    """A random-walk runner whose transition relation is built on the fly.

    ``mode`` selects the transition law.  ``STUTTER_FREE`` is the default
    PrGA walk, ``STUTTERING`` includes no-change leaves, and the two async
    modes select exactly one unstable variable per step (uniformly, or with
    probability proportional to the effective firing weights).

    Firing weights come from the flip driver.  ``weights`` overrides them for
    the named variables only; unspecified variables keep the driver value.
    """

    def __init__(self, net: BNet, drv: Driver,
                 mode: SamplingMode | str = SamplingMode.STUTTER_FREE,
                 weights: Optional[Mapping[str, float]] = None) -> None:
        self.net = net
        self.driver = drv
        self.mode = SamplingMode(mode)
        self.aps: list[str] = list(net.names)
        self.weights: dict[str, float] = dict(weights or {})
        self._validate_weights()
        self.state: Optional[int] = None
        self._count = 0
        self._cache: dict[int, tuple[int, list[int]]] = {}
        self._dependents = self._build_dependents()
        self._values: list[bool] = []
        self._values_state: Optional[int] = None
        self._last_state: Optional[int] = None
        self._last_target = 0
        self.deadlock_actions: list[Action] = []
        self.nondet_actions: list[Action] = []
        self.transition_hooks: list[Hook] = []
        self.candidates: list[tuple[int, str]] = []
        self.pr = self.get_weights()
        self.pr.update(self.weights)

        if self.mode is SamplingMode.ASYNC_UNIFORM and self.weights:
            raise ValueError("uniform async does not use weight overrides")
        if self.mode is SamplingMode.ASYNC_WEIGHTED:
            if all(math.isclose(w, 0.5, rel_tol=0.0, abs_tol=1e-12)
                   for w in self.pr.values()):
                logger.warning(
                    "async-weighted mode has no non-uniform weights; "
                    "falling back to uniform async")
                self.mode = SamplingMode.ASYNC_UNIFORM
            elif not any(w > 0.0 for w in self.pr.values()):
                raise ValueError(
                    "async-weighted mode needs at least one positive weight")

    def _validate_weights(self) -> None:
        unknown = sorted(set(self.weights) - set(self.net.names))
        if unknown:
            raise ValueError(
                f"unknown variable(s) in weight overrides: {', '.join(unknown)}")
        for name, weight in self.weights.items():
            if (not isinstance(weight, (int, float))
                    or not math.isfinite(weight)
                    or not 0.0 <= weight <= 1.0):
                raise ValueError(
                    f"weight for {name} must be a finite number in [0, 1]")

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

    def _build_dependents(self) -> list[list[int]]:
        """Out-neighbours of each variable in the network's interaction graph."""
        dependents = [[] for _ in range(self.net.n)]
        for j, expr in enumerate(self.net.update):
            for name in vars_of(expr):
                dependents[self.net.index[name]].append(j)
        return dependents

    @staticmethod
    def _changes_from_target(target: int, state: int) -> list[int]:
        """Variables whose target differs from their current value."""
        diff = target ^ state
        changes: list[int] = []
        while diff:
            bit = diff & -diff
            changes.append(bit.bit_length() - 1)
            diff ^= bit
        return changes

    def _values_for(self, state: int) -> list[bool]:
        """Return the variable values of ``state``, rebuilding them if needed."""
        if self._values_state != state:
            self._values = [
                bool((state >> i) & 1) for i in range(self.net.n)]
            self._values_state = state
        return self._values

    def _target_of(self, values: list[bool]) -> int:
        """Evaluate every update function; used only to seed the cache."""
        target = 0
        for j, f in enumerate(self.net._compiled_update):
            if f(values):
                target |= 1 << j
        return target

    def _seed_cache(self, state: int) -> tuple[int, list[int]]:
        """Compute and cache the target and changes for ``state`` from scratch."""
        target = self._target_of(self._values_for(state))
        changes = self._changes_from_target(target, state)
        self._cache[state] = (target, changes)
        self._last_state = state
        self._last_target = target
        return target, changes

    def init(self, state: Optional[int] = None) -> None:
        if state is None:
            state = 0
            for i in range(self.net.n):
                if PRG_UNIFORM() < 0.5:
                    state |= 1 << i
        self.state = state
        self.count = 0
        self._values = []
        self._values_state = None
        hit = self._cache.get(state)
        if hit is None:
            self._seed_cache(state)
        else:
            self._last_state = state
            self._last_target = hit[0]

    def add_transition_hook(self, hook: Hook) -> None:
        self.transition_hooks.append(hook)

    def add_nondet_action(self, action: Action) -> None:
        self.nondet_actions.append(action)

    def add_deadlock_action(self, action: Action) -> None:
        self.deadlock_actions.append(action)

    def potential_changes(self, state: int) -> list[int]:
        """Indices of the variables that may fire in ``state``, cached.

        Cache hits return the stored change list directly.  On a miss, the
        target mask is updated only for variables whose update function reads
        one of the bits that differ from the last cached state.
        """
        hit = self._cache.get(state)
        if hit is not None:
            target, changes = hit
            self._last_state = state
            self._last_target = target
            return changes

        values = self._values_for(state)
        last_state = self._last_state
        if last_state is None:
            target = self._target_of(values)
        else:
            target = self._last_target
            flipped = last_state ^ state
            if flipped:
                if (flipped & (flipped - 1)) == 0:
                    i = flipped.bit_length() - 1
                    for j in self._dependents[i]:
                        if self.net._compiled_update[j](values):
                            target |= 1 << j
                        else:
                            target &= ~(1 << j)
                else:
                    affected: set[int] = set()
                    remaining = flipped
                    while remaining:
                        bit = remaining & -remaining
                        i = bit.bit_length() - 1
                        affected.update(self._dependents[i])
                        remaining ^= bit
                    for j in affected:
                        if self.net._compiled_update[j](values):
                            target |= 1 << j
                        else:
                            target &= ~(1 << j)

        changes = self._changes_from_target(target, state)
        self._cache[state] = (target, changes)
        self._last_state = state
        self._last_target = target
        return changes

    def is_steady(self, state: int) -> bool:
        """Whether ``state`` has no successor other than itself."""
        return not self.potential_changes(state)

    def step(self, inputs: Optional[set] = None) -> Iterable[tuple]:
        assert self.state is not None
        changes = self.potential_changes(self.state)

        if self.mode is SamplingMode.ASYNC_UNIFORM:
            r = 0.0 if len(changes) <= 1 else PRG_UNIFORM()
            model, m_str, next_state = self.net.sample_async(
                self.state, r, changes)
        elif self.mode is SamplingMode.ASYNC_WEIGHTED:
            r = 0.0 if len(changes) <= 1 else PRG_UNIFORM()
            model, m_str, next_state = self.net.sample_async_weighted(
                self.state, self.pr, r, changes)
        else:
            # Consume a draw exactly when ``trel()`` would have more than one
            # leaf, matching the reference ``pick()``: a steady state (one
            # stutter leaf) and a stutter-free state with a single changeable
            # variable (one non-empty leaf) both involve no choice at all.
            single = (len(changes) <= 1
                      if self.mode is SamplingMode.STUTTER_FREE
                      else not changes)
            r = 0.0 if single else PRG_UNIFORM()
            model, m_str, next_state = self.net.sample(
                self.state, self.pr, r, changes,
                stutter=self.mode is SamplingMode.STUTTERING)

        old_state, self.state = self.state, next_state
        self.count += 1
        for hook in self.transition_hooks:
            hook.run(self)  # type: ignore[arg-type]
        return ((old_state, model, m_str, self.state),)


def format_state(net: BNet, state: int, order: Optional[Sequence[str]] = None) -> str:
    """Render a state as a bit string, by default in sorted-variable order."""
    names = sorted(net.names) if order is None else order
    return "".join("1" if (state >> net.index[name]) & 1 else "0" for name in names)


def main() -> None:
    """Walk a ``.bnet`` network, printing one state per line (harness format)."""
    # Imported here rather than at module level: the runner is imported by the
    # benchmark walker and typer only matters for this developer entry point.
    import typer

    def walk(
        network: Path,
        bound: int,
        mode: SamplingMode = typer.Option(
            SamplingMode.STUTTER_FREE, "--mode",
            help="Sampling mode."),
        weight: list[str] = typer.Option(
            [], "--weight", "-w", metavar="KEY=VALUE",
            help="Override a firing weight; repeatable."),
    ) -> None:
        seed = os.environ.get("HOAX_SEED")
        if seed is not None:
            PRG_SEED(int(seed))

        tic = datetime.now()
        net = parse_bnet_file(network)
        try:
            weights = parse_weight_specs(weight, net.names)
        except ValueError as e:
            raise typer.BadParameter(str(e)) from e
        runner = BnetRunner(
            net, RandomDriver(list(net.names)), mode=mode, weights=weights)
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
        print(f"hoax,{network.stem},{x},{toc - tic}", file=sys.stderr)

    typer.run(walk)


if __name__ == "__main__":
    main()
