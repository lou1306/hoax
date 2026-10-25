"""Acceptance conditions over Boolean-network states.

This module is deliberately separate from hoax's HOA acceptance machinery.
It parses the small ``Fin``/``Inf`` language in
:file:`bnet_acceptance.lark`, where a pattern is a Boolean state vector with
``x``/``X`` don't-cares and a comma-separated list of patterns is a union.

The checker in this module implements the two cheap interaction-graph rules:

* outside ``sp`` with a frozen required mismatch blocks every pattern, so
  ``Fin(sp)`` is GOOD and ``Inf(sp)`` is BAD;
* inside a matching pattern with every required bit frozen, the run stays in
  ``sp``, so ``Inf(sp)`` is GOOD and ``Fin(sp)`` is BAD.

Those rules are sound but incomplete.  A verdict of ``None`` here means the
exact symbolic analysis in :mod:`hoax.bnet_symbolic` is still needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Iterable, Sequence

from ._bnet_acceptance_parser import Lark_StandAlone, Tree
from .verdict import PrefixType

if TYPE_CHECKING:
    from .bnet_runner import BnetRunner


@lru_cache(maxsize=1)
def _parser():
    # The generated module contains the same LALR tables as the development
    # Lark parser, so runtime users do not need the lark package.
    return Lark_StandAlone()


def _pattern_from_tree(node: Tree, order: Sequence[str],
                       index: dict[str, int]) -> "Pattern":
    return Pattern.from_text(str(node.children[0]), order, index)


def _compile(node: Tree, net, order: Sequence[str]) -> "Condition":
    match node.data:
        case "fin":
            patterns = tuple(
                _pattern_from_tree(p, order, net.index)
                for p in node.children[0].children)
            return Fin(patterns)
        case "inf":
            patterns = tuple(
                _pattern_from_tree(p, order, net.index)
                for p in node.children[0].children)
            return Inf(patterns)
        case "and_":
            return And((_compile(node.children[0], net, order),
                        _compile(node.children[1], net, order)))
        case "or_":
            return Or((_compile(node.children[0], net, order),
                       _compile(node.children[1], net, order)))
        case _:
            raise ValueError(f"unexpected acceptance node {node.data!r}")


def parse_condition(text: str, net, order: Sequence[str] | None = None
                    ) -> "Condition":
    """Parse an acceptance condition and bind it to ``net``.

    Pattern characters are read left to right in ``order``.  By default that
    is the sorted variable-name order used by :func:`format_state`, so a
    condition matches the state strings printed by the runner.  Pass
    ``order=net.names`` to use ``.bnet`` target order instead.
    """
    order = tuple(sorted(net.names)) if order is None else tuple(order)
    if len(order) != net.n or len(set(order)) != len(order):
        raise ValueError(
            f"pattern order must be a permutation of the {net.n} variables")
    if set(order) != set(net.names):
        raise ValueError(
            f"pattern order contains unknown variables: "
            f"{sorted(set(order) - set(net.names))}")
    tree = _parser().parse(text)
    return _compile(tree.children[0], net, order)


@dataclass(frozen=True)
class Pattern:
    """A fixed-width Boolean pattern, with ``required`` and ``values`` masks."""

    required: int
    values: int
    text: str

    @classmethod
    def from_text(cls, text: str, order: Sequence[str],
                  index: dict[str, int]) -> "Pattern":
        if len(text) != len(order):
            raise ValueError(
                f"pattern {text!r} has width {len(text)}, expected {len(order)}")
        required = 0
        values = 0
        for char, name in zip(text, order):
            bit = 1 << index[name]
            if char in "xX":
                continue
            if char not in "01":
                raise ValueError(f"invalid pattern character {char!r}")
            required |= bit
            if char == "1":
                values |= bit
        return cls(required, values, text)

    def matches(self, state: int) -> bool:
        return (state & self.required) == self.values

    def mismatch(self, state: int) -> int:
        """Required bits on which ``state`` disagrees with the pattern."""
        return (state ^ self.values) & self.required


class Condition(ABC):
    """A BNet acceptance condition over a sampled run."""

    @abstractmethod
    def check(self, runner: "BnetRunner") -> PrefixType | None:
        """Return a verdict for the current prefix, or ``None``."""

    @abstractmethod
    def terminal(self, state: int) -> PrefixType:
        """Exact verdict when a stutter-free run has reached a steady state."""

    @abstractmethod
    def atoms(self) -> Iterable["Condition"]:
        """Yield the ``Fin``/``Inf`` leaves of this condition."""

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class _PatternAtom(Condition):
    patterns: tuple[Pattern, ...]

    def matches(self, state: int) -> bool:
        return any(p.matches(state) for p in self.patterns)

    def atoms(self) -> Iterable["Condition"]:
        yield self

    def __str__(self) -> str:
        return f"{type(self).__name__}(" + ", ".join(
            p.text for p in self.patterns) + ")"


@dataclass(frozen=True)
class Fin(_PatternAtom):
    patterns: tuple[Pattern, ...]

    def check(self, runner: "BnetRunner") -> PrefixType | None:
        state = runner.state
        assert state is not None
        changeable = runner.changeable_mask(state)

        if all(p.mismatch(state) & ~changeable for p in self.patterns):
            return PrefixType.GOOD
        if any(p.matches(state) and not (p.required & changeable)
               for p in self.patterns):
            return PrefixType.BAD
        return None

    def terminal(self, state: int) -> PrefixType:
        return PrefixType.BAD if self.matches(state) else PrefixType.GOOD


@dataclass(frozen=True)
class Inf(_PatternAtom):
    patterns: tuple[Pattern, ...]

    def check(self, runner: "BnetRunner") -> PrefixType | None:
        state = runner.state
        assert state is not None
        changeable = runner.changeable_mask(state)

        if any(p.matches(state) and not (p.required & changeable)
               for p in self.patterns):
            return PrefixType.GOOD
        if all(p.mismatch(state) & ~changeable for p in self.patterns):
            return PrefixType.BAD
        return None

    def terminal(self, state: int) -> PrefixType:
        return PrefixType.GOOD if self.matches(state) else PrefixType.BAD


def _and(check: list[PrefixType | None]) -> PrefixType | None:
    if PrefixType.BAD in check:
        return PrefixType.BAD
    if all(value is PrefixType.GOOD for value in check):
        return PrefixType.GOOD
    return None


def _or(check: list[PrefixType | None]) -> PrefixType | None:
    if PrefixType.GOOD in check:
        return PrefixType.GOOD
    if all(value is PrefixType.BAD for value in check):
        return PrefixType.BAD
    return None


class _Binary(Condition):
    """Base for the ``And``/``Or`` combinators.

    Both share the tuple of sub-conditions, the flattened ``atoms()`` and the
    parenthesised ``__str__``; subclasses supply only the combining rules and
    the operator separator.
    """

    conditions: tuple[Condition, ...]
    operator: str

    def atoms(self) -> Iterable[Condition]:
        for condition in self.conditions:
            yield from condition.atoms()

    def __str__(self) -> str:
        return "(" + f" {self.operator} ".join(str(c) for c in self.conditions) + ")"


@dataclass(frozen=True)
class And(_Binary):
    conditions: tuple[Condition, ...]
    operator = "&"

    def check(self, runner: "BnetRunner") -> PrefixType | None:
        return _and([c.check(runner) for c in self.conditions])

    def terminal(self, state: int) -> PrefixType:
        values = [c.terminal(state) for c in self.conditions]
        return PrefixType.BAD if PrefixType.BAD in values else PrefixType.GOOD


@dataclass(frozen=True)
class Or(_Binary):
    conditions: tuple[Condition, ...]
    operator = "|"

    def check(self, runner: "BnetRunner") -> PrefixType | None:
        return _or([c.check(runner) for c in self.conditions])

    def terminal(self, state: int) -> PrefixType:
        values = [c.terminal(state) for c in self.conditions]
        return PrefixType.GOOD if PrefixType.GOOD in values else PrefixType.BAD
