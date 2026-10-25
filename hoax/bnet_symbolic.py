"""Exact symbolic acceptance checking for Boolean networks.

The checker builds a BDD for the projected transition relation of the
variables that can still change from the current state.  It never enumerates
the reachable state graph: reachability, pre/post images and SCC extraction are
BDD fixpoints.  The result is a three-valued prefix verdict:

* ``GOOD``: every infinite extension satisfies the condition;
* ``BAD``: every infinite extension violates it;
* ``UGLY``: both kinds of extension exist.

``Undecided`` marks a non-verdict outcome.  ``Undecided.INCONCLUSIVE`` is a
structural non-decision; ``Undecided.TIMEOUT`` means the analysis did not
finish within its budget.  Neither is an ``UNKNOWN`` verdict.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Iterable, Optional, Sequence

from oxidd.bdd import BDDManager, BDDFunction
from oxidd.util import DDMemoryError

from .bnet import BNet, Expr, And as ExprAnd, Const, Not, Or as ExprOr, Var
from .bnet_acceptance import (And, Condition, Fin, Inf, Or, Pattern)
from .bnet_runner import SamplingMode
from .verdict import PrefixType, Undecided

DEFAULT_MAX_NODES = 1_000_000
DEFAULT_MAX_SECONDS = 10.0
DEFAULT_MAX_SCCS = 100_000
DEFAULT_MAX_ITERATIONS = 1_000_000


class SymbolicBudgetExceeded(RuntimeError):
    """Raised internally when a BDD analysis exceeds its configured budget."""


def _dnf(condition: Condition) -> list[tuple[Fin | Inf, ...]]:
    """Return a DNF of ``condition`` as lists of positive Fin/Inf atoms."""
    if isinstance(condition, (Fin, Inf)):
        return [(condition,)]
    if isinstance(condition, Or):
        result: list[tuple[Fin | Inf, ...]] = []
        for child in condition.conditions:
            result.extend(_dnf(child))
        return result
    if isinstance(condition, And):
        result = [()]
        for child in condition.conditions:
            child_terms = _dnf(child)
            result = [left + right for left in result for right in child_terms]
        return result
    raise TypeError(f"unexpected acceptance condition {condition!r}")


def _dual(condition: Condition) -> Condition:
    """Return the De Morgan dual, swapping ``Fin``/``Inf`` and ``And``/``Or``."""
    if isinstance(condition, Fin):
        return Inf(condition.patterns)
    if isinstance(condition, Inf):
        return Fin(condition.patterns)
    if isinstance(condition, And):
        return Or(tuple(_dual(child) for child in condition.conditions))
    if isinstance(condition, Or):
        return And(tuple(_dual(child) for child in condition.conditions))
    raise TypeError(f"unexpected acceptance condition {condition!r}")


def _expr_bdd_net(expr: Expr, net: BNet, manager: BDDManager,
                  var_of: dict[int, int],
                  constants: dict[int, bool]) -> BDDFunction:
    """Build a BDD for ``expr`` using projected variables and frozen bits."""
    if isinstance(expr, Var):
        index = net.index[expr.name]
        if index in var_of:
            return manager.var(var_of[index])
        return manager.true() if constants[index] else manager.false()
    if isinstance(expr, Const):
        return manager.true() if expr.value else manager.false()
    if isinstance(expr, Not):
        return ~_expr_bdd_net(expr.arg, net, manager, var_of, constants)
    if isinstance(expr, ExprAnd):
        result = manager.true()
        for arg in expr.args:
            result &= _expr_bdd_net(arg, net, manager, var_of, constants)
        return result
    if isinstance(expr, ExprOr):
        result = manager.false()
        for arg in expr.args:
            result |= _expr_bdd_net(arg, net, manager, var_of, constants)
        return result
    raise TypeError(f"unexpected expression {expr!r}")


def _bfs_order(net: BNet, variables: set[int],
               starts: Iterable[int]) -> list[int]:
    """Choose a static variable order by walking the interaction graph."""
    adjacency: dict[int, set[int]] = {v: set() for v in variables}
    for target, expr in enumerate(net.update):
        if target not in variables:
            continue
        for name in _vars_of(expr):
            source = net.index[name]
            if source in variables and source != target:
                adjacency[source].add(target)
                adjacency[target].add(source)

    order: list[int] = []
    seen: set[int] = set()
    seeds = [v for v in starts if v in variables]
    seeds.extend(sorted(variables))
    for start in seeds:
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        while queue:
            v = queue.popleft()
            order.append(v)
            for w in sorted(adjacency[v]):
                if w not in seen:
                    seen.add(w)
                    queue.append(w)
    return order


def _vars_of(expr: Expr) -> set[str]:
    # Local copy to avoid importing the public helper in the hot module; this
    # is only used while building the static order.
    result: set[str] = set()
    stack = [expr]
    while stack:
        e = stack.pop()
        if isinstance(e, Var):
            result.add(e.name)
        elif isinstance(e, Not):
            stack.append(e.arg)
        elif isinstance(e, (ExprAnd, ExprOr)):
            stack.extend(e.args)
    return result


class SymbolicEngine:
    """BDD transition relation and exact Emerson-Lei checks for one prefix."""

    def __init__(self, net: BNet, mode: SamplingMode, pr: dict[str, float],
                 state: int, changeable_mask: int, order: Sequence[int],
                 *, max_nodes: int = DEFAULT_MAX_NODES,
                 max_seconds: float = DEFAULT_MAX_SECONDS,
                 max_sccs: int = DEFAULT_MAX_SCCS,
                 max_iterations: int = DEFAULT_MAX_ITERATIONS) -> None:
        self.net = net
        self.mode = SamplingMode(mode)
        self.pr = pr
        self.state = state
        self.changeable_mask = changeable_mask
        self.order = tuple(order)
        self.max_nodes = max_nodes
        self.max_seconds = max_seconds
        self.max_sccs = max_sccs
        self.max_iterations = max_iterations
        self._deadline = time.monotonic() + max_seconds
        self._iterations = 0

        self._variables = set(self.order)
        self.manager = BDDManager(max_nodes, max_nodes, 1)
        self._var_of: dict[int, int] = {}
        self._prime_of: dict[int, int] = {}
        self._x: dict[int, BDDFunction] = {}
        self._xp: dict[int, BDDFunction] = {}

        # Interleave x_v and x'_v.  OxiDD numbers variables in insertion order.
        names: list[str] = []
        for v in self.order:
            names.append(f"x{v}")
            names.append(f"x{v}'")
        var_numbers = list(self.manager.add_named_vars(names))
        for i, v in enumerate(self.order):
            unprimed = var_numbers[2 * i]
            primed = var_numbers[2 * i + 1]
            self._var_of[v] = unprimed
            self._prime_of[v] = primed
            self._x[v] = self.manager.var(unprimed)
            self._xp[v] = self.manager.var(primed)

        self._unprime_conj = self.manager.true()
        self._prime_conj = self.manager.true()
        for v in self.order:
            self._unprime_conj &= self._x[v]
            self._prime_conj &= self._xp[v]

        self._sub_prime_to_unprime = BDDFunction.make_substitution(
            [(self._prime_of[v], self._x[v]) for v in self.order])
        self._sub_unprime_to_prime = BDDFunction.make_substitution(
            [(self._var_of[v], self._xp[v]) for v in self.order])

        self._constants = {
            i: bool((state >> i) & 1) for i in range(net.n)
            if i not in self._variables}
        self._pattern_cache: dict[tuple[int, int], BDDFunction] = {}
        self._reach: BDDFunction
        self._transition = self._build_transition()
        self._reach = self._reach_from(self._initial_cube(), self.manager.true())

    # -- transition relation ------------------------------------------------

    def _build_transition(self) -> BDDFunction:
        rels: list[BDDFunction] = []
        flips: list[BDDFunction] = []
        stays: list[BDDFunction] = []

        for v in self.order:
            f = _expr_bdd_net(
                self.net.update[v], self.net, self.manager,
                self._var_of, self._constants)
            x = self._x[v]
            xp = self._xp[v]
            change = f ^ x
            stay = ~(xp ^ x)
            flip = xp ^ x
            stays.append(stay)

            name = self.net.names[v]
            if self.mode is SamplingMode.ASYNC_UNIFORM:
                enabled = change
            else:
                enabled = change & (
                    self.manager.true() if self.pr[name] > 0.0
                    else self.manager.false())

            actual_flip = enabled & flip
            flips.append(actual_flip)

            if (self.mode not in (SamplingMode.ASYNC_UNIFORM,
                                  SamplingMode.ASYNC_WEIGHTED)
                    and self.pr[name] >= 1.0):
                # p = 1 forces the variable to fire whenever it is unstable.
                rel = (enabled & flip) | (~enabled & stay)
            else:
                # Optional firing: if enabled, either value is allowed; if
                # disabled, the variable must keep its value.
                rel = enabled | stay
            rels.append(rel)

        if self.mode is SamplingMode.STUTTER_FREE:
            global_constraint = self.manager.false()
            for flip in flips:
                global_constraint |= flip
        elif self.mode is SamplingMode.STUTTERING:
            global_constraint = self.manager.true()
        else:
            one = self.manager.false()
            for flip in flips:
                one |= flip
            at_most_one = self.manager.true()
            for i in range(len(flips)):
                for j in range(i + 1, len(flips)):
                    at_most_one &= ~(flips[i] & flips[j])
            global_constraint = one & at_most_one

        transition = global_constraint
        for rel in rels:
            transition &= rel

        # Sinks get an analysis-only self-loop.  Operational terminal
        # reporting still uses SteadyState; this transition exists only so the
        # SCC analysis has an infinite path to talk about.
        has_successor = (
            transition.exists(self._prime_conj)
            if self.order else transition)
        sink = ~has_successor
        equality = self.manager.true()
        for stay in stays:
            equality &= stay
        transition |= sink & equality
        return transition

    # -- images and reachability -------------------------------------------

    def _check_budget(self) -> None:
        self._iterations += 1
        if self._iterations > self.max_iterations:
            raise SymbolicBudgetExceeded("iteration budget exhausted")
        if time.monotonic() > self._deadline:
            raise SymbolicBudgetExceeded("time budget exhausted")
        if self.manager.num_inner_nodes() > self.max_nodes:
            raise SymbolicBudgetExceeded("BDD node budget exhausted")

    def post(self, states: BDDFunction) -> BDDFunction:
        """Image of ``states`` under the transition relation."""
        result = self._transition & states
        if self.order:
            result = result.exists(self._unprime_conj)
            result = result.substitute(self._sub_prime_to_unprime)
        return result

    def pre(self, states: BDDFunction) -> BDDFunction:
        """Pre-image of ``states`` under the transition relation."""
        result = self._transition
        if self.order:
            primed_states = states.substitute(self._sub_unprime_to_prime)
            result = (result & primed_states).exists(self._prime_conj)
        return result

    def _initial_cube(self) -> BDDFunction:
        cube = self.manager.true()
        for v in self.order:
            bit = bool((self.state >> v) & 1)
            cube &= self._x[v] if bit else ~self._x[v]
        return cube

    def _pick_seed(self, states: BDDFunction) -> Optional[BDDFunction]:
        cube = states.pick_cube()
        if cube is None:
            return None
        seed = self.manager.true()
        for v in self.order:
            value = cube[self._var_of[v]]
            # ``pick_cube`` leaves irrelevant variables as don't-cares.  The
            # set depends only on unprimed variables, so choosing false for a
            # don't-care still yields a state in the set.
            bit = bool(value) if value is not None else False
            seed &= self._x[v] if bit else ~self._x[v]
        return seed

    def _reach_from(self, seed: BDDFunction,
                    within: BDDFunction) -> BDDFunction:
        reached = seed & within
        frontier = reached
        while frontier.satisfiable():
            self._check_budget()
            new = within & self.post(frontier) & ~reached
            if not new.satisfiable():
                break
            reached |= new
            frontier = new
        return reached

    def _co_reach(self, seed: BDDFunction,
                  within: BDDFunction) -> BDDFunction:
        reached = seed & within
        frontier = reached
        while frontier.satisfiable():
            self._check_budget()
            new = within & self.pre(frontier) & ~reached
            if not new.satisfiable():
                break
            reached |= new
            frontier = new
        return reached

    def _scc_of(self, seed: BDDFunction,
                within: BDDFunction) -> BDDFunction:
        forward = self._reach_from(seed, within)
        if not forward.satisfiable():
            return forward
        return forward & self._co_reach(seed, forward)

    def has_cycle(self, states: BDDFunction) -> bool:
        """Whether the subgraph induced by ``states`` contains a cycle."""
        if not states.satisfiable():
            return False
        current = states
        while True:
            self._check_budget()
            nxt = states & self.post(current)
            if nxt == current:
                return current.satisfiable()
            current = nxt

    # -- conditions ---------------------------------------------------------

    def _pattern_bdd(self, pattern: Pattern) -> BDDFunction:
        key = (pattern.required, pattern.values)
        hit = self._pattern_cache.get(key)
        if hit is not None:
            return hit

        # Bits outside the projection are frozen at the current state.
        if pattern.mismatch(self.state) & ~self.changeable_mask:
            result = self.manager.false()
        else:
            result = self.manager.true()
            for v in self.order:
                if pattern.required & (1 << v):
                    literal = (self._x[v]
                               if pattern.values & (1 << v)
                               else ~self._x[v])
                    result &= literal
        self._pattern_cache[key] = result
        return result

    def _term_sets(self, term: Sequence[Fin | Inf]
                   ) -> tuple[BDDFunction, list[BDDFunction]]:
        fin = self.manager.false()
        inf_sets: list[BDDFunction] = []
        for atom in term:
            if isinstance(atom, Fin):
                for pattern in atom.patterns:
                    fin |= self._pattern_bdd(pattern)
            elif isinstance(atom, Inf):
                inf = self.manager.false()
                for pattern in atom.patterns:
                    inf |= self._pattern_bdd(pattern)
                inf_sets.append(inf)
            else:
                raise TypeError(f"unexpected atom {atom!r}")
        return fin, inf_sets

    def _scc_satisfies(self, states: BDDFunction,
                       inf_sets: Sequence[BDDFunction]) -> bool:
        remaining = states
        scc_count = 0
        while remaining.satisfiable():
            self._check_budget()
            seed = self._pick_seed(remaining)
            if seed is None:
                return False
            component = self._scc_of(seed, remaining)
            if not component.satisfiable():
                return False
            if (self.has_cycle(component)
                    and all((component & inf).satisfiable()
                            for inf in inf_sets)):
                return True
            remaining &= ~component
            scc_count += 1
            if scc_count > self.max_sccs:
                raise SymbolicBudgetExceeded("SCC budget exhausted")
        return False

    def sat_some_term(self, term: Sequence[Fin | Inf]) -> bool:
        """Whether some infinite path satisfies a conjunction of atoms."""
        fin, inf_sets = self._term_sets(term)
        outside = self._reach & ~fin
        if not outside.satisfiable():
            return False
        if not inf_sets:
            return self.has_cycle(outside)
        if not self.has_cycle(outside):
            return False
        return self._scc_satisfies(outside, inf_sets)

    def verdict(self, condition: Condition) -> PrefixType | Undecided:
        """Exact three-valued verdict for the current prefix."""
        some = any(self.sat_some_term(term) for term in _dnf(condition))
        some_dual = any(self.sat_some_term(term)
                        for term in _dnf(_dual(condition)))
        if some and some_dual:
            return PrefixType.UGLY
        if some:
            return PrefixType.GOOD
        if some_dual:
            return PrefixType.BAD
        # With the sink self-loop there is always an infinite path, so this is
        # unreachable for a well-formed condition.  Treat it as structurally
        # undecided rather than inventing an UNKNOWN verdict.
        return Undecided.INCONCLUSIVE


def symbolic_verdict(condition: Condition, runner,
                     state: Optional[int] = None, *,
                     max_nodes: int = DEFAULT_MAX_NODES,
                     max_seconds: float = DEFAULT_MAX_SECONDS,
                     max_sccs: int = DEFAULT_MAX_SCCS,
                     max_iterations: int = DEFAULT_MAX_ITERATIONS
                     ) -> PrefixType | Undecided:
    """Run the exact BDD check, returning ``Undecided.TIMEOUT`` if the budget
    is exhausted and ``Undecided.INCONCLUSIVE`` for a structural non-decision.
    """
    if state is None:
        state = runner.state
    assert state is not None

    changeable_mask = runner.changeable_mask(state)
    variables = {v for v in range(runner.net.n)
                 if changeable_mask & (1 << v)}
    starts = runner.enabled_changes(state)
    base_order = _bfs_order(runner.net, variables, starts)
    orders: list[tuple[int, ...]] = []
    for candidate in (tuple(base_order), tuple(reversed(base_order))):
        if candidate not in orders:
            orders.append(candidate)

    for order in orders:
        try:
            engine = SymbolicEngine(
                runner.net, runner.mode, runner.pr, state, changeable_mask,
                order, max_nodes=max_nodes, max_seconds=max_seconds,
                max_sccs=max_sccs, max_iterations=max_iterations)
            return engine.verdict(condition)
        except (SymbolicBudgetExceeded, DDMemoryError):
            continue
    return Undecided.TIMEOUT
