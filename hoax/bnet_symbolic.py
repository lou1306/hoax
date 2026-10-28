"""Exact symbolic acceptance checking for Boolean networks.

The checker builds a BDD for the projected transition relation of the
variables that can still change from the current state.  It never enumerates
the reachable state graph: reachability, pre/post images and SCC extraction are
BDD fixpoints.  The result is a three-valued prefix verdict:

* ``GOOD``: every infinite extension satisfies the condition;
* ``BAD``: every infinite extension violates it;
* ``UGLY``: both kinds of extension exist and every reachable bottom SCC is
  itself mixed (a closed accepting-and-rejecting class the walk cannot leave).
* ``Undecided.INDETERMINATE``: both kinds of extension exist but some reachable
  bottom SCC is fully good or fully bad, so a continuation could still resolve
  the outcome.

``Undecided`` marks a non-verdict outcome.  ``Undecided.INCONCLUSIVE`` is a
structural non-decision; ``Undecided.TIMEOUT`` means the analysis did not
finish within its budget.  Neither is an ``UNKNOWN`` verdict.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Iterable, Optional, Sequence

from cachetools import LRUCache
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

# A transition cell holds a whole BDD manager, so it is far heavier than the
# runner/monitor verdict caches.  A few thousand distinct cells (changeable set
# x frozen assignment x variable order) is ample for any walk that stays in a
# handful of regions.
DEFAULT_CELL_CACHE_SIZE = 4_096


class SymbolicBudgetExceeded(RuntimeError):
    """Raised internally when a BDD analysis exceeds its configured budget."""


def _merge_row(row: tuple[Fin | Inf, ...]) -> tuple[Fin | Inf, ...]:
    """Merge the ``Fin`` conjuncts of a single row into one (exact).

    ``Fin(A) & Fin(B)`` is equivalent to ``Fin(A|B)``, so a row never carries
    more than one ``Fin`` conjunct.  ``Inf`` conjuncts stay separate: each is an
    independent hit-set that the witnessing cycle must intersect.
    """
    fin_patterns: list[Pattern] = []
    infs: list[Fin | Inf] = []
    for atom in row:
        if isinstance(atom, Fin):
            fin_patterns.extend(atom.patterns)
        else:
            infs.append(atom)
    if fin_patterns:
        infs.insert(0, Fin(tuple(fin_patterns)))
    return tuple(infs)


def _pnf_rows(rows: list[tuple[Fin | Inf, ...]]
              ) -> list[tuple[Fin | Inf, ...]]:
    """Reduce a DNF row list by positive-normal-form merging (exact).

    * Within a row, ``Fin(A) & Fin(B)`` collapses to ``Fin(A|B)``.
    * Across rows, ``Inf(A) | Inf(B)`` collapses to ``Inf(A|B)``: a cycle
      hitting ``A`` or ``B`` is exactly a cycle hitting the union.

    A disjunction that mixes functors (``Inf(a) | Fin(b)``) cannot be merged
    and is kept as separate rows.  Minimising the number of rows and of
    hit-sets keeps the often-expensive joint cycle test small.
    """
    merged_ = [_merge_row(row) for row in rows]
    result: list[tuple[Fin | Inf, ...]] = []
    lone_inf: list[Inf] = []
    for row in merged_:
        if len(row) == 1 and isinstance(row[0], Inf):
            lone_inf.append(row[0])
        else:
            result.append(row)
    if lone_inf:
        patterns = tuple(p for inf in lone_inf for p in inf.patterns)
        result.append((Inf(patterns),))
    return result


def _row_terms(condition: Condition) -> list[tuple[Fin | Inf, ...]]:
    """Return a PNF-reduced disjunction of conjunctive atom rows.

    Each row is a conjunction of ``Fin``/``Inf`` atoms; the list is a
    disjunction of them.  This is the smallest row-set needed for the exact
    infinite-path tests: equal-functor unions are merged and the remaining
    disjunction only splits when a single path must branch.
    """
    if isinstance(condition, (Fin, Inf)):
        return [(condition,)]
    if isinstance(condition, Or):
        result: list[tuple[Fin | Inf, ...]] = []
        for child in condition.conditions:
            result.extend(_row_terms(child))
        return _pnf_rows(result)
    if isinstance(condition, And):
        terms: list[tuple[Fin | Inf, ...]] = [()]
        for child in condition.conditions:
            child_terms = _row_terms(child)
            terms = [left + right for left in terms for right in child_terms]
        return _pnf_rows(terms)
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


_TRANSITION_CELL_CACHE: LRUCache[tuple, "_TransitionCell"] = LRUCache(
    maxsize=DEFAULT_CELL_CACHE_SIZE)


def _frozen_values(changeable_mask: int, net: BNet, state: int) -> tuple[int, ...]:
    """Return the bits of the frozen (non-changeable) variables, in index order."""
    return tuple(
        (state >> i) & 1 for i in range(net.n)
        if not (changeable_mask & (1 << i)))


def _get_cell(net: BNet, mode: SamplingMode, pr: dict[str, float],
              changeable_mask: int, order: Sequence[int],
              frozen_values: Sequence[int], max_nodes: int) -> "_TransitionCell":
    """Return the (cached) cell for this changeable set / frozen assignment /
    variable order.  ``mode``, ``pr``, ``net`` and ``max_nodes`` are folded into
    the key so caches never leak across runs or weightings.
    """
    key = (mode, id(net), changeable_mask, tuple(order), tuple(frozen_values),
           tuple(sorted(pr.items())), max_nodes)
    cell = _TRANSITION_CELL_CACHE.get(key)
    if cell is None:
        cell = _TransitionCell(net, mode, pr, changeable_mask, order,
                               frozen_values, max_nodes)
        _TRANSITION_CELL_CACHE[key] = cell
    return cell


class _TransitionCell:
    """Cell-reusable exact-symbolic gear.

    A *cell* is the set of states sharing the same changeable variables and the
    same values on the frozen (non-changeable) variables.  All of them share the
    projected transition relation, so the BDD manager, variable BDDs, the
    substitutions and the transition relation are built here once and reused
    across every state in the cell.  Only the initial cube and the reachability
    / verdict are state-specific (see :class:`SymbolicEngine`).
    """

    def __init__(self, net: BNet, mode: SamplingMode, pr: dict[str, float],
                 changeable_mask: int, order: Sequence[int],
                 frozen_values: Sequence[int],
                 max_nodes: int = DEFAULT_MAX_NODES) -> None:
        self.net = net
        self.mode = SamplingMode(mode)
        self.pr = pr
        self.changeable_mask = changeable_mask
        self.order = tuple(order)
        self.max_nodes = max_nodes
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

        frozen_idx = [i for i in range(net.n)
                      if not (changeable_mask & (1 << i))]
        self._constants = {i: bool(bit)
                           for i, bit in zip(frozen_idx, frozen_values)}
        self._frozen_mask = sum(
            (1 << i) for i, bit in zip(frozen_idx, frozen_values) if bit)

        self._pattern_cache: dict[tuple[int, int], BDDFunction] = {}
        self._transition = self._build_transition()

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

    # -- images and reachability primitives ---------------------------------

    def num_inner_nodes(self) -> int:
        return self.manager.num_inner_nodes()

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

    def initial_cube(self, state: int) -> BDDFunction:
        cube = self.manager.true()
        for v in self.order:
            bit = bool((state >> v) & 1)
            cube &= self._x[v] if bit else ~self._x[v]
        return cube

    def pick_seed(self, states: BDDFunction) -> Optional[BDDFunction]:
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

    def pattern_bdd(self, pattern: Pattern) -> BDDFunction:
        key = (pattern.required, pattern.values)
        hit = self._pattern_cache.get(key)
        if hit is not None:
            return hit

        # Bits outside the projection are frozen at the current state.  Because
        # the frozen values are part of the cell, this is cell-constant.
        frozen_mismatch = ((pattern.values ^ self._frozen_mask)
                           & pattern.required & ~self.changeable_mask)
        if frozen_mismatch:
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


class SymbolicEngine:
    """Exact Emerson-Lei checks for one prefix, reusing a cached cell's gear.

    The :class:`_TransitionCell` holds everything that is constant across the
    cell (manager, variable BDDs, transition relation, pattern cache).  This
    engine only adds the state-specific pieces: the initial cube and the
    reachability/SCC check derived from it.
    """

    def __init__(self, net: BNet, mode: SamplingMode, pr: dict[str, float],
                 state: int, changeable_mask: int, order: Sequence[int],
                 *, max_nodes: int = DEFAULT_MAX_NODES,
                 max_seconds: float = DEFAULT_MAX_SECONDS,
                 max_sccs: int = DEFAULT_MAX_SCCS,
                 max_iterations: int = DEFAULT_MAX_ITERATIONS) -> None:
        self.state = state
        self.max_seconds = max_seconds
        self.max_sccs = max_sccs
        self.max_iterations = max_iterations
        self._deadline = time.monotonic() + max_seconds
        self._iterations = 0

        frozen_values = _frozen_values(changeable_mask, net, state)
        self.cell = _get_cell(net, mode, pr, changeable_mask, order,
                              frozen_values, max_nodes)
        self._reach = self._reach_from(
            self.cell.initial_cube(state), self.cell.manager.true())

    # -- images and reachability -------------------------------------------

    def _check_budget(self) -> None:
        self._iterations += 1
        if self._iterations > self.max_iterations:
            raise SymbolicBudgetExceeded("iteration budget exhausted")
        if time.monotonic() > self._deadline:
            raise SymbolicBudgetExceeded("time budget exhausted")
        if self.cell.num_inner_nodes() > self.cell.max_nodes:
            raise SymbolicBudgetExceeded("BDD node budget exhausted")

    def post(self, states: BDDFunction) -> BDDFunction:
        """Image of ``states`` under the transition relation."""
        return self.cell.post(states)

    def pre(self, states: BDDFunction) -> BDDFunction:
        """Pre-image of ``states`` under the transition relation."""
        return self.cell.pre(states)

    def _initial_cube(self) -> BDDFunction:
        return self.cell.initial_cube(self.state)

    def _pick_seed(self, states: BDDFunction) -> Optional[BDDFunction]:
        return self.cell.pick_seed(states)

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

    def _bottom_sccs(self) -> list[BDDFunction]:
        """Decompose ``_reach`` and return its bottom (sink) SCCs.

        A bottom SCC ``k`` is closed (``post(k) <= k``), so a walk that enters
        it can never leave.  ``_reach`` is always non-empty and closed, so it
        admits at least one bottom SCC (every state has a successor).
        """
        remaining = self._reach
        bottoms: list[BDDFunction] = []
        scc_count = 0
        while remaining.satisfiable():
            self._check_budget()
            seed = self._pick_seed(remaining)
            if seed is None:
                break
            component = self._scc_of(seed, remaining)
            if not component.satisfiable():
                break
            remaining &= ~component
            scc_count += 1
            if scc_count > self.max_sccs:
                raise SymbolicBudgetExceeded("SCC budget exhausted")
            if not (self.post(component) & ~component).satisfiable():
                bottoms.append(component)
        return bottoms

    def _all_bottom_sccs_mixed(self, rows: Sequence[tuple[Fin | Inf, ...]],
                               dual_rows: Sequence[tuple[Fin | Inf, ...]]
                               ) -> bool:
        """Whether every reachable bottom SCC is internally mixed.

        A bottom SCC is mixed when it admits both a permanently-accepting and a
        permanently-rejecting extension.  When every such class is mixed the
        outcome is irrevocable (``UGLY``); if some bottom SCC is fully good or
        fully bad the mixed outcome is only transient (``INDETERMINATE``).
        GOOD/BAD themselves stay global over ``_reach``: a rejecting extension
        may come from a transient rough cycle, not only from a bottom SCC.
        """
        bottoms = self._bottom_sccs()
        if not bottoms:
            return False
        for component in bottoms:
            good = any(self.sat_some_term(term, region=component)
                       for term in rows)
            bad = any(self.sat_some_term(term, region=component)
                      for term in dual_rows)
            if not (good and bad):
                return False
        return True

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
        return self.cell.pattern_bdd(pattern)

    def _term_sets(self, term: Sequence[Fin | Inf]
                   ) -> tuple[BDDFunction, list[BDDFunction]]:
        fin = self.cell.manager.false()
        inf_sets: list[BDDFunction] = []
        for atom in term:
            if isinstance(atom, Fin):
                for pattern in atom.patterns:
                    fin |= self._pattern_bdd(pattern)
            elif isinstance(atom, Inf):
                inf = self.cell.manager.false()
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

    def sat_some_term(self, term: Sequence[Fin | Inf],
                      region: Optional[BDDFunction] = None) -> bool:
        """Whether some infinite path within ``region`` satisfies ``term``.

        ``region`` defaults to ``_reach``; pass a single bottom SCC to ask
        whether a permanently-resident extension can satisfy the term.
        """
        if region is None:
            region = self._reach
        fin, inf_sets = self._term_sets(term)
        outside = region & ~fin
        if not outside.satisfiable():
            return False
        if not inf_sets:
            return self.has_cycle(outside)
        if not self.has_cycle(outside):
            return False
        return self._scc_satisfies(outside, inf_sets)

    def verdict(self, condition: Condition) -> PrefixType | Undecided:
        """Exact three-valued verdict for the current prefix.

        ``GOOD``/``BAD`` are terminal (the dual extension does not exist).
        When both kinds of extension exist the outcome is mixed: it is
        terminal ``UGLY`` only if *every* reachable bottom SCC is itself mixed
        (irreversibly undecided); otherwise it is the transient non-decision
        ``Undecided.INDETERMINATE`` (a continuation could still settle into a
        fully good or fully bad class, resolving it to GOOD or BAD).
        """
        rows = _row_terms(condition)
        dual_rows = _row_terms(_dual(condition))
        some = any(self.sat_some_term(term) for term in rows)
        some_dual = any(self.sat_some_term(term) for term in dual_rows)
        if some and some_dual:
            if self._all_bottom_sccs_mixed(rows, dual_rows):
                return PrefixType.UGLY
            return Undecided.INDETERMINATE
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
    is exhausted, ``Undecided.INCONCLUSIVE`` for a structural non-decision and
    ``Undecided.INDETERMINATE`` when both extensions exist but some reachable
    bottom SCC is clean (so a continuation could still resolve it).
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
