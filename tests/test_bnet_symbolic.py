"""Exact symbolic acceptance checks for Boolean networks."""

import sys

import pytest

from hoax.bnet import parse_bnet
from hoax.bnet_acceptance import Fin, Inf, parse_condition
from hoax.bnet_runner import BnetRunner, SamplingMode
from hoax.bnet_symbolic import (_bfs_order, _dual, _row_terms, SymbolicEngine,
                                symbolic_verdict)
from hoax.drivers import RandomDriver
from hoax.verdict import PrefixType, Undecided


def make_runner(net, state, mode=SamplingMode.STUTTER_FREE, weights=None):
    runner = BnetRunner(net, RandomDriver(list(net.names)),
                        mode=mode, weights=weights)
    runner.init(state=state)
    return runner


def projected_order(runner):
    mask = runner.changeable_mask(runner.state)
    variables = {v for v in range(runner.net.n) if mask & (1 << v)}
    return _bfs_order(runner.net, variables, runner.enabled_changes(runner.state))


# ---------------------------------------------------------------------------
# Transition relation support
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", list(SamplingMode))
def test_post_matches_enabled_successors(mode):
    net = parse_bnet("targets, factors\na, !a\nb, a\n")
    weights = {"a": 0.2, "b": 0.8} if mode is SamplingMode.ASYNC_WEIGHTED else None
    runner = make_runner(net, 0, mode=mode, weights=weights)

    order = projected_order(runner)
    engine = SymbolicEngine(
        net, runner.mode, runner.pr, runner.state,
        runner.changeable_mask(runner.state), order)

    image = engine.post(engine._initial_cube())
    mask = runner.changeable_mask(runner.state)
    frozen = runner.state & ~mask
    reached = []
    for bits in range(1 << len(order)):
        args = [(engine.cell._var_of[v], bool(bits >> i & 1))
                for i, v in enumerate(order)]
        if image.eval(args):
            state = frozen
            for i, v in enumerate(order):
                if bits >> i & 1:
                    state |= 1 << v
            reached.append(state)
    assert sorted(reached) == sorted(set(runner.enabled_successors(runner.state)))


# ---------------------------------------------------------------------------
# Exact verdict against a brute-force SCC checker
# ---------------------------------------------------------------------------

def _tarjan_sccs(vertices, adjacency):
    index = {}
    low = {}
    onstack = set()
    stack = []
    comps = []
    counter = [0]

    def strong(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        onstack.add(v)
        for w in adjacency.get(v, ()):
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                onstack.discard(w)
                comp.append(w)
                if w == v:
                    break
            comps.append(comp)

    for v in vertices:
        if v not in index:
            strong(v)
    return comps


def _reachable(init, adjacency):
    seen = {init}
    stack = [init]
    while stack:
        v = stack.pop()
        for w in adjacency.get(v, ()):
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return seen


def _has_cycle(comp, adjacency):
    if len(comp) > 1:
        return True
    v = comp[0]
    return v in adjacency.get(v, ())


def _bottom_sccs_brute(init, adjacency):
    """Return the bottom (sink) SCCs of the reachable set from ``init``."""
    reach = _reachable(init, adjacency)
    sub = {v: {w for w in adjacency.get(v, ()) if w in reach} for v in reach}
    bottoms = []
    for comp in _tarjan_sccs(reach, sub):
        as_set = set(comp)
        if all(w in as_set for v in comp for w in adjacency.get(v, ())):
            bottoms.append(as_set)
    return bottoms


def _all_bottom_sccs_mixed_brute(condition, init, adjacency):
    """Whether every reachable bottom SCC is internally mixed."""
    rows = _row_terms(condition)
    dual_rows = _row_terms(_dual(condition))
    bottoms = _bottom_sccs_brute(init, adjacency)
    if not bottoms:
        return False
    for comp in bottoms:
        seed = next(iter(comp))
        good = any(_sat_some_term(t, seed, adjacency) for t in rows)
        bad = any(_sat_some_term(t, seed, adjacency) for t in dual_rows)
        if not (good and bad):
            return False
    return True


def _sat_some_term(term, init, adjacency):
    fin = set()
    inf = []
    for atom in term:
        if isinstance(atom, Fin):
            for p in atom.patterns:
                fin.add(p)
        else:
            inf.append(atom.patterns)

    reach = _reachable(init, adjacency)
    remains = {s for s in reach if not any(p.matches(s) for p in fin)}
    if not remains:
        return False
    sub = {s: {w for w in adjacency.get(s, ()) if w in remains} for s in remains}
    inf_sets = [{s for s in remains if any(p.matches(s) for p in pats)}
                for pats in inf]
    for comp in _tarjan_sccs(remains, sub):
        if not _has_cycle(comp, sub):
            continue
        as_set = set(comp)
        if all(as_set & inf_set for inf_set in inf_sets):
            return True
    return False


def _brute_force_verdict(condition, init, adjacency):
    positive = any(_sat_some_term(t, init, adjacency)
                   for t in _row_terms(condition))
    dual = any(_sat_some_term(t, init, adjacency)
               for t in _row_terms(_dual(condition)))
    if positive and dual:
        if _all_bottom_sccs_mixed_brute(condition, init, adjacency):
            return PrefixType.UGLY
        return Undecided.INDETERMINATE
    if positive:
        return PrefixType.GOOD
    if dual:
        return PrefixType.BAD
    return Undecided.INCONCLUSIVE


def test_cell_reused_across_states_in_same_region():
    net = parse_bnet("targets, factors\na, !a\nb, b\n")

    def make(state):
        runner = make_runner(net, state)
        order = projected_order(runner)
        return SymbolicEngine(net, runner.mode, runner.pr, state,
                              runner.changeable_mask(runner.state), order)

    # a always toggles; b is a self-loop (never changeable).  States 0 and 1
    # share the changeable set {a} and the frozen value b=0, so they live in the
    # same cell and must reuse the same transition relation.
    e0 = make(0)
    e1 = make(1)
    assert e0.cell is e1.cell
    assert e0.cell._transition is e1.cell._transition

    # State 2 freezes b=1, which is a different cell.
    e2 = make(2)
    assert e2.cell is not e0.cell


def test_exact_verdict_matches_brute_force():
    net = parse_bnet("targets, factors\na, !a\nb, a\n")
    conditions = ["Fin(0x)", "Inf(1x)", "Fin(00) & Inf(11)",
                  "Fin(01) | Inf(x0)"]

    for mode in (SamplingMode.STUTTER_FREE, SamplingMode.STUTTERING,
                 SamplingMode.ASYNC_UNIFORM):
        weights = None
        for state in range(1 << net.n):
            runner = make_runner(net, state, mode=mode, weights=weights)
            adjacency = {}
            for s in range(1 << net.n):
                succ = set(runner.enabled_successors(s)) or {s}
                adjacency[s] = succ
            for text in conditions:
                condition = parse_condition(text, net)
                expected = _brute_force_verdict(condition, state, adjacency)
                got = symbolic_verdict(condition, runner, state=state)
                assert got == expected, (mode, state, text)


def test_ugly_requires_bottom_scc():
    # a toggles alone; b and c form a 2-cycle.  Some states can still reach
    # either the accepting or the rejecting class (transient -> INDETERMINATE),
    # while states already committed to a closed mixed class are UGLY.
    net = parse_bnet("targets, factors\na, !a\nb, c\nc, b\n")
    conditions = ["Fin(000)", "Inf(111)", "Fin(00x) & Inf(11x)",
                  "Fin(01x) | Inf(x00)", "Fin(000) | Fin(111)",
                  "Inf(00x)", "Fin(0xx) & Inf(1xx)"]

    seen = set()
    for mode in (SamplingMode.STUTTER_FREE, SamplingMode.STUTTERING,
                 SamplingMode.ASYNC_UNIFORM):
        runner = make_runner(net, 0, mode=mode)
        adjacency = {
            s: set(runner.enabled_successors(s)) or {s}
            for s in range(1 << net.n)}
        for state in range(1 << net.n):
            for text in conditions:
                condition = parse_condition(text, net)
                expected = _brute_force_verdict(condition, state, adjacency)
                got = symbolic_verdict(condition, runner, state=state)
                assert got == expected, (mode, state, text)
                seen.add(expected)

    # The refinement must actually be exercised, not merely be a no-op.
    assert PrefixType.UGLY in seen
    assert Undecided.INDETERMINATE in seen


def test_inf_conjunction_vs_union():
    # Two independent toggles.  ``Inf(0x) & Inf(x1)`` (generalized Buchi: visit
    # a=0 and b=1 both infinitely often) is STRICTLY stronger than
    # ``Inf(0x, x1)`` (visit a=0 or b=1 infinitely often).  The conjunction is
    # mixed (UGLY): an accepting 4-cycle visits both sets, while a rejecting
    # 2-cycle on {10, 11} avoids {00, 01}.  The union has no rejecting
    # extension (any move leaves the union), so it is GOOD.
    net = parse_bnet("targets, factors\na, !a\nb, !b\n")
    runner = make_runner(net, 0, mode=SamplingMode.STUTTER_FREE)
    conjunction = parse_condition("Inf(0x) & Inf(x1)", net)
    union = parse_condition("Inf(0x, x1)", net)
    assert symbolic_verdict(conjunction, runner, state=0) is PrefixType.UGLY
    assert symbolic_verdict(union, runner, state=0) is PrefixType.GOOD


def test_symbolic_engine_not_imported_by_plain_runner():
    code = (
        "import sys, hoax.bnet_runner as r; "
        "assert 'hoax.bnet_symbolic' not in sys.modules; "
        "assert 'hoax.bnet_monitor' not in sys.modules; "
        "assert 'hoax.bnet_acceptance' not in sys.modules")
    subprocess = __import__("subprocess")
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=__import__("pathlib").Path(__file__).resolve().parent.parent,
        check=True)


def test_monitor_reports_terminal_and_verdict():
    from hoax.bnet_monitor import BnetMonitor

    net = parse_bnet("targets, factors\na, a\nb, b\n")
    runner = make_runner(net, 0)
    condition = parse_condition("Fin(00)", net)
    monitor = BnetMonitor(condition, runner)
    # A fixed point stays in 00 forever.
    assert monitor.check(runner.state) is PrefixType.BAD

    runner = make_runner(net, 0, mode=SamplingMode.ASYNC_UNIFORM)
    condition = parse_condition("Inf(11)", net)
    monitor = BnetMonitor(condition, runner)
    # 11 is unreachable, so Inf is false on the only infinite path.
    assert monitor.check(runner.state) is PrefixType.BAD
