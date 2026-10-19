import pytest

from hoax.bnet import (And, BNet, Const, Not, Or, Var, eval_expr, parse_bnet,
                       parse_expr, tokenize)
from hoax.bnet_runner import BnetRunner, format_state
from hoax.drivers import RandomDriver
from hoax.util import PRG_SEED, pick

# a is bit 0, b is bit 1
TINY = """
# a tiny network
targets, factors
a, !a
b, a & !b
"""

INPUT = """
targets,factors
a, a & b
"""


def all_states(net):
    return range(1 << net.n)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_tokenize():
    assert tokenize("a & !b | (c)") == ["a", "&", "!", "b", "|", "(", "c", ")"]
    assert tokenize("v_NAD____simple_molecule & v_x") == ["v_NAD____simple_molecule", "&", "v_x"]
    assert tokenize("  a&b  ") == ["a", "&", "b"]


def test_parse_precedence():
    assert parse_expr("a|b&c") == Or((Var("a"), And((Var("b"), Var("c")))))
    assert parse_expr("!a&b") == And((Not(Var("a")), Var("b")))
    assert parse_expr("!!a") == Not(Not(Var("a")))
    assert parse_expr("(a|b)&c") == And((Or((Var("a"), Var("b"))), Var("c")))
    assert parse_expr("1") == Const(True)
    assert parse_expr("0") == Const(False)


def test_parse_errors():
    with pytest.raises(ValueError):
        parse_expr("a &")
    with pytest.raises(ValueError):
        parse_expr("(a")
    with pytest.raises(ValueError):
        parse_expr("a b")


def test_parse_bnet_basic():
    net = parse_bnet(TINY)
    assert net.names == ["a", "b"]
    assert net.n == 2
    assert net.update[0] == Not(Var("a"))
    assert net.update[1] == And((Var("a"), Not(Var("b"))))


def test_parse_bnet_header_variants():
    net = parse_bnet("targets,factors\na, b\nb, a\n")
    assert net.names == ["a", "b"]
    net = parse_bnet("targets, factors\n\na, b\nb, a\n")
    assert net.names == ["a", "b"]


def test_parse_bnet_undeclared_input():
    net = parse_bnet(INPUT)
    assert net.n == 2
    assert "b" in net.names
    assert net.update[net.index["b"]] == Var("b")  # self-loop


def test_parse_bnet_constant_target():
    net = parse_bnet("targets, factors\na, b\nb,\n")
    assert net.update[net.index["b"]] == Var("b")


def test_eval_expr():
    net = parse_bnet(TINY)
    assert eval_expr(net.update[0], {"a": False, "b": False}) is True
    assert eval_expr(net.update[1], {"a": True, "b": False}) is True
    assert eval_expr(net.update[1], {"a": True, "b": True}) is False


# ---------------------------------------------------------------------------
# Reference semantics
# ---------------------------------------------------------------------------

def test_successors_tiny():
    net = parse_bnet(TINY)
    # state 0 = (a=0, b=0): f_a = 1 (changes), f_b = 0 (holds) -> {0, 1}
    assert net.successors(0) == [0, 1]
    # state 1 = (a=1, b=0): f_a = 0 and f_b = 1 both change -> all four
    assert net.successors(1) == [0, 1, 2, 3]
    # state 2 = (a=0, b=1): f_a = 1 and f_b = 0 both change -> all four
    assert net.successors(2) == [0, 1, 2, 3]
    # state 3 = (a=1, b=1): both change -> all four
    assert net.successors(3) == [0, 1, 2, 3]


def test_potential_changes():
    net = parse_bnet(TINY)
    rel = net
    assert rel.potential_changes(0) == [0]
    assert rel.potential_changes(1) == [0, 1]
    assert rel.potential_changes(2) == [0, 1]
    assert rel.potential_changes(3) == [0, 1]


# ---------------------------------------------------------------------------
# Successors and distribution
# ---------------------------------------------------------------------------

def test_successors_include_stutter():
    net = parse_bnet(TINY)
    rel = net
    for state in all_states(net):
        assert state in rel.successors(state)


def test_successors_box_structure():
    """Successors are exactly the flips of the variables that may change."""
    net = parse_bnet(TINY)
    rel = net
    for state in all_states(net):
        changes = rel.potential_changes(state)
        expected = set()
        for mask in range(1 << len(changes)):
            nxt = state
            for bit, i in enumerate(changes):
                if (mask >> bit) & 1:
                    nxt ^= 1 << i
            expected.add(nxt)
        assert set(rel.successors(state)) == expected
        assert len(rel.successors(state)) == 1 << len(changes)


def test_models_bijection_and_fields():
    net = parse_bnet(TINY)
    rel = net
    for state in all_states(net):
        entries = rel.models(state)
        assert {e[2] for e in entries} == set(rel.successors(state))
        assert len({e[0] for e in entries}) == len(entries)
        for model, m_str, nxt in entries:
            assert m_str.startswith("{") and m_str.endswith("}")
            # model mentions exactly the potential-change variables
            assert [name for name, _ in model] == [net.names[i] for i in rel.potential_changes(state)]
            # value is True iff the variable actually changed
            for name, fired in model:
                changed = bool((nxt >> net.index[name]) & 1) != bool((state >> net.index[name]) & 1)
                assert fired == changed


def test_trel_uniform_for_half_weights():
    net = parse_bnet(TINY)
    rel = net
    pr = {name: 0.5 for name in net.names}
    for state in all_states(net):
        trel = rel.trel(state, pr)
        k = len(trel)
        assert abs(trel[-1][0] - 1.0) < 1e-9
        for i, (cum, (model, m_str, nxt)) in enumerate(trel):
            assert abs(cum - (i + 1) / k) < 1e-9
            assert abs(1 / k - 2.0 ** -len(model)) < 1e-9


def test_trel_nonuniform_weights():
    net = parse_bnet(TINY)
    rel = net
    pr = {"a": 0.3, "b": 0.7}
    # state 1 = (1, 0): both a and b may change
    trel = rel.trel(1, pr)
    probs: dict[int, float] = {}
    previous = 0.0
    for cum, (model, m_str, nxt) in trel:
        probs[nxt] = cum - previous
        previous = cum
    # successor 1: nobody fires
    assert abs(probs[1] - (0.7 * 0.3)) < 1e-9
    # successor 0: only a fires
    assert abs(probs[0] - (0.3 * 0.3)) < 1e-9
    # successor 3: only b fires
    assert abs(probs[3] - (0.7 * 0.7)) < 1e-9
    # successor 2: both fire
    assert abs(probs[2] - (0.3 * 0.7)) < 1e-9
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_trel_steady_state():
    net = parse_bnet("targets, factors\na, a\nb, b\n")
    rel = net
    trel = rel.trel(0, {"a": 0.5, "b": 0.5})
    assert len(trel) == 1
    assert trel[0][0] == 1.0
    assert trel[0][1][0] == ()   # empty firing model
    assert trel[0][1][2] == 0


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _cumulative_scan(entries, r):
    """Reference: the leaf a cumulative scan of ``trel()`` would pick for ``r``."""
    for cumulative_prob, value in entries:
        if r < cumulative_prob:
            return value
    return entries[-1][1]


def test_sample_matches_cumulative_scan():
    """The descent must agree with a full cumulative scan for every draw."""
    net = parse_bnet(TINY)
    pr = {"a": 0.3, "b": 0.7}
    for state in all_states(net):
        changes = net.potential_changes(state)
        entries = net.trel(state, pr)
        for i in range(1000):
            r = i / 1000
            assert net.sample(state, pr, r, changes) == _cumulative_scan(entries, r)


def test_sample_degenerate_weights():
    net = parse_bnet(TINY)
    pr = {"a": 0.0, "b": 1.0}
    for state in all_states(net):
        entries = net.trel(state, pr)
        for r in (0.0, 0.25, 0.5, 0.75, 0.999):
            assert net.sample(state, pr, r) == _cumulative_scan(entries, r)


def test_sample_steady_state():
    net = parse_bnet("targets, factors\na, a\n")
    pr = {"a": 0.5}
    assert net.sample(0, pr, 0.7) == _cumulative_scan(net.trel(0, pr), 0.7)
    assert net.sample(0, pr, 0.7) == ((), "{}", 0)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def test_format_state_sorted():
    net = parse_bnet("targets, factors\nb, a\na, !a\n")
    assert net.names == ["b", "a"]      # file order
    # index: b = bit 0, a = bit 1; default output order is sorted: a then b
    assert format_state(net, 0b01) == "01"   # b = 1, a = 0
    assert format_state(net, 0b10) == "10"   # b = 0, a = 1


def test_runner_walk():
    PRG_SEED(1234)
    net = parse_bnet(TINY)
    runner = BnetRunner(net, RandomDriver(list(net.names)))
    runner.init(state=3)
    assert runner.state == 3
    runner.init()  # random start
    assert runner.state is not None and 0 <= runner.state < 4
    prev = runner.state
    seen_states = set()
    for _ in range(50):
        tr = runner.step()
        assert len(tr) == 1
        old, model, m_str, new = tr[0]
        assert old == prev
        assert new == runner.state
        assert m_str.startswith("{")
        seen_states.add(new)
        prev = new
    assert runner.count == 50
    assert len(seen_states) >= 1


def test_runner_caches_topology_only():
    net = parse_bnet(TINY)
    runner = BnetRunner(net, RandomDriver(list(net.names)))
    runner.init(state=3)
    runner.step()
    assert runner.potential_changes(3) == [0, 1]
    assert runner.potential_changes(3) is runner.potential_changes(3)
    assert not runner.is_steady(3)

    net = parse_bnet("targets, factors\na, a\n")
    runner = BnetRunner(net, RandomDriver(list(net.names)))
    runner.init(state=0)
    assert runner.is_steady(0)


def test_runner_trajectory_unchanged():
    """The sampler must reproduce the old ``trel()`` + ``pick()`` walk exactly."""
    net = parse_bnet(TINY)
    runner = BnetRunner(net, RandomDriver(list(net.names)))
    pr = runner.pr

    PRG_SEED(20260719)
    runner.init(state=0)
    walked = [runner.state]
    for _ in range(200):
        runner.step()
        walked.append(runner.state)

    PRG_SEED(20260719)
    state = 0
    reference = [state]
    for _ in range(200):
        _, _, state = pick(net.trel(state, pr))
        reference.append(state)

    assert walked == reference


def test_runner_trajectory_unchanged_through_steady():
    """Steady states must consume no draw, like the reference ``pick()``."""
    net = parse_bnet("targets, factors\na, a\nb, a\n")
    runner = BnetRunner(net, RandomDriver(list(net.names)))
    pr = runner.pr
    for start in all_states(net):
        PRG_SEED(4242 + start)
        runner.init(state=start)
        walked = [runner.state]
        for _ in range(20):
            runner.step()
            walked.append(runner.state)

        PRG_SEED(4242 + start)
        state = start
        reference = [state]
        for _ in range(20):
            _, _, state = pick(net.trel(state, pr))
            reference.append(state)
        assert walked == reference
