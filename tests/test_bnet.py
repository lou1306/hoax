import pytest

from hoax.bnet import (And, BNet, Const, Not, Or, Var, compile_expr,
                       eval_expr, parse_bnet, parse_expr, tokenize)
from hoax.bnet_runner import (BnetRunner, SamplingMode, format_state,
                              parse_weight_specs)
from hoax.drivers import CompositeDriver, RandomDriver
from hoax.util import PRG_SEED, PRG_UNIFORM, pick

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


def reference_potential_changes(net, state):
    values = net.state_values(state)
    return [
        i for i in range(net.n)
        if eval_expr(net.update[i], values) != values[net.names[i]]
    ]


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


def test_compile_expr_matches_eval_expr():
    index = {"a": 0, "b": 1, "c": 2}
    exprs = [
        Var("a"),
        Const(True),
        Const(False),
        Not(Var("a")),
        And((Var("a"), Var("b"))),
        And((Var("a"), Var("b"), Not(Var("c")), Const(True))),
        Or((Var("a"), Var("b"))),
        Or((Var("a"), Var("b"), Not(Var("c")), Const(False))),
        And(()),
        Or(()),
    ]
    for expr in exprs:
        compiled = compile_expr(expr, index)
        for state in range(1 << len(index)):
            values = [bool((state >> i) & 1) for i in range(len(index))]
            assert compiled(values) == eval_expr(expr, dict(zip(index, values)))


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


def test_potential_changes_matches_reference():
    for text in (TINY, INPUT):
        net = parse_bnet(text)
        for state in all_states(net):
            assert net.potential_changes(state) == reference_potential_changes(net, state)


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
# Stutter-free PrGA
# ---------------------------------------------------------------------------

def test_trel_stutter_free_renormalises():
    net = parse_bnet(TINY)
    pr = {"a": 0.3, "b": 0.7}
    # state 1 = (a=1, b=0): both variables may change.
    trel = net.trel(1, pr, stutter=False)
    mass = 1.0 - 0.7 * 0.3      # 1 minus P(nobody fires)
    assert len(trel) == 3       # the no-change leaf is gone
    probs: dict[int, float] = {}
    previous = 0.0
    for cum, (model, m_str, nxt) in trel:
        probs[nxt] = cum - previous
        previous = cum
    assert abs(probs[0] - 0.3 * 0.3 / mass) < 1e-9   # only a fires
    assert abs(probs[3] - 0.7 * 0.7 / mass) < 1e-9   # only b fires
    assert abs(probs[2] - 0.3 * 0.7 / mass) < 1e-9   # both fire
    assert 1 not in probs                            # no stutter leaf
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_trel_stutter_free_matches_enumeration():
    """Compare the conditioned distribution with explicit subset enumeration."""
    net = parse_bnet(TINY)
    for pa in (0.0, 0.2, 0.9, 1.0):
        for pb in (0.0, 0.2, 0.9, 1.0):
            pr = {"a": pa, "b": pb}
            for state in all_states(net):
                changes = net.potential_changes(state)
                weights: dict[int, float] = {}
                for mask in range(1 << len(changes)):
                    nxt = state
                    weight = 1.0
                    for bit, v in enumerate(changes):
                        fired = bool((mask >> bit) & 1)
                        p = pr[net.names[v]]
                        weight *= p if fired else 1.0 - p
                        if fired:
                            nxt ^= 1 << v
                    weights[nxt] = weight
                mass = sum(w for nxt, w in weights.items() if nxt != state)
                expected = (
                    {nxt: w / mass for nxt, w in weights.items() if nxt != state}
                    if mass > 0.0 else {})
                got: dict[int, float] = {}
                previous = 0.0
                for cum, (_, _, nxt) in net.trel(state, pr, stutter=False):
                    got[nxt] = cum - previous
                    previous = cum
                assert got.keys() == expected.keys()
                for nxt, probability in expected.items():
                    assert abs(got[nxt] - probability) < 1e-12


def test_trel_stutter_free_steady_state_is_empty():
    net = parse_bnet("targets, factors\na, a\nb, b\n")
    assert net.trel(0, {"a": 0.5, "b": 0.5}, stutter=False) == []


def test_sample_stutter_free_matches_cumulative_scan():
    net = parse_bnet(TINY)
    pr = {"a": 0.3, "b": 0.7}
    for state in all_states(net):
        entries = net.trel(state, pr, stutter=False)
        for i in range(1000):
            r = i / 1000
            assert net.sample(state, pr, r, stutter=False) == _cumulative_scan(entries, r)


def test_sample_stutter_free_never_stutters():
    net = parse_bnet(TINY)
    pr = {"a": 0.3, "b": 0.7}
    for state in all_states(net):
        for i in range(100):
            _, _, nxt = net.sample(state, pr, i / 100, stutter=False)
            assert nxt != state


def test_sample_stutter_free_single_change_is_deterministic():
    net = parse_bnet(TINY)
    pr = {"a": 0.3, "b": 0.7}
    # state 0 = (0,0): only a may change, so it must fire for every draw.
    for r in (0.0, 0.25, 0.5, 0.999):
        assert net.sample(0, pr, r, stutter=False) == ((("a", True),), "{a}", 1)


def test_sample_stutter_free_degenerate_weights():
    net = parse_bnet(TINY)
    pr = {"a": 0.0, "b": 1.0}
    # a may change but never fires; b always fires.
    assert net.sample(1, pr, 0.5, stutter=False)[2] == 3
    assert net.sample(2, pr, 0.5, stutter=False)[2] == 0
    assert net.sample(3, pr, 0.5, stutter=False)[2] == 1
    # state 0: a is the only changeable variable and it cannot fire.
    with pytest.raises(ValueError):
        net.sample(0, pr, 0.5, stutter=False)


def test_sample_stutter_free_steady_state_raises():
    net = parse_bnet("targets, factors\na, a\n")
    with pytest.raises(ValueError):
        net.sample(0, {"a": 0.5}, 0.5, stutter=False)


def test_sample_async_uniform_picks_one_change():
    net = parse_bnet(TINY)
    state = 1                       # a=1, b=0: both variables are unstable
    assert net.potential_changes(state) == [0, 1]
    assert net.sample_async(state, 0.25)[2] == 0
    assert net.sample_async(state, 0.75)[2] == 3
    for i in range(10):
        _, _, nxt = net.sample_async(state, (i + 0.5) / 10)
        assert nxt in (0, 3)


def test_sample_async_single_change_is_deterministic():
    net = parse_bnet("targets, factors\na, !a\n")
    assert net.sample_async(0, 0.0) == ((("a", True),), "{a}", 1)
    assert net.sample_async(0, 0.99) == ((("a", True),), "{a}", 1)


def test_sample_async_steady_state_raises():
    net = parse_bnet("targets, factors\na, a\n")
    with pytest.raises(ValueError):
        net.sample_async(0, 0.5)


def test_sample_async_weighted_uses_relative_weights():
    net = parse_bnet(TINY)
    pr = {"a": 0.25, "b": 0.75}
    assert net.sample_async_weighted(1, pr, 0.1)[2] == 0
    assert net.sample_async_weighted(1, pr, 0.5)[2] == 3
    assert net.sample_async_weighted(1, pr, 0.99)[2] == 3


def test_sample_async_weighted_equal_weights_match_uniform():
    net = parse_bnet(TINY)
    pr = {"a": 0.5, "b": 0.5}
    for i in range(20):
        r = (i + 0.5) / 20
        assert net.sample_async(1, r) == net.sample_async_weighted(1, pr, r)


def test_sample_async_weighted_skips_zero_weights():
    net = parse_bnet(TINY)
    for i in range(10):
        _, _, nxt = net.sample_async_weighted(
            1, {"a": 0.0, "b": 1.0}, (i + 0.5) / 10)
        assert nxt == 3
    # r = 1.0 must still land on the last positive-weight variable, not a
    # trailing zero.
    _, _, nxt = net.sample_async_weighted(1, {"a": 1.0, "b": 0.0}, 1.0)
    assert nxt == 0


def test_sample_async_weighted_all_zero_raises():
    net = parse_bnet(TINY)
    with pytest.raises(ValueError):
        net.sample_async_weighted(1, {"a": 0.0, "b": 0.0}, 0.5)


def test_sample_stutter_free_small_p_limits_to_async():
    """With p -> 0 the conditioned draw picks a single variable, uniformly."""
    net = parse_bnet(TINY)
    pr = {"a": 1e-9, "b": 1e-9}
    counts = {0: 0, 2: 0, 3: 0}   # {a}, both, {b} from state 1 = (1,0)
    n = 10000
    for i in range(n):
        # Offset the grid so that no draw lands exactly on a branch boundary;
        # at r = 0.5 the tiny "both fire" interval has measure zero but is
        # still on the closed side of the split.
        _, _, nxt = net.sample(1, pr, (i + 0.5) / n, stutter=False)
        counts[nxt] += 1
    assert counts[0] / n == pytest.approx(0.5, abs=0.02)
    assert counts[3] / n == pytest.approx(0.5, abs=0.02)
    assert counts[2] == 0          # two fires are O(p) unlikely


def test_stutter_free_tv_approaches_async_linearly():
    """TV(stutter-free, async) / p -> (|U| - 1) / 2 as p -> 0."""
    net = parse_bnet(TINY)
    state = 1                      # (a, b) = (1, 0): both variables unstable
    assert len(net.potential_changes(state)) == 2

    def one_step(pr):
        probs: dict[int, float] = {}
        previous = 0.0
        for cum, (_, _, nxt) in net.trel(state, pr, stutter=False):
            probs[nxt] = cum - previous
            previous = cum
        return probs

    def tv(probs, target):
        keys = probs.keys() | target.keys()
        return 0.5 * sum(abs(probs.get(k, 0.0) - target.get(k, 0.0))
                         for k in keys)

    # Equal p: the asynchronous target is uniform over the two single flips.
    uniform = {0: 0.5, 3: 0.5}     # a fires alone, b fires alone
    for p in (1e-3, 1e-4, 1e-5):
        ratio = tv(one_step({"a": p, "b": p}), uniform) / p
        assert ratio == pytest.approx(0.5, abs=1e-3)

    # p_i = eps * a_i keeps the a_i proportions; the leading coefficient is
    # e2 / sum(p) = (1 * 3) / (1 + 3) * eps = 3/4 eps.
    weighted = {0: 0.25, 3: 0.75}
    for eps in (1e-3, 1e-4, 1e-5):
        ratio = tv(one_step({"a": eps, "b": 3 * eps}), weighted) / eps
        assert ratio == pytest.approx(0.75, abs=1e-3)


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


def test_runner_stutter_free_is_the_default():
    net = parse_bnet(TINY)
    runner = BnetRunner(net, RandomDriver(list(net.names)))
    assert runner.mode is SamplingMode.STUTTER_FREE


def test_runner_stutter_free_never_stutters():
    PRG_SEED(7)
    net = parse_bnet(TINY)
    runner = BnetRunner(net, RandomDriver(list(net.names)))
    runner.init(state=3)
    prev = runner.state
    for _ in range(200):
        runner.step()
        assert runner.state != prev
        prev = runner.state
    assert runner.count == 200


def test_runner_stutter_free_trajectory_matches_reference():
    """The runner must follow ``trel(..., stutter=False)`` + ``pick()``."""
    net = parse_bnet(TINY)
    runner = BnetRunner(net, RandomDriver(list(net.names)))
    pr = runner.pr

    PRG_SEED(20261021)
    runner.init(state=0)
    walked = [runner.state]
    for _ in range(100):
        runner.step()
        walked.append(runner.state)

    PRG_SEED(20261021)
    state = 0
    reference = [state]
    for _ in range(100):
        _, _, state = pick(net.trel(state, pr, stutter=False))
        reference.append(state)

    assert walked == reference


def test_runner_caches_potential_changes():
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


def test_runner_incremental_target_matches_reference():
    """Cache misses must update only the interaction graph's out-neighbours."""
    net = parse_bnet("""
targets, factors
a, !a
b, a & !b
c, b | a
d, 0
e, e
""")
    runner = BnetRunner(net, RandomDriver(list(net.names)))
    runner.init(state=0)

    # Arbitrary consecutive states exercise multi-bit target updates; the
    # second lookup checks that cache hits return the same list.
    for state in all_states(net):
        changes = runner.potential_changes(state)
        assert changes == reference_potential_changes(net, state)
        assert runner.potential_changes(state) is changes


@pytest.mark.parametrize("mode", list(SamplingMode))
def test_runner_target_cache_matches_reference_during_walk(mode):
    net = parse_bnet(TINY)
    weights = {"a": 0.2, "b": 0.8} if mode is SamplingMode.ASYNC_WEIGHTED else None
    runner = BnetRunner(
        net, RandomDriver(list(net.names)), mode=mode, weights=weights)

    PRG_SEED(20261022)
    runner.init(state=0)
    for _ in range(100):
        assert runner.potential_changes(runner.state) == (
            reference_potential_changes(net, runner.state))
        runner.step()
    assert runner.potential_changes(runner.state) == (
        reference_potential_changes(net, runner.state))


def test_runner_trajectory_unchanged():
    """The stuttering sampler must reproduce the old ``trel()`` + ``pick()``."""
    net = parse_bnet(TINY)
    runner = BnetRunner(
        net, RandomDriver(list(net.names)), mode=SamplingMode.STUTTERING)
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
    runner = BnetRunner(
        net, RandomDriver(list(net.names)), mode=SamplingMode.STUTTERING)
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


def test_runner_async_uniform_matches_reference():
    net = parse_bnet(TINY)
    runner = BnetRunner(
        net, RandomDriver(list(net.names)), mode=SamplingMode.ASYNC_UNIFORM)

    PRG_SEED(90210)
    runner.init(state=0)
    walked = [runner.state]
    for _ in range(100):
        runner.step()
        walked.append(runner.state)

    PRG_SEED(90210)
    state = 0
    reference = [state]
    for _ in range(100):
        changes = net.potential_changes(state)
        r = 0.0 if len(changes) <= 1 else PRG_UNIFORM()
        idx = min(int(r * len(changes)), len(changes) - 1)
        state ^= 1 << changes[idx]
        reference.append(state)

    assert walked == reference


def test_runner_async_never_stutters():
    net = parse_bnet(TINY)
    runner = BnetRunner(
        net, RandomDriver(list(net.names)), mode=SamplingMode.ASYNC_UNIFORM)
    PRG_SEED(7)
    runner.init(state=0)
    prev = runner.state
    for _ in range(100):
        runner.step()
        assert runner.state != prev
        prev = runner.state


def test_runner_async_weighted_matches_reference():
    net = parse_bnet(TINY)
    pr = {"a": 0.1, "b": 0.9}
    runner = BnetRunner(
        net, RandomDriver(list(net.names)),
        mode=SamplingMode.ASYNC_WEIGHTED, weights=pr)
    assert runner.mode is SamplingMode.ASYNC_WEIGHTED

    PRG_SEED(12345)
    runner.init(state=0)
    walked = [runner.state]
    for _ in range(100):
        runner.step()
        walked.append(runner.state)

    PRG_SEED(12345)
    state = 0
    reference = [state]
    for _ in range(100):
        changes = net.potential_changes(state)
        r = 0.0 if len(changes) <= 1 else PRG_UNIFORM()
        weights = [pr[net.names[v]] for v in changes]
        target = r * sum(weights)
        cumulative = 0.0
        selected = len(weights) - 1
        for bit, weight in enumerate(weights):
            cumulative += weight
            if target < cumulative:
                selected = bit
                break
        state ^= 1 << changes[selected]
        reference.append(state)

    assert walked == reference


def test_runner_async_weighted_without_weights_falls_back(caplog):
    net = parse_bnet(TINY)
    with caplog.at_level("WARNING", logger="hoax"):
        runner = BnetRunner(
            net, RandomDriver(list(net.names)),
            mode=SamplingMode.ASYNC_WEIGHTED)
    assert runner.mode is SamplingMode.ASYNC_UNIFORM
    assert caplog.text.count("falling back to uniform") == 1


def test_runner_async_weighted_all_zero_raises():
    net = parse_bnet(TINY)
    with pytest.raises(ValueError, match="positive weight"):
        BnetRunner(
            net, RandomDriver(list(net.names)),
            mode=SamplingMode.ASYNC_WEIGHTED,
            weights={"a": 0.0, "b": 0.0})


def test_runner_uniform_async_rejects_weight_overrides():
    net = parse_bnet(TINY)
    with pytest.raises(ValueError, match="does not use"):
        BnetRunner(
            net, RandomDriver(list(net.names)),
            mode=SamplingMode.ASYNC_UNIFORM,
            weights={"a": 0.1})


def test_runner_weight_overrides_keep_driver_defaults():
    net = parse_bnet(TINY)
    drv = RandomDriver(list(net.names))
    drv.cum_weights = (0.2, 1)
    runner = BnetRunner(net, drv, weights={"a": 0.9})
    assert runner.pr == {"a": 0.9, "b": 0.2}


def test_runner_uses_flip_driver_weights_for_async_weighted():
    net = parse_bnet(TINY)
    drivers = CompositeDriver()
    a = RandomDriver(["a"])
    a.cum_weights = (0.2, 1)
    b = RandomDriver(["b"])
    b.cum_weights = (0.8, 1)
    drivers.append(a)
    drivers.append(b)

    runner = BnetRunner(net, drivers, mode=SamplingMode.ASYNC_WEIGHTED)
    assert runner.mode is SamplingMode.ASYNC_WEIGHTED
    assert runner.pr == {"a": 0.2, "b": 0.8}


def test_parse_weight_specs():
    assert parse_weight_specs(["a=0.1", "b=0.2"]) == {"a": 0.1, "b": 0.2}
    assert parse_weight_specs(["a=0.1", "a=0.2"]) == {"a": 0.2}
    assert parse_weight_specs(["a=0.1"], ["a", "b"]) == {"a": 0.1}


@pytest.mark.parametrize("spec", ["a", "=0.1", "a=", "a=x", "a=-0.1",
                                  "a=1.1", "a=nan", "a=inf"])
def test_parse_weight_specs_rejects_invalid(spec):
    with pytest.raises(ValueError):
        parse_weight_specs([spec])


def test_parse_weight_specs_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown"):
        parse_weight_specs(["z=0.1"], ["a", "b"])


def test_runner_rejects_invalid_weight_overrides():
    net = parse_bnet(TINY)
    with pytest.raises(ValueError, match="unknown"):
        BnetRunner(net, RandomDriver(list(net.names)), weights={"z": 0.1})
    with pytest.raises(ValueError, match="finite number"):
        BnetRunner(net, RandomDriver(list(net.names)), weights={"a": 1.1})
