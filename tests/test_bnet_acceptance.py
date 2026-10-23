import subprocess
import sys
from pathlib import Path

import pytest
from lark import Lark
from lark.exceptions import UnexpectedInput

from hoax.bnet import parse_bnet
from hoax.bnet_acceptance import And, Fin, Inf, Or, parse_condition
from hoax.bnet_runner import BnetRunner
from hoax.drivers import RandomDriver
from hoax.verdict import PrefixType

GRAMMAR = Path(__file__).parent.parent / "hoax" / "bnet_acceptance.lark"


@pytest.fixture(scope="module")
def parser():
    # Constructing the parser with parser="lalr" also verifies that the
    # grammar has no LR conflicts.
    return Lark(GRAMMAR.read_text(), parser="lalr", start="start")


def test_fin_inf_and_pattern_list(parser):
    tree = parser.parse("Fin(0x1, 10x) & Inf(1xx)")
    root = tree.children[0]
    assert root.data == "and_"
    left, right = root.children
    assert left.data == "fin"
    assert right.data == "inf"
    assert [p.children[0].value for p in left.children[0].children] == [
        "0x1", "10x"]
    assert [p.children[0].value for p in right.children[0].children] == [
        "1xx"]


def test_hoa_right_associative_equal_precedence(parser):
    # The HOA acceptance parser treats "&" and "|" as equal-precedence,
    # right-associative operators.  Parentheses override this.
    tree = parser.parse("Fin(0) & Inf(1) | Fin(10)")
    root = tree.children[0]
    assert root.data == "and_"
    assert root.children[1].data == "or_"

    tree = parser.parse("(Fin(0) & Inf(1)) | Fin(10)")
    root = tree.children[0]
    assert root.data == "or_"
    assert root.children[0].data == "and_"


@pytest.mark.parametrize("text", [
    "Fin()",
    "Fin(0x1,)",
    "Fin(2)",
    "!Fin(0x1)",
    "Fin(0x1",
])
def test_rejects_invalid_conditions(parser, text):
    with pytest.raises(UnexpectedInput):
        parser.parse(text)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def make_runner(net, state, weights=None):
    runner = BnetRunner(
        net, RandomDriver(list(net.names)), weights=weights)
    runner.init(state=state)
    return runner


def test_parse_condition_order_and_width():
    net = parse_bnet("targets, factors\nb, a\na, !a\n")
    cond = parse_condition("Fin(01)", net)
    assert isinstance(cond, Fin)
    # Sorted order is a, b: 01 means a=0, b=1, i.e. state 1.
    assert cond.patterns[0].matches(1)
    assert not cond.patterns[0].matches(2)

    # File order is b, a: 01 now means b=0, a=1, i.e. state 2.
    cond = parse_condition("Fin(01)", net, order=net.names)
    assert cond.patterns[0].matches(2)
    assert not cond.patterns[0].matches(1)

    with pytest.raises(ValueError):
        parse_condition("Fin(0)", net)
    with pytest.raises(ValueError):
        parse_condition("Fin(00)", net, order=["a", "a"])


def test_fin_frozen_mismatch():
    # a toggles, b is frozen at 0.
    net = parse_bnet("targets, factors\na, !a\nb, b\n")
    runner = make_runner(net, 0)

    assert parse_condition("Fin(x1)", net).check(runner) is PrefixType.GOOD
    assert parse_condition("Fin(10)", net).check(runner) is None
    assert parse_condition("Fin(00)", net).check(runner) is None
    assert parse_condition("Fin(0x)", net).check(runner) is None


def test_inf_frozen_invariant():
    net = parse_bnet("targets, factors\na, !a\nb, b\n")
    runner = make_runner(net, 0)

    assert parse_condition("Inf(x0)", net).check(runner) is PrefixType.GOOD
    assert parse_condition("Inf(0x)", net).check(runner) is None
    assert parse_condition("Inf(x1)", net).check(runner) is PrefixType.BAD
    assert parse_condition("Inf(1x)", net).check(runner) is None


def test_fin_bad_inside_invariant():
    # b is frozen at 0, so Fin(x0) is violated by staying in the pattern.
    net = parse_bnet("targets, factors\na, !a\nb, b\n")
    runner = make_runner(net, 0)

    assert parse_condition("Fin(x0)", net).check(runner) is PrefixType.BAD
    assert parse_condition("Inf(x0)", net).check(runner) is PrefixType.GOOD


def test_fin_uses_dependency_closure():
    # b is stable now but can become unstable after a fires, so it is not
    # frozen and Fin(01) is not decided.
    net = parse_bnet("targets, factors\na, !a\nb, a\n")
    runner = make_runner(net, 0)
    assert parse_condition("Fin(01)", net).check(runner) is None

    runner = make_runner(net, 0, weights={"b": 0.0})
    assert parse_condition("Fin(01)", net).check(runner) is PrefixType.GOOD


def test_terminal_classification():
    net = parse_bnet("targets, factors\na, a\nb, b\n")
    runner = make_runner(net, 0)

    fin = parse_condition("Fin(00)", net)
    inf = parse_condition("Inf(00)", net)
    assert fin.terminal(runner.state) is PrefixType.BAD
    assert inf.terminal(runner.state) is PrefixType.GOOD

    fin = parse_condition("Fin(01)", net)
    inf = parse_condition("Inf(01)", net)
    assert fin.terminal(runner.state) is PrefixType.GOOD
    assert inf.terminal(runner.state) is PrefixType.BAD


def test_combinators_are_conservative():
    net = parse_bnet("targets, factors\na, !a\nb, b\n")
    runner = make_runner(net, 0)

    assert parse_condition(
        "Fin(x1) & Inf(x0)", net).check(runner) is PrefixType.GOOD
    assert parse_condition(
        "Fin(10) & Inf(x0)", net).check(runner) is None
    assert parse_condition(
        "Fin(10) | Inf(x0)", net).check(runner) is PrefixType.GOOD
    assert parse_condition(
        "Fin(x1) | Inf(x1)", net).check(runner) is PrefixType.GOOD

    assert isinstance(parse_condition("Fin(x1)", net), Fin)
    assert isinstance(parse_condition("Inf(x0)", net), Inf)
    assert isinstance(parse_condition("Fin(x1) & Inf(x0)", net), And)
    assert isinstance(parse_condition("Fin(x1) | Inf(x0)", net), Or)



def test_runtime_parser_does_not_import_lark():
    code = """
import sys

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "lark" or name.startswith("lark."):
            raise ImportError("lark must not be imported at runtime")

sys.meta_path.insert(0, Blocker())
from hoax.bnet import parse_bnet
from hoax.bnet_acceptance import parse_condition

net = parse_bnet("targets, factors\\nb, a\\na, !a\\n")
cond = parse_condition("Fin(01)", net)
assert cond.patterns[0].matches(1)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parent.parent, check=True)
