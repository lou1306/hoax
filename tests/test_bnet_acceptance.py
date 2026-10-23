from pathlib import Path

import pytest
from lark import Lark
from lark.exceptions import UnexpectedInput

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
