from abc import ABC
from dataclasses import dataclass
from functools import reduce
from operator import and_, or_

import hoa.ast.boolean_expression as ast
import hoa.ast.label as ast_label
from hoa.core import HOA, Edge, State


def fmt_state(s: State) -> str:
    if s.name is not None:
        return f"{s.index} ('{s.name}')"
    return str(s.index)


def fmt_expr(node, aps: list[str]):
    def recurse_and_reduce(op: str):
        return op.join(fmt_expr(x, aps) for x in node.operands)
    if isinstance(node, ast_label.LabelAtom):
        return aps[node.proposition]
    if isinstance(node, ast.FalseFormula):
        return "false"
    if isinstance(node, ast.TrueFormula):
        return "true"
    if isinstance(node, ast.And):
        return recurse_and_reduce(" & ")
    if isinstance(node, ast.Or):
        return recurse_and_reduce(" | ")
    if isinstance(node, ast.Not):
        return "!" + fmt_expr(node.argument, aps)
    raise Exception(f"Unexpected node {node}")


@dataclass(frozen=True)
class AbstractTransition(ABC):
    src: State
    tgt: State


@dataclass(frozen=True)
class Transition(AbstractTransition):
    edge: Edge
    aps: list[str]

    def __str__(self) -> str:
        return f"{fmt_state(self.src)} -- {fmt_expr(self.edge.label, self.aps)} --> {fmt_state(self.tgt)}"  # noqa: E501


@dataclass(frozen=True)
class ForcedTransition(AbstractTransition):
    label: str

    def __str__(self) -> str:
        return f"{fmt_state(self.src)} -- {self.label} --> {fmt_state(self.tgt)}"  # noqa: E501


class Automaton:
    def __init__(self, aut: HOA) -> None:
        self.aut = aut
        self.int2states = {x.index: x for x in aut.body.state2edges}

    def get_aps(self):
        yield from self.aut.header.propositions

    def get_states(self):
        pass

    def get_initial_states(self):
        yield from self.aut.header.start_states

    def get_edges(self, state: State):
        yield from self.aut.body.state2edges[state]

    def evaluate(self, node, valuation):
        # TODO support aliases
        def recurse_and_reduce(op):
            return reduce(
                op, (self.evaluate(x, valuation) for x in node.operands))
        if isinstance(node, ast_label.LabelAtom):
            return valuation[node.proposition]
        if isinstance(node, ast.FalseFormula):
            return False
        if isinstance(node, ast.TrueFormula):
            return False
        if isinstance(node, ast.And):
            return recurse_and_reduce(and_)
        if isinstance(node, ast.Or):
            return recurse_and_reduce(or_)
        if isinstance(node, ast.Not):
            return not self.evaluate(node.argument, valuation)
        raise Exception(f"Unexpected node {node}")

    def get_candidates(self, state: State, inputs: dict):
        values = [inputs[x] for x in self.aut.header.propositions]
        for edge in self.get_edges(state):
            if self.evaluate(edge.label, values):
                yield edge
