from functools import reduce
from operator import and_, or_

import hoa.ast.boolean_expression as ast
import hoa.ast.label as ast_label
from hoa.core import HOA
from hoa.parsers import State


class Runner:
    def __init__(self, aut: HOA, s0: State, driver, stepper) -> None:
        self.aut = aut
        self.state = s0
        self.driver = driver()
        self.stepper = stepper()
        self.int2states = {x.index: x for x in aut.body.state2edges}

    def step(self):
        inputs: dict = self.driver()
        next_index: int = self.stepper(inputs, self.state)
        self.state = self.int2states[next_index]


def stutter(state):
    return state.index


def fail(state):
    raise Exception(f"No successor at {state}")


def first_match(aut: HOA, state: State, values: dict, on_fail=stutter) -> int:
    next_state = state
    found = False
    # Sort values
    values = [values[x] for x in aut.header.propositions]
    for edge in aut.body.state2edges[state]:
        if eval(edge.label, values):
            # TODO suport multiple states (alternating automata)
            found = True
            next_state = next(iter(edge.state_conj))
            break
    if not found:
        next_state = on_fail(state)
    return next_state


def eval(node, valuation):
    # TODO support aliases
    def recurse_and_reduce(op):
        return reduce(op, (eval(x, valuation) for x in node.operands))
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
        return not eval(node.argument, valuation)
    raise Exception(f"Unexpected node {node}")
