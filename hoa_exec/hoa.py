from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Collection, Iterable

import hoa.ast.boolean_expression as ast
import hoa.ast.label as ast_label
import networkit as nk
from networkit import Graph
from hoa.core import HOA, Edge, State
from hoa.parsers import HOAParser


def fmt_state(s: State) -> str:
    if s.name is not None:
        return f"{s.index} ('{s.name}')"
    return str(s.index)


def fmt_edge(e: Edge, aps: list[str]) -> str:
    lbl = fmt_expr(e.label, aps)
    tgt = ' '.join(str(i) for i in e.state_conj)
    return f"{lbl} --> {tgt}"


def extract_aps(all_aps: Iterable[str], aps: Collection[str | int]):
    tmp = (i if type(i) is str else all_aps[i] for i in aps)
    return [i for i in tmp if i in all_aps]


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
    def __init__(self, aut: HOA, filename: str = None, ctrl: set = None):
        self.aut = aut
        self.ctrl = ctrl
        self.cache = {}
        self.filename = filename
        self.int2states = {x.index: x for x in aut.body.state2edges}
        self.graph, self.cond, self.graph_node2scc, self.cond_node2scc = self.graph_and_cond()  # noqa: E501
        # Compute accepting sets
        self.acc_sets = defaultdict(set)
        for x in self.aut.body.state2edges:
            for s in (x.acc_sig or tuple()):
                self.acc_sets[s].add(x.index)

    def get_trap_set_of(self, index: int) -> tuple[set, bool]:
        k_id, k_nodes = self.graph_node2scc[index]
        is_minimal = self.cond.degree(k_id) == 0
        result = set(k_nodes)

        def callback(u):
            result.update(self.cond_node2scc[u])

        nk.graph.Traversal.DFSfrom(self.cond, k_id, callback)
        return result, is_minimal

    def get_aps(self):
        yield from self.aut.header.propositions

    def get_states(self):
        pass

    def get_initial_states(self):
        yield from self.aut.header.start_states

    def get_edges(self, state: State):
        yield from self.aut.body.state2edges[state]

    @staticmethod
    @lru_cache(maxsize=2048)
    def evaluate(node, valuation):
        match node:
            case ast_label.LabelAtom():
                return valuation[node.proposition]
            case ast.FalseFormula():
                return False
            case ast.TrueFormula():
                return True
            case ast.And(operands=ops):
                return all(Automaton.evaluate(x, valuation) for x in ops)
            case ast.Or(operands=ops):
                return any(Automaton.evaluate(x, valuation) for x in ops)
            case ast.Not(argument=arg):
                return not Automaton.evaluate(arg, valuation)
        raise Exception(f"Unexpected node {node}")

    def get_candidates(self, state: State, values: tuple):
        for edge in self.get_edges(state):
            if self.evaluate(edge.label, values):
                yield edge

    def get_first_candidate(self, state: State, values: tuple):
        key = ("get_first_candidate", state.index, values)
        if key not in self.cache:
            try:
                val = next(self.get_candidates(state, values))
            except StopIteration:
                val = None
            self.cache[key] = val
        return self.cache[key]

    def graph_and_cond(self) -> tuple[Graph, Graph, dict, dict]:
        # Build digraph of automaton
        states = len(self.int2states)
        g = nk.Graph(states, directed=True)
        
        for s, edges in self.aut.body.state2edges.items():
            for e in edges:
                g.addEdge(s.index, e.state_conj[0])
        # Compute SCCs

        sccs = nk.components.StronglyConnectedComponents(g)
        sccs.run()
        # Compute condensation of g
        cond = nk.Graph(directed=True)
        g_node2scc = {}
        cond_node2scc = {}
        for k in sccs.getComponents():
            s = cond.addNode()
            for node in k:
                g_node2scc[node] = s, k
                cond_node2scc[s] = k

        for (src, tgt) in g.iterEdges():
            src_k, tgt_k =  g_node2scc[src][0], g_node2scc[tgt][0]
            if src_k != tgt_k:
                cond.addEdge(src_k, tgt_k, checkMultiEdge=True)
        return g, cond, g_node2scc, cond_node2scc


def parse(file: str) -> Automaton:
    __parser = HOAParser()
    input_string = Path(file).read_text()
    input_lines = Path(file).read_text().splitlines()
    input_string = "\n".join(
        line for line in input_lines
        if not line.startswith("controllable"))
    # Read in Strix controllable APs
    control = [x for x in input_lines if x.startswith("controllable")]
    if control:
        control = control[0].split(":")[1].split()
        control = set(int(x) for x in control)
    else:
        control = set()
    hoa_obj: HOA = __parser(input_string)
    return Automaton(hoa_obj, filename=file, ctrl=control)
