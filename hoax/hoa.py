from collections import defaultdict
from pathlib import Path
from sys import intern
from typing import Any, Collection, Iterable, Optional, Sequence

import hoa.ast.boolean_expression as ast  # type: ignore
import hoa.ast.label as ast_label  # type: ignore
import networkit as nk  # type: ignore
from networkit import Graph
from hoa.core import HOA, Edge, State  # type: ignore
from hoa.parsers import HOAParser  # type: ignore


get_first_candidate = intern("get_first_candidate")
get_trap_set_of = intern("get_trap_set_of")


def fmt_state(s: State) -> str:
    if s.name is not None:
        return f"{s.index} ('{s.name}')"
    return str(s.index)


def fmt_edge(e: Edge, aps: list[str]) -> str:
    lbl = fmt_expr(e.label, aps)
    tgt = ' '.join(str(i) for i in e.state_conj)
    return f"{lbl} --> {tgt}"


def extract_aps(all_aps: Sequence[str], aps: Collection[str | int]) -> list[str]:  # noqa: E501
    """Return only the propositions from `all_aps` that are in `aps` \
        (by name or index)."""
    tmp = (i if type(i) is str else all_aps[int(i)] for i in aps)
    return [i for i in tmp if i in all_aps]


def fmt_expr(node, aps: list[str]) -> str:
    """Format an expression into a string."""
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


Transition = tuple[int, set, int]
"""A transition is a triple (source state, valuation, target state)"""


class Automaton:
    def __init__(self, aut: HOA, filename: Optional[str] = None, ctrl: Optional[set] = None):  # noqa: E501
        self.hoa = aut
        self.ctrl = ctrl or set()
        self.cache: dict[tuple, Any] = {}
        self.filename = filename
        self.states = max(x.index for x in aut.body.state2edges)
        self.int2edges: list[Sequence[Edge]] = [()] * (self.states + 1)
        for x, edges in aut.body.state2edges.items():
            self.int2edges[x.index] = edges

        self.graph, self.cond, self.graph_node2scc, self.cond_node2scc = self.graph_and_cond()  # noqa: E501

        # Compute accepting sets
        self.acc_sets = defaultdict(set)
        for x in self.hoa.body.state2edges:
            for s in (x.acc_sig or tuple()):
                self.acc_sets[s].add(x.index)

    def get_trap_set_of(self, index: int) -> tuple[set, bool]:
        key = (get_trap_set_of, index)
        try:
            return self.cache[key]
        except KeyError:
            k_id, k_nodes = self.graph_node2scc[index]
            is_minimal = self.cond.degree(k_id) == 0
            result = set(k_nodes)

            def callback(u):
                result.update(self.cond_node2scc[u])

            nk.graph.Traversal.DFSfrom(self.cond, k_id, callback)
            self.cache[key] = result, is_minimal
            return result, is_minimal

    def get_aps(self):
        """Yield all the atomic propositions of the automaton."""
        yield from self.hoa.header.propositions

    def get_states(self):
        pass

    def get_initial_states(self):
        yield from self.hoa.header.start_states

    def get_edges(self, index: int) -> Sequence[Edge]:
        """Yield all the edges from state `index`."""
        return self.int2edges[index]

    def evaluate(self, node, valuation: set) -> bool:
        """Evaluate an expression `node` against `valuation`."""
        match node:
            case ast_label.LabelAtom():
                return self.hoa.header.propositions[node.proposition] in valuation  # noqa: E501
            case ast.FalseFormula():
                return False
            case ast.TrueFormula():
                return True
            case ast.And(operands=ops):
                return all(self.evaluate(x, valuation) for x in ops)
            case ast.Or(operands=ops):
                return any(self.evaluate(x, valuation) for x in ops)
            case ast.Not(argument=arg):
                return not self.evaluate(arg, valuation)
        raise Exception(f"Unexpected node {node}")

    def get_candidates(self, index: int, values: set) -> Iterable[Edge]:
        """Yield all candidate edges rooted in `index` for a given valuation.

        Args:
            index (int): A state in the automaton.
            values (set): A valuation.

        Yields:
            Iterable[Edge]: Edges that `valuation` may trigger.
        """
        for edge in self.get_edges(index):
            if self.evaluate(edge.label, values):
                yield edge

    def get_first_candidate(self, index: int, values: set):
        """Return the first potential edge for the valuation `values`"""
        key = (get_first_candidate, index, *sorted(values))
        try:
            return self.cache[key]
        except KeyError:
            for edge in self.get_edges(index):
                if self.evaluate(edge.label, values):
                    self.cache[key] = edge
                    return edge
            self.cache[key] = None
            return self.cache[key]

    def graph_and_cond(self) -> tuple[Graph, Graph, dict, dict]:
        # Build digraph of automaton
        g = nk.Graph(directed=True)
        for i, edges in enumerate(self.int2edges):
            if edges is None:
                continue
            for e in edges:
                g.addEdge(i, e.state_conj[0], addMissing=True)
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
            src_k, tgt_k = g_node2scc[src][0], g_node2scc[tgt][0]
            if src_k != tgt_k:
                cond.addEdge(src_k, tgt_k, checkMultiEdge=True)
        return g, cond, g_node2scc, cond_node2scc


def parse(file: str) -> Automaton:
    """Wrapper around the base HOA parser."""
    __parser = HOAParser()
    input_string = Path(file).read_text()
    input_lines = Path(file).read_text().splitlines()
    input_string = "\n".join(
        line for line in input_lines
        if not line.startswith("controllable"))
    # Read in Strix controllable APs
    control_line = [x for x in input_lines if x.startswith("controllable")]
    if control_line:
        control_split = control_line[0].split(":")[1].split()
        control = set(int(x) for x in control_split)
    else:
        control = set()
    hoa_obj: HOA = __parser(input_string)
    return Automaton(hoa_obj, filename=file, ctrl=control)
