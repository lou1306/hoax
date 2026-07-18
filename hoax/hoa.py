from collections import defaultdict
from pathlib import Path
import re
from sys import intern
from os import PathLike
from typing import TYPE_CHECKING, Any, Collection, Iterable, Optional, Sequence

import hoa.ast.boolean_expression as ast  # type: ignore
import hoa.ast.label as ast_label  # type: ignore
import networkit as nk  # type: ignore
from networkit import Graph
from hoa.core import HOA, Edge, State  # type: ignore
from hoa.parsers import HOAParser  # type: ignore
import sympy  # type: ignore
from sympy.logic.boolalg import BooleanAtom, BooleanFunction  # type: ignore

from bnet2hoa.main import get_worker_fn  # type: ignore

from .util import Model


get_first_candidate = intern("get_first_candidate")
get_trap_set_of = intern("get_trap_set_of")


def fmt_state(s: State) -> str:
    if s.name is not None:
        return f"{s.index} ('{s.name}')"
    return str(s.index)


def fmt_edge(e: Edge, aps: list[str]) -> str:
    if isinstance(e.label, (BooleanAtom, BooleanFunction, sympy.Symbol)):
        lbl = e.label
    else:
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


Transition = tuple[int, set, str, int]
PartialTransition = tuple[int, Model, str, int]
"""A transition is a triple (source state, valuation, target state)"""


class Automaton:
    def __init__(self, aut: HOA, filename: Optional[str] = None, ctrl: Optional[set] = None):  # noqa: E501
        self.hoa = aut
        self.symbols: list[sympy.Symbol] = [sympy.symbols(ap) for ap in self.get_aps()]  # noqa: E501
        self.ctrl = ctrl or set()
        self.cache: dict[tuple, Any] = {}
        self.candidate_cache: dict[tuple, Edge | None] = {}
        self.filename = filename
        self.states = aut.header.nb_states
        if self.states is not None:
            self.int2edges: list[Sequence[Edge]] = [()] * (self.states + 1)
        if aut.body.state2edges is not None:
            for x, edges in aut.body.state2edges.items():
                self.int2edges[x.index] = edges

        self.graph, self.cond = None, None
        self.graph_node2scc, self.cond_node2scc = None, None

        # Compute accepting sets
        self.acc_sets = defaultdict(set)
        for x in self.hoa.body.state2edges:
            for s in (x.acc_sig or tuple()):
                self.acc_sets[s].add(x.index)

    def init_monitor_structures(self):
        if self.graph is None:
            self.graph, self.cond, self.graph_node2scc, self.cond_node2scc = self.graph_and_cond()  # noqa: E501

    def get_trap_set_of(self, index: int) -> tuple[set, bool]:
        self.init_monitor_structures()
        key = (get_trap_set_of, index)
        try:
            return self.cache[key]
        except KeyError:
            assert self.graph_node2scc is not None
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

    def get_edges(self, index: int):
        """Yield all the edges from state `index`."""
        yield from self.int2edges[index]

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
        subs = {sym: (sym.name in values) for sym in self.symbols}  # noqa: E501
        for edge in self.get_edges(index):
            if isinstance(edge.label, (BooleanFunction, BooleanAtom, sympy.Symbol)):  # noqa: E501
                if edge.label.subs(subs) is sympy.true:
                    yield edge
            elif self.evaluate(edge.label, values):
                yield edge

    def get_first_candidate(self, index: int, values: set):
        """Return the first potential edge for the valuation `values`"""
        key = (get_first_candidate, index, *sorted(values))
        try:
            return self.candidate_cache[key]
        except KeyError:
            for edge in self.get_edges(index):
                if self.evaluate(edge.label, values):
                    self.candidate_cache[key] = edge
                    return edge
            self.candidate_cache[key] = None
            return self.candidate_cache[key]

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


def to_sympy(node, symbols=list[sympy.core.symbol.Symbol]) -> sympy.core.expr.Expr:  # noqa: E501
    """Turn an AST node into a Sympy expression."""
    match node:
        case ast_label.LabelAtom():
            return symbols[node.proposition]
        case ast.FalseFormula():
            return False
        case ast.TrueFormula():
            return True
        case ast.And(operands=ops):  # type: ignore
            return sympy.And(*(to_sympy(x, symbols) for x in ops))
        case ast.Or(operands=ops):
            return sympy.Or(*(to_sympy(x, symbols) for x in ops))
        case ast.Not(argument=arg):
            return ~to_sympy(arg, symbols)
    raise Exception(f"Unexpected node {node}")


class LazyHOA:
    int_re = re.compile(r"(\d+)")

    def __init__(self, states: int, header: str, states_txts: Sequence[str]):
        self.int2edges: dict[int, (Sequence[Edge] | None)] = defaultdict(lambda: None)  # noqa: E501
        self.header = header
        self.num_states = states
        self.states = {}
        for s in states_txts:
            m = LazyHOA.int_re.search(s)
            if m is None:
                raise Exception(f"Unexpected state line {s}")
            index = int(m.group(1))
            self.states[index] = s

    def __getitem__(self, index: int) -> Sequence[Edge]:
        if index >= self.num_states:
            raise StopIteration
        if self.int2edges[index] is None:
            aut = LazyAutomaton._parser(
                "HOA:v1\n"
                "Acceptance: 0 t\n"
                f"--BODY--\nState: {self.states[index]}\n--END--")
            _, edges = aut.body.state2edges.popitem()
            self.int2edges[index] = edges
        result = self.int2edges[index]
        if TYPE_CHECKING:
            assert result is not None
        return result

    def __iter__(self):
        for i in range(self.num_states):
            yield self[i]


class LazyBNet:

    def __init__(self, fname: PathLike):
        self.int2edges: dict[int, (Sequence[Edge] | None)] = defaultdict(lambda: None)  # noqa: E501
        self.worker, aps = get_worker_fn(fname, allow_stuttering=True)  # noqa: E501
        self.aps = tuple(aps)

        self.header_hoa = LazyAutomaton._parser(
            "HOA:v1\n"
            f"Acceptance: 0 t\n"
            f"""AP: {len(self.aps)} {" ".join(f'"{ap}"' for ap in self.aps)}"""
            "--BODY--\n--END--")

        self.symbols: list[sympy.Symbol] = [sympy.symbols(ap) for ap in self.aps]  # noqa: E501

    def var_to_symbol(self, i: int) -> str:
        if i == 0:
            return sympy.true
        return self.symbols[i-1] if i > 0 else ~self.symbols[-i-1]

    def dnf_to_sympy(self, dnf: Sequence[Sequence[int]]) -> sympy.core.expr.Expr:
        return sympy.Or(*(
            sympy.And(*(self.var_to_symbol(i) for i in clause))
            for clause in dnf))

    def make_edge(self, s: int, dnf: list[list[int]]) -> Edge:
        lbl = self.dnf_to_sympy(dnf)
        return Edge(state_conj=(s,), label=lbl, acc_sig=())

    def __getitem__(self, index: int):
        if self.int2edges[index] is not None:
            return self.int2edges[index]
        tr = self.worker(index)
        self.int2edges[index] = [self.make_edge(k, v) for k, v in tr.items()]

        result = self.int2edges[index]
        if TYPE_CHECKING:
            assert result is not None
        return result


class LazyAutomaton(Automaton):
    _parser = HOAParser()

    def __init__(self, aut, i2e, filename=None, ctrl=None):
        super().__init__(aut, filename, ctrl)
        self.int2edges = i2e

    @staticmethod
    def from_file(file: PathLike) -> "LazyAutomaton":
        content = Path(file).read_text()
        header, body = content.split("--BODY--")
        header_hoa = LazyAutomaton._parser(header + "--BODY--\n--END--")
        states_txts = body.split("State: ")[1:]
        states_txts[-1] = states_txts[-1][:states_txts[-1].rfind("--END--")]
        states = header_hoa.header.nb_states or len(states_txts)
        i2e = LazyHOA(states, header, states_txts)

        return LazyAutomaton(header_hoa, i2e, filename=file)

    def from_bnet(file: PathLike) -> "LazyAutomaton":
        i2e = LazyBNet(file)
        aut = LazyAutomaton(i2e.header_hoa, i2e, filename=file)
        aut.states = 2**len(i2e.aps)
        return aut
