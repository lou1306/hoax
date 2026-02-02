from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from itertools import chain, product
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

import msgpack  # type: ignore
import networkit as nk  # type: ignore
import sympy  # type: ignore
from bidict import MutableBidirectionalMapping, bidict
from hoa.ast.acceptance import AcceptanceAtom, AtomType  # type: ignore
from hoa.ast.boolean_expression import PositiveAnd, PositiveOr  # type: ignore
from networkit.graph import Graph  # type: ignore
from sympy.logic.inference import satisfiable  # type: ignore
from sympy.utilities.autowrap import autowrap  # type: ignore

from .drivers import Driver, RandomDriver, UserDriver
from .hoa import (Automaton, PartialTransition, Transition, fmt_edge,
                  fmt_state, parse, to_sympy)
from .util import PRG_BOUNDED, allsat, includes, pick, powerset


class Action(ABC):
    @abstractmethod
    def run(self, runner: "SingleRunner") -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        return type(self).__name__


class Condition(ABC):
    @abstractmethod
    def check(self, runner: "SingleRunner"):
        pass

    def __repr__(self) -> str:
        return f"{self} (at {hex(id(self))})"


class StopRunner(Exception):
    pass


class Runner(ABC):
    @abstractmethod
    def init(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, values=Optional[set]) -> Iterable[Transition | PartialTransition]:  # noqa: E501
        raise NotImplementedError

    @abstractmethod
    def add_transition_hook(self, hook):
        raise NotImplementedError

    @abstractmethod
    def add_nondet_action(self, action):
        raise NotImplementedError

    @abstractmethod
    def add_deadlock_action(self, action):
        raise NotImplementedError

    @property
    @abstractmethod
    def count(self):
        raise NotImplementedError

    @staticmethod
    def factory(a: Sequence[Automaton], drv: Driver, mon: bool = False) -> "Runner":  # noqa: E501
        if len(a) > 1:
            return CompositeRunner(a, drv, mon)
        aut = a[0]
        prp = aut.hoa.header.properties or []
        if all(x in prp for x in ("complete", "deterministic")):
            return DetCompleteSingleRunner(aut, drv, mon)
        return (
            AllsatRunner(aut, drv, mon)
            if all(type(d) is RandomDriver for d in drv.get_drivers())
            else SingleRunner(aut, drv, mon))


def spawn_runner(pipe: Connection):  # type: ignore
    fname, monitor = pipe.recv(), pipe.recv()
    aut = parse(fname)
    runner = SympyRunner(aut, UserDriver([]), monitor)
    while True:
        data = pipe.recv_bytes()
        msg = msgpack.loads(data)
        match msg:
            case "INIT":
                runner.init()
            case "QUIT":
                break
            case "GET_COUNT":
                pipe.send(runner.count)
            case set(values):
                x = runner.step(values)
                pipe.send_bytes(msgpack.dumps(x))
                # pipe.send(x)
            case ("ADD_HOOK", hook):
                runner.add_transition_hook(hook)
            case ("ADD_NONDET", action):
                x = runner.add_nondet_action(action)
            case ("ADD_DEADLOCK", action):
                x = runner.add_deadlock_action(action)


class MPCompositeRunner(Runner):
    INIT_MSG = msgpack.dumps("INIT")
    QUIT_MSG = msgpack.dumps("QUIT")
    GET_COUNT_MSG = msgpack.dumps("GET_COUNT")

    def __init__(self, automata: Sequence[Automaton], drv: Driver,
                 monitor: bool = False) -> None:
        self.driver = drv
        self.runners: list[Runner] = []
        self.procs = []
        for a in automata:
            parent_pipe, child_pipe = Pipe()
            proc = Process(target=spawn_runner, args=(child_pipe,))
            self.procs.append((proc, parent_pipe))
            parent_pipe.send(a.filename)
            parent_pipe.send(monitor)
            proc.start()

    def __del__(self):
        for proc, pipe in self.procs:
            pipe.close()
            proc.kill()

    @property
    def count(self):
        _, pipe = self.procs[0]
        pipe.send_bytes(MPCompositeRunner.GET_COUNT_MSG)
        return pipe.recv()

    def init(self):
        for _, pipe in self.procs:
            pipe.send_bytes(MPCompositeRunner.INIT_MSG)

    def step(self, inputs: Optional[set] = None):
        values = inputs or self.driver.get()
        msg = msgpack.dumps(values)
        for _, pipe in self.procs:
            pipe.send_bytes(msg)
        result = [tr for _, pipe in self.procs for tr in msgpack.loads(pipe.recv_bytes())]  # noqa: E501
        return result

    def add_transition_hook(self, hook):
        msg = msgpack.dumps(("ADD_HOOK", hook))
        for _, pipe in self.procs:
            pipe.send_bytes(msg)

    def add_nondet_action(self, action):
        msg = msgpack.dumps(("ADD_NONDET", action))
        for _, pipe in self.procs:
            pipe.send_bytes(msg)

    def add_deadlock_action(self, action):
        msg = msgpack.dumps(("ADD_DEADLOCK", action))
        for _, pipe in self.procs:
            pipe.send_bytes(msg)


class CompositeRunner(Runner):
    """Runner for multiple automata, fed by the same driver."""
    def __init__(self, automata: Sequence[Automaton], drv: Driver,
                 monitor: bool = False) -> None:
        self.driver = drv
        self.runners: list[Runner] = []
        for a in automata:
            self.runners.append(Runner.factory((a,), drv, monitor))

    @property
    def count(self):
        return self.runners[0].count

    def init(self) -> None:
        for runner in self.runners:
            runner.init()

    def step(self, inputs: Optional[set] = None) -> Iterable[Transition | PartialTransition]:  # noqa: E501
        values = inputs or self.driver.get()
        return [tr for r in self.runners for tr in r.step(values)]


    def add_transition_hook(self, hook):
        for runner in self.runners:
            runner.add_transition_hook(hook)

    def add_nondet_action(self, action):
        for runner in self.runners:
            runner.add_nondet_action(action)

    def add_deadlock_action(self, action):
        for runner in self.runners:
            runner.add_deadlock_action(action)


class SingleRunner(Runner):
    """Runner for a single automaton."""
    def __init__(self, aut: Automaton, drv: Driver, mon: bool = False) -> None:
        self.aut = aut
        self.aps = list(aut.get_aps())
        self.driver = drv
        self.state: int | None = None
        self.count = 0
        # self.trace = []
        self.deadlock_actions: list[Action] = []
        self.nondet_actions: list[Action] = []
        self.transition_hooks: list[Hook] = []
        self.candidates: list[tuple[int, str]] = []
        prp = self.aut.hoa.header.properties or []
        self.deterministic = "deterministic" in prp
        # TODO make this configurable
        if mon:
            chk = AcceptanceChecker.make_checker(self.aut)
            self.transition_hooks.append(Hook(chk, Reset()))

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, value: int):
        self._count = value

    def init(self) -> None:
        # TODO support initial state conjunction (alternating automata)
        # TODO support nondeterministic initial states
        for x in self.aut.hoa.header.start_states:
            for y in x:
                self.state = y
                assert self.state is not None
                return

    def add_transition_hook(self, hook):
        self.transition_hooks.append(hook)

    def add_nondet_action(self, action):
        self.nondet_actions.append(action)

    def add_deadlock_action(self, action):
        self.deadlock_actions.append(action)

    def step(self, inputs: Optional[set] = None) -> Iterable[Transition | PartialTransition]:  # noqa: E501
        if TYPE_CHECKING:
            assert self.state is not None
        if inputs is None:
            inputs = self.driver.get()
        if self.deterministic:
            candidate = self.aut.get_first_candidate(self.state, inputs)
            if candidate is not None:
                self.candidates = [
                    (candidate.state_conj[0], fmt_edge(candidate, self.aps))]
            else:
                self.candidates = []
        else:
            candidates = list(self.aut.get_candidates(self.state, inputs))
            if not candidates:
                for action in self.deadlock_actions:
                    action.run(self)
                return ()
            self.candidates = [
                (edge.state_conj[0], fmt_edge(edge, self.aps))
                for edge in candidates]
        if len(self.candidates) > 1:
            for action in self.nondet_actions:
                action.run(self)
        old_state, next_state = self.state, self.candidates[0][0]

        self.count += 1
        self.state = next_state
        for hook in self.transition_hooks:
            hook.run(self)
        return ((old_state, inputs, "[" + " ".join(inputs) + "]", next_state),)


class DetCompleteSingleRunner(SingleRunner):
    """Optimized `SingleRunner` for deterministic complete automata."""
    def step(self, inputs: Optional[set] = None) -> Iterable[Transition]:
        if TYPE_CHECKING:
            assert self.state is not None
        if inputs is None:
            inputs = self.driver.get()
        edge = self.aut.get_first_candidate(self.state, inputs)
        next_state = edge.state_conj[0]
        self.count += 1
        old_state, self.state = self.state, next_state
        for hook in self.transition_hooks:
            hook.run(self)
        return ((old_state, inputs, "[" + " ".join(inputs) + "]", next_state),)


class Reach(Condition):
    """Condition triggered by reaching a target state."""
    def __str__(self) -> str:
        return f"Reach {self.target}"

    def __init__(self, target: int) -> None:
        self.target = target

    def check(self, runner: SingleRunner):
        return runner.state == self.target


class Bound(Condition):
    """Condition triggered when an execution reaches a certain length."""
    def __str__(self) -> str:
        return f"Bound: {self.bound}"

    def __init__(self, bound) -> None:
        self.bound = bound

    def check(self, runner: SingleRunner):
        return runner.count > self.bound


class Always(Condition):
    """Trivial condition, always triggered."""
    def check(self, _):
        return True


class Hook:
    """A Hook is a combination of a triggering condition and an action."""
    def __init__(self, condition: Condition, action: Action) -> None:
        self.condition = condition
        self.action = action

    def run(self, runner: SingleRunner):
        if self.condition.check(runner):
            msg = f"Hook: {self.condition} triggered {self.action} at step {runner.count}"  # noqa: E501
            print(msg)
            self.action.run(runner)


class Reset(Action):
    """Reset the runner to its initial state."""
    def run(self, runner: SingleRunner) -> None:
        runner.init()


class Log(Action):
    """Write a message to standard output."""
    def __init__(self, msg: str) -> None:
        self.msg = msg

    def __str__(self) -> str:
        return f"Log {self.msg}"

    def run(self, runner: SingleRunner) -> None:
        print(f"{self.msg} at {fmt_state(runner.state)}")


class PressEnter(Action):
    """Ask for user confirmation before proceeding."""
    def run(self, _) -> None:
        input("Press [Enter] to continue...")


class RandomChoice(Action):
    """Randomly choose a candidate."""
    def run(self, runner: SingleRunner) -> None:
        idx = PRG_BOUNDED(len(runner.candidates))
        runner.candidates[0], runner.candidates[idx] = runner.candidates[idx], runner.candidates[0]  # noqa: E501


class UserChoice(Action):
    """Let the user choose a candidate."""
    def run(self, runner: SingleRunner) -> None:
        for i, candidate in enumerate(runner.candidates):
            print(f"[{i}]\t{candidate}")
            # print(f"[{i}]\t{fmt_edge(edge, runner.aps)}")
        choice_int = -1
        while not 0 <= choice_int < len(runner.candidates):
            choice = input("Choose a transition from above: ")
            choice_int = int(choice) if choice.isdecimal() else -1
        runner.candidates[0], runner.candidates[choice_int] = runner.candidates[choice_int], runner.candidates[0]  # noqa: E501


class Quit(Action):
    """Terminate the runner (and HOAX)."""
    def __init__(self, cause=None):
        super().__init__()
        self.cause = cause

    def run(self, runner):
        raise StopRunner(self.cause)


class PrefixType(Enum):
    GOOD = 1
    BAD = 2
    UGLY = 3

    def invert(self) -> "PrefixType":
        if self == PrefixType.GOOD:
            return PrefixType.BAD
        if self == PrefixType.BAD:
            return PrefixType.GOOD
        return self


class AcceptanceChecker(Condition):
    """Checks for satisfaction of an acceptance condition.

    For details, see: https://doi.org/10.48550/arXiv.2507.11126
    """
    @abstractmethod
    def check(self, runner: SingleRunner) -> PrefixType | None:
        pass

    @staticmethod
    def make_checker(aut: Automaton) -> "AcceptanceChecker":
        """Generate an acceptance checker from the automaton's acceptance \
            condition.

        Args:
            aut (Automaton): A HOA automaton

        Returns:
            AcceptanceChecker: An acceptance checker
        """
        acond = aut.hoa.header.acceptance.condition
        all_states = aut.states

        def get_uglies(accept: set[int]):
            result = set()
            for k_id in aut.cond.iterNodes():
                if aut.cond.degree(k_id) > 0:
                    continue
                k = set(aut.graph_node2scc[k_id][1])
                k_minus_accept = k - accept
                if not k_minus_accept:
                    continue
                # Build graph for k_minus_accept
                graph = Graph(directed=True)

                for e in aut.graph.iterEdges():
                    if e[0] in k_minus_accept and e[1] in k_minus_accept:
                        graph.addEdge(*e, addMissing=True)
                for x in accept:
                    graph.removeNode(x)

                # Check whether k_minus_accept is acyclic
                seen = set()
                for node in k_minus_accept:
                    # Skip nodes visited in a previous DFS
                    if node in seen:
                        continue
                    dfs = []

                    def callback(u):
                        dfs.append(u)
                        for v in graph.iterNeighbors(u):
                            dfs.extend(v)
                            seen.add(v)

                    nk.graph.Traversal.DFSfrom(graph, node, callback)
                    if len(dfs) != len(set(dfs)):
                        result.add(k_minus_accept)
                        break
            return result

        def _mk(cond):
            def get_acceptance_set(index: int):
                accept = aut.acc_sets[index]
                if cond.negated:
                    accept = all_states - accept
                return frozenset(accept)
            match cond:
                case AcceptanceAtom(atom_type=AtomType.INFINITE):
                    accept = get_acceptance_set(cond.acceptance_set)
                    return Inf(accept, aut, get_uglies(accept))
                case AcceptanceAtom(atom_type=AtomType.FINITE):
                    accept = get_acceptance_set(cond.acceptance_set)
                    return Fin(accept, aut, get_uglies(accept))
                case PositiveAnd():
                    return And([_mk(c) for c in cond.operands])
                case PositiveOr():
                    return Or([_mk(c) for c in cond.operands])
        chk = _mk(acond)
        chk.set_filename(aut.filename)
        return chk


class BaseChecker(AcceptanceChecker):
    """Base class for Fin/Inf checkers."""
    def __init__(self, aset: set[int], aut: Automaton, uglies):
        self.aset = aset
        self.aut = aut
        self.uglies = uglies
        self.name: str | None = None
        self.cache: dict = {}

    def set_filename(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def check_state(self, state: int) -> PrefixType | None:
        raise NotImplementedError

    def check(self, runner: SingleRunner) -> PrefixType | None:
        assert runner.state is not None
        try:
            return self.cache[runner.state]
        except KeyError:
            value = self.check_state(runner.state)
            self.cache[runner.state] = value
            return value


class Inf(BaseChecker):
    def __str__(self):
        return f"Inf({{{self.aset}}}){'@' if self.name else ""}{self.name or ""}"  # noqa: E501

    def check_state(self, state: int) -> PrefixType | None:
        t, t_minimal = self.aut.get_trap_set_of(state)
        if t <= self.aset:
            return PrefixType.GOOD
        if not (t & self.aset):
            return PrefixType.BAD
        if t_minimal:
            return (
                PrefixType.UGLY
                if any(t & u for u in self.uglies)
                else PrefixType.GOOD)
        return None


class Fin(BaseChecker):
    def __str__(self):
        return f"Fin({{{self.aset}}}){'@' if self.name else ""}{self.name or ""}"  # noqa: E501

    def check_state(self, state: int) -> PrefixType | None:
        t, t_minimal = self.aut.get_trap_set_of(state)
        if t <= self.aset:
            return PrefixType.BAD
        if not (t & self.aset):
            return PrefixType.GOOD
        if t_minimal:
            return (
                PrefixType.UGLY
                if any(t & u for u in self.uglies)
                else PrefixType.BAD)
        return None


class Neg(AcceptanceChecker):
    def __init__(self, mon: AcceptanceChecker):
        self.mon = mon

    def __str__(self):
        if isinstance(self.mon, Inf):
            return f"Fin({{{self.mon.aset}}})"
        return f"!({self.mon})"

    def check(self, runner: SingleRunner):
        p = self.mon.check(runner)
        if p is not None:
            return p.invert()
        return None


class And(AcceptanceChecker):
    def __init__(self, monitors):
        self.monitors = monitors

    def __str__(self):
        return " & ".join(f"({m})" for m in self.monitors)

    def check(self, runner: SingleRunner):
        checks = set()
        for m in self.monitors:
            check = m.check(runner)
            if check == PrefixType.UGLY:
                return PrefixType.UGLY
            checks.add(check)
        # If prefix is bad for at least one operand, it's bad
        if PrefixType.BAD in checks:
            return PrefixType.BAD
        if PrefixType.GOOD in checks:
            return PrefixType.GOOD
        return None


class Or(AcceptanceChecker):
    def __init__(self, monitors):
        self.monitors = monitors

    def __str__(self):
        return " | ".join(f"({m})" for m in self.monitors)

    def check(self, runner: SingleRunner):
        checks = set()
        for m in self.monitors:
            check = m.check(runner)
            checks.add(check)
        # If prefix is good for at least one operand, it's good
        if PrefixType.GOOD in checks:
            return PrefixType.GOOD
        # Prefix is bad for all operands = it's bad
        if PrefixType.BAD in checks and PrefixType.UGLY not in checks:
            return PrefixType.BAD
        # Prefix is bad for some operands and ugly for others = it's ugly
        if PrefixType.UGLY not in checks:
            return PrefixType.UGLY
        return None


class SympyRunner(SingleRunner):
    def __init__(self, aut: Automaton, drv: Driver, mon: bool = False) -> None:
        super().__init__(aut, drv, mon)
        states = aut.hoa.header.nb_states or max(x.index for x in aut.hoa.body.state2edges)  # noqa: E501
        self.transitions = [None] * (states + 1)
        self.symbols = [sympy.symbols(ap) for ap in self.aps]
        self.states_to_index: MutableBidirectionalMapping[int, tuple[int, ...]] = bidict()  # noqa: E501
        self.cache: dict[tuple, int] = {}

    def init(self) -> None:
        super().init()
        next_index = 1
        for state, edges in self.aut.hoa.body.state2edges.items():
            pieces = []
            for sub in sorted(powerset(edges), key=len, reverse=True):
                if len(sub) == 0:
                    break
                lbls = [to_sympy(e.label or True, self.symbols) for e in sub]
                conj = sympy.And(*lbls)
                if satisfiable(conj, algorithm="pycosat") is not False:
                    tgts = tuple(sorted(set(e.state_conj[0] for e in sub)))
                    if tgts not in self.states_to_index.inverse:
                        self.states_to_index[next_index] = tgts
                        next_index += 1
                    idx = self.states_to_index.inverse[tgts]
                    pieces.append((idx, conj))

            pieces.append((0, True))
            tr_fun = sympy.Piecewise(*pieces)
            f = tr_fun.expand()
            self.transitions[state.index] = autowrap(f, args=self.symbols, backend="cython")  # noqa: E501

    def step(self, inputs: Optional[set] = None) -> Iterable[Transition]:
        assert self.state is not None
        if inputs is None:
            inputs = self.driver.get()
        key = (self.state, *sorted(inputs))
        try:
            next_states_id = self.cache[key]
        except KeyError:
            tr_fun = self.transitions[self.state]
            if TYPE_CHECKING:
                assert tr_fun is not None
            next_states_id = tr_fun(*[ap in inputs for ap in self.aps])
            self.cache[key] = next_states_id
            if tr_fun is None:
                for action in self.deadlock_actions:
                    action.run(self)
                return ()
        if next_states_id == 0:
            for action in self.deadlock_actions:
                action.run(self)
            return ()
        candidates = self.states_to_index.get(next_states_id, ())
        self.candidates = [(s, "") for s in candidates]
        if len(self.candidates) > 1:
            self.candidates = list(self.candidates)
            for action in self.nondet_actions:
                action.run(self)
        old_state, next_state = self.state, self.candidates[0][0]
        self.count += 1
        self.state = next_state
        for hook in self.transition_hooks:
            hook.run(self)
        return ((old_state, inputs, "[" + " ".join(inputs) + "]", next_state),)


class AllsatRunner(SingleRunner):
    def __init__(self, aut, drv, mon=False) -> None:
        super().__init__(aut, drv, mon)
        self.trel: list[list[tuple[dict, str, int]]] = [list() for _ in range(aut.states + 1)]  # noqa: E501
        self.symbols = [sympy.symbols(ap) for ap in self.aps]
        self.symbols_inv = {ap: i for i, ap in enumerate(self.symbols)}

    def init(self):
        super().init()
        # Retrieve AP probabilities
        pr = {}
        for ap in self.driver.aps:
            drv = self.driver.find_driver(ap)
            assert type(drv) is RandomDriver
            pr[ap] = drv.cum_weights[0]

        def prob(model: dict) -> float:
            """Compute the probability of a model."""
            p = 1.0
            for ap, val in model.items():
                p *= pr[str(ap)] if val else (1 - pr[str(ap)])
            return p

        def compute_models(state):
            edges = self.aut.get_edges(state)
            states2models = defaultdict(list)
            disj_lbls = sympy.false
            for e in edges:
                lbl = to_sympy(e.label or True, self.symbols)
                tgt = e.state_conj[0]
                disj_lbls = sympy.Or(disj_lbls, lbl)
                for m in allsat(lbl):
                    print(state, fmt_edge(e, self.aps), m)
                    states2models[e.state_conj[0]].append(m)
            # Add implicit stuttering transitions
            for m in allsat(sympy.Not(disj_lbls)):
                states2models[state].append(m)

            # "Flesh out" models to cover all (relevant) APs
            relevant_aps = set(a for m in chain.from_iterable(states2models.values()) for a in m)  # noqa: E501
            print(relevant_aps)
            for s in states2models:
                new_models = []
                for m in states2models[s]:
                    missing_aps = relevant_aps - set(m.keys())
                    if not missing_aps:
                        new_models.append(m)
                        continue
                    for combo in product((True, False), repeat=len(missing_aps)):
                        m_ext = m.copy()
                        for ap, val in zip(missing_aps, combo):
                            m_ext[ap] = val
                        new_models.append(m_ext)
                states2models[s] = new_models

            models2states = defaultdict(set)
            next_states = set(states2models.keys())
            for s1, s2 in product(next_states, repeat=2):
                for m1 in states2models[s1]:
                    k = tuple(sorted(m1.items(), key=lambda x: (str(x[0]), x[1])))  # noqa: E501
                    models2states[k].update((s1,))
                    if s1 == s2:
                        continue
                    for m2 in states2models[s2]:
                        if m1 != m2 and includes(m2, m1):
                            models2states[k].update((s2,))

            cumulative_sum = 0

            trel = []
            for s in states2models:
                for m in states2models[s]:
                    k = tuple(sorted(m.items(), key=lambda x: (str(x[0]), x[1])))  # noqa: E501
                    m_states = models2states[k]
                    p_m = prob(m)
                    p_s_and_m = p_m * (1 / len(m_states))
                    cumulative_sum += p_s_and_m

                    m_str = ";".join(("!" if not value else "")+str(sym) for sym, value in m.items())  # noqa: E501
                    m_str = "{" + m_str + "}"

                    trel.append((cumulative_sum, (m, m_str, s)))

            trel[-1] = (1.0, trel[-1][1])
            self.trel[state] = trel

        for x in range(self.aut.states + 1):
            compute_models(x)
        input()

    def step(self, _: Optional[set] = None) -> Iterable[PartialTransition]:
        if TYPE_CHECKING:
            assert self.state is not None
        if not self.trel[self.state]:
            for action in self.deadlock_actions:
                action.run(self)
            return []
        m, m_str, next_state = pick(self.trel[self.state])
        old_state, self.state = self.state, next_state
        self.count += 1
        for hook in self.transition_hooks:
            hook.run(self)
        return ((old_state, m, m_str, self.state),)
