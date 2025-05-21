import multiprocessing as mp
from abc import ABC, abstractmethod
from enum import Enum
from random import choice
from typing import Optional, Sequence

import msgpack
import networkit as nk
from hoa.ast.acceptance import AcceptanceAtom, AtomType
from hoa.ast.boolean_expression import PositiveAnd, PositiveOr
from hoa.core import Edge, State
from networkit.graph import Graph

from .drivers import Driver
from .hoa import Automaton, Transition, fmt_edge, fmt_state, parse


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

    def __str__(self) -> str:
        return type(self).__name__


class StopRunner(Exception):
    pass


class Runner(ABC):
    @abstractmethod
    def init(self):
        raise NotImplementedError

    @abstractmethod
    def step(self):
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


def spawn_runner(pipe: mp.Pipe):  # type: ignore
    fname, monitor = pipe.recv(), pipe.recv()
    aut = parse(fname)
    runner = SingleRunner(aut, None, monitor)
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
            case dict(values):
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
        self.runners = []
        self.procs = []
        for a in automata:
            parent_pipe, child_pipe = mp.Pipe()
            proc = mp.Process(target=spawn_runner, args=(child_pipe,))
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

    def step(self):
        values = self.driver.get()
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
    def __init__(self, automata: Sequence[Automaton], drv: Driver,
                 monitor: bool = False) -> None:
        self.driver = drv
        self.runners = []
        for a in automata:
            if all(
                x in a.hoa.header.properties
                for x in ("complete", "deterministic")
            ):
                self.runners.append(DetCompleteSingleRunner(a, drv, monitor))
            else:
                self.runners.append(SingleRunner(a, drv, monitor))

    @property
    def count(self):
        return self.runners[0].count

    def init(self) -> None:
        for runner in self.runners:
            runner.init()

    def step(self) -> list[Transition]:
        values = self.driver.get()
        result = [tr for r in self.runners for tr in r.step(values)]
        return result

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
        self.candidates: list[Edge] = []
        prp = self.aut.hoa.header.properties or []
        self.deterministic = "deterministic" in prp
        # TODO make this configurable
        if mon:
            chk = AcceptanceChecker.make_checker(self.aut)
            self.transition_hooks.append(Hook(chk, Reset()))

    def init(self) -> None:
        # TODO support initial state conjunction (alternating automata)
        for x in self.aut.hoa.header.start_states:
            for y in x:
                self.state = y
                return

    def add_transition_hook(self, hook):
        self.transition_hooks.append(hook)

    def add_nondet_action(self, action):
        self.nondet_actions.append(action)

    def add_deadlock_action(self, action):
        self.deadlock_actions.append(action)

    def step(self, inputs: Optional[dict] = None) -> list[Transition]:
        """return False iff automaton stuttered"""
        inputs = inputs or self.driver.get()
        if self.deterministic:
            candidate = self.aut.get_first_candidate(self.state, inputs)
            self.candidates = [candidate] if candidate is not None else []
        else:
            self.candidates = list(self.aut.get_candidates(self.state, inputs))
        if not self.candidates:
            for action in self.deadlock_actions:
                action.run(self)
            return []
        elif len(self.candidates) > 1:
            for action in self.nondet_actions:
                action.run(self)
        if len(self.candidates) >= 1:
            edge = self.candidates[0]
            self.candidates = []
            old_state, next_state = self.state, edge.state_conj[0]

            self.count += 1
            self.state = next_state
            for hook in self.transition_hooks:
                hook.run(self)
            return [(old_state, inputs, next_state)]


class DetCompleteSingleRunner(SingleRunner):
    def step(self, inputs: Optional[dict] = None) -> list[Transition]:
        """return False iff automaton stuttered"""
        inputs = inputs or self.driver.get()
        # values = tuple(inputs[x] for x in self.aut.hoa.header.propositions)
        edge = self.aut.get_first_candidate(self.state, inputs)
        next_state = edge.state_conj[0]
        self.count += 1
        old_state, self.state = self.state, next_state
        for hook in self.transition_hooks:
            hook.run(self)
        return [(old_state, inputs, next_state)]


class Reach(Condition):
    def __str__(self) -> str:
        return f"Reach {self.target}"

    def __init__(self, target: State) -> None:
        self.target = target

    def check(self, runner: SingleRunner):
        return runner.state.index == self.target


class Bound(Condition):
    def __str__(self) -> str:
        return f"Bound: {self.bound}"

    def __init__(self, bound) -> None:
        self.bound = bound

    def check(self, runner: SingleRunner):
        return runner.count > self.bound


class Always(Condition):
    def check(self, _):
        return True


class Hook:
    def __init__(self, condition: Condition, action: Action) -> None:
        self.condition = condition
        self.action = action

    def run(self, runner: SingleRunner):
        if self.condition.check(runner):
            msg = f"Hook: {self.condition} triggered {self.action} at step {runner.count}"  # noqa: E501
            print(msg)
            self.action.run(runner)


class Reset(Action):
    def run(self, runner: SingleRunner) -> None:
        runner.init()


class Log(Action):
    def __init__(self, msg: str) -> None:
        self.msg = msg

    def __str__(self) -> str:
        return f"Log {self.msg}"

    def run(self, runner: SingleRunner) -> None:
        print(f"{self.msg} at {fmt_state(runner.state)}")


class PressEnter(Action):
    def run(self, _) -> None:
        input("Press [Enter] to continue...")


class RandomChoice(Action):
    def run(self, runner: SingleRunner) -> None:
        chosen = choice(runner.candidates)
        runner.candidates = [chosen]


class UserChoice(Action):
    def run(self, runner: SingleRunner) -> None:
        for i, edge in enumerate(runner.candidates):
            print(f"[{i}]\t{fmt_edge(edge, runner.aps)}")
        choice = -1
        while not 0 <= choice < len(runner.candidates):
            choice = input("Choose a transition from above: ")
            choice = int(choice) if choice.isdecimal() else -1
        chosen = runner.candidates[choice]
        runner.candidates = [chosen]


class Quit(Action):
    def __init__(self, cause=None):
        super().__init__()
        self.cause = cause

    def run(self, runner):
        raise StopRunner()


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
    @abstractmethod
    def check(self, state: int) -> PrefixType | None:
        pass

    @staticmethod
    def make_checker(aut: Automaton):
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
    def __init__(self, aset: set[int], aut: Automaton, uglies):
        self.aset = aset
        self.aut = aut
        self.uglies = uglies
        self.name = None
        self.cache = {}

    def set_filename(self, name: str) -> None:
        self.name = name

    def check(self, runner: SingleRunner) -> PrefixType | None:
        state = runner.state
        if state not in self.cache:
            self.cache[state] = self.check_state(runner.state)
        return self.cache[state]

    def check_state(state: int) -> PrefixType | None:
        raise NotImplementedError


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
            return PrefixType.BAD
        return None
