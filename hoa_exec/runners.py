from abc import ABC, abstractmethod
from enum import Enum
from random import choice
from typing import Optional, Sequence

import networkit as nk
from hoa.ast.acceptance import AcceptanceAtom, AtomType
from hoa.ast.boolean_expression import PositiveAnd, PositiveOr
from hoa.core import Edge, State
from networkit.graph import Graph

from .drivers import Driver
from .hoa import Automaton, Transition, fmt_edge, fmt_state



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


class CompositeRunner(Runner):
    def __init__(self, automata: Sequence[Automaton], drv: Driver,
                 monitor: bool = False) -> None:
        self.driver = drv
        self.runners = [SingleRunner(a, drv, monitor) for a in automata]
        self.count = 0

    def init(self) -> None:
        for runner in self.runners:
            runner.init()

    def step(self) -> list[Transition]:
        values = self.driver.get()
        result = [tr for r in self.runners for tr in r.step(values)]
        self.count += 1
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
        self.state: State = None
        self.count = 0
        # self.trace = []
        self.deadlock_actions: list[Action] = []
        self.nondet_actions: list[Action] = []
        self.transition_hooks: list[Hook] = []
        self.candidates: list[Edge] = []
        prp = self.aut.aut.header.properties or []
        self.deterministic = "deterministic" in prp
        # TODO make this configurable
        if mon:
            chk = AcceptanceChecker.make_checker(self.aut)
            self.transition_hooks.append(Hook(chk, Reset()))

    def init(self) -> None:
        self.state = next(iter(self.aut.get_initial_states()))
        # TODO support initial state conjunction (alternating automata)
        self.state = next(iter(self.state))
        self.state = self.aut.int2states[self.state]

    def add_transition_hook(self, hook):
        self.transition_hooks.append(hook)

    def add_nondet_action(self, action):
        self.nondet_actions.append(action)

    def add_deadlock_action(self, action):
        self.deadlock_actions.append(action)

    def step(self, inputs: Optional[dict] = None) -> list[Transition]:
        """return False iff automaton stuttered"""
        inputs = inputs or self.driver.get()
        values = tuple(inputs[x] for x in self.aut.aut.header.propositions)
        if self.deterministic:
            candidate = self.aut.get_first_candidate(self.state, values)
            self.candidates = [candidate] if candidate is not None else []
        else:
            self.candidates = list(self.aut.get_candidates(self.state, values))
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
            next_state_index = next(iter(edge.state_conj))
            next_state = self.aut.int2states[next_state_index]

            tr = Transition(self.state, next_state, edge, self.aps)
            self.count += 1
            self.state = next_state
            for hook in self.transition_hooks:
                hook.run(self)
            return [tr]


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
        acond = aut.aut.header.acceptance.condition
        all_states = set(aut.int2states)

        def get_uglies(accept: set[int]):
            graph = Graph(directed=True)
            for e in aut.graph.iterEdges():
                graph.addEdge(*e, addMissing=True)
            for x in accept:
                graph.removeNode(x)
            result = set()
            for trap in aut.traps:
                t = trap - accept
                if t in result:
                    continue
                for node in t:
                    dfs = []

                    def callback(u):
                        dfs.append(u)
                        dfs.extend(graph.iterNeighbors(u))

                    nk.graph.Traversal.DFSfrom(graph, node, callback)
                    if len(dfs) != len(set(dfs)):
                        result.add(t)
            return frozenset(result)

        def _mk(cond):
            def get_acceptance_set(index: int):
                accept = aut.acc_sets[index]
                if cond.negated:
                    accept = all_states - accept
                return frozenset(accept)
            match cond:
                case AcceptanceAtom(atom_type=AtomType.INFINITE):
                    accept = get_acceptance_set(cond.acceptance_set)
                    return Inf(accept, aut.traps, get_uglies(accept))
                case AcceptanceAtom(atom_type=AtomType.FINITE):
                    accept = get_acceptance_set(cond.acceptance_set)
                    return Fin(accept, aut.traps, get_uglies(accept))
                case PositiveAnd():
                    return And([_mk(c) for c in cond.operands])
                case PositiveOr():
                    return Or([_mk(c) for c in cond.operands])
        chk = _mk(acond)
        chk.set_filename(aut.filename)
        return chk


class BaseChecker(AcceptanceChecker):
    def __init__(self, aset, traps, uglies):
        self.aset = aset
        self.traps = traps
        self.uglies = uglies
        self.name = None
        self.cache = {}

    def set_filename(self, name: str) -> None:
        self.name = name

    def check(self, runner: SingleRunner) -> PrefixType | None:
        state = runner.state.index
        if state not in self.cache:
            self.cache[state] = self.check_state(runner.state.index)
        return self.cache[state]

    def check_state(state: int) -> PrefixType | None:
        raise NotImplementedError


class Inf(BaseChecker):
    def __str__(self):
        return f"Inf({{{self.aset}}}){'@' if self.name else ""}{self.name or ""}"  # noqa: E501

    def check_state(self, state: int) -> PrefixType | None:
        for t in self.traps:
            if state in t:
                if t <= self.aset:
                    return PrefixType.GOOD
                if not (t & self.aset):
                    return PrefixType.BAD
                if any(t & u for u in self.uglies):
                    return PrefixType.UGLY
        return None


class Fin(BaseChecker):
    def __str__(self):
        return f"Fin({{{self.aset}}}){'@' if self.name else ""}{self.name or ""}"  # noqa: E501

    def check_state(self, state: int) -> PrefixType | None:
        for t in self.traps:
            if state in t:
                if t <= self.aset:
                    return PrefixType.BAD
                if not (t & self.aset):
                    return PrefixType.GOOD
                if any(t & u for u in self.uglies):
                    return PrefixType.UGLY
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
