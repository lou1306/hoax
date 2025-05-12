from abc import ABC, abstractmethod
from random import choice
from typing import Optional, Sequence

from hoa.core import Edge, State

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


class CompositeRunner(Runner):
    def __init__(self, automata: Sequence[Automaton], drv: Driver) -> None:
        self.driver = drv
        self.runners = [SingleRunner(a, drv) for a in automata]
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


class SingleRunner(Runner):
    def __init__(self, aut: Automaton, drv: Driver) -> None:
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

    def init(self) -> None:
        self.state = next(iter(self.aut.get_initial_states()))
        # TODO support initial state conjunction (alternating automata)
        self.state = next(iter(self.state))
        self.state = self.aut.int2states[self.state]

    def add_transition_hook(self, hook):
        self.transition_hooks.append(hook)

    def add_nondet_action(self, action):
        self.nondet_actions.append(action)

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
