from abc import ABC, abstractmethod
from random import choice

from hoa.core import State, Edge

from .drivers import Driver
from .hoa import Automaton, Transition


class Action(ABC):
    @abstractmethod
    def run(self, runner: "Runner") -> None:
        raise NotImplementedError


class Condition(ABC):
    @abstractmethod
    def check(self, runner: "Runner"):
        pass


class StopRunner(Exception):
    pass


class Runner:
    def __init__(self, aut: Automaton, drv: Driver) -> None:
        self.aut = aut
        self.aps = list(aut.get_aps())
        self.driver = drv
        self.state: State = None
        self.trace = []
        self.deadlock_actions: list[Action] = []
        self.nondet_actions: list[Action] = []
        self.transition_hooks: list[Hook] = []
        self.candidates: list[Edge] = []

    def init(self) -> None:
        self.state = next(iter(self.aut.get_initial_states()))
        # TODO support initial state conjunction (alternating automata)
        self.state = next(iter(self.state))
        self.state = self.aut.int2states[self.state]

    def step(self):
        values = self.driver.get()
        self.candidates = list(self.aut.get_candidates(self.state, values))
        if not self.candidates:
            for action in self.deadlock_actions:
                action.run(self)
        elif len(self.candidates) > 1:
            for action in self.nondet_actions:
                action.run(self)
        if len(self.candidates) >= 1:
            edge = self.candidates[0]
            next_state_index = next(iter(edge.state_conj))
            next_state = self.aut.int2states[next_state_index]

            tr = Transition(self.state, next_state, edge, self.aps)
            self.trace.append(tr)
            self.state = next_state
            for hook in self.transition_hooks:
                hook.run(self)
        self.candidates = []


class Reach(Condition):
    def __init__(self, target: State) -> None:
        self.target = target

    def check(self, runner: Runner):
        return runner.state.index == self.target


class Always(Condition):
    def check(self, _):
        return True


class Hook:
    def __init__(self, condition: Condition, action: Action) -> None:
        self.condition = condition
        self.action = action

    def run(self, runner: Runner):
        if self.condition.check(runner):
            self.action.run(runner)


class Reset(Action):
    def run(self, runner: Runner) -> None:
        runner.init()


class Log(Action):
    def run(self, runner: Runner) -> None:
        tr = runner.trace[-1] if runner.trace else None
        if tr is not None:
            print(str(tr))


class PressEnter(Action):
    def run(self, _) -> None:
        input("Press [Enter] to continue...")


class RandomChoice(Action):
    def run(self, runner: Runner) -> None:
        runner.candidates = [choice(runner.candidates)]


class Quit(Action):
    def run(self, runner: Runner) -> None:
        raise StopRunner()
