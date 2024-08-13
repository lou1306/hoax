from abc import ABC, abstractmethod
import logging
from random import choice

from hoa.core import State, Edge

from .drivers import Driver
from .hoa import Automaton, ForcedTransition, Transition, fmt_expr, fmt_state

log = logging.getLogger(__name__)


class Action(ABC):
    @abstractmethod
    def run(self, runner: "Runner") -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        return type(self).__name__


class Condition(ABC):
    @abstractmethod
    def check(self, runner: "Runner"):
        pass

    def __str__(self) -> str:
        return type(self).__name__


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
    def __str__(self) -> str:
        return f"Reach {self.target}"

    def __init__(self, target: State) -> None:
        self.target = target

    def check(self, runner: Runner):
        return runner.state.index == self.target


class Bound(Condition):
    def __str__(self) -> str:
        return f"Bound {self.bound}"

    def __init__(self, bound) -> None:
        self.bound = bound

    def check(self, runner: Runner):
        return len(runner.trace) >= self.bound


class Always(Condition):
    def check(self, _):
        return True


class Hook:
    def __init__(self, condition: Condition, action: Action) -> None:
        self.condition = condition
        self.action = action

    def run(self, runner: Runner):
        if self.condition.check(runner):
            log.debug(f"Hook: {self.condition} triggered {self.action}")
            self.action.run(runner)


class Reset(Action):
    def run(self, runner: Runner) -> None:
        last_state = runner.state
        runner.init()
        runner.trace.append(ForcedTransition(last_state, label="reset", tgt=runner.state))  # noqa: E501


class Log(Action):
    def __init__(self, msg: str) -> None:
        self.msg = msg

    def __str__(self) -> str:
        return f"Log {self.msg}"

    def run(self, runner: Runner) -> None:
        print(f"{self.msg} at {fmt_state(runner.state)}")


class PressEnter(Action):
    def run(self, _) -> None:
        input("Press [Enter] to continue...")


class RandomChoice(Action):
    def run(self, runner: Runner) -> None:
        chosen = choice(runner.candidates)
        lbl = fmt_expr(chosen.label, runner.aps)
        log.debug(f"Randomly picked {lbl} --> {chosen.state_conj}")
        runner.candidates = [chosen]


class UserChoice(Action):
    def run(self, runner: Runner) -> None:
        for i, edge in enumerate(runner.candidates):
            lbl = fmt_expr(edge.label, runner.aps)
            print(f"[{i}]\t{lbl} --> {edge.state_conj}")
        choice = -1
        while not 0 <= choice < len(runner.candidates):
            choice = input("Choose a transition from above: ")
            choice = int(choice) if choice.isdecimal() else -1
        chosen = runner.candidates[choice]
        lbl = fmt_expr(chosen.label, runner.aps)
        log.debug(f"User picked {lbl} --> {chosen.state_conj}")
        runner.candidates = [chosen]


class Quit(Action):
    def run(self, runner: Runner) -> None:
        raise StopRunner()
