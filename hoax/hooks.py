from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .hoa import fmt_state
from .util import PRG_BOUNDED
if TYPE_CHECKING:
    from .runners import SingleRunner


class StopRunner(Exception):
    pass


class Action(ABC):
    @abstractmethod
    def run(self, runner: SingleRunner) -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        return type(self).__name__


class Condition(ABC):
    @abstractmethod
    def check(self, runner: "SingleRunner"):
        pass

    def __repr__(self) -> str:
        return f"{self} (at {hex(id(self))})"


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
