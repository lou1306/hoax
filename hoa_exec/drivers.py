from abc import ABC, abstractmethod
from io import TextIOWrapper
from random import choices
from typing import Iterable
from json import loads


class Driver(ABC):

    def __init__(self, aps: Iterable[str]) -> None:
        self.aps = aps

    @abstractmethod
    def get(self) -> dict:
        raise NotImplementedError


class CompositeDriver(Driver):
    def __init__(self) -> None:
        self.aps = []
        self.drivers = []

    def append(self, driver: Driver):
        self.drivers.append(driver)
        assert all(ap not in self.aps for ap in driver.aps), \
            "Some APs have multiple drivers: " \
            f"{[ap for ap in self.aps if ap in driver.aps]}"
        self.aps.extend(driver.aps)

    def get(self) -> dict:
        result = {}
        for d in self.drivers:
            result |= d.get()
        return result


class UserDriver(Driver):
    NAME = "user"

    def __init__(self, aps: Iterable[str]) -> None:
        super().__init__(aps)

    def handle(in_str: str) -> bool:
        in_str = in_str.strip().lower()
        try:
            in_int = int(in_str)
            return bool(in_int)
        except ValueError:
            pass
        return False if in_str in ('false', 'null', '') else bool(in_str)

    def get(self):
        result = {}
        for ap in self.aps:
            value = handle(input(f"{ap}? "))
            result[ap] = value
        return result


class RandomDriver(Driver):
    NAME = "flip"

    def __init__(self, aps) -> None:
        super().__init__(aps)

    def get(self) -> dict:
        result = choices((True, False), k=len(self.aps))
        return {ap: value for ap, value in zip(self.aps, result)}


class StreamDriver(Driver):
    def __init__(self, aps, stream: TextIOWrapper, diff: bool = False) -> None:
        super().__init__(aps)
        self.diff = diff
        self.stream = stream
        self.prev = {}

    def get(self) -> dict:
        line = self.read_next()
        result = ({**self.prev} if self.diff else {}) | self.to_dict(line)
        if self.diff:
            self.prev = result
        return result

    def read_next(self) -> dict:
        raise NotImplementedError


class JSONDriver(StreamDriver):
    def read_next(self) -> dict:
        line = self.stream.readline()
        return loads(line)


def handle(in_str: str) -> bool:
    in_str = in_str.strip().lower()
    try:
        in_int = int(in_str)
        return bool(in_int)
    except ValueError:
        pass
    return False if in_str in ('false', 'null', '') else bool(in_str)


DRIVERS = {
    cls.NAME: cls
    for cls in Driver.__subclasses__()
    if hasattr(cls, "NAME")
}
