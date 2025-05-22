import logging
from abc import ABC, abstractmethod
from io import TextIOWrapper
from json import loads
from random import choices
from typing import Iterable

log = logging.getLogger(__name__)


class EndOfFiniteTrace(Exception):
    pass


class Driver(ABC):
    def __init__(self, aps: Iterable[str]) -> None:
        log.debug(f"Instantiating {type(self).__name__} for {aps=}")
        self.aps = aps

    @abstractmethod
    def get(self) -> set:
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

    def get(self) -> set:
        result = set()
        for d in self.drivers:
            result |= d.get()
        return result


class UserDriver(Driver):
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
        result = set()
        for ap in self.aps:
            value = handle(input(f"{ap}? "))
            if value:
                result.add(ap)
        return result


class RandomDriver(Driver):
    pop = True, False

    def __init__(self, aps) -> None:
        self.aps = aps
        self.k = len(self.aps)
        self.cum_weights = None

    def get(self) -> set:
        result = choices(self.pop, k=self.k, cum_weights=self.cum_weights)
        return set(ap for ap, value in zip(self.aps, result) if value)


class StreamDriver(Driver):
    def __init__(self, aps, stream: TextIOWrapper, diff: bool = False) -> None:
        super().__init__(aps)
        self.diff = diff
        self.stream = stream
        self.prev = {}

    def get(self) -> set:
        return self.read_next()

    def read_next(self) -> set:
        raise NotImplementedError


class JSONDriver(StreamDriver):
    def read_next(self) -> set:
        line = self.stream.readline()
        return set(x for x, y in loads(line) if y)


class SimpleTxtDriver(StreamDriver):
    def read_next(self):
        line = self.stream.readline()
        if not line:
            self.stream.close()
            raise EndOfFiniteTrace(self.stream.name)
        data = set()
        while line:
            if line == "\n":
                return data
            ap = line[:-1]
            if ap in self.aps:
                data.add(ap)
            line = self.stream.readline()
        return data


def handle(in_str: str) -> bool:
    in_str = in_str.strip().lower()
    try:
        in_int = int(in_str)
        return bool(in_int)
    except ValueError:
        pass
    return False if in_str in ('false', 'null', '') else bool(in_str)
