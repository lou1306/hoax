from abc import ABC, abstractmethod
from io import TextIOWrapper
import logging
from random import choices
from typing import Collection, Iterable
from json import loads

from .config.toml_v1 import TomlV1

log = logging.getLogger(__name__)


class Driver(ABC):
    def __init__(self, aps: Iterable[str]) -> None:
        log.debug(f"Instantiating {type(self).__name__} for {aps=}")
        self.aps = aps

    @abstractmethod
    def get(self) -> dict:
        raise NotImplementedError

    @classmethod
    def of_toml_v1(cls, all_aps: Iterable[str], conf):
        raise NotImplementedError

    @staticmethod
    def extract_aps(all_aps: Iterable[str], aps: Collection[str | int]):
        return [i if type(i) is str else all_aps[i] for i in aps]


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

    @classmethod
    def of_toml_v1(cls, all_aps: Iterable[str], conf: TomlV1.DriverSection.UserDriver):  # noqa: E501
        aps = cls.extract_aps(all_aps, conf.aps)
        return UserDriver(aps)


class RandomDriver(Driver):
    pop = True, False

    def __init__(self, aps) -> None:
        self.aps = aps
        self.k = len(self.aps)
        self.cum_weights = None

    def get(self) -> dict:
        result = choices(self.pop, k=self.k, cum_weights=self.cum_weights)
        return {ap: value for ap, value in zip(self.aps, result)}

    @classmethod
    def of_toml_v1(cls, all_aps: Iterable[str], conf: TomlV1.DriverSection.RandomDriver):  # noqa: E501
        result = RandomDriver(cls.extract_aps(all_aps, conf.aps))
        if conf.bias is not None:
            result.cum_weights = (conf.bias, 1)
        log.debug(f"Instantiating RandomDriver for {result.aps=} with bias={conf.bias}")  # noqa: E501
        return result


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
