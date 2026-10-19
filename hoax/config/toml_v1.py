from itertools import chain, combinations
import logging
from pathlib import Path
import sys
from typing import Annotated, Collection, Optional

from msgspec import Meta, Struct, field

from ..hoa import extract_aps
from .. import drivers, runners
from ..util import PRG_DEFAULT_SEED


def invalid(field: str, valid: Collection[str]):
    valid = ', '.join(valid)
    raise ValueError(f"{field} must be one of {valid}") from None


def resolve(path: str, base_path: Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (base_path / resolved).resolve()
    return resolved


class TomlV1(Struct):
    """The HOAX TOML configuration format, v1.

    This class defines a `msgspec` structured format, which directly \
    encodes to/decodes from TOML.
    """

    class HoaxSection(Struct):
        DRIVERS = {
            "flip": drivers.RandomDriver,
            "user": drivers.UserDriver,
            "json": drivers.JSONDriver,
            "txt": drivers.SimpleTxtDriver}

        version: Annotated[int, Meta(ge=1, le=1)]
        name: Optional[str] = None
        default_driver: Optional[str] = field(name="default-driver", default="user")  # noqa: E501
        seed: Annotated[int, Meta(ge=0)] = field(name="seed", default=PRG_DEFAULT_SEED)  # noqa: E501

        def get_default_driver(self):
            return self.DRIVERS[self.default_driver]

        def __post_init__(self):
            if self.default_driver not in self.DRIVERS:
                invalid("default-driver", self.DRIVERS)

    class LogSection(Struct):
        LOG_LEVELS = {
            "none": logging.FATAL,
            "error": logging.ERROR,
            "warning": logging.WARNING,
            "info": logging.INFO,
            "debug": logging.DEBUG}
        name: Optional[str] = None
        level: Optional[str] = field(default="info")

        def get_level(self) -> int:
            return self.LOG_LEVELS[self.level or "info"]

        def get_handler(self) -> logging.Handler:
            handler = (
                logging.StreamHandler(sys.stderr)
                if self.name is None
                else logging.FileHandler(self.name))
            # TODO add format customization
            handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
            handler.setLevel(self.get_level())
            return handler

        def __post_init__(self):
            if self.level not in self.LOG_LEVELS:
                invalid("level", self.LOG_LEVELS)

    class DriverSection(Struct):
        class RandomDriver(Struct):
            aps: set[str | int]
            bias: Annotated[float, Meta(ge=0, le=1)] = -1

            def get_driver(self, aps, _) -> drivers.RandomDriver:
                result = drivers.RandomDriver(extract_aps(aps, self.aps))
                if self.bias >= 0:
                    result.cum_weights = (self.bias, 1)
                return result

        class UserDriver(Struct):
            aps: set[str | int]

            def get_driver(self, aps, _) -> drivers.UserDriver:
                return drivers.UserDriver(extract_aps(aps, self.aps))

        class JSONDriver(Struct):
            aps: set[str | int]
            filename: str

            def get_driver(self, aps, base_path: Path) -> drivers.JSONDriver:
                stream = open(resolve(self.filename, base_path))
                return drivers.JSONDriver(extract_aps(aps, self.aps), stream)

        class SimpleTxtDriver(Struct):
            aps: set[str | int]
            filename: str

            def get_driver(self, aps, base_path: Path) -> drivers.SimpleTxtDriver:    # noqa: E501
                stream = open(resolve(self.filename, base_path))
                return drivers.SimpleTxtDriver(extract_aps(aps, self.aps), stream)  # noqa: E501

        flip: list[RandomDriver] = field(default_factory=list)
        user: list[UserDriver] = field(default_factory=list)
        json: list[JSONDriver] = field(default_factory=list)
        txt: list[SimpleTxtDriver] = field(default_factory=list)

    class RunnerSection(Struct):
        NONDET_VALUES = {
            "first": None,
            "random": runners.RandomChoice(),
            "user": runners.UserChoice()}

        bound: Annotated[int, Meta(gt=0)] = 0
        nondet: Optional[str] = field(default="first")

        def get_nondet(self) -> runners.Action | None:
            return self.NONDET_VALUES[self.nondet or "first"]

        def __post_init__(self):
            if self.nondet not in self.NONDET_VALUES:
                invalid("nondet", self.NONDET_VALUES)

    hoax: HoaxSection = field(name="hoax")
    driver: DriverSection = field(default_factory=DriverSection)
    runner: RunnerSection = field(default_factory=RunnerSection)
    log: list[LogSection] = field(default_factory=list)

    def __post_init__(self):
        dups = set()
        for d1, d2 in combinations(self.drivers(), 2):
            dups.update(d1.aps.intersection(d2.aps))
        if dups:
            raise ValueError(f"APs have multiple drivers: {dups}")

    def drivers(self):
        yield from chain(self.driver.flip, self.driver.user, self.driver.json, self.driver.txt)  # noqa: E501
