from typing import Annotated, Collection, Optional

from msgspec import Meta, Struct, field


def invalid(field: str, valid: Collection[str]):
    valid = ', '.join(valid)
    raise ValueError(f"{field} must be one of {valid}") from None


class TomlV1(Struct):

    class HoaExecSection(Struct):
        DRIVERS = "flip", "user"
        version: Annotated[int, Meta(ge=1, le=1)]
        name: Optional[str] = None
        default_driver: Optional[str] = field(name="default-driver", default="user")  # noqa: E501

        def __post_init__(self):
            if self.default_driver not in self.DRIVERS:
                invalid("default-driver", self.DRIVERS)

    class LogSection(Struct):
        LOG_LEVELS = "none", "error", "warning", "info", "debug"
        name: Optional[str] = None
        level: Optional[str] = field(default="info")
        level_obj: int = field(init=False)

        def __post_init__(self):
            if self.level not in self.LOG_LEVELS:
                invalid("level", self.LOG_LEVELS)

    class DriverSection(Struct):
        class RandomDriver(Struct):
            aps: set[str | int]
            bias: Annotated[float, Meta(ge=0, le=1)] = None

        class UserDriver(Struct, tag=True):
            aps: set[str | int]

        flip: list[RandomDriver] = field(default_factory=list)
        user: list[UserDriver] = field(default_factory=list)

    class RunnerSection(Struct):
        NONDET_VALUES = ("first", "random", "user")

        bound: Annotated[int, Meta(gt=0)] = None
        nondet: Optional[str] = field(default="first")

        def __post_init__(self):
            if self.nondet not in self.NONDET_VALUES:
                invalid("nondet", self.NONDET_VALUES)

    hoa_exec: HoaExecSection = field(name="hoa-exec")
    driver: DriverSection = field(default_factory=DriverSection)
    runner: RunnerSection = field(default_factory=RunnerSection)
    log: list[LogSection] = field(default_factory=list)
