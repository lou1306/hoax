from typing import Annotated, Optional

from msgspec import Meta, Struct, field


class TomlV1(Struct):
    class HoaExec(Struct):
        DRIVERS = "flip", "user"
        version: Annotated[int, Meta(ge=1, le=1)]
        name: Optional[str] = None
        default_driver: Optional[str] = field(name="default-driver", default="user")  # noqa: E501

        def __post_init__(self):
            if self.default_driver not in self.DRIVERS:
                raise ValueError(
                    f"default-driver must be one of {', '.join(self.DRIVERS)}")

    class DriverSection(Struct):
        class RandomDriver(Struct):
            aps: set[str | int]
            bias: Annotated[float, Meta(ge=0, le=1)] = None

        class UserDriver(Struct, tag=True):
            aps: set[str | int]

        flip: list[RandomDriver] = field(default_factory=list)
        user: list[UserDriver] = field(default_factory=list)

    hoa_exec: HoaExec = field(name="hoa-exec")
    driver: DriverSection
