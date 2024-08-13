from abc import ABC
from pathlib import Path

import msgspec
import tomli

from ..drivers import CompositeDriver, RandomDriver, UserDriver
from .toml_v1 import TomlV1


class ConfigurationError(Exception):
    pass


class Configuration(ABC):
    def get_driver(self):
        return self.driver

    @staticmethod
    def factory(fname: Path, aps: list):
        if fname.suffix != ".toml":
            raise NotImplementedError(f"Unsupported config format {fname.suffix}")  # noqa: E501
        with open(fname, "rb") as conf_file:
            try:
                toml = tomli.load(conf_file)
                assert "hoa-exec" in toml, "Missing mandatory section [hoa-exec]"  # noqa: E501
                assert "version" in toml["hoa-exec"], "Missing mandatory field [hoa-exec].version"  # noqa: E501
                conf_version = toml["hoa-exec"]["version"]
                if conf_version == 1:
                    conf = msgspec.convert(toml, type=TomlV1)
                    return TomlConfigV1(fname, conf, aps)
                else:
                    raise ConfigurationError(f"Unsupported version {conf_version}")  # noqa: E501
            except (AssertionError, tomli.TOMLDecodeError, msgspec.ValidationError) as err:  # noqa: E501
                raise ConfigurationError(err) from None


class DefaultConfig(Configuration):
    def __init__(self, aps: list[str]) -> None:
        self.driver = UserDriver(aps)


class TomlConfigV1(Configuration):
    DRIVERS = {"flip": RandomDriver, "user": UserDriver}

    def __init__(self, fname: Path, conf: TomlV1, aps: list[str]) -> None:
        self.fname = fname
        d = CompositeDriver()
        default_driver = self.DRIVERS[conf.hoa_exec.default_driver]
        for drv_conf in conf.driver.flip:
            drv = RandomDriver.of_toml_v1(aps, drv_conf)
            d.append(drv)
        for drv_conf in conf.driver.user:
            drv = UserDriver.of_toml_v1(aps, drv_conf)
            d.append(drv)
        aps_left = [ap for ap in aps if ap not in set(d.aps)]
        if aps_left:
            d.append(default_driver(aps_left))
        self.driver = d

    def get_driver(self):
        return self.driver
