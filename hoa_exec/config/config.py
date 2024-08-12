from abc import ABC
from pathlib import Path

import tomlkit
from tomlkit.exceptions import TOMLKitError
from tomlkit import TOMLDocument

from ..drivers import CompositeDriver, DRIVERS, UserDriver


class Configuration(ABC):
    def get_driver(self):
        return self.driver

    @staticmethod
    def factory(fname: Path, aps: list):
        if fname.suffix != ".toml":
            raise NotImplementedError(f"Unsupported config format {fname.suffix}.")  # noqa: E501
        with open("config.toml") as conf_file:
            try:
                conf = tomlkit.load(conf_file)
            except TOMLKitError as err:
                raise IOError(err) from None
            print(conf)
        assert "hoa-exec" in conf, "configuration file missing mandatory element '[hoa-exec]'"  # noqa: E501
        assert "version" in conf['hoa-exec'], "configuration file missing mandatory element '[hoa-exec]version'"  # noqa: E501
        conf_version = conf["hoa-exec"]["version"]
        assert conf_version == 1, f"unsupported configuration version: {conf_version}"  # noqa: E501
        if conf_version == 1:
            return TomlConfigV1(fname, conf, aps)
        raise Exception("???")


class DefaultConfig(Configuration):
    def __init__(self, aps: list[str]) -> None:
        self.driver = UserDriver(aps)


class TomlConfigV1(Configuration):

    def __init__(self, fname: Path, conf: TOMLDocument, aps: list[str]) -> None:  # noqa: E501
        d = CompositeDriver()
        # If no default driver is given, pick user driver
        default = conf["hoa-exec"].get("default-driver", "user")
        default_driver = DRIVERS[default]
        for key in DRIVERS:
            if key in conf.get("driver", []):
                drv = DRIVERS[key].of_toml_v1(aps, conf["driver"][key])
                d.append(drv)
        aps_left = [ap for ap in aps if ap not in set(d.aps)]
        if aps_left:
            d.append(default_driver(aps_left))
        self.driver = d

    def get_driver(self):
        return self.driver
