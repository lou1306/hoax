from abc import ABC
from pathlib import Path
import logging
import sys

import msgspec
import tomli

from ..hoa import Automaton
from ..drivers import CompositeDriver, RandomDriver, UserDriver, Driver
from ..runners import Bound, Hook, Log, Quit, RandomChoice, Runner, UserChoice
from .toml_v1 import TomlV1


class ConfigurationError(Exception):
    pass


class Configuration(ABC):
    def get_driver(self) -> Driver:
        return self.driver

    def get_runner(self) -> Runner:
        return self.runner

    @staticmethod
    def factory(fname: Path, aut: Automaton):
        if fname.suffix != ".toml":
            raise ConfigurationError(f"Unsupported config format {fname.suffix}")  # noqa: E501
        with open(fname, "rb") as conf_file:
            try:
                toml = tomli.load(conf_file)
                assert "hoa-exec" in toml, "Missing mandatory section [hoa-exec]"  # noqa: E501
                assert "version" in toml["hoa-exec"], "Missing mandatory field [hoa-exec].version"  # noqa: E501
                conf_version = toml["hoa-exec"]["version"]
                if conf_version == 1:
                    conf = msgspec.convert(toml, type=TomlV1)
                    return TomlConfigV1(fname, conf, aut)
                else:
                    raise ConfigurationError(f"Unsupported version {conf_version}")  # noqa: E501
            except (AssertionError, tomli.TOMLDecodeError, msgspec.ValidationError) as err:  # noqa: E501
                raise ConfigurationError(err) from None


class DefaultConfig(Configuration):
    def __init__(self, aut: Automaton) -> None:
        self.driver = UserDriver(list(aut.get_aps()))
        self.runner = Runner(aut=aut, drv=self.driver)
        self.runner.nondet_actions.append(UserChoice())


class TomlConfigV1(Configuration):
    DRIVERS = {"flip": RandomDriver, "user": UserDriver}

    def handle_log_section(self, log_conf: TomlV1.LogSection) -> None:
        handler = (
            logging.StreamHandler(sys.stdout)
            if log_conf.name is None
            else logging.FileHandler(log_conf.name))
        handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
        handler.setLevel({
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "none": logging.FATAL
             }.get(log_conf.level, "info"))
        logging.getLogger().addHandler(handler)

    def __init__(self, fname: Path, conf: TomlV1, aut: Automaton) -> None:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger().handlers.clear()
        for log_conf in conf.log:
            self.handle_log_section(log_conf)

        self.fname = fname
        aps = list(aut.get_aps())
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
        self.runner = Runner(aut, d)
        if conf.runner.nondet == "user":
            self.runner.nondet_actions.append(UserChoice())
        elif conf.runner.nondet == "random":
            self.runner.nondet_actions.append(RandomChoice())
        if conf.runner.bound:
            self.runner.transition_hooks.append(
                Hook(Bound(conf.runner.bound), Quit()))

    def get_driver(self):
        return self.driver
