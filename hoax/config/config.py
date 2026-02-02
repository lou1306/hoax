# import logging
from abc import ABC, abstractmethod
from pathlib import Path

import msgspec
import tomli

from ..drivers import CompositeDriver, Driver, UserDriver
from ..hoa import Automaton
from ..runners import (Bound, Hook,
                       Quit, Runner, UserChoice)
from ..util import PRG_DEFAULT_SEED, logger
from .toml_v1 import TomlV1


class ConfigurationError(Exception):
    pass


class Configuration(ABC):
    @property
    @abstractmethod
    def driver(self):
        raise NotImplementedError

    @driver.setter
    @abstractmethod
    def driver(self, value: Driver):
        raise NotImplementedError

    @property
    @abstractmethod
    def runner(self) -> Runner:
        raise NotImplementedError

    @runner.setter
    @abstractmethod
    def runner(self, value: Runner):
        raise NotImplementedError

    @property
    @abstractmethod
    def seed(self) -> int:
        raise NotImplementedError

    @seed.setter
    @abstractmethod
    def seed(self, value: int):
        raise NotImplementedError

    @staticmethod
    def factory(fname: Path, a: list[Automaton], monitor: bool = False) -> "Configuration":  # noqa: E501
        """Load a configuration file from `fname`.

        Args:
            fname (Path): Path of the configuration file
            a (list[Automaton]): Automata to run
            monitor (bool, optional): True iff automata should be monitored \
                for acceptance. Defaults to False.

        Raises:
            ConfigurationError: Something wrong with the provided \
                configuration file

        Returns:
            Configuration: An object with configuration details.
        """
        if fname.suffix != ".toml":
            raise ConfigurationError(f"Unsupported config format {fname.suffix}")  # noqa: E501
        with open(fname, "rb") as conf_file:
            try:
                toml = tomli.load(conf_file)
                assert "hoax" in toml, "Missing mandatory section [hoax]"  # noqa: E501
                assert "version" in toml["hoax"], "Missing mandatory field [hoax].version"  # noqa: E501
                conf_version = toml["hoax"]["version"]
                if conf_version == 1:
                    conf = msgspec.convert(toml, type=TomlV1)
                    return TomlConfigV1(fname, conf, a, monitor)
                else:
                    raise ConfigurationError(f"Unsupported version {conf_version}")  # noqa: E501
            except (AssertionError, tomli.TOMLDecodeError, msgspec.ValidationError) as err:  # noqa: E501
                raise ConfigurationError(err) from None


class DefaultConfig(Configuration):
    """Class for the default HOAX configuration."""

    @property
    def runner(self) -> Runner:
        return self._runner

    @runner.setter
    def runner(self, value: Runner):
        self._runner = value

    @property
    def driver(self) -> Driver:
        return self._driver

    @driver.setter
    def driver(self, value: Driver):
        self._driver = value

    @property
    def seed(self) -> int:
        return PRG_DEFAULT_SEED

    @seed.setter
    def seed(self, value: int):
        pass

    def __init__(self, a: list[Automaton], mon: bool = False) -> None:
        aps = list(set(ap for aut in a for ap in aut.get_aps()))
        self.driver = UserDriver(list(aps))
        self.runner = Runner.factory(a=a, drv=self.driver, mon=mon)
        self.runner.add_nondet_action(UserChoice())


class TomlConfigV1(Configuration):
    """Builds a configuration from a TOML file (version 1)"""

    @property
    def driver(self) -> Driver:
        return self._driver

    @driver.setter
    def driver(self, value: Driver):
        self._driver = value

    @property
    def runner(self) -> Runner:
        return self._runner

    @runner.setter
    def runner(self, value: Runner):
        self._runner = value

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int):
        self._seed = value

    def __init__(self, fname: Path, conf: TomlV1, a: list[Automaton],
                 monitor: bool = False) -> None:
        for log_conf in conf.log:
            logger.addHandler(log_conf.get_handler())
        self.fname = fname
        self.seed = conf.hoax.seed

        aps = list(ap for aut in a for ap in aut.get_aps())
        d = CompositeDriver()
        for drv_conf in conf.drivers():
            base_dir = Path(self.fname).parent
            drv = drv_conf.get_driver(aps, base_dir)
            d.append(drv)

        aps_left = [ap for ap in aps if ap not in set(d.aps)]
        if aps_left:
            default_driver = conf.hoax.get_default_driver()
            d.append(default_driver(aps_left))
        self.driver = d if len(d.drivers) > 1 else d.drivers[0]
        self.runner = Runner.factory(a, d, monitor)
        nondet_action = conf.runner.get_nondet()
        if nondet_action is not None:
            self.runner.add_nondet_action(nondet_action)
        if conf.runner.bound > 0:
            bound_cond = Bound(conf.runner.bound)
            self.runner.add_transition_hook(
                Hook(bound_cond, Quit(cause=bound_cond)))

    def get_driver(self):
        return self.driver
