import importlib.metadata
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from signal import SIG_DFL, SIGPIPE, signal
from typing import Annotated, Optional

import typer

from .config.config import Configuration, DefaultConfig
from .drivers import EndOfFiniteTrace
from .hoa import parse
from .runners import StopRunner
from .util import logger

signal(SIGPIPE, SIG_DFL)

app = typer.Typer()


def print_version(version: bool):
    if version:
        print(f"hoax {importlib.metadata.version("hoax-hoa-executor")}")
        raise typer.Exit()


@app.command()
def hoax(
        files: list[Path],
        config: Annotated[
            Optional[Path],
            typer.Option(help="Path to a TOML config file.")] = None,
        monitor: Annotated[
            bool,
            typer.Option(help="Monitor the automaton's acceptance condition.")
        ] = False,
        quiet: Annotated[
            bool,
            typer.Option(help="Suppress output of individual transitions.")
        ] = False,
        version: Annotated[
            Optional[bool],
            typer.Option(
                "--version",
                help="Print version number and exit.",
                callback=print_version, is_eager=True)] = None,
):
    """Execute HOA automata"""
    t = datetime.now()
    t0 = t
    with ThreadPoolExecutor() as exc:
        automata = list(exc.map(parse, files))
    logger.info(f"parsing done in {datetime.now() - t} s")

    t = datetime.now()
    conf = (
        Configuration.factory(config, automata, monitor)
        if config is not None
        else DefaultConfig(automata, monitor))

    logger.info(f"config read in {datetime.now() - t} s")
    logger.info(f"Using runner {type(conf.runner).__name__}")
    logger.info(f"Using seed {conf.seed}")

    t = datetime.now()
    run = conf.runner
    run.init()

    logger.info(f"init done in {datetime.now() - t} s")

    t = datetime.now()
    try:
        if quiet:
            while True:
                run.step()
        else:
            while True:
                tr = run.step()
                for i, (old_state, _, lbl, new_state) in enumerate(tr):
                    sys.stdout.write(f"{i}: {old_state} -- {lbl} --> {new_state}\n")  # noqa: E501
    except (StopRunner, KeyboardInterrupt, EndOfFiniteTrace) as e:
        logger.warning(f"Stopping due to {repr(e)}")
    end = datetime.now()
    logger.info(f"{run.count} steps done in {end - t} s")
    logger.info(f"total: {end - t0} s")
