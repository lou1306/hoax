import importlib.metadata
import os
import sys
from datetime import datetime
from pathlib import Path
from signal import SIG_DFL, SIGPIPE, signal
from typing import Annotated, Optional

import typer

from .config.config import Configuration, DefaultConfig
from .drivers import EndOfFiniteTrace
from .hoa import Automaton, LazyAutomaton, parse
from .hooks import StopRunner
from .util import PRG_SEED, logger

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

    automata: list[Automaton] = [
        LazyAutomaton.from_bnet(f) if f.suffix == ".bnet" else
        parse(str(f)) if monitor else LazyAutomaton.from_file(f)
        for f in files]
    print(f"Lazy parsing done in {datetime.now() - t} s", file=sys.stderr)

    t = datetime.now()
    conf = (
        Configuration.factory(config, automata, monitor)
        if config is not None
        else DefaultConfig(automata, monitor))

    logger.info(f"Config read in {datetime.now() - t} s")
    logger.info(f"Using runner {type(conf.runner).__name__}")
    logger.info(f"Using seed {conf.seed}")
    PRG_SEED(conf.seed)

    t = datetime.now()
    run = conf.runner
    run.init()
    logger.info(f"Init done in {datetime.now() - t} s")

    t = datetime.now()
    out_stream = open(os.devnull, 'w') if quiet else sys.stdout
    try:
        while True:
            tr = run.step()
            for i, (old_state, _, lbl, new_state) in enumerate(tr):
                out_stream.write(f"{i}: {old_state} -- {lbl} --> {new_state}\n")  # noqa: E501
    except (StopRunner, KeyboardInterrupt, EndOfFiniteTrace) as e:
        logger.warning(f"Stopping due to {repr(e)}")
    end = datetime.now()
    logger.info(f"{run.count-1} steps done in {end - t} s")
    logger.info(f"total: {end - t0} s")
