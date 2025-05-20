import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from signal import SIG_DFL, SIGPIPE, signal
from typing import Annotated, Optional

import typer

from . import __version__
from .config.config import Configuration, DefaultConfig
from .drivers import EndOfFiniteTrace
from .hoa import parse
from .runners import StopRunner

signal(SIGPIPE, SIG_DFL)

app = typer.Typer()


def print_version(version: bool):
    if version:
        print(f"hoa_exec {__version__}")
        raise typer.Exit()


@app.command()
def main(
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

    with ThreadPoolExecutor() as exc:
        automata = list(exc.map(parse, files))

    conf = (
        Configuration.factory(config, automata, monitor)
        if config is not None
        else DefaultConfig(automata, monitor))

    run = conf.get_runner()

    t = datetime.now()
    run.init()
    try:
        if quiet:
            while True:
                run.step()
        else:
            while True:
                tr = run.step()
                if tr:
                    print(*(f"{i}: {t}" for i, t in enumerate(tr)), sep="\n")
    except (StopRunner, KeyboardInterrupt, EndOfFiniteTrace) as e:
        print(f"Stopping due to {repr(e)}", file=sys.stderr)
    print(run.count, "steps,", datetime.now() - t, "seconds", file=sys.stderr)
