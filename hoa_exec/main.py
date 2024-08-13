import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from hoa.core import HOA
from hoa.parsers import HOAParser

from .config.config import Configuration, DefaultConfig
from .runners import Automaton, StopRunner

app = typer.Typer()

log = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

@app.command()
def main(
        file: Path,
        config: Annotated[Path, typer.Option(help="Path to a TOML config file")] = None  # noqa: E501
):
    """Execute a HOA automaton from FILE"""

    # Work around Strix extensions to the HOA format
    input_lines = Path(file).read_text().splitlines()
    input_string = "\n".join(
        line for line in input_lines
        if not line.startswith("controllable"))
    # Read in Strix controllable APs
    control = [x for x in input_lines if x.startswith("controllable")]
    if control:
        control = control[0].split(":")[1].split()
        control = sorted(set(int(x) for x in control))

    parser = HOAParser()
    hoa_obj: HOA = parser(input_string)
    aut = Automaton(hoa_obj)

    conf = (
        Configuration.factory(config, aut)
        if config is not None
        else DefaultConfig(aut))

    if control:
        pprint = ', '.join(hoa_obj.header.propositions[i] for i in control)
        log.info(f"Found {len(control)} controllable APs: {pprint}")

    run = conf.get_runner()
    run.init()

    while True:
        try:
            run.step()
        except (StopRunner, KeyboardInterrupt):
            print()
            log.debug("Stopping")
            log.debug("Printing trace:")
            for t in run.trace:
                print(t)
            break
