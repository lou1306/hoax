import logging
from pathlib import Path
from typing import Annotated

import typer
from hoa.core import HOA
from hoa.parsers import HOAParser

from .config import Configuration, DefaultConfig
from .stepping import first_match
from .config.config import Configuration, DefaultConfig

app = typer.Typer()


logging.getLogger().setLevel(logging.INFO)


@app.command()
def main(
        file: Path,
        config: Annotated[Path, typer.Option(help="Path to a TOML config file")] = None  # noqa: E501
):
    """Execute a HOA automaton from FILE"""

    # Work around Strix extensions to the HOA format
    logging.info(f"Parsing {file}")
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
    conf = (
        Configuration.factory(Path("config.toml"), hoa_obj.header.propositions)
        if config is not None
        else DefaultConfig(hoa_obj.header.propositions))
    driver = conf.get_driver()

    if control:
        pprint = ', '.join(hoa_obj.header.propositions[i] for i in control)
        logging.info(f"Found {len(control)} controllable APs: {pprint}")

    int2states = {x.index: x for x in hoa_obj.body.state2edges}

    # TODO support multiple initial states
    cur_state = next(iter(hoa_obj.header.start_states))
    # TODO support initial state conjunction (alternating automata)
    cur_state = next(iter(cur_state))
    logging.info(f"Initial state: {cur_state}")
    while True:
        cur_state = int2states[cur_state]
        valuation = driver.get()

        pprint_valuation = (
            f"{'' if valuation[ap] else '!'}{ap}"
            for ap in hoa_obj.header.propositions)
        logging.info(f"Values: {', '.join(pprint_valuation)}")
        cur_state = first_match(hoa_obj, cur_state, valuation)
        logging.info(f"New state: {cur_state}")
        input("Press [Enter] to continue...")
