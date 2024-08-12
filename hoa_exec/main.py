from pathlib import Path
import logging

from .drivers import random_values
from hoa.core import HOA
from hoa.parsers import HOAParser
from .stepping import first_match

FNAME = "GFA_and_GFb.hoa"

logging.getLogger().setLevel(logging.INFO)


def main():
    # Work around Strix extensions to the format
    logging.info(f"Parsing {FNAME}")
    input_lines = Path(FNAME).read_text().splitlines()
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
    conf = DefaultConfig(hoa_obj.header.propositions)

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
            f"{'' if val else '!'}{prop}"
            for prop, val in valuation.items())
        logging.info(f"Values: {', '.join(pprint_valuation)}")
        cur_state = first_match(hoa_obj, cur_state, valuation)
        logging.info(f"New state: {cur_state}")
        input("Press [Enter] to continue...")
