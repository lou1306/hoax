from io import TextIOWrapper
from random import choices, choice
from hoa.core import HOA
from json import loads

def flip():
    return choice((False, True))


class Driver:
    def __init__(self, aps) -> None:
        self.aps = aps
        # self.resolvers = {ap: flip for ap in self.aps}

    def get(self):
        raise NotImplementedError


class RandomDriver(Driver):
    def __init__(self, aps) -> None:
        super().__init__(aps)

    def get(self) -> dict:
        result = choices((True, False), k=len(self.aps))
        return {ap: value for ap, value in zip(self.aps, result)}


class StreamDriver(Driver):
    def __init__(self, aps, stream: TextIOWrapper, diff: bool = False) -> None:
        super().__init__(aps)
        self.diff = diff
        self.stream = stream
        self.prev = {}

    def get(self) -> dict:
        line = self.read_next()
        result = ({**self.prev} if self.diff else {}) | self.to_dict(line)
        if self.diff:
            self.prev = result
        return result

    def read_next(self) -> dict:
        raise NotImplementedError


class JSONDriver(StreamDriver):
    def read_next(self) -> dict:
        line = self.stream.readline()
        return loads(line)


def handle(in_str: str) -> bool:
    in_str = in_str.strip().lower()
    try:
        in_int = int(in_str)
        return bool(in_int)
    except ValueError:
        pass
    return False if in_str in ('false', 'null', '') else bool(in_str)


def random_values(aut: HOA, ctrl=None):
    result = choices((True, False), k=len(aut.header.propositions))
    for c in (ctrl or []):
        result[c] = handle(input(f"{aut.header.propositions[c]}?"))
    return result