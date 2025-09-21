import logging
from abc import ABC, abstractmethod
from io import TextIOWrapper
from json import loads
from random import choices
import subprocess
import re
import os
import tempfile

log = logging.getLogger(__name__)


class EndOfFiniteTrace(Exception):
    pass


class Driver(ABC):
    def __init__(self, aps: list[str]) -> None:
        log.debug(f"Instantiating {type(self).__name__} for {aps=}")
        self.aps = aps

    @abstractmethod
    def get(self) -> set:
        raise NotImplementedError


class CompositeDriver(Driver):
    def __init__(self) -> None:
        self.aps: list[str] = []
        self.drivers: list[Driver] = []

    def append(self, driver: Driver):
        self.drivers.append(driver)
        assert all(ap not in self.aps for ap in driver.aps), \
            "Some APs have multiple drivers: " \
            f"{[ap for ap in self.aps if ap in driver.aps]}"
        self.aps.extend(driver.aps)

    def get(self) -> set:
        result = set()
        for d in self.drivers:
            result |= d.get()
        return result


class UserDriver(Driver):
    def __init__(self, aps: list[str]) -> None:
        super().__init__(aps)

    @staticmethod
    def handle(in_str: str) -> bool:
        in_str = in_str.strip().lower()
        try:
            in_int = int(in_str)
            return bool(in_int)
        except ValueError:
            pass
        return False if in_str in ('false', 'null', '') else bool(in_str)

    def get(self):
        result = set()
        for ap in self.aps:
            value = handle(input(f"{ap}? "))
            if value:
                result.add(ap)
        return result


class RandomDriver(Driver):
    pop = True, False

    def __init__(self, aps) -> None:
        self.aps = aps
        self.k = len(self.aps)
        self.cum_weights: tuple[float, int] | None = None

    def get(self) -> set:
        result = choices(self.pop, k=self.k, cum_weights=self.cum_weights)
        return set(ap for ap, value in zip(self.aps, result) if value)


class StreamDriver(Driver):
    def __init__(self, aps, stream: TextIOWrapper, diff: bool = False) -> None:
        super().__init__(aps)
        self.diff = diff
        self.stream = stream

    def get(self) -> set:
        return self.read_next()

    def read_next(self) -> set:
        raise NotImplementedError


class JSONDriver(StreamDriver):
    def read_next(self) -> set:
        line = self.stream.readline()
        return set(x for x, y in loads(line) if y)


class SimpleTxtDriver(StreamDriver):
    def read_next(self):
        line = self.stream.readline()
        if not line:
            self.stream.close()
            raise EndOfFiniteTrace(self.stream.name)
        data = set()
        while line:
            if line == "\n":
                return data
            ap = line[:-1]
            if ap in self.aps:
                data.add(ap)
            line = self.stream.readline()
        return data

import os
import re
import subprocess
import tempfile
from io import TextIOWrapper

from hoax.drivers import StreamDriver, EndOfFiniteTrace


class AssumptionTxtDriver(StreamDriver):
    def __init__(self,
                 aps: list[str],
                 assumption: str = None,
                 assumption_ins: str = None,
                 assumption_outs: str = None,
                 original_hoa: str = None,
                 bound: int = 20,
                 hoax_config: str = "./examples/random.toml",
                 stream: TextIOWrapper = None,
                 diff: bool = False):
        self.assumption = assumption
        self.ins = assumption_ins
        self.outs = assumption_outs
        self.original_hoa = original_hoa
        self.bound = bound
        self.hoax_config = hoax_config

        # If no stream provided, generate one
        if stream is None:
            stream = self._generate_stream(aps)

        super().__init__(aps=aps, stream=stream, diff=diff)

    def _generate_stream(self, aps):
        """
        1. Build monitor HOA from assumption using Spot.
        2. Run HOAX on monitor HOA to generate random trace.
        3. Parse HOAX trace to StreamDriver format.
        4. Return file stream to parsed trace.
        """
        import shutil
        from pathlib import Path

        # Step 1: Create temporary HOA file for assumption system
        tmp_monitor = tempfile.NamedTemporaryFile(delete=False, suffix=".hoa")
        monitor_name = tmp_monitor.name
        tmp_monitor.close()

        subprocess.run(
            ["ltl2tgba", "-DM", self.assumption, "-o", monitor_name],
            check=True,
        )

        # Step 2: Generate random trace from monitor HOA
        tmp_trace = tempfile.NamedTemporaryFile(delete=False, suffix=".raw")
        trace_name = tmp_trace.name
        tmp_trace.close()

        subprocess.run(
            f"hoax {monitor_name} --config {self.hoax_config} > {trace_name}",
            shell=True,
            check=True,
        )

        # Step 3: Parse trace to SimpleTxtDriver format
        reformatted = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        reformatted_name = reformatted.name
        reformatted.close()
        self._parse_trace(trace_name, aps, reformatted_name)

        # --- NEW: save a permanent copy for inspection ---
        dest_dir = Path("./generated_traces")
        dest_dir.mkdir(exist_ok=True)
        dest_path = dest_dir / f"assumption_trace_{os.getpid()}.txt"
        shutil.copy(reformatted_name, dest_path)
        print(f"[AssumptionTxtDriver] Saved generated trace to {dest_path}")

        # Optional: clean up intermediate HOA and raw trace
        os.unlink(monitor_name)
        os.unlink(trace_name)

        # Step 4: Return stream handle to reformatted file
        return open(reformatted_name, "r")

    def _parse_trace(self, trace_file: str, ap_list: list[str], out_path: str):
        """
        Convert HOAX raw output into StreamDriver (SimpleTxtDriver) format.
        """
        set_pat = re.compile(r"\{(.*?)\}")
        with open(trace_file, "r") as infile, open(out_path, "w") as outfile:
            for raw in infile:
                raw = raw.strip()
                if not raw:
                    continue
                m = set_pat.search(raw)
                if m:
                    items = m.group(1).strip()
                    if items == "":
                        present = set()
                    else:
                        present = {
                            tok.strip().strip("'\"")
                            for tok in items.split(",")
                            if tok.strip()
                        }
                else:
                    present = set()

                # One line per AP, blank line between timesteps
                for ap in ap_list:
                    outfile.write(f"{ap if ap in present else ''}\n")
                outfile.write("\n")

    def read_next(self):
        """
        Read one timestep of APs from the stream.
        """
        line = self.stream.readline()
        if not line:
            self.stream.close()
            raise EndOfFiniteTrace(self.stream.name)

        data = set()
        while line:
            if line == "\n":
                return data
            ap = line.strip()
            if ap in self.aps:
                data.add(ap)
            line = self.stream.readline()
        return data



def handle(in_str: str) -> bool:
    in_str = in_str.strip().lower()
    try:
        in_int = int(in_str)
        return bool(in_int)
    except ValueError:
        pass
    return False if in_str in ('false', 'null', '') else bool(in_str)
