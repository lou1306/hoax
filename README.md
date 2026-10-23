## HOAX: Hanoi Omega-Automata eXecutor

This tool executes one or more automata expressed in HOA format.
Execution may be customised in several ways by means of config files.

You can obtain it from `pip`:

```sh
pip install hoax-hoa-executor
```

If you only need the command-line tool, we recommend using `pipx` or `uv`:

```sh
pipx install hoax-hoa-executor  # or
uv tool install hoax-hoa-executor
```

## Running from sources (developers)

The tool requires Python >= 3.12 and [`uv`](https://docs.astral.sh/uv/).

After cloning this repository:

```
cd hoax
# Run
uv run hoax examples/nondet.hoa --config examples/flip.toml
```

Use Ctrl-C to stop. Use

```
uv run hoax --help
```

For usage instructions.

## Boolean-network acceptance monitoring

The ``hoax-bn`` walker can monitor an Emerson-Lei acceptance condition over
Boolean state-vector patterns:

```sh
uv run hoax-bn network.bnet 1000 --acceptance 'Fin(0x1) & Inf(1xx)'
```

A pattern is a fixed-width vector with ``x``/``X`` wildcards and a
comma-separated list is a union; ``&`` and ``|`` have equal precedence and
associate right, exactly as in the HOA acceptance parser.  Patterns read left
to right in sorted variable order by default; ``--acceptance-order file`` uses
the ``.bnet`` target order instead.

The monitor reports a verdict ``GOOD`` (every extension satisfies),
``BAD`` (every extension violates) or ``UGLY`` (both exist).  ``None`` is not a
verdict: if the analysis cannot decide within its BDD budget it simply keeps
walking.  In stutter-free mode a steady state is reported through the
condition's terminal rule and does not pretend an infinite successor run
exists.

## Publications 

If you use `hoax` in an academic project, please cite this paper:

```
@inproceedings{distefano2025executionmonitoringhoaautomata,
    title={Execution and monitoring of {HOA} automata with {HOAX}}, 
    author={Di Stefano, Luca},
    year={2025},
    booktitle={25th International Conference on Runtime Verification ({RV})}
    url={https://arxiv.org/abs/2507.11126},
    pubstate={To appear}
}
```
