## HOAX: Hanoi Omega-Automata eXecutor

This tool executes one or more automata expressed in HOA format.
Execution may be customised in several ways by means of config files.

The tool requires Python >= 3.12 and [`uv`](https://docs.astral.sh/uv/).

After cloning this repository:

```
cd hoa-exec
# Run
uv run hoax examples/nondet.hoa --config examples/flip.toml
```

Use Ctrl-C to stop. Use

```
uv run hoax --help
```

For usage instructions.