# Repository setup

See [development](development.md) for executable commands and CI policy. Python 3.11–3.14 is supported;
3.14 is preferred. `pyproject.toml` pins tools and strict Pyright; `ruff.toml` sets 120-column formatting.
Use isolated environments and verify wheels outside checkout. The distribution name is `ffa-methods`,
the internal namespace is `ffa_methods`, and the installed command is `ffa`.
