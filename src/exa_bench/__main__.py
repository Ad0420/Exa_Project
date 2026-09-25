"""Command-line entry point: `python -m exa_bench probe [--yes]`."""

import sys

from exa_bench.probe_cli import main

sys.exit(main(sys.argv[1:]))
