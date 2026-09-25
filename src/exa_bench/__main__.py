"""Command-line entry point: `python -m exa_bench <command> [options]`."""

import sys

from exa_bench.cli import main

sys.exit(main(sys.argv[1:]))
