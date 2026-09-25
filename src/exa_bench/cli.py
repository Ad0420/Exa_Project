"""Command-line entry point: `python -m exa_bench <command> [options]`."""

import argparse
from collections.abc import Sequence

from exa_bench.probe_cli import DEFAULT_COUNT, DEFAULT_SEED, run_probe


def main(argv: Sequence[str]) -> int:
    args = _parse(argv)
    if args.command == "probe":
        return run_probe(count=args.count, seed=args.seed, yes=args.yes)
    raise AssertionError(f"unhandled command {args.command!r}")


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m exa_bench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe", help="measure typed-field coverage on real searches")
    probe.add_argument("--count", type=int, default=DEFAULT_COUNT)
    probe.add_argument("--seed", type=int, default=DEFAULT_SEED)
    probe.add_argument("--yes", action="store_true", help="actually call Exa (spends credits)")
    return parser.parse_args(argv)
