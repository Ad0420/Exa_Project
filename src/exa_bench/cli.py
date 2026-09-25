"""Command-line entry point: `python -m exa_bench <command> [options]`."""

import argparse
from collections.abc import Sequence

from exa_bench import grade_cli, probe_cli


def main(argv: Sequence[str]) -> int:
    args = _parse(argv)
    if args.command == "probe":
        return probe_cli.run_probe(count=args.count, seed=args.seed, yes=args.yes)
    if args.command == "grade":
        return grade_cli.run_grade(seed=args.seed, yes=args.yes)
    raise AssertionError(f"unhandled command {args.command!r}")


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m exa_bench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    spend_help = "actually call Exa (spends credits)"

    probe = subparsers.add_parser("probe", help="measure typed-field coverage on real searches")
    probe.add_argument("--count", type=int, default=probe_cli.DEFAULT_COUNT)
    probe.add_argument("--seed", type=int, default=probe_cli.DEFAULT_SEED)
    probe.add_argument("--yes", action="store_true", help=spend_help)

    grade = subparsers.add_parser("grade", help="grade every gradable benchmark query")
    grade.add_argument("--seed", type=int, default=grade_cli.DEFAULT_SEED)
    grade.add_argument("--yes", action="store_true", help=spend_help)
    return parser.parse_args(argv)
