"""Command-line entry point: `python -m exa_bench <command> [options]`."""

import argparse
from collections.abc import Sequence

from exa_bench import crosscheck_cli, grade_cli, probe_cli


def main(argv: Sequence[str]) -> int:
    args = _parse(argv)
    if args.command == "probe":
        return probe_cli.run_probe(count=args.count, seed=args.seed, yes=args.yes)
    if args.command == "grade":
        return grade_cli.run_grade(seed=args.seed, yes=args.yes)
    if args.command == "crosscheck":
        return crosscheck_cli.run_crosscheck(
            seed=args.seed, yes=args.yes, price_in=args.price_in, price_out=args.price_out
        )
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

    crosscheck = subparsers.add_parser(
        "crosscheck", help="check typed fields and our verdicts against page text"
    )
    crosscheck.add_argument("--seed", type=int, default=crosscheck_cli.DEFAULT_SEED)
    crosscheck.add_argument("--yes", action="store_true", help="fetch pages and call OpenAI")
    crosscheck.add_argument("--price-in", type=float, help="USD per 1M prompt tokens")
    crosscheck.add_argument("--price-out", type=float, help="USD per 1M completion tokens")
    return parser.parse_args(argv)
