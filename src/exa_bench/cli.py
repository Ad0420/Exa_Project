"""Command-line entry point: `python -m exa_bench <command> [options]`."""

import argparse
from collections.abc import Sequence

from exa_bench import crosscheck_cli, depth_cli, grade_cli, policy_cli, policy_eval_cli, probe_cli
from exa_bench.policy import POLICIES
from exa_bench.stats import DEFAULT_RESAMPLES
from exa_filters.results import NullPolicy


def main(argv: Sequence[str]) -> int:
    args = _parse(argv)
    if args.command == "policy-eval":
        return policy_eval_cli.run_policy_eval(seed=args.seed, resamples=args.resamples)
    if args.command == "policy":
        return policy_cli.run_policy_command(
            policy=args.policy,
            null_policy=args.null_policy,
            tolerance=args.tolerance,
            yes=args.yes,
        )
    if args.command == "probe":
        return probe_cli.run_probe(count=args.count, seed=args.seed, yes=args.yes)
    if args.command == "grade":
        return grade_cli.run_grade(seed=args.seed, yes=args.yes)
    if args.command == "crosscheck":
        return crosscheck_cli.run_crosscheck(
            seed=args.seed, yes=args.yes, price_in=args.price_in, price_out=args.price_out
        )
    if args.command == "stability":
        return depth_cli.run_stability(yes=args.yes, repeat=args.repeat)
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

    stability = subparsers.add_parser(
        "stability", help="check that Exa's top-10 is a stable prefix of its top-100"
    )
    stability.add_argument("--yes", action="store_true", help=spend_help)
    stability.add_argument(
        "--repeat",
        action="store_true",
        help="repeat the shallow searches unchanged instead of fetching deep ones",
    )

    policy = subparsers.add_parser(
        "policy", help="run one fetch policy over the evaluation workload"
    )
    policy.add_argument("--policy", required=True, choices=list(POLICIES))
    policy.add_argument(
        "--null-policy", default=NullPolicy.STRICT.value, choices=[p.value for p in NullPolicy]
    )
    policy.add_argument(
        "--tolerance", type=float, default=0.0, help="headcount tolerance, e.g. 0.2"
    )
    policy.add_argument("--yes", action="store_true", help=spend_help)

    policy_eval = subparsers.add_parser(
        "policy-eval", help="aggregate the policy run records into results/policy_eval.json"
    )
    policy_eval.add_argument("--seed", type=int, default=policy_eval_cli.DEFAULT_SEED)
    policy_eval.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    return parser.parse_args(argv)
