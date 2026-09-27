"""The policy-eval command: aggregate the run records into results/policy_eval.json. No spend."""

import json
from dataclasses import asdict
from pathlib import Path

from exa_bench.policy_analysis import Check, Evaluation, build_evaluation, load_record
from exa_bench.stats import DEFAULT_RESAMPLES, Rate

RECORDS_DIR = Path("results/policy")
RESULTS_PATH = Path("results/policy_eval.json")
DEFAULT_SEED = 20260926


def run_policy_eval(*, seed: int = DEFAULT_SEED, resamples: int = DEFAULT_RESAMPLES) -> int:
    paths = sorted(RECORDS_DIR.glob("*.json"))
    if not paths:
        print(f"No run records in {RECORDS_DIR}; run `policy --yes` first.")
        return 2
    records = {path.stem: load_record(path) for path in paths}
    evaluation = build_evaluation(records, seed=seed, resamples=resamples)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(asdict(evaluation), indent=2) + "\n", encoding="utf-8")
    _print_runs(evaluation)
    _print_checks(evaluation.checks)
    print(f"\nExa-reported spend over these runs: ${evaluation.metadata.spent_usd_total:.2f}")
    print(f"Wrote {RESULTS_PATH}")
    return 0


def _print_runs(evaluation: Evaluation) -> None:
    meta = evaluation.metadata
    print(
        f"{meta.queries} queries ({meta.clean_sample} from the clean sample), k={meta.k}; "
        f"95% cluster-bootstrap CIs by query, seed {meta.bootstrap_seed}"
    )
    header = f"{'run':18} {'fill (95% CI)':22} {'non-clean':>9} {'dynamic':>8} {'1 call':>6} "
    header += f"{'calls':>5} {'$ mean':>7} {'lat p50':>8} {'lat p95':>8} {'viol':>6} {'unk':>6}"
    print(header)
    for name, run in evaluation.runs.items():
        print(
            f"{name:18} {_rate(run.fill_rate):22} "
            f"{_pct(run.fill_rate_by_sample.get('non_clean')):>9} "
            f"{_pct(run.fill_rate_by_split.get('dynamic')):>8} {run.one_call_share:>6.0%} "
            f"{run.mean_calls:>5.2f} {run.cost_usd.mean:>7.4f} {run.latency_ms.p50:>8.0f} "
            f"{run.latency_ms.p95:>8.0f} {_pct(run.violating_share):>6} "
            f"{_pct(run.unevaluable_share):>6}"
        )
    print("\nAgainst the baseline (same null policy):")
    for name, c in evaluation.comparisons.items():
        print(
            f"  {name:18} fill {c.fill_gain:+d}; cost x{_num(c.cost_ratio_mean)} "
            f"(p50 x{_num(c.cost_ratio_p50)}); latency p50 x{_num(c.latency_ratio_p50)}, "
            f"p95 x{_num(c.latency_ratio_p95)}"
        )
    if evaluation.null_policy_effect:
        print(f"\nLenient fills beyond strict (H-N): {evaluation.null_policy_effect}")


def _print_checks(checks: list[Check]) -> None:
    print("\nPre-registered checks:")
    for check in checks:
        verdict = "n/a " if check.passed is None else ("PASS" if check.passed else "FAIL")
        print(
            f"  {verdict} {check.hypothesis:5} {check.claim}: "
            f"{_num(check.value)} {check.direction} {check.threshold:g}"
        )


def _rate(rate: Rate) -> str:
    if rate.value is None:
        return "n/a"
    if rate.ci_low is None or rate.ci_high is None:
        return f"{rate.value:.1%}"
    return f"{rate.value:.1%} [{rate.ci_low:.1%}-{rate.ci_high:.1%}]"


def _pct(rate: Rate | None) -> str:
    return "n/a" if rate is None or rate.value is None else f"{rate.value:.1%}"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"
