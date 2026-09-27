"""The agent-eval command: compare Agent with the policies from the stored records. No spend.

Only records at headcount tolerance 0 are read (file names `graded.<effort>.json` and
`<policy>.<null_policy>.json`), so every side is judged by the same filters.
"""

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from exa_bench.agent_analysis import AgentEvaluation, build_agent_evaluation, load_graded_record
from exa_bench.agent_cli import RESULTS_DIR as AGENT_RESULTS_DIR
from exa_bench.policy_analysis import load_record
from exa_bench.policy_eval_cli import DEFAULT_SEED, RECORDS_DIR
from exa_bench.stats import DEFAULT_RESAMPLES, Rate

RESULTS_PATH = Path("results/agent_comparison.json")


def run_agent_eval(*, seed: int = DEFAULT_SEED, resamples: int = DEFAULT_RESAMPLES) -> int:
    graded = {
        path.stem.removeprefix("graded."): load_graded_record(path)
        for path in _untolerant(AGENT_RESULTS_DIR.glob("graded.*.json"))
    }
    policies = {path.stem: load_record(path) for path in _untolerant(RECORDS_DIR.glob("*.json"))}
    if not graded or not policies:
        print(f"Needs graded records in {AGENT_RESULTS_DIR} and policy records in {RECORDS_DIR}.")
        return 2
    evaluation = build_agent_evaluation(policies, graded, seed=seed, resamples=resamples)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(asdict(evaluation), indent=2) + "\n", encoding="utf-8")
    _print(evaluation)
    print(f"Wrote {RESULTS_PATH}")
    return 0


def _untolerant(paths: Iterable[Path]) -> list[Path]:
    """Record files named `<a>.<b>.json`: a tolerance adds a third part, as in `.tol0.2`."""
    return sorted(path for path in paths if path.stem.count(".") == 1)


def _print(evaluation: AgentEvaluation) -> None:
    meta = evaluation.metadata
    print(f"{meta.queries} queries (seed {meta.subset_seed}), k={meta.k}")
    for effort, agent in evaluation.agents.items():
        print(
            f"  agent.{effort:9} fill {_pct(agent.fill_rate_strict)} strict, "
            f"{_pct(agent.fill_rate_lenient)} lenient; {agent.mean_companies:.1f} companies/run, "
            f"{_pct(agent.violating_share)} of those found violate; "
            f"${agent.cost_usd.mean:.4f}/query; p50 {agent.latency_ms.p50 / 1000:.1f} s"
        )
    for name, policy in evaluation.policies.items():
        print(
            f"  {name:15} fill {_pct(policy.fill_rate)}; ${policy.cost_usd.mean:.4f}/query; "
            f"p50 {policy.latency_ms.p50 / 1000:.1f} s"
        )
    for check in evaluation.checks:
        verdict = "n/a " if check.passed is None else ("PASS" if check.passed else "FAIL")
        value = "n/a" if check.value is None else f"{check.value:.2f}"
        print(f"  {verdict} {check.hypothesis} {check.claim}: {value}")


def _pct(rate: Rate) -> str:
    return "n/a" if rate.value is None else f"{rate.value:.1%}"
