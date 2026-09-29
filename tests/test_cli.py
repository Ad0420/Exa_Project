"""Tests for the command-line dispatcher: routing, defaults, and dry run by default."""

from types import ModuleType

import pytest

from exa_bench import cli
from exa_bench.agent import cli as agent_cli
from exa_bench.agent import eval_cli as agent_eval_cli
from exa_bench.agent import grade_cli as agent_grade_cli
from exa_bench.constraints import grade_cli, probe_cli
from exa_bench.core.stats import DEFAULT_RESAMPLES
from exa_bench.crosscheck import cli as crosscheck_cli
from exa_bench.depth import cli as depth_cli
from exa_bench.policies import cli as policy_cli
from exa_bench.policies import eval_cli as policy_eval_cli

CASES: list[tuple[list[str], ModuleType, str, dict[str, object]]] = [
    (
        ["probe"],
        probe_cli,
        "run_probe",
        {"count": probe_cli.DEFAULT_COUNT, "seed": probe_cli.DEFAULT_SEED, "yes": False},
    ),
    (["grade", "--yes"], grade_cli, "run_grade", {"seed": grade_cli.DEFAULT_SEED, "yes": True}),
    (
        ["crosscheck", "--price-in", "1.5"],
        crosscheck_cli,
        "run_crosscheck",
        {"seed": crosscheck_cli.DEFAULT_SEED, "yes": False, "price_in": 1.5, "price_out": None},
    ),
    (["stability", "--repeat"], depth_cli, "run_stability", {"yes": False, "repeat": True}),
    (
        ["policy", "--policy", "deep", "--null-policy", "lenient", "--tolerance", "0.2"],
        policy_cli,
        "run_policy_command",
        {"policy": "deep", "null_policy": "lenient", "tolerance": 0.2, "yes": False},
    ),
    (
        ["policy-eval"],
        policy_eval_cli,
        "run_policy_eval",
        {"seed": policy_eval_cli.DEFAULT_SEED, "resamples": DEFAULT_RESAMPLES},
    ),
    (["agent", "--effort", "medium"], agent_cli, "run_agent", {"effort": "medium", "yes": False}),
    (
        ["agent-grade", "--tolerance", "0.2"],
        agent_grade_cli,
        "run_agent_grade",
        {"effort": "low", "tolerance": 0.2, "yes": False},
    ),
    (
        ["agent-eval", "--resamples", "50"],
        agent_eval_cli,
        "run_agent_eval",
        {"seed": policy_eval_cli.DEFAULT_SEED, "resamples": 50},
    ),
]


@pytest.mark.parametrize(("argv", "module", "function", "expected"), CASES)
def test_each_command_calls_its_study_with_the_parsed_options(
    argv: list[str],
    module: ModuleType,
    function: str,
    expected: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def record(**kwargs: object) -> int:
        calls.append(kwargs)
        return 7

    monkeypatch.setattr(module, function, record)

    assert cli.main(argv) == 7
    assert calls == [expected]


def test_every_spending_command_is_a_dry_run_unless_asked() -> None:
    spending = ["probe", "grade", "crosscheck", "stability", "agent", "agent-grade"]
    for command in [*spending, "policy"]:
        argv = [command, "--policy", "baseline"] if command == "policy" else [command]
        assert cli._parse(argv).yes is False, command


@pytest.mark.parametrize(
    "argv",
    [
        ["policy", "--policy", "fixed-100"],
        ["policy"],
        ["agent", "--effort", "auto"],
        ["policy", "--policy", "baseline", "--null-policy", "loose"],
        ["unknown-command"],
    ],
)
def test_invalid_options_are_rejected(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 2
