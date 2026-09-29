# Hard constraints in Exa company search

**Problem.** On Exa's own company-search benchmark, 12.1% of the returned companies that can be
checked break a constraint in the query, and 24.1% when the constraint is about a fact that
changes, such as headcount or funding. The API returns those fields but cannot filter on them.

**Solution.** A filter layer on the public API that enforces the constraints, which takes hard
queries from 17% to 69% filled with one call, and a design for enforcing them inside Exa's index,
where the ceiling on filtering after retrieval goes away.

**Why it matters.** Company search is sold for list building, where every wrong company is a bad
lead. Filtering in the index makes every returned company match the query without over-fetching.

Every number below comes from the committed files in `results/` and can be regenerated with the
commands at the end. Queries come from Exa's own benchmark,
[exa-labs/benchmarks](https://github.com/exa-labs/benchmarks) (announced in Exa's post
[Introducing Exa's Company Search Benchmarks](https://exa.ai/blog/company-search-benchmarks)),
pinned to commit `c096f1a` and checked by SHA-256. Its company file holds 839 queries, 605 of them
search tests. The 303 graded here are the search tests with at least one constraint on a typed
field Exa returns; two whose only such constraint is a negation are left out.

## Findings

### 1. One result in eight breaks a constraint, one in four on facts that change

Each of the 303 gradable queries was sent to `/search` (`category: "company"`, `auto`, 10
results), and each returned company was graded against the query's constraints using Exa's own
typed entity data.

| results that could be evaluated | breaking a constraint | 95% CI |
|---|---|---|
| all: 2,856 of the 3,030 | 12.1% | 9.7–14.8% |
| queries on changing facts: headcount, funding, stage (1,267) | 24.1% | 19.6–28.8% |
| queries on fixed facts: founding year, headquarters (1,589) | 2.6% | 1.5–4.0% |

174 results lacked a field their query needed and are excluded from these rates. Over all 3,030
results the violation rate is at least 11.5%, and at most 17.2% if every excluded result were a
violation. 112 of the 303 queries (37%) return at least one violating company in the top 10.

By constraint, over the checks that had a value to compare, funding stage fails most often
(25.3%), then funding amount (18.3%), funding date (14.4%), headcount (11.5%), founding year
(3.0%) and headquarters country (1.8%). The stage field is missing for 24.9% of its checks and
the funding total for 10.7%; headcount is never missing. With the 20% tolerance Exa's own grader
allows, applied here to headcount, the overall rate is 10.8%.
Source: `results/company_constraint_benchmark.json`.

### 2. The grading holds up where anything can check it

A sample of 300 graded results was checked against the companies' own pages twice: an LLM
field-by-field fact extraction, and Exa's own retrieval grader from the benchmark repo, ported
verbatim. Pages rarely state headcount or funding, so 115 of 120 sampled violating results cannot
be confirmed or refuted from page text. Where a page does decide, 4 of 5 violations are
confirmed, and Exa's grader agrees with ours on 85% of the results it can decide (33 of 39).
Source: `results/company_crosscheck.json`.

### 3. A patch from the outside: filters on the public API

Company search cannot be filtered on the company fields it returns, and with
`category: "company"` it rejects `excludeDomains`. `src/exa_filters` adds the missing feature as
a layer over the public API:

- **Hard filters** on six typed fields: headcount, total funding, founding year, date of the
  latest round, headquarters country, and funding stage.
- **Exclusions** by Exa entity id or domain, so a follow-up search does not return companies
  already seen. A company returned twice is kept once.
- **No silent failures.** Every returned company satisfies every filter. Companies missing a
  filtered field are rejected by default, or returned flagged on request. When fewer than 10
  pass, the response returns fewer and says how many it is short; it never pads.
- **Fetch planning.** When too few candidates pass, it asks Exa for more within a call and cost
  budget, and states why it stopped.

```python
import os

from exa_filters.client import FilteredExa
from exa_filters.spec import Filters, In, Range

with FilteredExa(os.environ["EXA_API_KEY"]) as exa:
    response = exa.search(
        "fintech startups building payments infrastructure",
        Filters(employees=Range(lte=30), country=In(["SG", "MY"]), founded_year=Range(gte=2020)),
        exclude_entities=["acme.com"],
    )
```

### 4. The patch helps, but has a ceiling

On a 175-query workload weighted toward hard queries (145 whose top 10 were not all compliant,
30 whose top 10 were), with "filled" meaning 10 fully compliant companies:

| policy | filled | 95% CI | mean cost per query | p50 latency |
|---|---|---|---|---|
| baseline: one call of 10 | 17% | 12–23% | $0.007 | 1.1 s |
| `deep` search, one call of 10 | 26% | 19–32% | $0.012 | 5.7 s |
| prior: one call sized from benchmark pass rates | 59% | 51–66% | $0.014 | 1.1 s |
| fixed-25: one call of 25 | 69% | 62–76% | $0.022 | 1.1 s |
| adaptive: 10, then 25, then 100 | 77% | 71–83% | $0.051 | 2.5 s |

Deeper results pass less often: 70% of the results in a 10-result call pass, against 61% of the
results in a 25-result call. On the 52 hardest queries, 39% pass in the 10-result call, 21% in the
25-result call and 9% across the 100-result call. Exa's `deep` search returns 6.2 compliant
companies per query against 7.0 for `auto`. The prior policy sizes its call from pass rates on the
full benchmark, so on this harder workload it under-fetches by construction. Seven of the eleven
targets set for the policies failed.
Source: `results/policy_eval.json`, `results/policy/`.

### 5. Exa Agent fills less and runs slower on this task

On a seeded 40 of the 175 queries, Exa Agent was asked for up to 10 companies matching each
query, and each company it returned was graded the same way. Agent counts as filled only when all
10 pass, the same bar as one search call of 10:

| on the same 40 queries | filled | cost per query | p50 latency |
|---|---|---|---|
| one search call of 10 | 12.5% | $0.007 | 1.1 s |
| Agent, `low` effort | 2.5% | $0.025 | 20 s |
| Agent, `medium` effort | 22.5% | $0.10 | 35 s |
| filters, prior | 60% | $0.015 | 1.2 s |
| filters, adaptive | 75% | $0.057 | 2.6 s |

Grading uses the same typed data the filters select on, so this favors the filters by
construction; Agent may be right where the typed data is stale. 11–13% of the companies Agent
returned whose entity could be looked up break a constraint. Source: `results/agent_comparison.json`.

### 6. The fix belongs in the index

The ceiling in finding 4 is set by ranking: a post-filter can only keep what retrieval returned,
and the standard remedy for selective filters is to apply them during retrieval (arXiv 2507.21989,
section 3.1). Exa's vector database already filters before the vector scan, through inverted
indexes over filterable values (Exa blog, "How We Built a Web-Scale Vector Database"). The
proposal:

- `filters` and `excludeEntities` on `/search` for `category: "company"`, over the typed fields
  every result already carries;
- strict missing-value semantics by default, with an opt-in to include companies whose field is
  unknown; never pad with non-matching results;
- phase 1, a filter step over a deep internal candidate pool, behind a beta header, measuring
  how often the pool runs out;
- phase 2, if it does: the filter inside the candidate index, on every retrieval leg, switching
  between scanning the matches and probing clusters by exact match count;
- phase 3, freshness metadata for the fields that change.

## Method

- **Fixed in advance, reported as they came out.** Samples and seeds are constants in code. Each
  study's pass/fail checks are in code, and the results files report them as they fell: the
  stability gate failed, 7 of the 11 fetch-policy targets failed, 2 of the 4 Agent checks failed.
- **Graded on Exa's own data.** Constraints are checked against the typed fields in each response,
  so a violation is Exa disagreeing with itself, not with us.
- **No raw responses in git.** Exa's terms prohibit copying or distributing information obtained
  through its services, so responses are cached under `cache/` (gitignored) and `results/` holds
  aggregates only.
- **Tested.** Unit tests run on recorded fixtures and never touch the network. The fetch-policy
  and Agent analysis code, the workload sampler, the response cache, the coverage probe and the
  pricing table were mutation-tested: bugs were planted one at a time to confirm the tests catch
  them.

## Reproduce

```
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]" -c constraints.txt
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy && .venv/bin/pytest
```

Each study is a subcommand of `python -m exa_bench`. Commands that call a paid API are dry runs,
printing the calls and their list price, until given `--yes`. Keys come from the environment
(`EXA_API_KEY`, `OPENAI_API_KEY`).

| command | what it does |
|---|---|
| `probe` | how often company results carry each typed field |
| `grade` | finding 1: grade every gradable benchmark query |
| `crosscheck` | finding 2: check sampled verdicts against page text |
| `stability` | is the top 10 a stable prefix of the top 100, and is it repeatable |
| `policy`, `policy-eval` | finding 4: run each fetch policy, then aggregate |
| `agent`, `agent-grade`, `agent-eval` | finding 5: run Exa Agent, grade it, compare |

`src/exa_filters/` is the filter layer; `src/exa_bench/` holds the studies, one package each;
`data/` pinned inputs; `results/` aggregate outputs; `tests/` unit tests and fixtures.

## Limits

- The benchmark's company queries are Exa's, not a sample of real user traffic.
- Typed fields can be stale: a result graded as violating may be correct, and page text can
  rarely settle it.
- The fetch policies run through the public API, so their costs and latencies include network
  round trips that an in-index filter would not pay.
- The tolerant rate applies Exa's 20% window to headcount only. Funding near-misses, such as $82M
  against an $80M cap, remain violations.
