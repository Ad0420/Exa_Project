# Hard constraints in Exa company search

**TL;DR.** I've been using Exa more frequently recently to find cool companies, but I noticed
that when I ask for specific things (like companies under x employees), some of the results don't seem to match the query. I was
curious how often this happens, so I ran the 303 queries in Exa's "company-search" benchmark
that have checkable conditions, and checked every result against the company data Exa returns. Of the results that could be checked, 12.1%
broke a condition in the query, and 24.1% broke when the condition was about headcount or funding. So I
built a filter layer on top of the public API to fix it from the outside. Each benchmark query
asks Exa for 10 companies. For about half the queries (158 of 303), all 10 meet the conditions.
For the other 145 (112 have a company that breaks a condition, 33 have one that can't be
checked), the filter layer fixes 91 (63%) with a single search for 25 results; none of the 30
clean queries I re-ran was broken by it. That approach has a ceiling, so I also wrote up how Exa could enforce the conditions
inside its index.

All numbers below come from the committed files in `results/` and can be regenerated with the
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

### 2. Independent checks agree with the grading

A sample of 300 graded results was checked against the companies' own pages twice: an LLM
extracting facts field by field, and Exa's own retrieval grader, ported verbatim. Pages rarely
state headcount or funding, so 115 of 120 sampled violating results cannot be confirmed or refuted.
Where a page does decide, 4 of 5 violations are confirmed, and Exa's grader agrees with ours on
85% of the results it can decide (33 of 39).
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

On 175 benchmark queries: 145 where a plain search's top 10 were not all compliant, and 30 where
they were. A query succeeds when all 10 returned companies meet every condition:

| how the results were fetched | all 10 match | 95% CI | mean cost per query | p50 latency |
|---|---|---|---|---|
| a normal search for 10 results, no filtering | 17% | 12–23% | $0.007 | 1.1 s |
| Exa's `deep` search for 10 results, no filtering | 26% | 19–32% | $0.012 | 5.7 s |
| one search for as many results as the benchmark's pass rates suggest, then filter | 59% | 51–66% | $0.014 | 1.1 s |
| one search for 25 results, then filter | 69% | 62–76% | $0.022 | 1.1 s |
| search for 10, then 25, then 100, filtering until 10 pass | 77% | 71–83% | $0.051 | 2.5 s |

The 17% in the first row is not a finding about Exa: the 175 queries were chosen as 145 that a
plain search got wrong and 30 it got right, so a plain search passes exactly those 30. What the
filter changes is the 145: 91 fixed with one search of 25, 105 with up to three, and all 30 clean
queries stayed clean.

Deeper results pass less often: on these 175 queries, 70% of the results in a 10-result call
pass, against 61% of the results in a 25-result call. On the 52 queries where 25 results were still not enough, 39% pass in
the 10-result call, 21% in the 25-result call and 9% across the 100-result call. Exa's `deep`
search returns 6.2 compliant companies per query against 7.0 for `auto`. The third row sizes its search from pass rates on
the full benchmark, and these queries are harder than average, so it asks for too few by
construction. Seven of the eleven targets I set for these strategies were missed.
Source: `results/policy_eval.json`, `results/policy/`.

### 5. Exa Agent gets fewer queries fully right, and takes longer

On a seeded 40 of the 175 queries, Exa Agent was asked for up to 10 companies matching each
query, and each company it returned was graded the same way. Agent succeeds only when all 10
pass, the same bar as a normal search for 10:

| on the same 40 queries | all 10 match | cost per query | p50 latency |
|---|---|---|---|
| a normal search for 10 results, no filtering | 12.5% | $0.007 | 1.1 s |
| Exa Agent, `low` effort | 2.5% | $0.025 | 20 s |
| Exa Agent, `medium` effort | 22.5% | $0.10 | 35 s |
| filter layer, one search sized from pass rates | 60% | $0.015 | 1.2 s |
| filter layer, 10 then 25 then 100 | 75% | $0.057 | 2.6 s |

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
`results/` aggregate outputs; `tests/` unit tests and fixtures.

## Limits

- The benchmark's company queries are Exa's, not a sample of real user traffic.
- Typed fields can be stale: a result graded as violating may be correct, and page text can
  rarely settle it.
- The fetch policies run through the public API, so their costs and latencies include network
  round trips that an in-index filter would not pay.
- The tolerant rate applies Exa's 20% window to headcount only. Funding near-misses, such as $82M
  against an $80M cap, remain violations.
