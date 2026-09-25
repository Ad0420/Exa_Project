"""Match funding-stage names across the spellings Exa and its benchmark use.

Exa returns stages like "Series b", "Pre seed", "Venture"; queries say "Series B",
"Pre-Seed", "Seed". Matching is case-insensitive, treats hyphens as spaces, and
works on whole tokens so that "Seed" does not match "Pre seed".

Only names on the venture ladder (angel, pre-seed, seed, series X) say what stage a
company is at. Other round types Exa reports ("Venture" is Crunchbase's "series
unknown", plus grants, debt, private equity, post-IPO rounds) are not evidence for or
against a stage, so `is_stage` lets callers treat them as unknown.
"""

import re

_LADDER = re.compile(r"^(angel|pre seed|seed|series( [a-z])?)$")


def is_stage(text: str) -> bool:
    """True if the round name is a venture-ladder stage rather than another round type."""
    return _LADDER.match(" ".join(normalize_stage(text))) is not None


def normalize_stage(text: str) -> tuple[str, ...]:
    """Lower-case tokens with hyphens and underscores treated as spaces."""
    return tuple(text.casefold().replace("-", " ").replace("_", " ").split())


def stage_matches(actual: str, expected: str) -> bool:
    """True if `expected` appears in `actual` as a whole-token phrase not preceded by "pre"."""
    haystack, needle = normalize_stage(actual), normalize_stage(expected)
    if not needle:
        return False
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] != needle:
            continue
        if start > 0 and haystack[start - 1] == "pre":
            continue
        return True
    return False
