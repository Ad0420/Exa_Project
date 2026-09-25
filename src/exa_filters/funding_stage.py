"""Match funding-stage names across the spellings Exa and its benchmark use.

Exa returns stages like "Series b", "Pre seed", "Venture"; queries say "Series B",
"Pre-Seed", "Seed". Matching is case-insensitive, treats hyphens as spaces, and
works on whole tokens so that "Seed" does not match "Pre seed".
"""


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
