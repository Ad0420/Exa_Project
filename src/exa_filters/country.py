"""Country normalization: names, ISO 3166 codes, and common aliases resolve to one key.

Exa returns mostly full names ("Germany") but sometimes ISO codes ("DE"); queries say
"UK", "South Korea", "Turkey". Comparing raw strings would count those as mismatches.
"""

import pycountry

# Spellings that ISO 3166 lookup does not resolve on its own.
_ALIASES = {
    "uk": "GB",
    "u.k.": "GB",
    "great britain": "GB",
    "england": "GB",
    "usa": "US",
    "u.s.": "US",
    "u.s.a.": "US",
    "united states of america": "US",
    "america": "US",
    "south korea": "KR",
    "korea": "KR",
    "north korea": "KP",
    "russia": "RU",
    "taiwan": "TW",
    "vietnam": "VN",
    "iran": "IR",
    "turkey": "TR",
    "türkiye": "TR",
    "czech republic": "CZ",
    "czechia": "CZ",
    "the netherlands": "NL",
    "holland": "NL",
    "uae": "AE",
    "united arab emirates": "AE",
    "syria": "SY",
    "laos": "LA",
    "bolivia": "BO",
    "venezuela": "VE",
    "tanzania": "TZ",
    "moldova": "MD",
    "brunei": "BN",
    "macau": "MO",
    "palestine": "PS",
    "ivory coast": "CI",
    "cape verde": "CV",
    "burma": "MM",
    "the gambia": "GM",
    "the bahamas": "BS",
}


def country_key(text: str) -> str:
    """ISO alpha-2 code when `text` names a country; otherwise the cleaned text itself."""
    cleaned = " ".join(text.strip().casefold().split())
    if cleaned in _ALIASES:
        return _ALIASES[cleaned]
    try:
        return str(pycountry.countries.lookup(cleaned).alpha_2)
    except LookupError:
        return cleaned


def same_country(first: str, second: str) -> bool:
    return country_key(first) == country_key(second)
