"""
Utility functions for validating that critical tokens (numbers, URLs, etc.)
survive translation without alteration.

Assumptions
-----------
- Simple regular expressions can capture tokens we consider invariant.

Risks
-----
- Locale variants may break the regular expressions and yield false alarms.

Alternatives
------------
- A curated glossary or semantic comparison could provide stronger guarantees.

Rationale
---------
- Catch obvious mistranslations by verifying that important tokens appear in
  the output with the same counts as in the source text.
"""

from collections import Counter
import re

# Token categories and their matching patterns. The keys are stable so callers
# can request checks for specific categories.
PATTERNS: dict[str, str] = {
    "NUMBER": r"\b\d+(?:\.\d+)?\b",
    "DATE": r"\b\d{4}-\d{2}-\d{2}\b",
    "URL": r"https?://\S+",
    "EMAIL": r"[\w.-]+@[\w.-]+",
    "UNIT": r"\b(?:%|°C|MB|km)\b",
    "CURRENCY": r"\b(?:\$|€|원)\d+\b",
    "CODE": r"`[^`]+`",
}


def missing_tokens(src: str, tgt: str, types: list[str] | None = None) -> dict[str, list[str]]:
    """Return missing invariant tokens grouped by type.

    Parameters
    ----------
    src: str
        Source text before translation.
    tgt: str
        Target text produced by translation.
    types: list[str] | None
        Optional subset of token categories to check. If ``None`` all known
        categories are verified.

    Returns
    -------
    dict[str, list[str]]
        Mapping of token type to the list of tokens that were present in
        ``src`` but not in ``tgt``. An empty mapping means the check passed.
    """

    missing: dict[str, list[str]] = {}
    for name, pat in PATTERNS.items():
        if types is not None and name not in types:
            continue
        src_tokens = re.findall(pat, src)
        if not src_tokens:
            continue
        tgt_tokens = Counter(re.findall(pat, tgt))
        absent: list[str] = []
        for token in src_tokens:
            if tgt_tokens.get(token, 0):
                tgt_tokens[token] -= 1
            else:
                absent.append(token)
        if absent:
            missing[name] = absent
    return missing


def preserve_ok(src: str, tgt: str, types: list[str] | None = None) -> bool:
    """Return ``True`` when all requested tokens are preserved.

    ``types`` allows callers to restrict validation to a subset of token
    categories, for example only numbers or dates.
    """

    return not missing_tokens(src, tgt, types)


# Checklist:
# - [x] Think Harder
# - [x] Think Deeper
# - [x] More Information
# - [x] Check Again

