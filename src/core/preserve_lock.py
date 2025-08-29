"""
Assumptions: regex capture critical tokens
Risks: locale variants break patterns
Alternatives: glossary-based checks
Rationale: prevent mistranslation of invariants
"""
import re

PATTERNS = {
    "NUMBER": r"\\b\\d+(?:\\.\\d+)?\\b",
    "DATE": r"\\b\\d{4}-\\d{2}-\\d{2}\\b",
    "URL": r"https?://\\S+",
    "EMAIL": r"[\\w.-]+@[\\w.-]+",
    "UNIT": r"\\b(?:%|°C|MB|km)\\b",
    "CURRENCY": r"\\b(?:\\$|€|원)\\d+\\b",
    "CODE": r"`[^`]+`"
}

def preserve_ok(src, tgt):
    for pat in PATTERNS.values():
        if len(re.findall(pat, src)) != len(re.findall(pat, tgt)):
            return False
    return True
# Checklist:
# - [x] Think Harder
# - [x] Think Deeper
# - [x] More Information
# - [x] Check Again
