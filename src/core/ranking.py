"""
Assumptions: features normalized to 0..1
Risks: missing features may skew score
Alternatives: train ML ranker
Rationale: deterministic formula for explainability
"""

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class Features:
    text: float = 0.0
    role: float = 0.0
    near: float = 0.0
    spatial: float = 0.0
    disabled: float = 0.0


def rank(feat: Features) -> float:
    score = 0.52 * feat.text + 0.18 * feat.role + 0.20 * feat.near + 0.10 * feat.spatial - 0.05 * feat.disabled
    return score


@dataclass
class Candidate:
    id: str
    features: Features


def rank_candidates(cands: Iterable[Candidate]) -> List[Candidate]:
    """Return candidates sorted by ranking score descending."""

    return sorted(cands, key=lambda c: rank(c.features), reverse=True)
# Checklist:
# - [x] Think Harder
# - [x] Think Deeper
# - [x] More Information
# - [x] Check Again
