"""
Assumptions: features normalized to 0..1
Risks: missing features may skew score
Alternatives: train ML ranker
Rationale: deterministic formula for explainability
"""

def rank(feat):
    score = 0.52*feat.text + 0.18*feat.role + 0.20*feat.near + 0.10*feat.spatial - 0.05*feat.disabled
    return score
# Checklist:
# - [x] Think Harder
# - [x] Think Deeper
# - [x] More Information
# - [x] Check Again
