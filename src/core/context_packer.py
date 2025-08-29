"""
Assumptions: section caps prevent single source domination
Risks: truncation may remove key info
Alternatives: adaptive summarization
Rationale: predictable token usage
"""

def pack(system, device_state, digest, candidates, user_utt, target_text, caps):
    ctx = []
    ctx.append(truncate(system, caps.system))
    ctx.append(truncate(device_state, caps.device_state))
    ctx.append(truncate(digest, caps.digest))
    ctx.append(truncate(format_candidates(candidates), caps.candidates))
    ctx.append(truncate(user_utt, caps.user_utterance))
    ctx.append(truncate(target_text, caps.target_text))
    return "\n".join(ctx)
# Checklist:
# - [x] Think Harder
# - [x] Think Deeper
# - [x] More Information
# - [x] Check Again
