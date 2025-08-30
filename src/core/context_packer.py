"""Context packing helpers.

Assumptions
-----------
Section *caps* limit the number of characters contributed by each context
component.  This keeps any single source (e.g. screen digest) from dominating
the final prompt.

Risks
-----
Brute character counts are a naive stand‑in for real token accounting and may
truncate in the middle of words or multibyte characters.

Alternatives
------------
Adaptive summarisation could balance sections based on importance.

Rationale
---------
Predictable, deterministic truncation is easy to reason about and cheap to
compute.
"""

from dataclasses import dataclass
from typing import Iterable


@dataclass
class Caps:
    """Maximum character lengths for each packed section."""

    system: int = 0
    device_state: int = 0
    digest: int = 0
    candidates: int = 0
    user_utterance: int = 0
    target_text: int = 0


def truncate(text: str | None, cap: int) -> str:
    """Return ``text`` limited to ``cap`` characters.

    ``None`` is treated as an empty string.  The function is intentionally
    simple; higher level token‑aware truncation can replace it later.
    """

    if not text:
        return ""
    if cap <= 0:
        return ""
    return text[:cap]


def format_candidates(candidates: Iterable) -> str:
    """Return a newline separated ``id:text`` list for ``candidates``.

    Each item is expected to expose ``id`` and ``text`` attributes.  Unknown
    attributes fall back to an empty string, keeping the function forgiving for
    simple mocks in tests.
    """

    lines = []
    for c in candidates:
        cid = getattr(c, "id", "")
        txt = getattr(c, "text", "")
        lines.append(f"{cid}:{txt}")
    return "\n".join(lines)


def pack(
    system,
    device_state,
    digest,
    candidates,
    user_utterance,
    target_text,
    caps,
):
    """Pack context fragments into a newline‑delimited string.

    Parameters
    ----------
    system, device_state, digest, user_utterance, target_text:
        Individual text sections contributing to the final prompt.
    candidates:
        Iterable of objects exposing ``id`` and ``text`` attributes. They are
        formatted as ``id:text`` lines and then truncated.
    caps:
        :class:`Caps` instance limiting the number of characters drawn from each
        section.

    Returns
    -------
    str
        Concatenation of the clipped sections separated by newlines. Empty
        strings are inserted for ``None`` values or non‑positive caps.
    """

    ctx = []
    ctx.append(truncate(system, caps.system))
    ctx.append(truncate(device_state, caps.device_state))
    ctx.append(truncate(digest, caps.digest))
    ctx.append(truncate(format_candidates(candidates), caps.candidates))
    ctx.append(truncate(user_utterance, caps.user_utterance))
    ctx.append(truncate(target_text, caps.target_text))
    return "\n".join(ctx)


# Checklist:
# - [x] Think Harder
# - [x] Think Deeper
# - [x] More Information
# - [x] Check Again
