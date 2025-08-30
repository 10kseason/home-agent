"""Utilities for generating stable element identifiers.

Assumptions
----------
UIA (or a similar accessibility layer) exposes attributes such as
``process_id``, ``window`` handle, ``automation_id``, ``role`` and
``name``.  The element tree can change across application sessions so we
derive an identifier from a tuple of these attributes plus the position
within the accessibility tree.

Risks
-----
Dynamic UI updates may alter the tree path.  We cache the first value so
the ID remains stable for the lifetime of the object.

Alternatives
------------
Other systems use vendor specific runtime ids or image hashes.

Rationale
---------
A SHA1 hash of the attribute tuple gives us a short, deterministic id
that can be persisted or compared across runs.
"""

import hashlib
from weakref import WeakKeyDictionary

_CACHE: "WeakKeyDictionary" = WeakKeyDictionary()


def _tree_path(el) -> str:
    """Return a stable path for ``el`` based on its ancestry.

    Each segment is ``role[index]`` where ``index`` is the zero-based
    position amongst siblings.  Missing information is treated as
    best-effort; the function never raises.
    """

    path = []
    cur = el
    while getattr(cur, "parent", None) is not None:
        parent = cur.parent
        try:
            idx = parent.children.index(cur)
        except Exception:  # pragma: no cover - defensive
            idx = 0
        path.append(f"{getattr(parent, 'role', '?')}[{idx}]")
        cur = parent
    return "/".join(reversed(path))


def make_stable_id(el) -> str:
    """Generate (and cache) a stable id for a UI element.

    Parameters
    ----------
    el:
        An object exposing ``process_id``, ``window``, ``automation_id``,
        ``role``, ``name`` and navigation via ``parent``/``children``.
    """

    if el in _CACHE:
        return _CACHE[el]

    path = _tree_path(el)
    raw = (
        f"{getattr(el, 'process_id', '')}:"
        f"{getattr(el, 'window', '')}:"
        f"{getattr(el, 'automation_id', '')}:"
        f"{getattr(el, 'role', '')}:"
        f"{getattr(el, 'name', '')}:"
        f"{path}"
    )
    sid = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    _CACHE[el] = sid
    return sid
# Checklist:
# - [x] Think Harder
# - [x] Think Deeper
# - [x] More Information
# - [x] Check Again
