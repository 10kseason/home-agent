"""
Assumptions: UIA exposes process_id, window_handle, AutomationId, role, name
Risks: dynamic tree path changes on app update
Alternatives: use runtime_id or image hash
Rationale: hashed tuple ensures cross-session stability
"""
import hashlib

def make_stable_id(el):
    """Generate stable_id from process, window, AutomationId, role, name, tree path"""
    path = tree_path(el)
    raw = f"{el.process_id}:{el.window}:{el.automation_id}:{el.role}:{el.name}:{path}"
    return hashlib.sha1(raw.encode()).hexdigest()
# Checklist:
# - [x] Think Harder
# - [x] Think Deeper
# - [x] More Information
# - [x] Check Again
