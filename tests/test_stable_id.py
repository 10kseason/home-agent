import os, sys
sys.path.append(os.path.abspath('.'))
from src.core.stable_id import make_stable_id


class FakeElement:
    def __init__(self, process_id, window, automation_id, role, name, parent=None):
        self.process_id = process_id
        self.window = window
        self.automation_id = automation_id
        self.role = role
        self.name = name
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)


def test_stable_id_path_and_cache():
    root = FakeElement(1, 1, 'root', 'window', 'root')
    btn1 = FakeElement(1, 1, 'btn', 'button', 'OK', parent=root)
    btn2 = FakeElement(1, 1, 'btn', 'button', 'OK', parent=root)

    sid1 = make_stable_id(btn1)
    sid2 = make_stable_id(btn1)
    assert sid1 == sid2  # cached

    sid3 = make_stable_id(btn2)
    assert sid1 != sid3  # different sibling index

    btn1.name = 'Changed'
    sid4 = make_stable_id(btn1)
    assert sid4 == sid1  # cache unaffected
