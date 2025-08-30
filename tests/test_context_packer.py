import os, sys
from dataclasses import dataclass

sys.path.append(os.path.abspath('.'))
from src.core.context_packer import Caps, format_candidates, pack, truncate


@dataclass
class Cand:
    id: str
    text: str


def test_pack_truncates_and_formats():
    caps = Caps(system=5, device_state=5, digest=5, candidates=50, user_utterance=5, target_text=5)
    cands = [Cand("1", "hello"), Cand("2", "world")]
    res = pack("abcdef", "123456", "digest", cands, "utterance", "target-text", caps)
    parts = res.split("\n")
    assert parts[0] == "abcde"  # system truncated
    assert parts[1] == "12345"  # device_state truncated
    assert parts[2] == "diges"  # digest truncated
    assert parts[3] == "1:hello"  # candidate formatting line 1
    assert parts[4] == "2:world"  # candidate formatting line 2
    assert parts[5] == "utter"  # user utterance truncated
    assert parts[6] == "targe"  # target text truncated


def test_truncate_handles_none():
    assert truncate(None, 5) == ""


def test_format_candidates_missing_attrs():
    class Bare:
        pass

    formatted = format_candidates([Bare()])
    # Both id and text should gracefully fall back to empty strings
    assert formatted == ":"
