import os, sys

sys.path.append(os.path.abspath('.'))
from src.core.runner import run_two_stage


class DummyLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, *, max_tokens, stop):
        self.calls.append({'max_tokens': max_tokens, 'stop': stop})
        return {'text': '<think>t</think>out'}


def test_single_call_and_parse():
    llm = DummyLLM()
    res = run_two_stage(llm, 'p', 10, 20)
    assert res == {'think': 't', 'answer': 'out'}
    assert llm.calls == [{'max_tokens': 30, 'stop': ['</think>', '}\n\n']}]


class PlainLLM:
    def generate(self, prompt, *, max_tokens, stop):
        return {'text': 'hello'}


def test_no_think_segment():
    llm = PlainLLM()
    res = run_two_stage(llm, 'p', 5, 5)
    assert res == {'think': '', 'answer': 'hello'}
