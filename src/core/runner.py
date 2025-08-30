"""Lightweight two-stage LLM runner without latency management.

Assumes a local LLM backend such as LM Studio or Ollama is installed and
available. The runner simply asks the model once and parses out optional
``<think>`` and answer segments.

Risks
-----
Any model stall or failure surfaces directly to the caller without retries.

Alternatives
------------
A higher level supervisor could implement retries or clarification flows.

Rationale
---------
Keep the runner minimal so it works with locally hosted models.
"""

from __future__ import annotations

from typing import Dict, Protocol


class _LLM(Protocol):
    """Minimal protocol for the LLM used in ``run_two_stage``."""

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        stop: list[str],
    ) -> Dict[str, object]:
        ...


def _parse(text: str) -> Dict[str, str]:
    """Parse the LLM output into ``think`` and ``answer`` sections."""

    think = ""
    answer = text
    start = text.find("<think>")
    end = text.find("</think>")
    if start != -1 and end != -1 and end > start:
        think = text[start + len("<think>") : end].strip()
        answer = text[end + len("</think>") :].strip()
    return {"think": think, "answer": answer}


def run_two_stage(
    llm: _LLM,
    prompt: str,
    think_budget: int,
    answer_budget: int,
) -> Dict[str, str]:
    """Ask ``llm`` once and parse out ``think`` and ``answer`` segments."""

    result = llm.generate(
        prompt,
        max_tokens=think_budget + answer_budget,
        stop=["</think>", "}\n\n"],
    )
    return _parse(str(result.get("text", "")))
