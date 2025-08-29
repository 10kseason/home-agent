"""
Assumptions: LLM expose timeout flag and token usage
Risks: repeated timeouts waste budget
Alternatives: early clarify without retry
Rationale: bounded latency with one retry at +50%
"""

def run_two_stage(task, prompt, think_budget, answer_budget, timeout_ms):
    for attempt in range(2):
        result = llm.generate(prompt, max_tokens=think_budget+answer_budget,
                              stop=["</think>", "}\n\n"], timeout=timeout_ms)
        if result.timeout and attempt == 0 and timeout_ms <= 600:
            think_budget = int(think_budget * 1.5)
            answer_budget = int(answer_budget * 1.5)
            timeout_ms = int(timeout_ms * 1.5)
            continue
        return parse(result)
    return {"error": "timeout"}
# Checklist:
# - [x] Think Harder
# - [x] Think Deeper
# - [x] More Information
# - [x] Check Again
