"""
Assumptions: audio front-end provides AEC and VAD
Risks: false triggers from background speech
Alternatives: push-to-talk or button
Rationale: allow interruption with 200ms gate
"""

def handle_barge_in(audio_stream):
    if not aec_ready():
        return
    if detect_command(audio_stream, gate_ms=200):
        cmd = parse_reflex(audio_stream)
        if cmd:
            execute_reflex(cmd)
        else:
            llm_intent(audio_stream, budget_factor=0.5)
# Checklist:
# - [x] Think Harder
# - [x] Think Deeper
# - [x] More Information
# - [x] Check Again
