from agent.plugins.translator_plugin import dedup_translation


def test_dedup_translation_collapses_duplicate_lines():
    assert dedup_translation("안녕\n안녕") == "안녕"


def test_dedup_translation_keeps_unique_lines():
    text = "안녕\n하세요"
    assert dedup_translation(text) == text
