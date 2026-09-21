from nova.brain.llm_engine import LLMEngine


def test_dynamic_context_only_changes_last_user_message():
    engine = LLMEngine()
    first = engine._build_messages("hello", "first", [], [], "chat")
    second = engine._build_messages("hello", "second", [], [], "chat")
    assert first[0] == second[0]
    assert first[:-1] == second[:-1]
    assert first[-1] != second[-1]
