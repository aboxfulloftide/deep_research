from deep_research.kb.conversation import parse_conversation_turns


def test_parses_alternating_speaker_turns():
    text = (
        "User: Is the Eiffel Tower taller than the Statue of Liberty?\n"
        "Assistant: Yes, the Eiffel Tower is about 330 meters tall.\n"
        "User: What about the Great Pyramid?\n"
        "Assistant: The Great Pyramid is about 146 meters tall.\n"
    )
    turns = parse_conversation_turns(text)
    assert [speaker for speaker, _ in turns] == ["User", "Assistant", "User", "Assistant"]
    assert "Eiffel Tower" in turns[0][1]
    assert "330 meters" in turns[1][1]


def test_multiline_turn_stays_with_its_speaker():
    text = "Alice: This is line one\nand this continues the same turn.\nBob: A reply.\n"
    turns = parse_conversation_turns(text)
    assert turns[0] == ("Alice", "This is line one\nand this continues the same turn.")
    assert turns[1] == ("Bob", "A reply.")


def test_plain_prose_with_no_speaker_lines_returns_one_untagged_turn():
    text = "Just a plain paragraph with no speakers at all."
    assert parse_conversation_turns(text) == [(None, text)]


def test_empty_text_returns_no_turns():
    assert parse_conversation_turns("") == []
    assert parse_conversation_turns("   \n  ") == []
