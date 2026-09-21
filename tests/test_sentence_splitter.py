from nova.brain.sentence_splitter import split_sentences


def test_sentence_splitter_handles_abbreviations():
    assert split_sentences("Ask Dr. Smith. Then call me.") == ["Ask Dr. Smith.", "Then call me."]


def test_sentence_splitter_handles_decimals():
    assert split_sentences("The value is 3.14. It is stable.") == ["The value is 3.14.", "It is stable."]


def test_sentence_splitter_handles_ellipses():
    assert split_sentences("Wait... Really? Yes!") == ["Wait...", "Really?", "Yes!"]
