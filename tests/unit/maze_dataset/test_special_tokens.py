import pytest

from maze_dataset.constants import SPECIAL_TOKENS


def test_special_tokens_base():
    # Test the getitem method
    assert SPECIAL_TOKENS["ADJLIST_START"] == "<ADJLIST_START>"
    with pytest.raises(KeyError):
        SPECIAL_TOKENS["NON_EXISTENT_KEY"]

    # Test the len method
    assert len(SPECIAL_TOKENS) == 11

    # Test the contains method
    assert "ADJLIST_START" in SPECIAL_TOKENS
    assert "NON_EXISTENT_KEY" not in SPECIAL_TOKENS

    # Test the values method
    assert "<ADJLIST_START>" in SPECIAL_TOKENS.values()

    # Test the items method
    assert ("ADJLIST_START", "<ADJLIST_START>") in SPECIAL_TOKENS.items()

    # Test the keys method
    assert "ADJLIST_START" in SPECIAL_TOKENS.keys()
