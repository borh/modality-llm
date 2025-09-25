import pytest

from modality_llm.augmentation import remove_modality_transform

GOLD = [
    ("I can't wait to see them.", "I'm eager to see them."),
    ("Can you do that as well?", "Do that as well."),
    ("The future could be bright.", "The future is bright."),
    ("There might be issues.", "There are issues."),
    ("He must have been high.", "He was high."),
    ("Must have cost 70k.", "It cost 70k."),
    ("I can't sleep.", "I'm not able to sleep."),
]


@pytest.mark.parametrize("src,expected", GOLD)
def test_modality_removal_gold(src, expected):
    try:
        out = remove_modality_transform(
            src, (src.split()[1] if "*" not in src else src.split("*")[1]).lower(), None
        )
    except Exception:
        pytest.skip("spaCy model unavailable")
    assert out == expected
    bad = out.lower()
    assert " not not " not in bad
    assert " i are " not in bad
