from modality_llm.augmentation import generate_substitution_variants
from modality_llm.schema import Example, ModalExample


def _make_example(mv: str, utt: str) -> Example:
    # minimal ModalExample + Example wrapper used in other augmentation tests
    me = ModalExample(
        mv=mv,
        utt=utt,
        res={"palmer": [], "quirk": []},
        annotations={"palmer": [], "quirk": []},
    )
    return Example.from_modal_verb_example(me)


def test_remove_modality_must_fragment():
    entry = _make_example(
        "must", "That is really sweet of them. Must have been a big party."
    )
    variants = generate_substitution_variants(entry)
    # at least one substitution should remove 'must' and keep the 'party' mention;
    # accept either an epistemic paraphrase ('probably') or a cleaned past form ('was')
    assert any(
        ("must" not in v.english.lower())
        and ("party" in v.english.lower())
        and (("probably" in v.english.lower()) or (" was " in v.english.lower()))
        for v in variants
    )


def test_remove_modality_inner_future_clause():
    entry = _make_example(
        "will", "I bought a lottery ticket and have a feeling I will win."
    )
    variants = generate_substitution_variants(entry)
    # ensure the modal 'will' is removed from at least one variant,
    # the embedded "have a feeling" remains and "win" still appears.
    assert any(
        ("will" not in v.english.lower())
        and ("have a feeling" in v.english.lower())
        and ("win" in v.english.lower())
        for v in variants
    )
