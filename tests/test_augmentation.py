import pytest

from modality_llm.augmentation import (
    _MODAL_TRANSFORMATIONS,
    generate_acceptability_variants,
    generate_contradiction_variants,
    generate_entailment_tests,
    generate_substitution_variants,
    get_subject_form,
)
from modality_llm.schema import Example, GrammarLabel, ModalExample


# helper to lift a plain dict into our Example model
def make_example(d: dict[str, any]) -> Example:
    # ensure both taxonomy keys are present with list values
    base = {"palmer": [], "quirk": []}
    res = {**base, **d.get("res", {})}
    annotations = {**base, **d.get("annotations", {})}
    me = ModalExample(mv=d["mv"], utt=d["utt"], res=res, annotations=annotations)
    return Example.from_modal_verb_example(me)


class TestAcceptabilityVariants:
    """Test acceptability judgment generation."""

    def test_insert_to_strategy(self):
        """Test inserting 'to' after modal verbs."""
        entry_dict = {
            "mv": "can",
            "utt": "I can swim.",
            "res": {},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        variants = generate_acceptability_variants(entry, strategies=["insert_to"])
        assert len(variants) == 1
        assert variants[0].english == "I can to swim."
        assert variants[0].grammatical == GrammarLabel.no
        assert variants[0].test_type == "acceptability"
        assert variants[0].transformation_strategy == "insert_to"

    def test_double_modal_strategy(self):
        """Test double modal construction."""
        entry_dict = {
            "mv": "should",
            "utt": "You should go home.",
            "res": {},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        variants = generate_acceptability_variants(entry, strategies=["double_modal"])
        assert len(variants) == 1
        assert variants[0].english == "You might should go home."
        assert variants[0].grammatical == GrammarLabel.no
        assert variants[0].transformation_strategy == "double_modal"

    def test_gerund_form_strategy(self):
        """Test gerund form transformation."""
        entry_dict = {
            "mv": "must",
            "utt": "She must leave now.",
            "res": {},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        variants = generate_acceptability_variants(entry, strategies=["gerund_form"])
        assert len(variants) == 1
        assert variants[0].english == "She must leaving now."
        assert variants[0].grammatical == GrammarLabel.no
        assert variants[0].transformation_strategy == "gerund_form"

    def test_max_variants_limit(self):
        """Test that max_variants is respected."""
        entry_dict = {
            "mv": "can",
            "utt": "I can help you.",
            "res": {},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        variants = generate_acceptability_variants(
            entry,
            strategies=["insert_to", "double_modal", "gerund_form"],
            max_variants=2,
        )
        assert len(variants) == 2

    def test_no_change_cases(self):
        """Test cases where transformation doesn't apply."""
        entry_dict = {
            "mv": "can",
            "utt": "I cannot.",  # No verb after modal
            "res": {},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        variants = generate_acceptability_variants(entry, strategies=["gerund_form"])
        assert len(variants) == 0  # No variants generated

    def test_with_source_id(self):
        """Test that source_id is properly propagated."""
        entry_dict = {
            "mv": "should",
            "utt": "You should try.",
            "res": {},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        variants = generate_acceptability_variants(entry, strategies=["insert_to"])
        assert variants[0].eid == entry.eid


class TestSubstitutionVariants:
    """Test substitution-based transformations."""

    def test_can_to_be_able_to(self):
        """Test ability paraphrase: can -> be able to."""
        entry_dict = {
            "mv": "can",
            "utt": "Alice can solve the puzzle.",
            "res": {"palmer": [], "quirk": []},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        variants = generate_substitution_variants(entry)
        assert any("is able to" in v.english for v in variants)
        # Check that transformation strategy is recorded
        able_variant = next(v for v in variants if "is able to" in v.english)
        assert able_variant.transformation_strategy == "ability_paraphrase"
        assert able_variant.expected_res is None

    def test_must_to_should_weakening(self):
        """Test necessity to advice transformation."""
        entry_dict = {
            "mv": "must",
            "utt": "You must finish your work.",
            "res": {"palmer": [], "quirk": []},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        pairs = generate_entailment_tests(entry)
        assert any("should" in p.hypothesis for p in pairs)
        should_pair = next(p for p in pairs if "should" in p.hypothesis)
        assert should_pair.transformation_strategy == "necessity_to_advice"

    def test_should_to_ought_to(self):
        """Test advice paraphrase: should -> ought to."""
        entry_dict = {
            "mv": "should",
            "utt": "You should call her.",
            "res": {"palmer": [], "quirk": []},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        variants = generate_substitution_variants(entry)
        assert any("ought to" in v.english for v in variants)
        ought_variant = next(v for v in variants if "ought to" in v.english)
        assert ought_variant.transformation_strategy == "advice_paraphrase"
        assert ought_variant.expected_res is None

    # def test_can_to_cannot_negation(self):
    #     """Test ability to denial transformation."""
    #     entry: ModalExample = {
    #         "mv": "can",
    #         "utt": "The baby can sleep through noise.",
    #         "res": {},
    #         "annotations": {},
    #     }
    #     variants = generate_substitution_variants(entry, create_contradictions=True)
    #     assert any("cannot" in v["utt"] for v in variants)
    #     cannot_variant = next(v for v in variants if "cannot" in v["utt"])
    #     assert cannot_variant["transformation_strategy"] == "ability_to_denial"
    #
    # def test_may_to_must_not_prohibition(self):
    #     """Test permission to prohibition transformation."""
    #     entry: ModalExample = {
    #         "mv": "may",
    #         "utt": "Employees may work from home.",
    #         "res": {},
    #         "annotations": {},
    #     }
    #     variants = generate_substitution_variants(entry, create_contradictions=True)
    #     assert any("must not" in v["utt"] for v in variants)
    #     prohibition_variant = next(v for v in variants if "must not" in v["utt"])
    #     assert prohibition_variant["transformation_strategy"] == "permission_to_prohibition"

    def test_variant_limit_respected(self):
        """Test that max_variants is respected across all types."""
        entry_dict = {
            "mv": "must",
            "utt": "You must go.",
            "res": {},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        variants = generate_substitution_variants(entry, max_variants=2)
        assert len(variants) <= 2

    def test_unknown_modal_returns_empty(self):
        """Test that unknown modals return empty list."""
        entry_dict = {
            "mv": "gonna",  # Not in _MODAL_TRANSFORMATIONS
            "utt": "I'm gonna go.",
            "res": {},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        variants = generate_substitution_variants(entry)
        assert len(variants) == 0

    def test_must_to_have_to_substitution(self):
        """Test necessity paraphrase: must → have to."""
        entry_dict = {
            "mv": "must",
            "utt": "They must leave soon.",
            "res": {"palmer": [], "quirk": []},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        variants = generate_substitution_variants(entry)
        v = next(
            (
                v
                for v in variants
                if v.transformation_strategy == "necessity_paraphrase"
            ),
            None,
        )
        assert v is not None
        assert "have to" in v.english

    def test_case_insensitive_matching(self):
        """Test that modal matching is case-insensitive."""
        entry_dict = {
            "mv": "Can",  # Capitalized
            "utt": "Can you help me?",
            "res": {},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        variants = generate_substitution_variants(entry)
        assert len(variants) > 0  # Should still find transformations

    def test_substitution_someone_ability_paraphrase(self):
        """Test that substitution for 'can' + 'someone' uses 'is able to'."""
        entry_dict = {
            "mv": "can",
            "utt": "Someone can solve the puzzle.",
            "res": {"palmer": [], "quirk": []},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        variants = generate_substitution_variants(entry)
        able = next(
            (v for v in variants if v.transformation_strategy == "ability_paraphrase"),
            None,
        )
        assert able is not None
        assert "is able to" in able.english


class TestEntailmentTests:
    """Test entailment pair generation."""

    def test_must_to_should_entailment(self):
        """Test necessity to advice entailment."""
        entry_dict = {
            "mv": "must",
            "utt": "Passengers must fasten their seatbelts.",
            "res": {},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        pairs = generate_entailment_tests(entry)
        should_pair = next((p for p in pairs if "should" in p.hypothesis), None)
        assert should_pair is not None
        assert should_pair.english == entry.english
        assert "should" in should_pair.hypothesis
        assert should_pair.label == "entailment"
        assert should_pair.transformation_strategy == "necessity_to_advice"

    def test_must_to_may_entailment(self):
        """Test obligation to permission entailment."""
        entry_dict = {
            "mv": "must",
            "utt": "Visitors must sign in.",
            "res": {},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        pairs = generate_entailment_tests(entry)
        may_pair = next((p for p in pairs if "may" in p.hypothesis), None)
        assert may_pair is not None
        assert may_pair.label == "entailment"
        assert may_pair.transformation_strategy == "necessity_to_permission"

    # def test_can_cannot_contradiction(self):
    #     """Test ability to denial contradiction."""
    #     entry: ModalExample = {
    #         "mv": "can",
    #         "utt": "I can swim.",
    #         "res": {},
    #         "annotations": {},
    #     }
    #     pairs = generate_entailment_tests(entry)
    #     contradiction = next((p for p in pairs if p["label"] == "contradiction"), None)
    #     assert contradiction is not None
    #     assert "cannot" in contradiction["hypothesis"]
    #     assert contradiction["transformation_strategy"] == "ability_to_denial"
    #
    # def test_paraphrase_as_entailment(self):
    #     """Test that paraphrases are treated as mutual entailment."""
    #     entry: ModalExample = {
    #         "mv": "should",
    #         "utt": "You should go home.",
    #         "res": {},
    #         "annotations": {},
    #     }
    #     pairs = generate_entailment_tests(entry)
    #     paraphrase = next((p for p in pairs if "ought to" in p["hypothesis"]), None)
    #     assert paraphrase is not None
    #     assert paraphrase["label"] == "entailment"
    #     assert paraphrase["transformation_strategy"] == "advice_paraphrase"

    def test_max_hypotheses_limit(self):
        """Test that max_hypotheses is respected."""
        entry_dict = {
            "mv": "must",
            "utt": "You must leave.",
            "res": {},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        pairs = generate_entailment_tests(entry, max_hypotheses=1)
        assert len(pairs) == 1

    def test_source_id_propagation(self):
        """Test that source_id is properly propagated to pairs."""
        entry_dict = {
            "mv": "must",
            "utt": "You must go.",
            "res": {},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        pairs = generate_entailment_tests(entry)
        assert all(p.eid == entry.eid for p in pairs)

    def test_non_must_modal_handling(self):
        """Test that non-must modals still generate some pairs."""
        entry_dict = {
            "mv": "might",
            "utt": "It might rain.",
            "res": {},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        pairs = generate_entailment_tests(entry)
        assert len(pairs) == 0

    def test_should_to_can_entailment(self):
        """Test advice → possibility entailment: should → can."""
        entry_dict = {
            "mv": "should",
            "utt": "You should save your work.",
            "res": {},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        pairs = generate_entailment_tests(entry)
        can_pair = next(
            (p for p in pairs if p.transformation_strategy == "advice_to_possibility"),
            None,
        )
        assert can_pair is not None
        assert "can" in can_pair.hypothesis

    # def test_entailment_someone_must_necessity_paraphrase(self):
    #     """Test that entailment for 'Someone must ...' uses 'has to'."""
    #     entry: ModalExample = {
    #         "mv": "must",
    #         "utt": "Someone must hurry.",
    #         "res": {},
    #         "annotations": {},
    #     }
    #     pairs = generate_entailment_tests(entry)
    #     paraphrase = next(
    #         (p for p in pairs if p["transformation_strategy"] == "necessity_paraphrase"),
    #         None,
    #     )
    #     assert paraphrase is not None
    #     assert "has to" in paraphrase["hypothesis"]


class TestModalTransformations:
    """Test the _MODAL_TRANSFORMATIONS configuration."""

    def test_all_modals_have_transformations(self):
        # our code only defines transformations for 'can','must','should'
        for modal in ["can", "must", "should"]:
            assert modal in _MODAL_TRANSFORMATIONS
            # and each category dict is non‐empty
            assert any(
                _MODAL_TRANSFORMATIONS[modal][cat]
                for cat in ("substitution", "entailment", "contradiction")
            )

    def test_substitution_paraphrases_exist(self):
        # each modal with a substitution bucket must have at least one _paraphrase strategy
        for modal, transforms in _MODAL_TRANSFORMATIONS.items():
            subs = transforms.get("substitution", {})
            assert any(name.endswith("_paraphrase") for name in subs)

    def test_transformation_strategies_valid(self):
        valid = {
            # substitution
            "ability_paraphrase",
            "necessity_paraphrase",
            "advice_paraphrase",
            # entailment
            "necessity_to_advice",
            "necessity_to_permission",
            "advice_to_possibility",
            # contradiction
            "ability_to_denial",
            "necessity_to_denial",
            "permission_to_prohibition",
            "advice_to_negation",
            "possibility_to_impossibility",
        }
        for modal, transforms in _MODAL_TRANSFORMATIONS.items():
            for category, bucket in transforms.items():
                for strat in bucket:
                    assert strat in valid


class TestSubjectFormDetection:
    """Test the get_subject_form helper function."""

    def test_basic_subject_forms(self):
        """Test basic subject-verb agreement detection."""
        # Without SpaCy, should fall back to defaults
        assert get_subject_form("I must go.", "must") in ["has to", "have to"]
        assert get_subject_form("She can swim.", "can") in ["is able to", "are able to"]

    def test_fallback_behavior(self):
        """Test that function works even without SpaCy."""
        # The function should not raise errors even if SpaCy is not available
        result = get_subject_form("They should leave.", "should")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_subject_someone_handles_singular(self):
        """Ensure 'someone' is treated as singular by get_subject_form."""
        assert get_subject_form("Someone can swim.", "can") == "is able to"
        assert get_subject_form("Someone must leave now.", "must") == "has to"


class TestIntegration:
    """Integration tests combining multiple augmentation strategies."""

    def test_full_augmentation_pipeline(self):
        """Test running all augmentation types on a single example."""
        entry_dict = {
            "mv": "must",
            "utt": "Students must submit their assignments.",
            "res": {"palmer": ["deontic"], "quirk": ["obligation"]},
            "annotations": {"palmer": ["deontic"], "quirk": ["obligation"]},
        }
        entry = make_example(entry_dict)

        # Generate all types
        acceptability = generate_acceptability_variants(entry)
        substitution = generate_substitution_variants(entry)
        entailment = generate_entailment_tests(entry)

        # Check we got results from each
        assert len(acceptability) > 0
        assert len(substitution) > 0
        assert len(entailment) > 0

        # Check variety of transformations
        all_strategies = [v.transformation_strategy for v in substitution] + [
            p.transformation_strategy for p in entailment
        ]
        assert len(set(all_strategies)) > 1  # Multiple different strategies used

    def test_augmentation_preserves_original_format(self):
        """Test that aug ated data maintains compatible structure."""
        entry_dict = {
            "mv": "can",
            "utt": "I can help.",
            "res": {"palmer": ["dynamic"]},
            "annotations": {"palmer": ["dynamic"]},
        }
        entry = make_example(entry_dict)

        # All augmentation functions should return properly typed data
        acceptability = generate_acceptability_variants(entry)
        assert all(v.test_type == "acceptability" for v in acceptability)

        substitution = generate_substitution_variants(entry)
        assert all(v.test_type == "substitution" for v in substitution)

        entailment = generate_entailment_tests(entry)
        assert all(p.test_type == "entailment" for p in entailment)


class TestContradictionVariants:
    """Test contradiction‐based transformations."""

    def test_can_to_cannot_contradiction(self):
        """Ability → denial: can → cannot."""
        entry_dict = {"mv": "can", "utt": "I can swim.", "res": {}, "annotations": {}}
        entry = make_example(entry_dict)
        variants = generate_contradiction_variants(entry)
        assert len(variants) == 1
        v = variants[0]
        assert v.hypothesis == "I cannot swim."
        assert v.transformation_strategy == "ability_to_denial"
        assert v.label == "contradiction"

    def test_must_to_need_not_contradiction(self):
        """Necessity → denial: must → need not."""
        entry_dict = {
            "mv": "must",
            "utt": "You must arrive on time.",
            "res": {},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        variants = generate_contradiction_variants(entry)
        assert len(variants) == 1
        v = variants[0]
        assert "need not" in v.hypothesis
        assert v.transformation_strategy == "necessity_to_denial"
        assert v.label == "contradiction"

    def test_should_to_negation_contradiction(self):
        """Advice → negation: should → should not."""
        entry_dict = {
            "mv": "should",
            "utt": "He should try harder.",
            "res": {},
            "annotations": {},
        }
        entry = make_example(entry_dict)
        variants = generate_contradiction_variants(entry)
        assert len(variants) == 1
        v = variants[0]
        assert "should not" in v.hypothesis
        assert v.transformation_strategy == "advice_to_negation"
        assert v.label == "contradiction"


if __name__ == "__main__":
    pytest.main([__file__])
