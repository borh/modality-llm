from modality_llm.prompts import grammar_prompts, modal_prompts


def test_grammar_prompts_english_only():
    class Dummy:
        pass

    sent = "Say *hello* world"
    ex = Dummy()
    ex.english = sent
    ex.japanese = ""
    eng, jap = grammar_prompts([ex], "english")
    assert len(eng) == 1
    assert '"hello"' in eng[0]
    assert jap == []


def test_grammar_prompts_both_languages():
    class Dummy:
        pass

    eng_sent = "Hello *world*!"
    jap_sent = "こんにちは*世界*！"
    ex = Dummy()
    ex.english = eng_sent
    ex.japanese = jap_sent
    eng, jap = grammar_prompts([ex], "both")
    assert len(eng) == 1 and len(jap) == 1
    assert "world" in eng[0]
    assert "世界" in jap[0]


def test_modal_prompts_for_taxonomies():
    from modality_llm.schema import Taxonomy

    class Dummy:
        pass

    example = Dummy()
    example.english = "You *must* go."
    example.english_target = "must"
    example.expected_categories = {"palmer": ["deontic"], "quirk": ["necessity"]}
    p_palmer, e_palmer = modal_prompts([example], Taxonomy.palmer)
    p_quirk, e_quirk = modal_prompts([example], Taxonomy.quirk)
    assert e_palmer == [["deontic"]]
    assert e_quirk == [["necessity"]]
    assert "must" in p_palmer[0]
    assert "must" in p_quirk[0]
