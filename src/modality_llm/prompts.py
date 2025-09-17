from typing import List, Literal, Tuple

from modality_llm.llm_utils import get_common_instruction
from modality_llm.schema import (
    Example,
    Taxonomy,
)

# https://aclanthology.org/2023.conll-1.13.pdf
# Proceedings of the 27th Conference on Computational Natural Language Learning (CoNLL), pages 183–199 December 6–7, 2023. Quirk or Palmer: A Comparative Study of Modal Verb Frameworks with Annotated Datasets
# p. 193: Figure 4 – General instructions given to MTurk workers (with slight formatting modifications)
PAPER_INSTRUCTIONS = """
# Instructions
Modal verbs are a group of words that convey important semantic information about a situation that is being described, or the speaker's perspective related to the likelihood of the proposition. Although there is some variation, most sources define them to be the following words:

*can*, *could*, *may*, *might*, *must*, *shall*, *should*, *will*, *would*

carry information about the likelihood of a proposition or the speaker’s stance. Despite this, linguists have struggled to agree on a framework for categorizing modal verbs due to their flexibility and wide range of potential meanings.

**Please do the following**

1. Read through the instructions, examples, and label description
2. Read each provided sentence
3. Understand how the modal verbs (can, could, may, might, must, shall, should, will, would) are being used.
4. Label them accordingly

If there is not enough context to determine the meaning of the modal verb, answer "unknown".
""".strip()

# p. 193: Figure 5 - Descriptions given to MTurk workers for Quirk's categories (with slight formatting modifications)
QUIRK_DESCRIPTIONS = """
# Descriptions

1. *Possibility*: Does the modal verb contain information on the likelihood of something happening?
   Ex. It *may* rain today.
2. *Ability*: Does the modal verb contain information about a person's physical, mental, legal, moral, financial, or qualification-wise capabilities?
   Ex. I know I *can* do this since I've been practicing for months!
3. *Permission*: Does the modal verb contain information about receiving or giving permission? 
   Ex. *Can* I borrow your book?
4. *(Logical) Necessity*: Does the modal verb refer to something that must be true given the information available to the speaker?
   Ex. He *must* have gone already since his coat is gone.
5. *Obligation/Compulsion*: Does the modal verb contain information on some rules or expectations that someone has or has to abide to?
   Ex. I *must* submit my work by tonight.
6. *Tentative Inference*: Does the modal verb refer to something that can be guessed given the information available to the speaker?
   Ex. You *should* be able to solve the problem now.
7. *Prediction*: Does the modal verb refer to some prediction?
   Ex. I was told they *would* be here by now.
8. *Volition*: Does the modal verb refer to one's decision or choice? 
   Ex. I *will* do it as soon as possible.

# Examples

Input: As a member of the team, you *must* participate in all our meetings.
Answer: obligation

Input: Life *can* be cruel at times.
Answer: possibility

Input: There *must* be a mistake!
Answer: necessity

Input: They left before me so they *should* be here by now.
Answer: inference

Input: Oil *will* float on water.
Answer: prediction

Input: I *will* be gone by then.
Answer: volition
""".strip()

PALMER_DESCRIPTIONS = """
# Descriptions

1. *Deontic*: Influences a thought, action, or event by giving permission, expressing an obligation, or making a promise or threat.
   Ex. You *should* go home now.
2. *Epistemic*: Concerning with matters of knowledge or belief. Making a decision about the possibility of whether or not something is true.
   Ex. It *may* rain tomorrow.
3. *Dynamic*: Related to the volition or ability of the speaker or subject. Can also refer to circumstantial possibility involving an individual.
   Ex. If your friend *will* help you, ask them to drive the car tomorrow.

# Examples

Input: Look at all her accomplishments! She *may* be nominated for the award.
Answer: epistemic

Input: Taylor *can* do crosswords faster than you.
Answer: dynamic

Input: You *can* get all kinds of vegetables at the market.
Answer: dynamic

Input: You *may* use your phone here.
Answer: deontic

Input: You *must* be excited about tomorrow's trip.
Answer: epistemic

Input: You *can* just put my name down for two.
Answer: deontic
"""

# Unneeded? Could use as extra examples
TODO = """
### Figures 9 & 10 – Annotation screen-shots (interface examples)

**Quirk screen** – each row shows an *Input Text* followed by an eight-option drop-down (Possibility, Ability, …, Volition). Example rows:

* “As a member of the team, you **must** participate …”
* “There **must** be a mistake!”
* “Oil **will** float on water.”

**Palmer screen** – identical layout but the drop-down has three options (Deontic, Epistemic, Dynamic). Example rows:

* “She **may** be nominated for the award.”
* “You **may** use your phone here.”
* “Sorry about that – you **could** have called a friend.

""".strip()


def grammar_prompts(
    examples: List[Example],
    lang: Literal["english", "japanese", "both"],
) -> Tuple[List[str], List[str]]:
    """
    Build yes/no grammar‐check prompts from marked sentences.

    Args:
        examples: List of Example models, each with an `.english`
            (a marked English sentence) and optional `.japanese`.
        lang:  Which language(s) to produce prompts for.

    Returns:
        A pair (english_prompts, japanese_prompts).
    """

    english_prompts = [
        f'Given the following English utterance: "{ex.english}"\n\n'
        f'Is the usage of "{ex.english.split("*")[1].split("*")[0]}" '
        f"(marked between asterisks) grammatically correct in this context? "
        f"Answer with only 'yes' or 'no'.\n\nAnswer:"
        for ex in examples
        if lang in ("english", "both") and ex.english
    ]

    japanese_prompts = [
        f'Given the following Japanese utterance: "{ex.japanese}"\n\n'
        f'Is the usage of "{ex.japanese.split("*")[1].split("*")[0]}" '
        f"(marked between asterisks) grammatically correct in this context? "
        f"Answer with only 'yes' or 'no'.\n\nAnswer:"
        for ex in examples
        if lang in ("japanese", "both") and ex.japanese
    ]

    return english_prompts, japanese_prompts


def modal_prompts(
    examples: List[Example],
    taxonomy: Taxonomy,
    zero_shot: bool = False,
) -> Tuple[List[str], List[List[str]]]:
    """
    Build classification prompts and collect expected answers.

    Args:
        examples: List of Example models with .english, .english_target, and .expected_categories.
        taxonomy: Either "palmer" or "quirk".
        zero_shot: If True, use the old get_common_instruction; else use PAPER_INSTRUCTIONS.

    Returns:
        A tuple (prompt_strings, expected_answers).
    """

    # Accept either a Taxonomy enum or a plain string; normalize to string
    tax_str = taxonomy.value if isinstance(taxonomy, Taxonomy) else taxonomy

    prompts: List[str] = []
    expected_answers: List[List[str]] = []
    if zero_shot:
        common_instruction = get_common_instruction(tax_str)
    else:
        from modality_llm.schema import PALMER_CATEGORIES, QUIRK_CATEGORIES

        cats = PALMER_CATEGORIES if tax_str == "palmer" else QUIRK_CATEGORIES
        common_instruction = (
            PAPER_INSTRUCTIONS + "\n\nChoose from: " + ", ".join(cats) + "\n\n"
        )

    for ex in examples:
        utterance = ex.english
        modal_verb = ex.english_target
        # expected_answers might be keyed by string ("palmer") or Taxonomy enum
        expected = []
        if ex.expected_categories:
            # try string key first
            if tax_str in ex.expected_categories:
                expected = ex.expected_categories.get(tax_str, [])
            else:
                try:
                    expected = ex.expected_categories.get(Taxonomy(tax_str), [])
                except Exception:
                    expected = []
        # ensure we always append a list
        expected = expected or []

        prompt = (
            f"{common_instruction}"
            f'Input: "{utterance}"\n'
            f'Modal verb: "{modal_verb}"\n'
            f"Answer:"
        )
        prompts.append(prompt)
        expected_answers.append(expected)
    return prompts, expected_answers
