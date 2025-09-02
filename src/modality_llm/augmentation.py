"""
Data‐augmentation routines for modal‐verb examples:
  - Acceptability judgments
  - Substitution tests
  - Entailment checks
"""

import re
from typing import (
    Callable,
    List,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
)

from modality_llm.schema import (
    AcceptabilityTestEntry,
    EntailmentTestEntry,
    Example,
    GrammarLabel,
    SubstitutionTestEntry,
)

R = TypeVar("R")


class TransformationInfo(TypedDict, total=False):
    pattern: str
    replacement: str
    replacement_fn: Callable[[str, str], str]


TransformationCategory = Literal["substitution", "entailment", "contradiction"]
TransformationStrategy = Literal[
    # substitution: equivalent paraphrases
    "ability_paraphrase",
    "necessity_paraphrase",
    "advice_paraphrase",
    "permission_paraphrase",
    "possibility_paraphrase",
    # entailment
    "necessity_to_advice",
    "necessity_to_permission",
    "possibility_to_certainty",
    "advice_to_possibility",
    # contradiction
    "ability_to_denial",
    "necessity_to_denial",
    "permission_to_prohibition",
    "advice_to_negation",
    "possibility_to_impossibility",
]


def generate_acceptability_variants(
    entry: Example,
    strategies: Optional[
        List[Literal["insert_to", "double_modal", "gerund_form"]]
    ] = None,
    max_variants: int = 1,
) -> List[AcceptabilityTestEntry]:
    # pull from our unified Example
    utt = entry.english or ""
    modal = entry.english_target or ""
    sid = entry.eid
    choices = strategies or ["insert_to", "double_modal", "gerund_form"]
    out: List[AcceptabilityTestEntry] = []

    def insert_to(s: str) -> str:
        return re.sub(rf"\b{modal}\s+", f"{modal} to ", s, count=1)

    def double_modal(s: str) -> str:
        return re.sub(rf"\b{modal}\b", f"might {modal}", s, count=1)

    def gerund_form(s: str) -> str:
        m = re.search(rf"{modal}\s+(\w+)", s)
        if not m:
            return s
        verb = m.group(1)
        # Handle common verb endings for gerund formation
        if verb.endswith("e") and not verb.endswith("ee"):
            # drop final 'e' before adding 'ing' (e.g., leave -> leaving)
            gerund = verb[:-1] + "ing"
        elif verb.endswith("ie"):
            # change 'ie' to 'y' before adding 'ing' (e.g., lie -> lying)
            gerund = verb[:-2] + "ying"
        else:
            # just add 'ing' for most verbs
            gerund = verb + "ing"
        return s.replace(f"{modal} {verb}", f"{modal} {gerund}", 1)

    funcs: dict[str, Callable[[str], str]] = {
        "insert_to": insert_to,
        "double_modal": double_modal,
        "gerund_form": gerund_form,
    }

    # Update grammaticality assignment based on strategy
    grammaticality_map = {
        "insert_to": GrammarLabel.no,  # Always ungrammatical
        "double_modal": GrammarLabel.no,  # Always ungrammatical
        "gerund_form": GrammarLabel.partial,  # Context-dependent
    }

    for strat in choices:
        if len(out) >= max_variants:
            break
        fn = funcs.get(strat)
        if not fn:
            continue
        new_utt = fn(utt)
        if new_utt != utt:
            out.append(
                AcceptabilityTestEntry(
                    # all Example fields
                    eid=sid,
                    english=new_utt,
                    japanese=None,
                    grammatical=grammaticality_map.get(strat, GrammarLabel.partial),
                    english_target=modal,
                    japanese_target=None,
                    human_annotations=entry.human_annotations,
                    expected_categories=entry.expected_categories,
                    # test‐entry fields
                    test_type="acceptability",
                    transformation_strategy=strat,
                )
            )
    return out


# Testing spaCy implementation
try:
    import spacy

    _nlp = None
    gpu_use = spacy.prefer_gpu()
    print(f"Using spaCy {spacy.__version__} (gpu: {gpu_use})")

    def get_spacy_model():
        global _nlp
        if _nlp is None:
            try:
                _nlp = spacy.load("en_core_web_sm")
            except Exception:
                _nlp = None
        return _nlp

    def get_subject_form(sentence: str, modal: str) -> str:
        """
        Choose correct paraphrase auxiliary based on the modal’s grammatical subject.

        Examples:
            # first‐person pronouns use 'have to' for must, 'am able to' for can
            >>> get_subject_form("I must go.", "must")
            'have to'
            >>> get_subject_form("I can swim.", "can")
            'am able to'

            # third‐person singular (pronouns or nouns)
            >>> get_subject_form("She must leave now.", "must")
            'has to'
            >>> get_subject_form("The cat can meow.", "can")
            'is able to'

            # plural subjects use 'have to' or 'are able to'
            >>> get_subject_form("They must work late.", "must")
            'have to'
            >>> get_subject_form("We can play games.", "can")
            'are able to'

            # fallback on empty or ambiguous subjects
            >>> get_subject_form("", "must") in ('has to', 'have to')
            True
            >>> get_subject_form("", "can") in ('is able to', 'are able to')
            True
        """
        nlp = get_spacy_model()
        if nlp:
            doc = nlp(sentence)
            for token in doc:
                if token.lower_ == modal and token.tag_ == "MD":
                    # find the nominal subject
                    for child in token.head.children:
                        if child.dep_ == "nsubj":
                            subj = child.lower_
                            pos = child.tag_
                            # first‐person singular
                            if subj == "i":
                                return "have to" if modal == "must" else "am able to"
                            # third‐person singular pronouns or singular nouns
                            if subj in ("he", "she", "it") or pos in ("NN", "NNP"):
                                return "has to" if modal == "must" else "is able to"
                            # plural pronouns or plural nouns
                            if subj in ("we", "they", "you") or pos in ("NNS", "NNPS"):
                                return "have to" if modal == "must" else "are able to"
                    break
        # Fallback: look at the first token as a heuristic subject
        first = sentence.split()[0].strip(".,").lower() if sentence else ""
        if first == "i":
            return "have to" if modal == "must" else "am able to"
        if first in ("he", "she", "it") or not first.endswith("s"):
            return "has to" if modal == "must" else "is able to"
        # default to plural
        return "have to" if modal == "must" else "are able to"
except ImportError:

    def get_subject_form(sentence: str, modal: str) -> str:
        return "has to" if modal == "must" else "is able to"


_MODAL_TRANSFORMATIONS: dict[
    str,
    dict[TransformationCategory, dict[str, TransformationInfo]],
] = {
    "can": {
        "substitution": {
            "ability_paraphrase": {
                "pattern": r"\bcan\b",
                "replacement_fn": lambda utt, sf: f"{sf} able to",
            },
        },
        "entailment": {},
        "contradiction": {
            "ability_to_denial": {
                "pattern": r"\bcan\b(?!\s*not)",
                "replacement": "cannot",
            },
        },
    },
    "may": {
        "substitution": {
            "permission_paraphrase": {"pattern": r"\bmay\b", "replacement": "can"},
        },
        "entailment": {
            "permission_to_possibility": {
                "pattern": r"\bmay\b",
                "replacement": "might",
            },
        },
        "contradiction": {
            "permission_to_prohibition": {
                "pattern": r"\bmay\b",
                "replacement": "may not",
            },
        },
    },
    "might": {
        "substitution": {
            "possibility_paraphrase": {"pattern": r"\bmight\b", "replacement": "may"},
        },
        "entailment": {},
        "contradiction": {
            "possibility_to_impossibility": {
                "pattern": r"\bmight\b",
                "replacement": "cannot",
            },
        },
    },
    "will": {
        "substitution": {
            "prediction_paraphrase": {"pattern": r"\bwill\b", "replacement": "shall"},
        },
        "entailment": {
            "prediction_to_possibility": {
                "pattern": r"\bwill\b",
                "replacement": "might",
            },
        },
        "contradiction": {
            "prediction_to_impossibility": {
                "pattern": r"\bwill\b",
                "replacement": "will not",
            },
        },
    },
    "would": {
        "substitution": {
            "conditional_paraphrase": {"pattern": r"\bwould\b", "replacement": "could"},
        },
        "entailment": {
            "conditional_to_possibility": {
                "pattern": r"\bwould\b",
                "replacement": "might",
            },
        },
        "contradiction": {
            "conditional_to_impossibility": {
                "pattern": r"\bwould\b",
                "replacement": "would not",
            },
        },
    },
    "could": {
        "substitution": {
            "ability_paraphrase": {"pattern": r"\bcould\b", "replacement": "might"},
        },
        "entailment": {},
        "contradiction": {
            "ability_to_denial": {"pattern": r"\bcould\b", "replacement": "could not"},
        },
    },
    "must": {
        "substitution": {
            "necessity_paraphrase": {
                "pattern": r"\bmust\b",
                "replacement_fn": lambda utt, sf: f"{sf} have to",
            },
        },
        "entailment": {
            "necessity_to_advice": {"pattern": r"\bmust\b", "replacement": "should"},
            "necessity_to_permission": {"pattern": r"\bmust\b", "replacement": "may"},
        },
        "contradiction": {
            "necessity_to_denial": {"pattern": r"\bmust\b", "replacement": "need not"},
        },
    },
    "should": {
        "substitution": {
            "advice_paraphrase": {"pattern": r"\bshould\b", "replacement": "ought to"},
        },
        "entailment": {
            "advice_to_possibility": {"pattern": r"\bshould\b", "replacement": "can"},
        },
        "contradiction": {
            "advice_to_negation": {
                "pattern": r"\bshould\b(?!\s*not)",
                "replacement": "should not",
            },
        },
    },
    "shall": {
        "substitution": {
            "obligation_paraphrase": {"pattern": r"\bshall\b", "replacement": "will"},
        },
        "entailment": {
            "obligation_to_advice": {"pattern": r"\bshall\b", "replacement": "should"},
        },
        "contradiction": {
            "obligation_to_denial": {
                "pattern": r"\bshall\b",
                "replacement": "shall not",
            },
        },
    },
    "ought to": {
        "substitution": {
            "advice_paraphrase": {"pattern": r"\bought to\b", "replacement": "should"},
        },
        "entailment": {
            "advice_to_possibility": {"pattern": r"\bought to\b", "replacement": "can"},
        },
        "contradiction": {
            "advice_to_negation": {
                "pattern": r"\bought to\b",
                "replacement": "ought not to",
            },
        },
    },
}


# Generic helper to apply all strategies of one category for a given modal example
def generate_variants_by_category(
    entry: Example,
    modal: str,
    category: TransformationCategory,
    build_entry_fn: Callable[[Example, str, str, TransformationCategory], R],
    max_variants: int = 3,
) -> list[R]:
    """
    Look up all transforms under `_MODAL_TRANSFORMATIONS[modal][category]`,
    apply at most `max_variants`, and build TypedDicts via `build_entry_fn`.
    Adds debug logging for modal, category, and transformation attempts.
    """
    bucket = _MODAL_TRANSFORMATIONS.get(modal, {}).get(category, {})
    print(f"DEBUG: Modal '{modal}', category '{category}' has {len(bucket)} strategies: {list(bucket.keys())}")
    out: list[R] = []
    # always use the marked‐English form for augmentation
    utt = entry.english or ""
    subj_form = get_subject_form(utt, modal)
    for strat, info in bucket.items():
        if len(out) >= max_variants:
            break
        pattern = info["pattern"]
        if "replacement_fn" in info:
            replacement = info["replacement_fn"](utt, subj_form)
        else:
            replacement = info["replacement"]
        new_utt = re.sub(pattern, replacement, utt, count=1, flags=re.IGNORECASE)
        if new_utt != utt:
            out.append(build_entry_fn(entry, new_utt, strat, category))
        else:
            print(f"DEBUG: No transformation for '{modal}' with '{strat}' in: '{utt}'")
    return out


def generate_substitution_variants(
    entry: Example,
    max_variants: int = 3,
) -> List[SubstitutionTestEntry]:
    """
    Generate meaning-preserving paraphrases (substitution) for a modal example.
    """
    out: List[SubstitutionTestEntry] = []
    for ent, utt, strat, cat in generate_variants_by_category(
        entry,
        (entry.english_target or "").lower(),
        "substitution",
        lambda ent, utt, strat, cat: (ent, utt, strat, cat),
        max_variants,
    ):
        out.append(
            SubstitutionTestEntry(
                eid=ent.eid,
                english=utt,
                japanese=None,
                grammatical=GrammarLabel.yes,
                english_target=strat.split("_")[0],
                japanese_target=None,
                human_annotations=ent.human_annotations,
                expected_categories=ent.expected_categories,
                test_type="substitution",
                transformation_strategy=strat,
                expected_res=None,
            )
        )
    return out


def generate_entailment_tests(
    entry: Example,
    max_hypotheses: int = 3,
) -> List[EntailmentTestEntry]:
    """Generate minimal one‐way entailment pairs for a modal example."""
    out: List[EntailmentTestEntry] = []
    for ent, utt, strat, cat in generate_variants_by_category(
        entry,
        entry.english_target,
        "entailment",
        lambda ent, utt, strat, cat: (ent, utt, strat, cat),
        max_hypotheses,
    ):
        out.append(
            EntailmentTestEntry(
                eid=ent.eid,
                english=ent.english,
                japanese=None,
                grammatical=GrammarLabel.yes,
                english_target=ent.english_target,
                japanese_target=None,
                human_annotations=ent.human_annotations,
                expected_categories=ent.expected_categories,
                test_type="entailment",
                hypothesis=utt,
                label=cat,
                transformation_strategy=strat,
            )
        )
    return out


def generate_contradiction_variants(
    entry: Example,
    max_variants: int = 3,
) -> List[EntailmentTestEntry]:
    """
    Generate contradiction variants for a modal example.
    """
    out: List[EntailmentTestEntry] = []
    modal = (entry.english_target or "").lower()
    for ent, utt, strat, cat in generate_variants_by_category(
        entry,
        modal,
        "contradiction",
        lambda ent, utt, strat, cat: (ent, utt, strat, cat),
        max_variants,
    ):
        out.append(
            EntailmentTestEntry(
                eid=ent.eid,
                english=ent.english,
                japanese=None,
                grammatical=GrammarLabel.yes,
                english_target=ent.english_target,
                japanese_target=None,
                human_annotations=ent.human_annotations,
                expected_categories=ent.expected_categories,
                test_type="entailment",
                hypothesis=utt,
                label="contradiction",
                transformation_strategy=strat,
            )
        )
    return out
