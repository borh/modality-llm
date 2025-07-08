import hashlib
from collections import Counter
from enum import Enum
from typing import List, Literal, Optional, TypeAlias, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    confloat,
    conint,
    model_validator,
)
from typing_extensions import Self


class _Model(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", validate_default=True)


class GrammarLabel(str, Enum):
    yes = "yes"
    no = "no"


class Taxonomy(str, Enum):
    palmer = "palmer"
    quirk = "quirk"


# Taxonomy = Literal["palmer", "quirk"]

PalmerCategory: TypeAlias = Literal[
    "deontic",
    "epistemic",
    "dynamic",
    "unknown",
]
QuirkCategory: TypeAlias = Literal[
    "possibility",
    "ability",
    "permission",
    "necessity",
    "obligation",
    "inference",
    "prediction",
    "volition",
    "unknown",
]

PALMER_CATEGORIES: List[PalmerCategory] = [
    "deontic",
    "epistemic",
    "dynamic",
    "unknown",
]
QUIRK_CATEGORIES: List[QuirkCategory] = [
    "possibility",
    "ability",
    "permission",
    "necessity",
    "obligation",
    "inference",
    "prediction",
    "volition",
    "unknown",
]


class ModalVerbsResultDict(TypedDict, total=False):
    """
    Human‐annotation or consensus labels, keyed by taxonomy.
    We allow arbitrary strings here; downstream code will validate actual categories.
    """
    palmer: List[str]
    quirk: List[str]


class ModalExample(BaseModel):
    """
    A modal_verbs.jsonl record for parsing purposes only.
    """

    mv: str
    annotations: ModalVerbsResultDict
    res: ModalVerbsResultDict
    utt: str


class Example(BaseModel):
    """
    Catch-all example model. Must supply at least one of `english` or `japanese`.
    NOTE: Currently defined in terms of English examples in modal_verbs.jsonl.
    """

    model_config = ConfigDict(populate_by_name=True)

    eid: str = Field(
        ...,
        description="unique id for record-keeping across translations and augmentations (we just use the english example)",
    )

    english: Optional[str] = Field(None, description="marked sentence in English")
    japanese: Optional[str] = Field(None, description="marked sentence in Japanese")

    english_target: Optional[str] = Field(
        None,
        description="target modal verb; corresponds to `mv` field in modal_verbs.jsonl",
    )
    japanese_target: Optional[str] = Field(
        None, description="target modality expression"
    )
    grammatical: GrammarLabel = Field(
        ...,
        description="expected grammaticality for marked modality; 'yes' unless ungrammatical augmented example",
    )
    human_annotations: Optional[ModalVerbsResultDict] = Field(
        None, description="Labels provided by human annotators keyed by taxonomy"
    )
    expected_categories: Optional[ModalVerbsResultDict] = Field(
        None,
        description="Majority classification category from human annotators keyed by taxonomy; same as `human_annotations` if no majority",
    )

    @classmethod
    def from_modal_verb_example(
        cls,
        example: ModalExample,
    ) -> Self:
        # use SHA-256 of the English utterance as stable unique ID
        eid = hashlib.sha256(example.utt.encode("utf-8")).hexdigest()
        return cls(
            eid=eid,
            english=example.utt,
            japanese=None,
            grammatical=GrammarLabel["yes"],
            english_target=example.mv,
            japanese_target=None,
            human_annotations=example.annotations,
            expected_categories=example.res,
        )

    @model_validator(mode="after")
    def require_language(cls, m):
        if not (m.english or m.japanese):
            raise ValueError(
                "must supply at least one of `english` or `japanese` example"
            )
        if m.english and not m.english_target:
            raise ValueError(f"target modality must be set for '{m.english}'")
        if m.japanese and not m.japanese_target:
            raise ValueError(f"target modality must be set for '{m.japanese}'")
        return m


class AcceptabilityTestEntry(Example):
    """
    A single "is this acceptable?" variant derived from an Example.
    """

    test_type: Literal["acceptability"]
    transformation_strategy: str


class SubstitutionTestEntry(Example):
    """
    A single substitution‐variant derived from an Example.
    """

    test_type: Literal["substitution"]
    transformation_strategy: str
    expected_res: Optional[ModalVerbsResultDict] = None


class EntailmentTestEntry(Example):
    """
    A single entailment‐pair test derived from an Example.
    """

    test_type: Literal["entailment"]
    # Example.english is the premise; we carry hypothesis and label here
    hypothesis: str
    label: Literal["entailment", "neutral", "contradiction"]
    transformation_strategy: str


NonNegInt = conint(ge=0)
Percent = confloat(ge=0, le=100)

CategoryDistribution: TypeAlias = dict[str, NonNegInt]
CategoryPercentages: TypeAlias = dict[str, Percent]


class LanguageResult(_Model):
    model_config = ConfigDict(frozen=True)
    prompt: str
    answers: List[str]

    @computed_field
    @property
    def distribution(self) -> CategoryDistribution:
        cnt = Counter(a.strip().lower() for a in self.answers)
        return {k: int(v) for k, v in cnt.items()}

    @computed_field
    @property
    def percentages(self) -> CategoryPercentages:
        dist = self.distribution
        total = sum(dist.values()) or 1
        return {k: (count / total) * 100 for k, count in dist.items()}


class TaskResult(_Model):
    """
    Container for LLM outputs per language.
    """

    english: Optional[LanguageResult] = Field(
        None, description="LLM results for English"
    )
    japanese: Optional[LanguageResult] = Field(
        None, description="LLM results for Japanese"
    )


class UnifiedResult(_Model):
    """
    Combined results entry representing either grammar‐check and/or
    modal‐classification outcomes for a single example, in English
    and/or Japanese.
    """

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    # Identification & raw input
    eid: str = Field(..., description="Unique example identifier")
    english: Optional[str] = Field(None, description="Original English sentence")
    japanese: Optional[str] = Field(None, description="Original Japanese sentence")
    english_target: Optional[str] = Field(
        None, description="Target modal verb for English sentence"
    )
    japanese_target: Optional[str] = Field(
        None, description="Target modality expression for Japanese sentence"
    )

    grammatical: GrammarLabel = Field(
        ..., description="Expected grammaticality for marked modality"
    )
    grammar: Optional[TaskResult] = Field(
        None, description="Grammar‐check results for English and/or Japanese"
    )

    human_annotations: Optional[ModalVerbsResultDict] = Field(
        None, description="Labels provided by human annotators keyed by taxonomy"
    )
    expected_categories: Optional[ModalVerbsResultDict] = Field(
        None,
        description="Majority classification from human annotators keyed by taxonomy",
    )
    classification: Optional[dict[Taxonomy, TaskResult]] = Field(
        None, description="Modal‐classification results keyed by taxonomy"
    )

    @model_validator(mode="after")
    def require_some_output(cls, m):
        """
        Must include at least one of grammar‐check or classification
        outputs in at least one language.
        """
        if not (
            (m.grammar and (m.grammar.english or m.grammar.japanese))
            or (
                m.classification
                and any(v.english or v.japanese for v in m.classification.values())
            )
        ):
            raise ValueError(
                "UnifiedResult must carry at least one grammar or classification result"
            )
        return m

    @model_validator(mode="after")
    def _validate_targets(self):
        if self.english and not self.english_target:
            raise ValueError("English target required for English examples")
        if self.japanese and not self.japanese_target:
            raise ValueError("Japanese target required for Japanese examples")
        return self
