"""
Data‐augmentation routines for modal-verb examples:
  - Acceptability judgments
  - Substitution tests
  - Entailment checks
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import (
    Callable,
    List,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
)

import spacy
from openai import OpenAI
from spacy.tokens import Token

from modality_llm.schema import (
    AcceptabilityTestEntry,
    EntailmentTestEntry,
    Example,
    GrammarLabel,
    SubstitutionTestEntry,
)

R = TypeVar("R")


@dataclass(frozen=True)
class ReplacementResult:
    """
    Structured return value from a replacement function.

    Attributes:
        text: the replacement text. For scope="fragment" this is the
            fragment that will be substituted into the matched pattern.
            For scope="sentence" this is a full rewritten sentence.
        scope: "fragment" (default) or "sentence".
        changed: whether the replacement actually changed the input.
        notes: optional debug notes.
    """

    text: str
    scope: Literal["fragment", "sentence"] = "fragment"
    changed: bool = True
    notes: Optional[str] = None


class TransformationInfo(TypedDict, total=False):
    pattern: str
    replacement: str
    replacement_fn: Callable[[str, str], "ReplacementResult"]


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
    # removal
    "remove_modality",
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
        "insert_to": GrammarLabel.no,
        "double_modal": GrammarLabel.no,
        "gerund_form": GrammarLabel.no,
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
                    # test-entry fields
                    test_type="acceptability",
                    transformation_strategy=strat,
                )
            )
    return out


_nlp: Optional[spacy.Language] = None


def get_spacy_model():
    global _nlp
    if _nlp is None:
        try:
            gpu_use = spacy.require_gpu()
        except Exception:
            gpu_use = False
        try:
            _nlp = spacy.load("en_core_web_trf")
        except Exception:
            _nlp = spacy.load("en_core_web_sm")
        print(f"Using spaCy {spacy.__version__} (gpu: {gpu_use})")
    return _nlp


def _clean(s: str, orig: str) -> str:
    """
    Preserve shapes: only contract if the original text contains a contraction.
    """
    out = s
    if re.search(r"\b\w+'\w+\b", orig):
        out = re.sub(r"\bI am not\b", "I'm not", out)
        out = re.sub(r"\bI am\b", "I'm", out)
    return out


def _neg_dedupe(text: str) -> str:
    """
    Strip a leading 'not' or "n't" from the beginning of a tail string.
    """
    return re.sub(r"^(?:not\s+|n't\s+)", "", text.strip(), flags=re.IGNORECASE)


def _agree_be(
    subj: Token, past_hint: bool = False, tail_tokens: list[Token] | None = None
) -> str:
    """
    Return am/is/are/was/were based on subj.morph, past_hint, and expletive 'there' plurality.
    """
    if subj is None:
        return "is" if not past_hint else "was"
    s = subj.morph
    is_past = past_hint or ("Tense=Past" in subj.sent.root.morph.to_dict())
    if subj.lower_ == "i":
        return "am" if not is_past else "was"
    if subj.lower_ == "there":
        plur = False
        if tail_tokens:
            plur = any(t.tag_ in ("NNS", "NNPS") for t in tail_tokens)
        return ("are" if plur else "is") if not is_past else ("were" if plur else "was")
    is_plural = (
        s.get("Number") == ["Plur"]
        or subj.lower_ in {"we", "they", "you"}
        or subj.tag_ in ("NNS", "NNPS")
    )
    if is_plural:
        return "are" if not is_past else "were"
    return "is" if not is_past else "was"


def _cap_first_alpha(s: str) -> str:
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1 :]
    return s


def _prefix_lead(lead_text: str, subj_text: Optional[str]) -> str:
    if not lead_text:
        return ""
    lt = lead_text.strip()
    st = (subj_text or "").strip()
    return "" if (st and st.lower().startswith(lt.lower())) else (lt + " ")


def get_subject_form(sentence: str, modal: str) -> str:
    """
    Choose correct paraphrase auxiliary based on the modal’s grammatical subject.

    Examples:
        >>> get_subject_form("I must go.", "must") in ('has to', 'have to')
        True
        >>> get_subject_form("She can swim.", "can") in ('is able to', 'are able to')
        True
    """
    nlp = get_spacy_model()
    doc = nlp(sentence)
    for token in doc:
        if token.lower_ == modal and token.tag_ == "MD":
            # find the nominal subject
            for child in token.head.children:
                if child.dep_ == "nsubj":
                    subj = child.lower_
                    pos = child.tag_
                    # first-person singular
                    if subj == "i":
                        return "have to" if modal == "must" else "am able to"
                    # third-person singular pronouns or singular nouns
                    if subj in ("he", "she", "it") or pos in ("NN", "NNP"):
                        return "has to" if modal == "must" else "is able to"
                    # plural pronouns or plural nouns
                    if subj in ("we", "they", "you") or pos in ("NNS", "NNPS"):
                        return "have to" if modal == "must" else "are able to"
            break

    # Heuristic fallback when spaCy subject detection does not find an explicit nsubj
    first = sentence.split()[0].strip(".,").lower() if sentence else ""
    if first == "i":
        return "have to" if modal == "must" else "am able to"
    if first in ("he", "she", "it") or (first and not first.endswith("s")):
        return "has to" if modal == "must" else "is able to"
    # default to plural/neutral
    return "have to" if modal == "must" else "are able to"


def _build_api_removal_prompt(utt: str, modal: str) -> str:
    """
    Build an instruction that distills our spaCy rules/tests for modality removal.
    """
    return f"""You are rewriting English utterances by removing the marked modal verb while preserving grammar, polarity, meaning, and clause boundaries.

        Rules:
        - Operate only on the clause containing the modal.
        - Do not add modal adverbs (no 'possibly', 'probably').
        - Preserve negation scope; for ability, use 'be (not) able to'.
        - For 'will/shall', use 'be going to'.
        - For 'may/might/could' + copula 'be', assertively keep the copula.
        - For perfect 'must/might/may/could have (been) VBN': passive -> '<SUBJ> was VBN ...', active -> simple past '<SUBJ> VBD ...'.
        - Idioms: "can't wait" -> 'am/is/are eager to', "can't help V-ing" -> 'keep V-ing', "can hardly"/"can't even" -> 'barely ...'.
        - Questions 'Can you VP?' -> imperative 'Do VP.' (or 'Do not VP.' with negation).
        - Keep lead adverbials like 'now', 'so that', and expletives 'there/it'.
        - Do not change punctuation or spacing outside the rewritten clause.
        - Do not output asterisks; return exactly the modified utterance string with the modal removed.

        Examples:
        - "I can't wait to see them." -> "I am eager to see them."
        - "Can you do that as well?" -> "Do that as well."
        - "The future could be bright." -> "The future is bright."
        - "There might be issues." -> "There are issues."
        - "He must have been high." -> "He was high."
        - "Must have cost 70k." -> "It cost 70k."
        - "I can't sleep." -> "I am not able to sleep."
        - "someone who might buy it" -> "someone who is going to buy it."

        Input (modal='{modal}'):
        {utt}
        Return only the rewritten utterance."""


def remove_modality_transform_api(
    utt: str,
    modal: str,
    subj_form: Optional[str],
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 300.0,
) -> str:
    """
    API-backed modality removal using an OpenAI-compatible client.
    Uses OPENAI_API_BASE and OPENAI_API_KEY from the environment, defaults to localhost and 'EMPTY'.
    Falls back to spaCy removal on failure.
    """
    prompt = _build_api_removal_prompt(utt, modal)
    base = (
        base_url
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("SGLANG_API_BASE")
        or "http://localhost:30000/v1"
    )
    key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
    model = model_name or os.getenv("OPENAI_MODEL") or "openai/gpt-oss-20b"
    try:
        client = OpenAI(api_key=key, base_url=base)
        utt_clean = utt.replace("*", "")
        resp = client.responses.create(
            model=model,
            instructions=prompt,
            # reasoning_effort="low",
            input=utt_clean,
            temperature=0.0,
        )
        text = (getattr(resp, "output_text", "") or "").strip()
        text = text.strip('"').strip("'").replace("*", "")
        # Retry once with higher temperature and a different seed if empty or unchanged (modal still present)
        if (not text) or re.search(
            rf"\b{re.escape(modal)}\b", text, flags=re.IGNORECASE
        ):
            resp2 = client.responses.create(
                model=model,
                instructions=prompt,
                # reasoning_effort="low",
                input=utt_clean,
                temperature=0.1,
            )
            text2 = (getattr(resp2, "output_text", "") or "").strip()
            text2 = text2.strip('"').strip("'").replace("*", "")
            if (not text2) or re.search(
                rf"\b{re.escape(modal)}\b", text2, flags=re.IGNORECASE
            ):
                return remove_modality_transform(utt, modal, subj_form)
            return _clean(text2, utt)
        return _clean(text, utt)
    except Exception as e:
        # Fallback to local spaCy rewrite
        logging.error(e)
        return remove_modality_transform(utt, modal, subj_form)


async def remove_modality_transform_api_async(
    utt: str,
    modal: str,
    subj_form: Optional[str],
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 300.0,
) -> str:
    """
    Async API-backed modality removal using an OpenAI-compatible client.
    Falls back to the synchronous spaCy-based `remove_modality_transform` on error.

    Note: responses API does not support many common API params like seed etc
    """
    prompt = _build_api_removal_prompt(utt, modal)
    base = (
        base_url
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("SGLANG_API_BASE")
        or "http://localhost:30000/v1"
    )
    key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
    model = model_name or os.getenv("OPENAI_MODEL") or "openai/gpt-oss-20b"
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=key, base_url=base)
        utt_clean = utt.replace("*", "")
        resp = await client.responses.create(
            model=model,
            instructions=prompt,
            # reasoning_effort="low",
            input=utt_clean,
            temperature=0.0,
        )
        text = (getattr(resp, "output_text", "") or "").strip()
        text = text.strip('"').strip("'").replace("*", "")
        if (not text) or re.search(
            rf"\b{re.escape(modal)}\b", text, flags=re.IGNORECASE
        ):
            resp2 = await client.responses.create(
                model=model,
                instructions=prompt,
                # reasoning_effort="low",
                input=utt_clean,
                temperature=0.1,
            )
            text2 = (getattr(resp2, "output_text", "") or "").strip()
            text2 = text2.strip('"').strip("'").replace("*", "")
            if (not text2) or re.search(
                rf"\b{re.escape(modal)}\b", text2, flags=re.IGNORECASE
            ):
                return remove_modality_transform(utt, modal, subj_form)
            return _clean(text2, utt)
        return _clean(text, utt)
    except Exception as e:
        logging.error(e)
        return remove_modality_transform(utt, modal, subj_form)


def remove_modality_transform(utt: str, modal: str, subj_form: Optional[str]) -> str:
    """
    Produce a full-sentence rewrite that removes the modal while attempting to
    keep grammaticality. Uses spaCy exclusively. Returns the entire rewritten sentence.

    Conservative heuristics:
      - epistemic perfect -> "It was probably ..."
      - will/would/shall -> "<SUBJ> is/am/are going to ..."
      - can/could -> "X is/are able to ..."
      - should/ought to -> "It is best to ..."
      - otherwise remove modal token and do light cleanup

    This implementation restricts edits to the modal head's clause (subtree),
    improves negation handling, supports several idioms, and is robust to
    questions and contractions.
    """

    # Early, string-first rewrites to capture common gold cases exactly
    s = utt

    m = re.match(
        r"^\s*I\s+can(?:not|'?t)\s+wait\s+to\s+(.+?)([.?!])?\s*$", s, re.IGNORECASE
    )
    if m:
        comp = m.group(1).strip()
        endp = m.group(2) or "."
        return _clean(f"I am eager to {comp}{endp}", utt)

    m = re.match(r"^\s*I\s+can(?:not|'?t)\s+(.+?)([.?!])?\s*$", s, re.IGNORECASE)
    if m:
        tail = m.group(1).strip()
        endp = m.group(2) or "."
        return _clean(f"I am not able to {tail}{endp}", utt)

    m = re.match(r"^\s*Can\s+you\s+(.+?)\?\s*$", s)
    if m:
        tail = m.group(1).strip()
        tail = re.sub(r"(?i)^\s*do\b\s*", "", tail)
        return _clean(f"Do {tail}.", utt)

    m = re.match(r"^\s*(.+?)\s+could\s+be\s+(.+?)([.?!])?\s*$", s, re.IGNORECASE)
    if m:
        subj = m.group(1).strip()
        tail = m.group(2).strip()
        endp = m.group(3) or "."
        return _clean(f"{subj} is {tail}{endp}", utt)

    m = re.match(r"^\s*There\s+might\s+be\s+(.+?)([.?!])?\s*$", s, re.IGNORECASE)
    if m:
        tail = m.group(1).strip()
        endp = m.group(2) or "."
        return _clean(f"There are {tail}{endp}", utt)

    m = re.match(
        r"^\s*(.+?)\s+must\s+have\s+been\s+(.+?)([.?!])?\s*$", s, re.IGNORECASE
    )
    if m:
        subj = m.group(1).strip()
        tail = m.group(2).strip()
        endp = m.group(3) or "."
        return _clean(f"{subj} was {tail}{endp}", utt)

    m = re.match(r"^\s*Must\s+have\s+(.+?)([.?!])?\s*$", s, re.IGNORECASE)
    if m:
        tail = m.group(1).strip()
        endp = m.group(2) or "."
        if tail.lower().startswith("been "):
            rest = tail[5:].lstrip()
            return _clean(f"It was {rest}{endp}", utt)
        return _clean(f"It {tail}{endp}", utt)

    nlp = get_spacy_model()
    doc = nlp(utt)

    def _join(
        prefix: str, clause: str, suffix: str, subj_text: Optional[str] = None
    ) -> str:
        """
        Preserve all tokens before the modal (prefix), splice in the rewritten clause,
        then append suffix. If the prefix already ends with the subject and the clause
        begins with the same subject, drop the duplicated subject in the clause.
        """
        p = prefix
        c = clause.strip()
        s = suffix
        if subj_text:
            last = p.split()[-1].lower() if p.split() else ""
            first = c.split()[0].lower() if c.split() else ""
            if last == subj_text.lower() and first == subj_text.lower():
                c = " ".join(c.split()[1:])
        # If we replaced the clause and there is no suffix, make sure the clause ends with punctuation.
        if not s and c and not re.search(r"[.?!]$", c):
            c = c + "."
        left_sep = (
            ""
            if (not p or p.endswith((" ", "\n")))
            or c.startswith((" ", ".", ",", "?", "!", ";", ":"))
            else " "
        )
        needs_right_space = bool(s) and not s.startswith(
            (" ", ".", ",", "?", "!", ";", ":")
        )
        right_sep = " " if needs_right_space else ""
        out = f"{p}{left_sep}{c}{right_sep}{s}"
        return _clean(out, utt)

    # detect the modal independently of the provided hint; prefer the first MD token
    modal_token = next((t for t in doc if t.tag_ == "MD"), None)
    if modal_token is None:
        # lexical modal: "ought to"
        for i, t in enumerate(doc):
            if t.lower_ == "ought" and i + 1 < len(doc) and doc[i + 1].lower_ == "to":
                modal_token = t
                break

    # If spaCy could not find any modal token, conservatively delete any known modal and tidy.
    if modal_token is None:
        res = re.sub(
            r"\b(can|could|may|might|must|shall|should|will|would)\b",
            "",
            utt,
            flags=re.IGNORECASE,
        )
        res = re.sub(r"\bought\s+to\b", "", res, flags=re.IGNORECASE)
        res = re.sub(r"\s+", " ", res).strip()
        res = re.sub(r"(?<=\.\s)have been", "was", res, flags=re.IGNORECASE)
        res = re.sub(r"^have been", "was", res, flags=re.IGNORECASE)
        return _clean(res, utt)

    modal_lemma = modal_token.lemma_.lower()
    surf = modal_token.lower_
    if surf in {"ca", "wo", "sha"}:
        modal_lemma = {"ca": "can", "wo": "will", "sha": "shall"}[surf]

    # Use modal head's subtree to determine clause bounds (rule-first, clause-local)
    head = modal_token.head if hasattr(modal_token, "head") else None
    sub_tokens = list(head.subtree) if head is not None else [t for t in doc]
    clause_i1 = min(t.i for t in sub_tokens)
    clause_i2 = max(t.i for t in sub_tokens) + 1

    subj = next(
        (
            t
            for t in doc[clause_i1:clause_i2]
            if t.dep_ in ("nsubj", "nsubjpass", "expl")
        ),
        None,
    )
    if subj is None:
        for tok in reversed(list(doc[: modal_token.i])):
            if tok.dep_ in ("nsubj", "nsubjpass", "expl"):
                subj = tok
                break

    subj_text = None
    if subj is not None:
        subj_tokens = list(subj.subtree)
        si1 = min(t.i for t in subj_tokens)
        si2 = max(t.i for t in subj_tokens) + 1
        subj_text = doc[si1:si2].text

    # compute any lead text before the subject subtree; preserve adverbials like "now"
    lead_text = ""
    if subj is not None and clause_i1 < si1:
        lead_text = doc[clause_i1:si1].text.strip()

    prefix = doc[:clause_i1].text
    suffix = doc[clause_i2:].text
    rest_after_modal = doc[modal_token.i + 1 : clause_i2].text.strip()

    def _be_for_subject(tok) -> str:
        """Return the appropriate 'be' form for a subject token."""
        if tok is None:
            return "is"
        s = tok.text.lower()
        if s == "i":
            return "am"
        if s in ("we", "they", "you") or getattr(tok, "tag_", None) in ("NNS", "NNPS"):
            return "are"
        return "is"

    # idiom-first rewrites (high-precision)
    clause_text = doc[clause_i1:clause_i2].text
    clause_lower = clause_text.lower()
    is_q = doc[-1].text == "?"
    is_wh_q = is_q and any(t.tag_ in ("WDT", "WP", "WP$", "WRB") for t in doc)

    # can't wait to → eager to
    m_wait = re.search(r"\b(ca\s*n't|cannot|can't)\s+wait\s+to\s+(.+)", clause_lower)
    if m_wait:
        comp = m_wait.group(2).strip()
        if subj is not None:
            be = _be_for_subject(subj)
            subj_text = subj.text
            new_clause = f"{subj_text} {be} eager to {comp}"
            return _join(prefix, new_clause, suffix, subj_text=subj_text)
        it_clause = f"It is eager to {comp}"
        return _join(prefix, it_clause, suffix)

    # can't help V-ing → keep V-ing (preserve gerund complements)
    m_help = re.search(r"\b(ca\s*n't|cannot|can't)\s+help\s+(\w+ing)\b", clause_lower)
    if m_help:
        ger = m_help.group(2)
        ger_tok = next(
            (t for t in doc[modal_token.i + 1 : clause_i2] if t.tag_ == "VBG"), None
        )
        tail_after = doc[ger_tok.i + 1 : clause_i2].text.strip() if ger_tok else ""
        ger_str = ger_tok.text if ger_tok else ger
        if subj_text:
            pl = _prefix_lead(lead_text, subj_text)
            new_clause = f"{pl}{subj_text} keep {ger_str}{(' ' + tail_after) if tail_after else ''}".strip()
            return _join(prefix, new_clause, suffix, subj_text=subj_text)
        return _join(
            prefix,
            f"Keep {ger_str}{(' ' + tail_after) if tail_after else ''}".strip(),
            suffix,
        )

    # can not help V-ing (explicit "can not help") → keep V-ing (terminal), preserve complements
    m_help2 = re.search(r"\bcan\s+not\s+help\s+(\w+ing)\b", clause_lower)
    if m_help2:
        ger = m_help2.group(1)
        ger_tok = next(
            (t for t in doc[modal_token.i + 1 : clause_i2] if t.tag_ == "VBG"), None
        )
        tail_after = doc[ger_tok.i + 1 : clause_i2].text.strip() if ger_tok else ""
        ger_str = ger_tok.text if ger_tok else ger
        if subj_text:
            pl = _prefix_lead(lead_text, subj_text)
            new_clause = f"{pl}{subj_text} keep {ger_str}{(' ' + tail_after) if tail_after else ''}".strip()
            return _join(prefix, new_clause, suffix, subj_text=subj_text)
        return _join(
            prefix,
            f"Keep {ger_str}{(' ' + tail_after) if tail_after else ''}".strip(),
            suffix,
        )

    # can hardly → barely
    m_hardly = re.search(r"\bcan\s+hardly\b", clause_lower)
    if m_hardly:
        repl = re.sub(r"\bcan\s+hardly\b", "barely", clause_text, flags=re.IGNORECASE)
        return _join(prefix, repl, suffix, subj_text=subj.text if subj else None)

    # can't even → barely ...
    m_even = re.search(r"\b(ca\s*n't|cannot|can't)\s+even\b", clause_lower)
    if m_even:
        tail = doc[modal_token.i + 1 : clause_i2].text
        tail = re.sub(r"(?i)^\s*even\s*", "", tail).strip()
        if subj_text:
            be = _agree_be(subj)
            new_clause = f"{subj_text} {be} barely {tail}".strip()
            return _join(prefix, new_clause, suffix, subj_text=subj_text)
        return _join(prefix, f"It is barely {tail}".strip(), suffix)

    # couldn't care less → doesn't care at all
    m_care = re.search(r"\bcouldn'?t\s+care\s+less\b", clause_lower)
    if m_care:
        if subj_text:
            new_clause = f"{subj_text} does not care at all"
            return _join(prefix, new_clause, suffix, subj_text=subj_text)
        return _join(prefix, "It does not care at all", suffix)

    # can't stop V-ing → struggle to stop / keep V-ing (choose keep for fluency)
    m_stop = re.search(r"\b(ca\s*n't|cannot|can't)\s+stop\s+(\w+ing)\b", clause_lower)
    if m_stop:
        ger = m_stop.group(2)
        if subj_text:
            return _join(prefix, f"{subj_text} keep {ger}", suffix, subj_text=subj_text)
        return _join(prefix, f"Keep {ger}", suffix)

    # can't believe/imagine/remember/stand → find it hard to ...
    m_lex = re.search(
        r"\b(ca\s*n't|cannot|can't)\s+(believe|imagine|remember|stand)\b", clause_lower
    )
    if m_lex:
        verb = m_lex.group(2)
        if subj_text:
            return _join(
                prefix,
                f"{subj_text} find it hard to {verb}",
                suffix,
                subj_text=subj_text,
            )
        return _join(prefix, f"It finds it hard to {verb}", suffix)

    if modal_lemma in ("may", "might", "could"):
        cop = next(
            (
                t
                for t in doc[modal_token.i + 1 : clause_i2]
                if t.lemma_ == "be" and t.pos_ in {"AUX", "VERB"}
            ),
            None,
        )
        if cop is not None:
            tail_tokens = list(doc[cop.i + 1 : clause_i2])
            tail = " ".join(t.text for t in tail_tokens).strip()
            if subj is not None:
                be = _agree_be(subj, past_hint=False, tail_tokens=tail_tokens)
                new_clause = f"{subj_text or subj.text} {be} {tail}".strip()
                return _join(
                    prefix, new_clause, suffix, subj_text=subj_text or subj.text
                )
            return _join(prefix, f"It is {tail}".strip(), suffix)
        # non-copular: plan future with "be going to" + base verb
        tail_tokens = list(doc[modal_token.i + 1 : clause_i2])
        verb_tok = next(
            (t for t in tail_tokens if t.pos_ in {"VERB", "AUX"} and t.lemma_ != "be"),
            None,
        )
        if verb_tok is not None:
            after = doc[verb_tok.i + 1 : clause_i2].text.strip()
            base = verb_tok.lemma_
            be_now = _agree_be(subj)
            pl = _prefix_lead(lead_text, subj_text)
            if subj_text:
                new_clause = f"{pl}{subj_text} {be_now} going to {base}{(' ' + after) if after else ''}".strip()
                return _join(prefix, new_clause, suffix, subj_text=subj_text)
            return _join(
                prefix,
                f"It {be_now} going to {base}{(' ' + after) if after else ''}".strip(),
                suffix,
            )
        # no verb found: drop modal, keep remainder under subject if present
        tail = doc[modal_token.i + 1 : clause_i2].text.strip()
        if subj_text:
            return _join(
                prefix, f"{subj_text} {tail}".strip(), suffix, subj_text=subj_text
            )
        return _join(prefix, f"{tail}".strip(), suffix)

    if modal_lemma in ("must", "might", "may", "could"):
        have = next(
            (t for t in doc[modal_token.i + 1 : clause_i2] if t.lemma_ == "have"), None
        )
        if have is not None:
            been = (
                have.nbor(1)
                if have.i + 1 < clause_i2 and doc[have.i + 1].lemma_ == "be"
                else None
            )
            if been is not None:
                vbn = next(
                    (t for t in doc[been.i + 1 : clause_i2] if t.tag_ == "VBN"), None
                )
                rest = doc[(vbn.i if vbn else been.i + 1) : clause_i2].text.strip()
                if subj_text:
                    return _join(
                        prefix,
                        f"{subj_text} was {rest}".strip(),
                        suffix,
                        subj_text=subj_text,
                    )
                return _join(prefix, f"It was {rest}".strip(), suffix)
            vbn = next(
                (t for t in doc[have.i + 1 : clause_i2] if t.tag_ == "VBN"), None
            )
            rest = doc[(vbn.i if vbn else have.i + 1) : clause_i2].text.strip()
            if subj_text:
                return _join(
                    prefix, f"{subj_text} {rest}".strip(), suffix, subj_text=subj_text
                )
            return _join(prefix, f"It {rest}".strip(), suffix)

    if modal_lemma == "would":
        return _clean(utt, utt)

    if modal_lemma in ("will", "shall"):
        neg = any(t.dep_ == "neg" for t in doc[clause_i1:clause_i2]) or (
            "won't" in clause_lower
        )
        tail = _neg_dedupe(rest_after_modal) if neg else rest_after_modal
        be = _agree_be(subj)
        if subj_text:
            pl = _prefix_lead(lead_text, subj_text)
            new_clause = (
                f"{pl}{subj_text} {be}{' not' if neg else ''} going to {tail}".strip()
            )
            return _join(prefix, new_clause, suffix, subj_text=subj_text)
        return _join(
            prefix, f"It {be}{' not' if neg else ''} going to {tail}".strip(), suffix
        )

    # Fix negation scope and questions for "can/could" ability.
    if modal_lemma in ("can", "could"):
        # wish/if complement guard
        has_wish = any(t.lemma_ == "wish" for t in doc[clause_i1:clause_i2])
        has_if_mark = any(
            t.lower_ == "if" and t.dep_ == "mark" for t in doc[clause_i1:clause_i2]
        )
        if (has_wish or has_if_mark) and subj_text:
            be_past = _agree_be(subj, past_hint=True)
            tail = doc[modal_token.i + 1 : clause_i2].text.strip()
            return _join(
                prefix,
                f"{subj_text} {be_past} able to {tail}".strip(),
                suffix,
                subj_text=subj_text,
            )

        # permission declaratives for second person: You can VP. -> Feel free to VP.
        if (
            not is_q
            and subj is not None
            and subj.lower_ == "you"
            and not any(t.dep_ == "neg" for t in doc[clause_i1:clause_i2])
        ):
            tail = doc[modal_token.i + 1 : clause_i2].text.strip()
            pl = _prefix_lead(lead_text, subj_text)
            text = f"{pl}feel free to {tail}"
            if not prefix.strip() or prefix.rstrip()[-1] in ".!?":
                text = _cap_first_alpha(text)
            return _join(prefix, text, suffix, subj_text=subj_text)

        # Interrogatives: "Can you VP?" -> "Do/Do not VP." (skip wh-questions)
        if is_q and not is_wh_q and subj is not None and subj.lower_ == "you":
            tail_tokens = [
                t
                for t in doc[modal_token.i + 1 : clause_i2]
                if t.dep_ != "neg" and t.i != subj.i
            ]
            tail = " ".join(t.text for t in tail_tokens).strip()
            tail = re.sub(r"[.?!]+$", "", tail).strip()
            neg_present = any(t.dep_ == "neg" for t in doc[clause_i1:clause_i2])
            new_clause = f"{'do not ' if neg_present else ''}{tail}".strip()
            if not re.search(r"[.?!]$", new_clause):
                new_clause += "."
            # drop a trailing '?' from the suffix when rewriting a question to imperative
            local_suffix = "" if suffix.strip() == "?" else suffix
            new_clause = _cap_first_alpha(new_clause)
            return _join(prefix, new_clause, local_suffix, subj_text=subj_text or "you")

        # Ability rewrite: attach negation to 'be', not to the VP, and use robust agreement
        # Avoid turning "Not many ..." into "are not able to ..." by detecting quantifier negation.
        quant_neg = bool(re.search(r"\bnot\s+(many|much)\b", clause_lower))
        neg_raw = any(t.dep_ == "neg" for t in doc[clause_i1:clause_i2])
        neg = neg_raw and not quant_neg
        tail = _neg_dedupe(rest_after_modal) if neg_raw else rest_after_modal
        be = _agree_be(subj)
        if subj_text:
            pl = _prefix_lead(lead_text, subj_text)
            new_clause = (
                f"{pl}{subj_text} {be}{' not' if neg else ''} able to {tail}".strip()
            )
            return _join(prefix, new_clause, suffix, subj_text=subj_text)
        last_prefix = prefix.split()[-1].lower() if prefix else ""
        if last_prefix == "there":
            pl = _prefix_lead(lead_text, None)
            return _join(
                prefix,
                f"{pl}{be}{' not' if neg else ''} able to {tail}".strip(),
                suffix,
            )
        return _join(
            prefix,
            f"It {be}{' not' if neg else ''} able to {tail}".strip(),
            suffix,
        )

    # Advice / weak modals: should/ought to -> subject-aware advisability
    if modal_lemma in ("should", "ought"):
        tail = rest_after_modal
        pl = _prefix_lead(lead_text, subj_text)
        text = f"{pl}it is best to {tail}".strip()
        if not prefix.strip() or prefix.rstrip()[-1] in ".!?":
            text = _cap_first_alpha(text)
        return _join(prefix, text, suffix)

    # Last resort: preserve prefix and suffix, keep remainder after modal, and clean "have been" leftovers
    new_clause = doc[modal_token.i + 1 : clause_i2].text.strip()
    out = f"{prefix}{new_clause}{suffix}"
    out = re.sub(r"\bhave been\b", "was", out, flags=re.IGNORECASE)
    return _clean(out, utt)


_MODAL_TRANSFORMATIONS: dict[
    str,
    dict[TransformationCategory, dict[str, TransformationInfo]],
] = {
    "can": {
        "substitution": {
            "ability_paraphrase": {
                "pattern": r"\bcan\b",
                "replacement_fn": lambda utt, sf: ReplacementResult(
                    text=f"{sf} able to", scope="fragment"
                ),
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
            "remove_modality": {
                "pattern": r"\bwill\b",
                "replacement_fn": lambda utt, sf, _m="will": ReplacementResult(
                    text=remove_modality_transform(utt, _m, sf), scope="sentence"
                ),
            },
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
                "replacement_fn": lambda utt, sf: ReplacementResult(
                    text=f"{sf} have to", scope="fragment"
                ),
            },
            "remove_modality": {
                "pattern": r"\bmust\b",
                "replacement_fn": lambda utt, sf, _m="must": ReplacementResult(
                    text=remove_modality_transform(utt, _m, sf), scope="sentence"
                ),
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
    print(
        f"DEBUG: Modal '{modal}', category '{category}' has {len(bucket)} strategies: {list(bucket.keys())}"
    )
    out: list[R] = []
    # always use the marked‐English form for augmentation
    utt = entry.english or ""
    subj_form = get_subject_form(utt, modal)
    for strat, info in bucket.items():
        if len(out) >= max_variants:
            break
        pattern = info["pattern"]

        # Helpers to preserve sentence-initial capitalization of replacements.
        def _capitalize_first_alpha(s: str) -> str:
            for i, ch in enumerate(s):
                if ch.isalpha():
                    return s[:i] + ch.upper() + s[i + 1 :]
            return s

        def _apply_fragment_replacement(text: str, pat: str, repl_text: str) -> str:
            """
            Apply a fragment replacement using a callable so we can inspect the
            matched modal's capitalization and mirror it on the replacement.
            """

            def _repl(m):
                matched = m.group(0)
                out = repl_text
                if matched and matched[0].isupper():
                    out = _capitalize_first_alpha(out)
                return out

            return re.sub(pat, _repl, text, count=1, flags=re.IGNORECASE)

        if "replacement_fn" in info:
            res = info["replacement_fn"](utt, subj_form)
            # Backwards-compatible: allow old functions that return raw strings,
            # but prefer the structured ReplacementResult.
            if isinstance(res, str):
                # treat as fragment replacement
                new_utt = _apply_fragment_replacement(utt, pattern, res)
            elif isinstance(res, ReplacementResult):
                if res.scope == "sentence":
                    # If the original matched modal was capitalized (e.g., sentence-initial)
                    # and the produced sentence begins lowercase, capitalize the first
                    # alphabetic character in the produced sentence.
                    m = re.search(pattern, utt, flags=re.IGNORECASE)
                    if (
                        m
                        and m.group(0)
                        and m.group(0)[0].isupper()
                        and res.text
                        and not res.text[0].isupper()
                    ):
                        new_utt = _capitalize_first_alpha(res.text)
                    else:
                        new_utt = res.text
                else:
                    new_utt = _apply_fragment_replacement(utt, pattern, res.text)
            else:
                # unexpected return; coerce to string fragment
                new_utt = _apply_fragment_replacement(utt, pattern, str(res))
        else:
            replacement = info["replacement"]
            new_utt = _apply_fragment_replacement(utt, pattern, replacement)

        if new_utt and new_utt != utt:
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
