import csv
import difflib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Literal, TypeVar

from pydantic import BaseModel

from modality_llm.schema import ModalExample

"""
Utility functions for model identifiers and caching.
"""

_ESCAPE_MAP = {
    "/": "+",
    ":": "%",
}
_UNESCAPE_MAP = {tok: ch for ch, tok in _ESCAPE_MAP.items()}


def sanitize_model_name(name: str) -> str:
    """
    Lowercase and escape special characters in a model-name for safe filenames.

    Reversible via `unsanitize_model_name`.

    Examples:
        >>> sanitize_model_name("meta-LLAMA/Llama-3.2-3B.Instruct")
        'meta-llama+llama-3.2-3b.instruct'
    """
    s = name.lower()
    for ch, tok in _ESCAPE_MAP.items():
        s = s.replace(ch, tok)
    return s


def unsanitize_model_name(s: str) -> str:
    """
    Reverse `sanitize_model_name`, restoring the original separators (all lowercased).

    Examples:
        >>> unsanitize_model_name('meta-llama+llama-3.2-3b.instruct')
        'meta-llama/llama-3.2-3b.instruct'
    """
    out = s
    for tok, ch in _UNESCAPE_MAP.items():
        out = out.replace(tok, ch)
    return out


T = TypeVar("T")


def chunked(xs: List[T], size: int) -> Iterator[List[T]]:
    """
    Yield successive size‐chunks of xs.

    >>> list(chunked([1, 2, 3, 4, 5], 2))
    [[1, 2], [3, 4], [5]]
    """
    for i in range(0, len(xs), size):
        yield xs[i : i + size]


def unanimous_examples(
    examples: List[ModalExample],
    which: Literal["palmer", "quirk", "both"] | None,
) -> List[ModalExample]:
    """
    Return only those ModalExample records where all three annotators
    agreed on the category for the given taxonomy.
    """
    if not which:
        return examples
    taxes = ["palmer", "quirk"] if which == "both" else [which]
    out: List[ModalExample] = []
    for ex in examples:
        anns = ex.annotations or {}
        ok = True
        for t in taxes:
            votes = anns.get(t, [])
            if len(votes) < 3 or len(set(votes)) != 1:
                ok = False
                break
        if ok:
            out.append(ex)
    return out


"""
I/O utilities for JSONL and CSV data.
"""


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_csv(path: str) -> List[Dict[str, str]]:
    """Load a CSV file into a list of rows (dicts)."""
    with open(path, "r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]


def write_json(path: str | Path, data: Any) -> None:
    """
    Dump a Python object (e.g. List[Dict] or List[BaseModel]) to a pretty-printed JSON file.
    """

    def _default(o: Any):
        # If it's a Pydantic model, dump to plain dict using aliases
        if isinstance(o, BaseModel):
            return o.model_dump(by_alias=True)
        # fallback to standard JSONEncoder
        return json.JSONEncoder().default(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            default=_default,
        )


def load_jsonl_models(path: Path | str, model_cls: type[BaseModel]) -> list[BaseModel]:
    """
    Load a JSONL file into a list of Pydantic models.
    """
    p = Path(path)
    out: list[BaseModel] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            out.append(model_cls.model_validate(data))
    return out


def write_json_models(path: Path | str, models: list[BaseModel]) -> None:
    """
    Pretty‐print a list of BaseModel instances to a JSON file.
    """
    # reuse existing write_json under the hood
    serialized = [m.model_dump(by_alias=True) for m in models]
    write_json(path, serialized)


def write_jsonl(path: Path | str, records: Iterable[Any]) -> None:
    """
    Write an iterable of JSON‐serializable objects (dict or BaseModel) to JSONL.
    """
    p = Path(path)
    with open(p, "w", encoding="utf-8") as f:
        for rec in records:
            if isinstance(rec, BaseModel):
                line = rec.model_dump_json(by_alias=True)
                f.write(line + "\n")
            else:
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")


def write_jsonl_models(path: Path | str, models: list[BaseModel]) -> None:
    """
    Write a list of BaseModel instances to a JSONL file.
    """
    write_jsonl(path, models)


def mark_changed_tokens(
    original: str,
    transformed: str,
    focus_span: tuple[int, int] | None = None,
) -> str:
    """
    Mark only inserted or replaced token spans in `transformed` relative to `original`.
    If the diff shows only deletions (e.g., pure removal), return `transformed` unchanged.

    Tokenization uses a simple regex: words and standalone punctuation.

    Args:
        original: original sentence (pre-transformation)
        transformed: transformed sentence
        focus_span: optional (start, end) character-span in `original` to anchor
            the diff (e.g., the modal span). When provided, only consider diffs
            that overlap tokens within this span.

    Returns:
        A string where a single contiguous changed span is wrapped in *...*,
        or the unmarked `transformed` when only deletions occurred.

    Examples:
        >>> mark_changed_tokens("I will win.", "I am going to win.", (2, 6))
        'I *am going to* win.'
        >>> mark_changed_tokens("It may rain.", "It rain.", (3, 6))
        'It rain.'
    """
    # 1) tokenize into words and punctuation, with spans preserved
    tok_re = re.compile(r"\w+|[^\w\s]")
    orig_matches = list(tok_re.finditer(original))
    rem_matches = list(tok_re.finditer(transformed))
    orig_tokens = [m.group(0) for m in orig_matches]
    rem_tokens = [m.group(0) for m in rem_matches]

    # 2) if focus_span is given, find token-index range that overlaps it
    modal_i1 = 0
    modal_i2 = len(orig_tokens)
    if focus_span is not None:
        start, end = focus_span
        idxs = [
            i for i, m in enumerate(orig_matches) if m.start() < end and m.end() > start
        ]
        if idxs:
            modal_i1 = idxs[0]
            modal_i2 = idxs[-1] + 1

    # 3) diff tokens and pick the first inserted/replaced range overlapping the focus
    sm = difflib.SequenceMatcher(a=orig_tokens, b=rem_tokens)
    changed_j: tuple[int, int] | None = None
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        # skip non-overlapping ranges if focus_span was provided
        if focus_span is not None and (i2 <= modal_i1 or i1 >= modal_i2):
            continue
        if tag in ("replace", "insert") and j2 > j1:
            changed_j = (j1, j2)
            break
        # deletions should not produce marking; skip

    if changed_j is None:
        # only deletions or no detectable change → no marking
        out = transformed
    else:
        j1, j2 = changed_j
        # map token indices back to character spans in transformed
        rem_start = rem_matches[j1].start()
        rem_end = rem_matches[j2 - 1].end()
        # wrap that span in asterisks
        out = (
            transformed[:rem_start]
            + "*"
            + transformed[rem_start:rem_end]
            + "*"
            + transformed[rem_end:]
        )

    # 4) normalize spacing around punctuation
    out = re.sub(r"\s+([?.!,;:])", r"\1", out)
    # squish multiple spaces
    out = re.sub(r"\s+", " ", out).strip()
    return out
