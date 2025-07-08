import csv
import json
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
