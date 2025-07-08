import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional, Type, TypeAlias, TypedDict

from pydantic import BaseModel

from modality_llm.utils import sanitize_model_name

# Define type aliases for better readability
CacheKey: TypeAlias = str
CachePath: TypeAlias = Path
ModelName: TypeAlias = str
TaskType: TypeAlias = str
Params: TypeAlias = dict[str, Any]


# Define structured types for cache data
class CacheMetadata(TypedDict, total=False):
    model_name: str
    task_type: str
    params: Params
    timestamp: float
    date: str
    # Optional fields used for modal‐classification metadata
    categories: list[str]
    num_examples: int
    taxonomy: str


class CacheData(TypedDict):
    metadata: CacheMetadata
    results: object


# a small wrapper for on-disk format
class CacheEntry(BaseModel):
    metadata: CacheMetadata
    results: list[Any]


class LLMCache:
    """Cache for LLM computations."""

    def __init__(self, cache_dir: str = "cache") -> None:
        """Initialize the cache.

        Args:
            cache_dir: Directory to store cache files
        """
        # use pathlib for all path operations
        self.cache_dir: Path = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_cache_key(
        self, model_name: ModelName, task_type: TaskType, params: Params
    ) -> CacheKey:
        """Generate a unique cache key based on model name, task type, and parameters.

        Args:
            model_name: Name of the LLM model
            task_type: Type of task
            params: Parameters used for the task

        Returns:
            A unique hash key for the cache
        """
        param_str = json.dumps(params, sort_keys=True)
        key = hashlib.md5(f"{model_name}_{task_type}_{param_str}".encode()).hexdigest()
        return key

    def _get_cache_path(self, model_name: ModelName, cache_key: CacheKey) -> CachePath:
        """Get the file path for a cache key, including sanitized model name."""
        sanitized_name = sanitize_model_name(model_name)
        filename = f"{sanitized_name}_{cache_key}.json"
        # return a Path, not a str
        return self.cache_dir / filename

    def get(
        self,
        model_name: ModelName,
        task_type: TaskType,
        params: Params,
        model_cls: Type[BaseModel] | None = None,
    ) -> Optional[list[Any]]:
        """
        Get cached results if they exist.
        If model_cls is None, return raw entry.results.
        Otherwise rehydrate each element into model_cls.
        """
        cache_key = self._generate_cache_key(model_name, task_type, params)
        cache_path = Path(self._get_cache_path(model_name, cache_key))

        if not cache_path.exists():
            return None
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            entry = CacheEntry.model_validate(raw)
        except Exception as e:
            print(f"Error reading cache file {cache_path}: {e}")
            return None

        print(f"Cache hit: Using cached results from {cache_path}")
        # If no model_cls requested AND this is modal_classification,
        # patch missing metadata.categories/taxonomy/num_examples.
        if model_cls is None and task_type == "modal_classification":
            md = entry.metadata
            if not md.get("categories"):
                md["categories"] = params.get("categories", [])
                md["taxonomy"] = params.get("taxonomy", "")
                md["num_examples"] = len(entry.results)
                try:
                    cache_path.write_text(entry.model_dump_json(indent=2), encoding="utf-8")
                    print("Updated cache metadata")
                except Exception as e:
                    print(f"Error updating cache metadata: {e}")
        # If caller didn't ask for a model, return raw JSON results
        if model_cls is None:
            return entry.results

        out: list[BaseModel] = []
        for d in entry.results:
            # convert any strings → enums
            if "grammatical" in d and isinstance(d["grammatical"], str):
                from modality_llm.schema import GrammarLabel

                d["grammatical"] = GrammarLabel(d["grammatical"])
            # convert any strings → enums for classification,
            # but only if it’s actually a dict (skip None)
            if isinstance(d.get("classification"), dict):
                from modality_llm.schema import Taxonomy

                d["classification"] = {
                    Taxonomy(k): v for k, v in d["classification"].items()
                }
            # strip out computed fields
            for task in (d.get("classification") or {}).values():
                if isinstance(task, dict):
                    for lang in ("english", "japanese"):
                        lr = task.get(lang)
                        if isinstance(lr, dict):
                            lr.pop("distribution", None)
                            lr.pop("percentages", None)
            if isinstance(d.get("grammar"), dict):
                for lang in ("english", "japanese"):
                    lr = d["grammar"].get(lang)
                    if isinstance(lr, dict):
                        lr.pop("distribution", None)
                        lr.pop("percentages", None)
            out.append(model_cls.model_validate(d))
        return out

    def save(
        self,
        model_name: ModelName,
        task_type: TaskType,
        params: Params,
        results: list[BaseModel],
    ) -> str:
        """Save results to cache."""
        cache_key = self._generate_cache_key(model_name, task_type, params)
        cache_path = self._get_cache_path(model_name, cache_key)

        metadata: CacheMetadata = {
            "model_name": model_name,
            "task_type": task_type,
            "params": params,
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # serialize each result (BaseModel → dict, else pass through)
        serialized: list[dict[str, Any]] = []
        for r in results:
            if isinstance(r, BaseModel):
                # do not persist computed fields like `distribution` / `percentages`
                serialized.append(
                    r.model_dump(
                        by_alias=True,
                        exclude_none=True,
                        exclude_computed=True,
                    )
                )
            else:
                serialized.append(r)  # assume already JSON-safe

        # build the CacheEntry and write it in one shot
        entry = CacheEntry(metadata=metadata, results=serialized)
        try:
            cache_path.write_text(entry.model_dump_json(indent=2), encoding="utf-8")
            print(f"Saved results to cache: {cache_path}")
            return str(cache_path)
        except IOError as e:
            print(f"Error saving to cache file {cache_path}: {e}")
            return ""
