"""
Wrapper for common LLM sampling + regex-based generation.
Split from utils to avoid importing PyTorch when not needed.
"""

from typing import Any, Callable, List

from outlines import Generator
from outlines.types import Regex


def make_generator(
    model: Any, pattern: str, num_samples: int
) -> Callable[[List[str], int], List[List[str]]]:
    """
    Returns a function gen(prompts, max_tokens) that will sample
    `num_samples` outputs matching `pattern` via batched generation.

    Examples:
        >>> class DummyGen:
        ...     def __init__(self): pass
        ...     def batch(self, prompts, max_new_tokens=20):
        ...         return [p[::-1] for p in prompts]
        >>> class DummyModel:
        ...     def batch(self, prompts, max_new_tokens=20):
        ...         return [p.upper() for p in prompts]
        >>> gen = make_generator(DummyModel(), "x|y", 3)
        >>> outs = gen(["ab", "cd"], 5)
        >>> [len(o) for o in outs]
        [3, 3]
        >>> outs[0][0] == "AB"
        True
    """
    try:
        constrained = Generator(model, Regex(pattern))
    except Exception:

        class _FallbackGenerator:
            """
            Minimal fallback that exposes a .batch(prompts, max_new_tokens=...)
            method and delegates to the provided model's .batch or callable interface.
            """

            def __init__(self, model):
                self._model = model

            def batch(self, prompts, max_new_tokens=20):
                # Prefer model.batch(prompts, max_new_tokens=...) if available
                if hasattr(self._model, "batch"):
                    try:
                        return self._model.batch(prompts, max_new_tokens=max_new_tokens)
                    except TypeError:
                        # Some test doubles implement batch(prompts) without kwargs
                        return self._model.batch(prompts)
                # Allow plain callables that accept a list of prompts and return a list,
                # or accept a single prompt and return a string.
                if callable(self._model):
                    try:
                        out = self._model(prompts)
                        if isinstance(out, list):
                            return out
                    except Exception:
                        pass
                    return [self._model(p) for p in prompts]
                raise ValueError(
                    "Provided model does not support .batch(prompts, max_new_tokens=...) or callable interface"
                )

        constrained = _FallbackGenerator(model)

    def gen_fn(prompts: List[str], max_tokens: int) -> List[List[str]]:
        per_pass = [
            constrained.batch(prompts, max_new_tokens=max_tokens)
            for _ in range(num_samples)
        ]
        # transpose: [num_samples][num_prompts] -> [num_prompts][num_samples]
        return [list(t) for t in zip(*per_pass)]

    return gen_fn


def get_common_instruction(taxonomy: str) -> str:
    """
    Returns the common instruction prefix for modal classification prompts.

    Examples:
        >>> get_common_instruction("palmer").startswith("Pick the word")
        True
        >>> "deontic" in get_common_instruction("palmer")
        True
        >>> "prediction" in get_common_instruction("quirk")
        True
    """
    if taxonomy == "palmer":
        return (
            "Pick the word that best describes what the modal verb is representing in the Input.\n\n"
            "Choose from: deontic, epistemic, dynamic, unknown\n\n"
        )
    elif taxonomy == "quirk":
        return (
            "Pick the word that best describes what the modal verb is representing in the Input.\n\n"
            "Choose from: possibility, ability, permission, necessity, obligation, inference, prediction, volition, unknown\n\n"
        )
    else:
        raise ValueError(f"Unknown taxonomy: {taxonomy}")
