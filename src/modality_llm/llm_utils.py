"""
Wrapper for common LLM sampling + regex-based generation.
Split from utils to avoid importing PyTorch when not needed.
"""

from typing import Callable, List

from outlines.generate import regex
from outlines.samplers import multinomial
from transformers import AutoTokenizer, PreTrainedModel


def make_generator(
    model: PreTrainedModel, pattern: str, num_samples: int
) -> Callable[[List[str], int], List[List[str]]]:
    """
    Returns a function gen(prompts, max_tokens) that will sample
    `num_samples` outputs matching `pattern` via the given `model`.
    """
    sampler = multinomial(num_samples)
    return regex(model, pattern, sampler)


def get_common_instruction(taxonomy: str) -> str:
    """
    Returns the common instruction prefix for modal classification prompts.
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


def get_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Returns the tokenizer for the given model name.
    """
    return AutoTokenizer.from_pretrained(model_name)
