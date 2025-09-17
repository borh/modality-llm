"""
Modal verb classification functions and analyses.
"""

import itertools
import random
from typing import List, Optional

import torch
from tqdm import tqdm

from modality_llm.augmentation import (
    generate_contradiction_variants,
    generate_entailment_tests,
    generate_substitution_variants,
)
from modality_llm.llm_cache import LLMCache
from modality_llm.prompts import modal_prompts
from modality_llm.schema import (
    PALMER_CATEGORIES,
    QUIRK_CATEGORIES,
    Example,
    LanguageResult,
    ModalExample,
    PalmerCategory,
    QuirkCategory,
    TaskResult,
    Taxonomy,
    UnifiedResult,
)
from modality_llm.utils import (
    chunked,
    load_jsonl,
    sanitize_model_name,
    write_jsonl_models,
)


def get_categories(taxonomy: Taxonomy) -> List[PalmerCategory] | List[QuirkCategory]:
    """
    Return the list of category labels for a given taxonomy.

    Examples:
        >>> get_categories("palmer") == ['deontic', 'epistemic', 'dynamic', 'unknown']
        True
        >>> get_categories("quirk")[0]
        'possibility'
        >>> get_categories("invalid")  # not a supported taxonomy
        Traceback (most recent call last):
        ...
        ValueError: Unknown taxonomy: invalid
    """
    # Accept either a Taxonomy enum or a plain string; normalize to string
    tax = taxonomy.value if isinstance(taxonomy, Taxonomy) else taxonomy
    if tax == "palmer":
        return PALMER_CATEGORIES
    elif tax == "quirk":
        return QUIRK_CATEGORIES
    else:
        raise ValueError(f"Unknown taxonomy: {taxonomy}")


def compute_modal_results(
    data_path: str,
    taxonomy: Taxonomy = Taxonomy["palmer"],
    num_samples: int = 5,
    batch_size: int = 50,
    use_cache: bool = True,
    force_refresh: bool = False,
    num_examples_to_sample: int = 0,
    random_seed: int = 42,
    model=None,
    model_name="model",
    cache: Optional[LLMCache] = None,
    augment_substitution: bool = False,
    augment_entailment: bool = False,
    augment_contradiction: bool = False,
    report_augmentations: bool = False,
    zero_shot: bool = False,
    require_consensus: str | None = None,
) -> List[UnifiedResult]:
    """Compute modal verb classification results (pure, no printing or plotting).

    Args:
        data_path: Path to the JSONL file
        taxonomy: Either "palmer" or "quirk"
        num_samples: Number of samples to generate per prompt
        batch_size: Number of examples to process in each batch
        use_cache: Whether to use cache
        force_refresh: Whether to force refresh the cache
        num_examples_to_sample: Number of examples to sample (0 for all)
        random_seed: Seed for sampling
        model: The initialized model to use
        model_name: Name of the model (for cache and plots)
        cache: Cache instance to use

    Returns:
        List of modal verb classification results
    """

    if cache is None:
        cache = LLMCache()

    categories = get_categories(taxonomy)
    cache_params = {
        "data_path": data_path,
        "taxonomy": taxonomy,
        "num_samples": num_samples,
        "model_config": {
            "name": model_name,
        },
        "categories": categories,
    }

    # 1) load & filter raw JSON
    from modality_llm.utils import unanimous_examples

    raw_dicts = load_jsonl(data_path)
    raw_models = [ModalExample.model_validate(d) for d in raw_dicts]
    raw_models = unanimous_examples(raw_models, require_consensus)
    all_examples = [Example.from_modal_verb_example(rec) for rec in raw_models]

    if augment_substitution:
        # now generate Example→SubstitutionTestEntry→Example
        subs: list[Example] = []
        for ex in all_examples:
            for v in generate_substitution_variants(ex):
                subs.append(v)
        all_examples.extend(subs)

    if augment_entailment:
        pairs = list(
            itertools.chain.from_iterable(
                generate_entailment_tests(ex) for ex in all_examples
            )
        )
        out_fn = f"{sanitize_model_name(model_name)}_{taxonomy}_entailment_tests.jsonl"
        write_jsonl_models(out_fn, pairs)
        all_examples.extend(pairs)
    # ---------- new: contradiction augmentation ----------
    if augment_contradiction:
        contrs: list[Example] = []
        for ex in all_examples:
            contrs.extend(generate_contradiction_variants(ex))
        all_examples.extend(contrs)

    num_available = len(all_examples)
    examples_to_process: List[Example] = all_examples
    actual_num_sampled = num_available

    if 0 < num_examples_to_sample < num_available:
        random.seed(random_seed)
        examples_to_process = random.sample(all_examples, num_examples_to_sample)
        actual_num_sampled = len(examples_to_process)
        cache_params["sampling_info"] = {
            "num_sampled": actual_num_sampled,
            "random_seed": random_seed,
        }

    if use_cache and not force_refresh:
        cached = cache.get(
            model_name,
            "modal_classification",
            cache_params,
            model_cls=UnifiedResult,
        )
        if cached and len(cached) == actual_num_sampled:
            return cached

    pattern = "|".join(categories)
    # use our utility so tests can stub make_generator
    from modality_llm.llm_utils import make_generator

    generator = make_generator(model, pattern, num_samples)
    
    # Log device info before starting generation
    print(f"\nStarting generation for {len(examples_to_process)} examples...")
    from modality_llm.model_manager import log_model_device_info
    log_model_device_info(model)
    all_results = []

    n_batches = (len(examples_to_process) + batch_size - 1) // batch_size
    for batch_examples in tqdm(
        chunked(examples_to_process, batch_size),
        desc="Processing batches",
        total=n_batches,
    ):
        # Convert taxonomy string to enum instance if needed
        taxonomy_enum = (
            taxonomy if isinstance(taxonomy, Taxonomy) else Taxonomy(taxonomy)
        )
        prompts, _ = modal_prompts(batch_examples, taxonomy_enum, zero_shot=zero_shot)
        with torch.inference_mode():
            answers = generator(prompts, max_tokens=20)
        for ex, prompt, ans in zip(batch_examples, prompts, answers):
            # build nested TaskResult for classification outputs
            class_task = TaskResult(english=LanguageResult(prompt=prompt, answers=ans))
            result = UnifiedResult(
                eid=ex.eid,
                english=ex.english,
                japanese=ex.japanese,
                english_target=ex.english_target,
                japanese_target=ex.japanese_target,
                grammatical=ex.grammatical,
                human_annotations=ex.human_annotations,
                expected_categories=ex.expected_categories,
                grammar=None,
                classification={taxonomy_enum: class_task},
            )
            all_results.append(result)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cache.save(
        model_name,
        "modal_classification",
        cache_params,
        [r.model_dump() for r in all_results],
    )
    return all_results
