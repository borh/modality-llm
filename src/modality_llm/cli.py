"""
Command‐line interface for grammar and modal tasks.
"""

import argparse
import random
from argparse import Namespace
from typing import Any, Callable

import numpy as np

import modality_llm.model_manager as mm
from modality_llm.settings import DEFAULT_MODEL_NAME, DEFAULT_QUANTIZATION_MODE
from modality_llm.tasks.generate_task import run as generate_run
from modality_llm.tasks.grammar_task import run as grammar_run
from modality_llm.tasks.modal_task import run as modal_run

# A Task is any callable taking (args, model, model_name) -> None
Task = Callable[[Namespace, Any, str], None]


def run_all(args: argparse.Namespace, model: Any, model_name: str) -> None:
    """
    Run the full pipeline:
      1) modal classification (with augmentations + analysis)
      2) grammar checks & Bayesian analyses
    """
    modal_run(args, model, model_name)
    grammar_run(args, model, model_name)


TASK_REGISTRY: dict[str, Task] = {
    "generate": generate_run,
    "modal": modal_run,
    "grammar": grammar_run,
    "run": run_all,
}


def add_common_compute_args(parser: argparse.ArgumentParser) -> None:
    """Inject arguments shared by 'compute' and 'run'."""
    parser.add_argument(
        "--data-path",
        type=str,
        default="modal_verbs.jsonl",
        help="Path to the modal-verbs JSONL file (default: modal_verbs.jsonl)",
    )
    parser.add_argument(
        "--taxonomy",
        choices=["palmer", "quirk", "both"],
        default="both",
        help="Taxonomy to use (palmer, quirk, or both)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for LLM calls",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples per prompt",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        dest="use_cache",
        default=True,
        help="Enable caching of LLM results",
    )
    parser.add_argument(
        "--no-cache",
        action="store_false",
        dest="use_cache",
        help="Disable caching",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        default=False,
        help="Force refresh cache even if a hit exists",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=0,
        help="Number of examples to sample (0 = use all)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--require-consensus",
        choices=["palmer", "quirk", "both"],
        default=None,
        help="Only use examples where all three annotators fully agreed",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="modality-LLM task runner")
    sub = parser.add_subparsers(dest="mode", required=True, help="Which task to run")

    # generate task
    gen = sub.add_parser("generate", help="Download modal dataset & make grammar CSV")
    gen.add_argument("data_path", help="Path or target for modal_verbs.jsonl")
    gen.add_argument("output_csv", help="Where to write the CSV")
    gen.add_argument(
        "--gen-include-alternatives",
        action="store_true",
        default=False,
        help="Also emit alternative substitution rows",
    )
    gen.add_argument(
        "--require-consensus",
        choices=["palmer", "quirk", "both"],
        default=None,
        help="If set, only include unanimously‐labeled examples",
    )

    # modal‐classification task
    modal = sub.add_parser("modal", help="Compute modal‐verb classification results")
    add_common_compute_args(modal)
    modal.add_argument(
        "--augmentations-file",
        type=str,
        default=None,
        help="Path to JSONL of precomputed augmentation Examples; if given, skip regenerating augmentations",
    )
    modal.add_argument(
        "--zero-shot",
        action="store_true",
        dest="zero_shot",
        default=False,
        help="Use zero‐shot prompts (old get_common_instruction) instead of PAPER_INSTRUCTIONS",
    )
    modal.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="LLM model name",
    )
    modal.add_argument(
        "--quantization",
        choices=["bf16", "int8", "int4"],
        default=DEFAULT_QUANTIZATION_MODE,
        help="Quantization mode",
    )
    modal.add_argument(
        "--no-augment-acceptability",
        dest="augment_acceptability",
        action="store_false",
        default=True,
        help="Disable acceptability augmentation",
    )
    modal.add_argument(
        "--no-augment-substitution",
        dest="augment_substitution",
        action="store_false",
        default=True,
        help="Disable substitution augmentation",
    )
    modal.add_argument(
        "--no-augment-entailment",
        dest="augment_entailment",
        action="store_false",
        default=True,
        help="Disable entailment augmentation",
    )
    modal.add_argument(
        "--no-augment-contradiction",
        dest="augment_contradiction",
        action="store_false",
        default=True,
        help="Disable contradiction augmentation",
    )
    modal.add_argument(
        "--no-augmentation-analysis",
        dest="run_augmentation_analysis",
        action="store_false",
        default=True,
        help="Disable automatic augmentation analysis",
    )

    # grammar‐checking task
    grammar = sub.add_parser("grammar", help="Run grammar checks & Bayes analyses")
    grammar.add_argument(
        "--grammar-source",
        choices=["modal", "file"],
        default="modal",
        help="Where to get examples: modal JSONL or a CSV",
    )
    grammar.add_argument(
        "--data-path",
        type=str,
        default="modal_verbs.jsonl",
        help="JSONL file for 'modal' source",
    )
    grammar.add_argument(
        "--examples-file",
        type=str,
        default="",
        help="CSV file for 'file' source",
    )
    grammar.add_argument(
        "--num-examples",
        type=int,
        default=0,
        help="Sample this many examples (0=all)",
    )
    grammar.add_argument(
        "--grammar-method",
        choices=["sampling", "yesno_prob", "intemplate_lp", "all"],
        default="all",
        help="Which grammar methods to run",
    )
    grammar.add_argument(
        "--grammar-language",
        choices=["english", "japanese", "both"],
        default="english",
        help="Language(s) to check",
    )
    grammar.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="LLM samples per prompt",
    )
    grammar.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for LLM calls",
    )
    grammar.add_argument(
        "--use-advi",
        action="store_true",
        dest="use_advi",
        default=True,
        help="Use ADVI in Bayes steps",
    )
    grammar.add_argument(
        "--no-advi",
        action="store_false",
        dest="use_advi",
        help="Use MCMC instead of ADVI",
    )
    grammar.add_argument(
        "--mcmc-samples",
        type=int,
        default=1,
        help="MCMC sample count",
    )
    grammar.add_argument(
        "--mcmc-chains",
        type=int,
        default=1,
        help="MCMC chain count",
    )
    grammar.add_argument(
        "--mcmc-cores",
        type=int,
        default=None,
        help="MCMC core count",
    )
    grammar.add_argument(
        "--require-consensus",
        choices=["palmer", "quirk", "both"],
        default=None,
        help="Only use examples where all three annotators agreed",
    )

    run_parser = sub.add_parser(
        "run",
        help="Run full pipeline: modal classification (+ augmentation analysis) and grammar checks",
    )
    add_common_compute_args(run_parser)
    run_parser.add_argument(
        "--augmentations-file",
        type=str,
        default=None,
        help="Path to JSONL of precomputed augmentation Examples; if given, skip regenerating augmentations",
    )
    run_parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="LLM model name",
    )
    run_parser.add_argument(
        "--quantization",
        choices=["bf16", "int8", "int4"],
        default=DEFAULT_QUANTIZATION_MODE,
        help="Quantization mode",
    )
    run_parser.add_argument(
        "--no-augment-acceptability",
        dest="augment_acceptability",
        action="store_false",
        default=True,
        help="Disable acceptability augmentation",
    )
    run_parser.add_argument(
        "--no-augment-substitution",
        dest="augment_substitution",
        action="store_false",
        default=True,
        help="Disable substitution augmentation",
    )
    run_parser.add_argument(
        "--no-augment-entailment",
        dest="augment_entailment",
        action="store_false",
        default=True,
        help="Disable entailment augmentation",
    )
    run_parser.add_argument(
        "--no-augment-contradiction",
        dest="augment_contradiction",
        action="store_false",
        default=True,
        help="Disable contradiction augmentation",
    )
    run_parser.add_argument(
        "--no-augmentation-analysis",
        dest="run_augmentation_analysis",
        action="store_false",
        default=True,
        help="Disable automatic augmentation analysis",
    )
    # reuse grammar flags
    run_parser.add_argument(
        "--grammar-source",
        choices=["modal", "file"],
        default="modal",
        help="Where to get examples: modal JSONL or a CSV",
    )
    run_parser.add_argument(
        "--examples-file",
        type=str,
        default="",
        help="CSV file for 'file' source",
    )
    run_parser.add_argument(
        "--grammar-method",
        choices=["sampling", "yesno_prob", "intemplate_lp", "all"],
        default="all",
        help="Which grammar methods to run",
    )
    run_parser.add_argument(
        "--grammar-language",
        choices=["english", "japanese", "both"],
        default="english",
        help="Language(s) to check",
    )
    run_parser.add_argument(
        "--use-advi",
        action="store_true",
        dest="use_advi",
        default=True,
        help="Use ADVI in Bayes steps",
    )
    run_parser.add_argument(
        "--no-advi",
        action="store_false",
        dest="use_advi",
        help="Use MCMC instead of ADVI",
    )
    run_parser.add_argument(
        "--mcmc-samples",
        type=int,
        default=1,
        help="MCMC sample count",
    )
    run_parser.add_argument(
        "--mcmc-chains",
        type=int,
        default=1,
        help="MCMC chain count",
    )
    run_parser.add_argument(
        "--mcmc-cores",
        type=int,
        default=None,
        help="MCMC core count",
    )

    return parser.parse_args()


def main_cli() -> None:
    args = parse_args()

    # dispatch to the selected task runner
    task_fn = TASK_REGISTRY.get(args.mode)
    if task_fn is None:
        raise ValueError(f"Unknown task mode: {args.mode}")

    # for non‐generate tasks, init model & seed
    model = None
    model_name = ""
    if args.mode != "generate":
        mm.quantization_mode = getattr(args, "quantization", DEFAULT_QUANTIZATION_MODE)
        mm.use_flash_attn = getattr(args, "use_flash_attn", True)
        random.seed(getattr(args, "random_seed", 42))
        np.random.seed(getattr(args, "random_seed", 42))
        model = mm.initialize_model(getattr(args, "model", DEFAULT_MODEL_NAME))
        model_name = getattr(args, "model", DEFAULT_MODEL_NAME)

    # run the task
    task_fn(args, model, model_name)


if __name__ == "__main__":
    main_cli()
