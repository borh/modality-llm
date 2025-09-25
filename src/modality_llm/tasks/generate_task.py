from typing import Any

import modality_llm.examples as examples_mod


# module‐level wrappers so tests can stub either here or in examples_mod
def download_modal_verb_dataset(path: str) -> str:
    return examples_mod.download_modal_verb_dataset(path)


def generate_grammar_examples_for_annotation(
    modal_data_path: str,
    output_csv_path: str,
    include_alternatives: bool,
    require_consensus: str | None = None,
    output_format: str = "csv",
    existing_csv_path: str | None = None,
    freeze_completed: bool = True,
    mark_diff: bool = True,
    removal_backend: str = "spacy",
    removal_model: str = "openai/gpt-oss-20b",
    removal_concurrency: int = 8,
    judge_grammaticality: bool = False,
    judge_model: str = "openai/gpt-oss-20b",
    judge_concurrency: int = 8,
    judge_overwrite: bool = False,
    judge_only_augmented: bool = True,
) -> list[Any]:
    return examples_mod.generate_grammar_examples_for_annotation(
        modal_data_path=modal_data_path,
        output_csv_path=output_csv_path,
        include_alternatives=include_alternatives,
        require_consensus=require_consensus,
        output_format=output_format,
        existing_csv_path=existing_csv_path,
        freeze_completed=freeze_completed,
        mark_diff=mark_diff,
        removal_backend=removal_backend,
        removal_model=removal_model,
        removal_concurrency=removal_concurrency,
        judge_grammaticality=judge_grammaticality,
        judge_model=judge_model,
        judge_concurrency=judge_concurrency,
        judge_overwrite=judge_overwrite,
        judge_only_augmented=judge_only_augmented,
    )


def run(
    args,
    model=None,
    model_name=None,
    *,
    download_fn=download_modal_verb_dataset,
    generate_fn=generate_grammar_examples_for_annotation,
    convert_fn=None,
) -> None:
    """
    Backwards-compatible wrapper for the generate subcommand.

    Accepts:
      - CLI-style Namespace with .submode in ("csv","jsonl") and attributes
        .data_path / .output (as produced by parse_args), or
      - legacy/test callers that provide .output_csv (or pass a class with
        class attributes).

    Normalizes missing attributes so the rest of the function can assume
    .submode and .output exist.
    """
    # Normalize submode/output for legacy callers and class-objects
    submode = getattr(args, "submode", None)

    # If caller provided output_csv (tests), map it to .output and default to csv
    if not submode:
        out_csv = getattr(args, "output_csv", None)
        out = getattr(args, "output", None)
        if out_csv is not None:
            # support both class objects and Namespace instances
            try:
                setattr(args, "output", out_csv)
            except Exception:
                # fallback: if args is immutable for some reason, ignore — we will use out_csv directly below
                pass
            submode = "csv"
        elif out is not None:
            submode = "csv"
        else:
            # Default to CSV for maximal backward compatibility with tests
            submode = "csv"

    if submode == "csv":
        print("Starting task: Generate Grammar CSV for Annotation")
        modal_data_path = download_fn(getattr(args, "data_path"))
        if modal_data_path:
            generate_fn(
                modal_data_path=modal_data_path,
                output_csv_path=getattr(
                    args, "output", getattr(args, "output_csv", None)
                ),
                include_alternatives=getattr(args, "gen_include_alternatives", False),
                require_consensus=getattr(args, "require_consensus", None),
                output_format=getattr(args, "format", "csv"),
                existing_csv_path=getattr(args, "existing_csv", None),
                freeze_completed=getattr(args, "freeze_completed", True),
                mark_diff=getattr(args, "mark_diff", True),
                removal_backend=getattr(args, "removal_backend", "spacy"),
                removal_model=getattr(args, "removal_model", "openai/gpt-oss-20b"),
                removal_concurrency=getattr(args, "removal_concurrency", 8),
                judge_grammaticality=getattr(args, "judge_grammaticality", False),
                judge_model=getattr(args, "judge_model", "openai/gpt-oss-20b"),
                judge_concurrency=getattr(args, "judge_concurrency", 8),
                judge_overwrite=getattr(args, "judge_overwrite", False),
                judge_only_augmented=getattr(args, "judge_only_augmented", True),
            )
        else:
            print("Warning: Could not load modal dataset. Cannot generate CSV.")

    elif submode == "jsonl":
        print("Starting task: Convert Annotated CSV to JSONL")
        if convert_fn is None:
            convert_fn = examples_mod.convert_annotated_csv_to_jsonl
        convert_fn(
            csv_path=getattr(args, "data_path"),
            output_jsonl_path=getattr(args, "output"),
            completed_only=getattr(args, "completed_only", False),
            jsonl_format=getattr(args, "jsonl_format", "minimal"),
        )
    elif submode == "judge":
        print("Starting task: Judge Grammaticality in CSV")
        examples_mod.judge_grammaticality_in_csv(
            csv_path=getattr(args, "data_path"),
            output_csv_path=getattr(args, "output"),
            model_name=getattr(args, "judge_model", "openai/gpt-oss-20b"),
            concurrency=getattr(args, "judge_concurrency", 8),
            overwrite=getattr(args, "judge_overwrite", False),
            only_augmented=getattr(args, "judge_only_augmented", True),
        )
    elif submode == "repair":
        print("Starting task: Repair Grammaticality in CSV")
        examples_mod.repair_grammaticality_in_csv(
            csv_path=getattr(args, "data_path"),
            output_csv_path=getattr(args, "output"),
            model_name=getattr(args, "repair_model", "openai/gpt-oss-20b"),
            concurrency=getattr(args, "repair_concurrency", 8),
            only_augmented=getattr(args, "repair_only_augmented", True),
            only_bad=getattr(args, "repair_only_bad", True),
            rejudge=not getattr(args, "no_rejudge", False),
        )
    else:
        raise ValueError(f"Unknown submode: {submode}")
