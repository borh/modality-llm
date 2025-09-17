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
) -> list[Any]:
    return examples_mod.generate_grammar_examples_for_annotation(
        modal_data_path=modal_data_path,
        output_csv_path=output_csv_path,
        include_alternatives=include_alternatives,
        require_consensus=require_consensus,
        output_format=output_format,
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
        )
    else:
        raise ValueError(f"Unknown submode: {submode}")
