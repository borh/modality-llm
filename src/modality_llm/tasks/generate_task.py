from typing import Any

import modality_llm.examples as examples_mod


# module‐level wrappers so tests can stub either here or in examples_mod
def download_modal_verb_dataset(path: str) -> str:
    return examples_mod.download_modal_verb_dataset(path)


def generate_grammar_examples_for_annotation(
    modal_data_path: str, output_csv_path: str, include_alternatives: bool
) -> list[Any]:
    return examples_mod.generate_grammar_examples_for_annotation(
        modal_data_path=modal_data_path,
        output_csv_path=output_csv_path,
        include_alternatives=include_alternatives,
    )


def run(
    args,
    model=None,
    model_name=None,
    *,
    download_fn=download_modal_verb_dataset,
    generate_fn=generate_grammar_examples_for_annotation,
) -> None:
    print("Starting task: Generate Grammar CSV for Annotation")
    modal_data_path = download_fn(args.data_path)
    if modal_data_path:
        generate_fn(
            modal_data_path=modal_data_path,
            output_csv_path=args.output_csv,
            include_alternatives=args.gen_include_alternatives,
            require_consensus=getattr(args, "require_consensus", None),
        )
    else:
        print("Warning: Could not load modal dataset. Cannot generate CSV.")
