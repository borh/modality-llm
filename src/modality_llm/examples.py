"""
Functions for loading, generating, and managing examples.
"""

import hashlib
import re
from pathlib import Path
from typing import List

import polars as pl
import requests

from modality_llm.schema import Example, GrammarLabel
from modality_llm.utils import load_jsonl_models


def download_modal_verb_dataset(target_path="modal_verbs.jsonl") -> str:
    path = Path(target_path)
    if path.exists():
        print(f"Dataset already exists at {path}")
        return str(path)

    url = "https://raw.githubusercontent.com/minnesotanlp/moverb/main/data/moVerb_all.jsonl"
    print(f"Downloading dataset from {url}...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        print(f"Dataset successfully downloaded to {path}")
        return str(path)
    except Exception as e:
        print(f"Error: {e}")
        # Ensure parent directories exist before creating file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return str(path)


def generate_grammar_examples_for_annotation(
    modal_data_path: str,
    output_csv_path: str,
    include_alternatives: bool = True,
    require_consensus: str | None = None,
) -> list[Example]:
    modal_path = Path(modal_data_path)
    output_path = Path(output_csv_path)

    if not modal_path.exists():
        print(f"Error: File not found at {modal_path}")
        return []

    output_path.parent.mkdir(parents=True, exist_ok=True)

    from modality_llm.schema import ModalExample
    from modality_llm.utils import unanimous_examples

    modal_data = load_jsonl_models(str(modal_path), ModalExample)
    # filter if requested
    modal_data = unanimous_examples(modal_data, require_consensus)
    if not modal_data:
        print("Error: No data loaded from modal dataset.")
        return []

    ENGLISH_MODALS: List[str] = [
        "can",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "will",
        "would",
        "ought to",
    ]

    print(
        f"Generating grammar examples for annotation from {len(modal_data)} modal sentences..."
    )
    print(f"Output will be saved to: {output_path}")
    if include_alternatives:
        print("Including alternative modal verb substitutions.")

    examples: list[Example] = []
    csv_rows = []

    for entry in modal_data:
        original_utt = entry.utt
        original_mv = entry.mv.lower()

        pattern = r"\b" + re.escape(original_mv) + r"\b"
        match = re.search(pattern, original_utt, re.IGNORECASE)

        if not match:
            print(
                f"Warning: Could not find modal '{original_mv}' in utterance: {original_utt}"
            )
            continue

        start, end = match.span()
        found_modal = original_utt[start:end]
        eid = f"modal_{len(examples) + 1:04d}"

        # Create Example for original, using sha256(utt) as eid
        human_annotations = entry.annotations
        expected_categories = entry.res
        eid = hashlib.sha256(original_utt.encode("utf-8")).hexdigest()
        original_example = Example(
            eid=eid,
            english=f"{original_utt[:start]}*{found_modal}*{original_utt[end:]}",
            japanese=None,
            grammatical=GrammarLabel.yes,
            english_target=original_mv,
            japanese_target=None,
            human_annotations=human_annotations,
            expected_categories=expected_categories,
        )
        examples.append(original_example)
        csv_rows.append(
            {
                "ID": f"{eid}_orig",
                "EID": eid,
                "Original_Sentence": original_utt,
                "Marked_Sentence_English": original_example.english,
                "Marked_Sentence_Japanese": "",
                "Expected_Grammaticality": "yes",
                "Source_Modal": original_mv,
                "Tested_Modal": original_mv,
                "Annotation_Notes": "Original sentence from dataset.",
            }
        )

        if include_alternatives:
            for alt_mv in ENGLISH_MODALS:
                if alt_mv.lower() == original_mv.lower():
                    continue

                # keep same base id but suffix it so alt rows remain unique
                alt_eid = f"{eid}_alt_{alt_mv.replace(' ', '_')}"
                human_annotations = entry.annotations
                expected_categories = entry.res
                alt_example = Example(
                    eid=alt_eid,
                    english=f"{original_utt[:start]}*{alt_mv}*{original_utt[end:]}",
                    japanese=None,
                    grammatical=GrammarLabel.yes,
                    english_target=alt_mv,
                    japanese_target=None,
                    human_annotations=human_annotations,
                    expected_categories=expected_categories,
                )
                examples.append(alt_example)
                csv_rows.append(
                    {
                        "ID": alt_eid,
                        "EID": eid,
                        "Original_Sentence": original_utt,
                        "Marked_Sentence_English": alt_example.english,
                        "Marked_Sentence_Japanese": "",
                        "Expected_Grammaticality": "yes",
                        "Source_Modal": original_mv,
                        "Tested_Modal": alt_mv,
                        "Annotation_Notes": "Defaulted to 'yes'. Please verify grammaticality.",
                    }
                )

    try:
        df = pl.DataFrame(csv_rows)
        df.write_csv(str(output_path))
        print(
            f"Successfully generated {len(csv_rows)} examples for annotation in {output_path}"
        )
        return examples
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        return []
