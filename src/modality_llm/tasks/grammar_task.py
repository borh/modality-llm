import hashlib
import random
import re
from itertools import islice

import transformers

from modality_llm.analysis_grammar_bayes import (
    bayesian_analysis as _bayesian_analysis,
)
from modality_llm.analysis_grammar_bayes import (
    bayesian_analysis_hierarchical_grammar as _bayesian_analysis_hierarchical_grammar,
)
from modality_llm.analysis_grammar_bayes import (
    bayesian_analysis_with_pymc as _bayesian_analysis_with_pymc,
)
from modality_llm.grammar import (
    compute_grammar_results as _compute_grammar_results,
)
from modality_llm.grammar import (
    run_grammar_check_intemplate_lp as _run_intemplate_lp,
)
from modality_llm.grammar import (
    run_grammar_check_yesno_prob as _run_yesno_prob,
)
from modality_llm.utils import (
    load_csv,
    load_jsonl,
    sanitize_model_name,
    write_json_models,
    write_jsonl,
)

# how many examples to print per grammar method
MAX_DISPLAY: int = 3


def run(
    args,
    model,
    model_name,
    *,
    compute_fn=_compute_grammar_results,
    yesno_fn=_run_yesno_prob,
    intemplate_fn=_run_intemplate_lp,
    bayes_fn=_bayesian_analysis,
    bayes_pymc_fn=_bayesian_analysis_with_pymc,
    bayes_hier_fn=_bayesian_analysis_hierarchical_grammar,
    load_jsonl_fn=load_jsonl,
    load_csv_fn=load_csv,
    write_json_models_fn=write_json_models,
    write_jsonl_fn=write_jsonl,
    sanitize_fn=sanitize_model_name,
    tokenizer_factory=None,
) -> None:
    # Use grammar_source, grammar_method, etc. directly from args (no getattr needed)
    grammar_source = args.grammar_source
    grammar_method = args.grammar_method

    # Determine the source of examples
    if grammar_source == "modal":
        from modality_llm.schema import Example, GrammarLabel, ModalExample
        from modality_llm.utils import unanimous_examples

        # load‐and‐filter raw JSONL
        raw = load_jsonl_fn(args.data_path)
        models = [ModalExample.model_validate(d) for d in raw]
        models = unanimous_examples(models, getattr(args, "require_consensus", None))
        examples_data_for_processing = []
        for mod in models:
            mv = mod.mv
            utt = mod.utt
            # wrap the first case‐insensitive match of mv in *…*
            pattern = rf"\b{re.escape(mv)}\b"
            marked = re.sub(
                pattern,
                lambda m: f"*{m.group(0)}*",
                utt,
                count=1,
                flags=re.IGNORECASE,
            )
            human_annotations = mod.annotations
            expected_categories = mod.res
            # compute eid = sha256 of the marked sentence
            eid = hashlib.sha256(utt.encode("utf-8")).hexdigest()
            examples_data_for_processing.append(
                Example(
                    eid=eid,
                    english=marked,
                    japanese=None,
                    grammatical=GrammarLabel.yes,
                    english_target=mv,
                    japanese_target=None,
                    human_annotations=human_annotations,
                    expected_categories=expected_categories,
                )
            )
    elif grammar_source == "file":
        from modality_llm.schema import Example, GrammarLabel

        loaded = load_csv_fn(args.examples_file)
        required_cols = [
            "Marked_Sentence_English",
            "Expected_Grammaticality",
            "EID",
            "Tested_Modal",
        ]
        # ensure we have a concrete list[str] for formatting
        fieldnames: list[str] = list(loaded[0].keys()) if loaded else []
        if not all(col in fieldnames for col in required_cols):
            print(
                f"CSV file missing required columns. Need: {required_cols}. Found: {fieldnames}"
            )
            print("Exiting.")
            return

        examples_data_for_processing = []
        for i, row in enumerate(loaded):
            eng_marked = row.get("Marked_Sentence_English", "")
            expected = row.get("Expected_Grammaticality", "").lower()
            if not eng_marked:
                print(
                    f"Warning: Skipping row {i + 2} due to missing 'Marked_Sentence_English'."
                )
                continue
            if expected not in ["yes", "no"]:
                print(
                    f"Warning: Skipping row {i + 2} due to invalid 'Expected_Grammaticality': '{expected}'. Must be 'yes' or 'no'."
                )
                continue

            # Fill in default palmer/quirk lists for file-based examples
            examples_data_for_processing.append(
                Example(
                    eid=row["EID"],
                    english=eng_marked,
                    japanese=row.get("Marked_Sentence_Japanese", None),
                    grammatical=GrammarLabel(expected),
                    english_target=row["Tested_Modal"],
                    japanese_target=None,
                    human_annotations=None,
                    expected_categories=None,
                )
            )

        if not examples_data_for_processing:
            print("Warning: No valid examples loaded from CSV file.")
    else:
        print(f"Error: Unknown grammar source: {grammar_source}")
        return

    if not examples_data_for_processing:
        print("Error: No grammar examples loaded or generated. Exiting.")
        return

    # detect whether we ever got a non‐empty Japanese sentence
    has_japanese = any(ex.japanese for ex in examples_data_for_processing)

    # Apply sampling logic for grammar task

    num_available = len(examples_data_for_processing)
    num_to_sample = args.num_examples

    if 0 < num_to_sample < num_available:
        print(
            f"Sampling {num_to_sample} examples from {num_available} available examples..."
        )
        examples_data_for_processing = random.sample(
            examples_data_for_processing, num_to_sample
        )
        print(f"Using {len(examples_data_for_processing)} sampled examples.")
    elif num_to_sample >= num_available:
        print(
            f"Requested {num_to_sample} examples, but only {num_available} available. Using all."
        )
    else:
        print(f"Using all {num_available} available examples.")

    print(f"Running grammar check on {len(examples_data_for_processing)} examples.")

    # load tokenizer if needed, but never crash on bad model_name
    tokenizer = None
    model_ref = None
    if grammar_method in ["yesno_prob", "intemplate_lp", "all"]:
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Failed to load tokenizer: {e}")
            tokenizer = None
        model_ref = getattr(model, "model", model)

    results_sampling = None
    results_yesno = None
    results_lp = None

    if grammar_method in ["sampling", "all"]:
        print("\n--- Running Grammar Check: Sampling Method ---")
        results_sampling = compute_fn(
            examples_data_for_processing,
            num_samples=args.num_samples,
            grammar_language=args.grammar_language,
            model=model,
            model_name=model_name,
            augment_acceptability=getattr(args, "augment_acceptability", True),
            batch_size=getattr(args, "batch_size", 50),
        )
        # Flat Bayesian analyses
        bayes_fn(results_sampling)
        bayes_pymc_fn(
            results_sampling,
            model_name,
            use_advi=args.use_advi,
            n_samples=args.mcmc_samples,
        )
        # Hierarchical analyses (only for the 'file' source)
        if grammar_source == "file":
            print("\n--- Hierarchical Bayesian Analysis (English) ---")
            bayes_hier_fn(
                results_sampling,
                language="English",
                use_advi=args.use_advi,
                n_samples=args.mcmc_samples,
            )
            print("\n--- Hierarchical Bayesian Analysis (Japanese) ---")
            bayes_hier_fn(
                results_sampling,
                language="Japanese",
                use_advi=args.use_advi,
                n_samples=args.mcmc_samples,
            )

    if grammar_method in ["yesno_prob", "all"]:
        print("\n--- Running Grammar Check: Yes/No Probability Method ---")
        if tokenizer_factory:
            tokenizer = tokenizer_factory(model_name)
        try:
            results_yesno = yesno_fn(
                model_ref,
                tokenizer,
                examples_data_for_processing,
                args.grammar_language,
            )
        except Exception as e:
            print(f"Warning: Yes/No probability check failed: {e}")
            results_yesno = []
        print("\n=== YES/NO PROBABILITY RESULTS ===\n")
        for i, r in enumerate(islice(results_yesno, MAX_DISPLAY)):
            print(f"\nExample {i + 1}:")
            print(f"  English: {r['English']}")
            print(f"  English P(yes): {r['English_P_yes']:.3f}")
            # Only print Japanese fields if they exist
            jap_sent = r.get("Japanese")
            jap_p = r.get("Japanese_P_yes")
            if jap_sent is not None and jap_p is not None:
                print(f"  Japanese: {jap_sent}")
                print(f"  Japanese P(yes): {jap_p:.3f}")
            print(f"  Expected: {r['Expected']}")
            print("-" * 50)
        if len(results_yesno) > MAX_DISPLAY:
            print(
                f"...Displayed first {MAX_DISPLAY} of {len(results_yesno)} examples\n"
            )

    if grammar_method in ["intemplate_lp", "all"]:
        print("\n--- Running Grammar Check: In-Template LP Method ---")
        results_lp = intemplate_fn(
            model_ref,
            tokenizer,
            examples_data_for_processing,
            "The following sentence is grammatically acceptable.\n\n{sentence}",
            args.grammar_language,
        )
        print("\n=== IN-TEMPLATE LP RESULTS ===\n")
        for i, r in enumerate(islice(results_lp, MAX_DISPLAY)):
            print(f"\nExample {i + 1}:")
            print(f"  English: {r['English']}")
            print(f"  English LP: {r['English_LP']:.2f}")
            if "Japanese_LP" in r:
                # only print Japanese fields when present
                print(f"  Japanese: {r.get('Japanese', '')}")
                print(f"  Japanese LP: {r['Japanese_LP']:.2f}")
            print(f"  Expected: {r['Expected']}")
            print("-" * 50)
        if len(results_lp) > MAX_DISPLAY:
            print(f"...Displayed first {MAX_DISPLAY} of {len(results_lp)} examples\n")

    if grammar_method == "all" and examples_data_for_processing:
        print("\n--- Comparison Summary (Example 1) ---")
        if results_sampling:
            # read percentages off the nested TaskResult
            eng_perc = results_sampling[0].grammar.english.percentages.get("yes", 0.0)
            jap_perc = 0.0
            if results_sampling[0].grammar.japanese:
                jap_perc = results_sampling[0].grammar.japanese.percentages.get(
                    "yes", 0.0
                )
            print(f"Sampling: Eng%={eng_perc:.1f}, Jap%={jap_perc:.1f}")
        if results_yesno:
            # always show English, Japanese only if present
            yesno_line = f"Yes/No Prob: EngP={results_yesno[0]['English_P_yes']:.3f}"
            if "Japanese_P_yes" in results_yesno[0]:
                yesno_line += f", JapP={results_yesno[0]['Japanese_P_yes']:.3f}"
            print(yesno_line)
        if results_lp:
            # always show English LP, Japanese LP only if present
            lp_line = f"In-Template LP: EngLP={results_lp[0]['English_LP']:.2f}"
            if "Japanese_LP" in results_lp[0]:
                lp_line += f", JapLP={results_lp[0]['Japanese_LP']:.2f}"
            print(lp_line)
        # show the expected grammaticality from the Example model
        print(f"Expected: {examples_data_for_processing[0].grammatical.value}")
        print("-" * 50)

    # if there were no Japanese inputs, strip out Japanese fields
    if not has_japanese:
        # 1) dict‐based results (yesno & intemplate)
        for lst in (results_yesno or [], results_lp or []):
            for entry in lst:
                for k in [key for key in entry.keys() if key.startswith("Japanese")]:
                    entry.pop(k, None)

        # 2) Pydantic UnifiedResult sampling outputs: clear the .japanese slot
        if results_sampling:
            for r in results_sampling:
                if r.grammar and r.grammar.japanese is not None:
                    r.grammar.japanese = None

    # --- dump grammar sampling results to JSON ---
    if results_sampling is not None:
        out_path = f"{sanitize_model_name(model_name)}_grammar_results.json"
        write_json_models_fn(out_path, results_sampling)
        print(f"Full Grammar results written to {out_path}")

    # --- SAVE COMBINED GRAMMAR‐CHECK RESULTS AS JSONL ---
    combined_path = f"{sanitize_model_name(model_name)}_grammar_all_results.jsonl"
    entries = []
    N = len(examples_data_for_processing)
    for i in range(N):
        entry: dict = {}
        if results_sampling is not None and i < len(results_sampling):
            # serialize UnifiedResult to a JSON‐safe dict
            entry["sampling"] = results_sampling[i].model_dump(
                by_alias=True, exclude_none=True
            )
        if results_yesno is not None and i < len(results_yesno):
            entry["yesno_prob"] = results_yesno[i]
        if results_lp is not None and i < len(results_lp):
            entry["intemplate_lp"] = results_lp[i]
        entries.append(entry)
    write_jsonl_fn(combined_path, entries)
    print(f"Combined grammar results written to {combined_path}")
    return
