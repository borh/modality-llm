import modality_llm.schema as schema_mod
from modality_llm.examples import download_modal_verb_dataset as _download_modal_dataset
from modality_llm.modal import compute_modal_results
from modality_llm.utils import sanitize_model_name, write_json


def run(
    args,
    model,
    model_name,
    *,
    download_fn=_download_modal_dataset,
    compute_fn=compute_modal_results,
    sanitize_fn=sanitize_model_name,
    write_json_fn=write_json,
    report_augmentations=False,
) -> None:
    # Validate that we only ever get palmer, quirk, or both
    if args.taxonomy not in ("palmer", "quirk", "both"):
        raise ValueError(f"Unknown taxonomy: {args.taxonomy}")

    data_path = download_fn(args.data_path)
    common_eval_args = {
        "data_path": data_path,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "use_cache": args.use_cache,
        "force_refresh": args.force_refresh,
        "num_examples_to_sample": args.num_examples,
        "random_seed": args.random_seed,
        "model": model,
        "model_name": model_name,
        "augment_substitution": getattr(args, "augment_substitution", True),
        "augment_entailment": getattr(args, "augment_entailment", True),
        "augment_contradiction": getattr(args, "augment_contradiction", True),
        "report_augmentations": report_augmentations,
        "zero_shot": getattr(args, "zero_shot", False),
        "require_consensus": getattr(args, "require_consensus", None),
    }
    if args.taxonomy in ["palmer", "both"]:
        palmer_results = compute_fn(
            taxonomy=schema_mod.Taxonomy.palmer, **common_eval_args
        )
        out = f"{sanitize_fn(model_name)}_palmer_results.json"
        write_json_fn(out, palmer_results)
        print(f"Full Palmer results written to {out}")

        # === generate classification & uncertainty dashboards for Palmer ===
        from modality_llm.analysis_compare import compare_with_human_annotators
        from modality_llm.analysis_modal_bayes import bayesian_analysis_modal
        from modality_llm.modal import get_categories

        print("\n--- Generating Palmer classification & uncertainty visualizations ---")
        pal_cols = get_categories("palmer")
        try:
            compare_with_human_annotators(palmer_results, pal_cols, "palmer", model_name)
        except Exception as e:
            print(f"Warning: Skipping human comparison for Palmer: {e}")
        try:
            bayesian_analysis_modal(
                results=palmer_results,
                categories=pal_cols,
                model_name=model_name,
                taxonomy="palmer",
            )
        except Exception as e:
            print(f"Warning: Skipping Bayesian analysis for Palmer: {e}")

    if args.taxonomy in ["quirk", "both"]:
        quirk_results = compute_fn(
            taxonomy=schema_mod.Taxonomy.quirk, **common_eval_args
        )
        out = f"{sanitize_fn(model_name)}_quirk_results.json"
        write_json_fn(out, quirk_results)
        print(f"Full Quirk results written to {out}")

        # === generate classification & uncertainty dashboards for Quirk ===
        from modality_llm.analysis_compare import compare_with_human_annotators
        from modality_llm.analysis_modal_bayes import bayesian_analysis_modal
        from modality_llm.modal import get_categories

        print("\n--- Generating Quirk classification & uncertainty visualizations ---")
        qk_cols = get_categories("quirk")
        try:
            compare_with_human_annotators(quirk_results, qk_cols, "quirk", model_name)
        except Exception as e:
            print(f"Warning: Skipping human comparison for Quirk: {e}")
        try:
            bayesian_analysis_modal(
                results=quirk_results,
                categories=qk_cols,
                model_name=model_name,
                taxonomy="quirk",
            )
        except Exception as e:
            print(f"Warning: Skipping Bayesian analysis for Quirk: {e}")

    # --- one-shot augmentation analysis? ---
    if getattr(args, "run_augmentation_analysis", False):
        import polars as pl
        import torch

        from modality_llm.analysis_augmentation import (
            analyze_augmentation_effects,
            plot_aug_dashboard,
        )
        from modality_llm.augmentation import (
            generate_acceptability_variants,
            generate_contradiction_variants,
            generate_entailment_tests,
            generate_substitution_variants,
        )
        from modality_llm.llm_utils import make_generator
        from modality_llm.modal import get_categories
        from modality_llm.prompts import modal_prompts
        from modality_llm.schema import Example, LanguageResult, ModalExample, Taxonomy
        from modality_llm.utils import chunked, load_jsonl, sanitize_model_name

        # 1) reify the original Examples
        raw = load_jsonl(data_path)
        examples = [
            Example.from_modal_verb_example(ModalExample.model_validate(d)) for d in raw
        ]

        # 2) load or build all variants
        from modality_llm.utils import load_jsonl_models
        if args.augmentations_file:
            variants = load_jsonl_models(args.augmentations_file, Example)
        else:
            variants: list[Example] = []
            for ex in examples:
                variants.extend(generate_acceptability_variants(ex))
                variants.extend(generate_substitution_variants(ex))
                variants.extend(generate_entailment_tests(ex))
                variants.extend(generate_contradiction_variants(ex))

        if not variants:
            print("No variants generated; skipping augmentation analysis.")
        else:
            # pick the taxonomy for classification
            tax = (
                schema_mod.Taxonomy.palmer
                if args.taxonomy in ("palmer", "both")
                else schema_mod.Taxonomy.quirk
            )
            categories = get_categories(tax.value)
            pattern = "|".join(categories)
            gen = make_generator(model, pattern, args.num_samples)

            # 3) classify each batch of variants
            rows = []
            orig_results = palmer_results if tax is Taxonomy.palmer else quirk_results

            for batch in chunked(variants, args.batch_size):
                prompts, _ = modal_prompts(batch, tax)
                with torch.inference_mode():
                    answers = gen(prompts, max_tokens=20)
                for ex, ans in zip(batch, answers):
                    # find the original distribution by eid
                    orig_dist = {}
                    for r in orig_results:
                        if r.eid == ex.eid:
                            orig_dist = r.classification[tax].english.distribution
                            break

                    # new distribution
                    var_dist = LanguageResult(prompt="", answers=ans).distribution

                    # top-label & confidence
                    def top_conf(d):
                        if not d:
                            return None, 0.0
                        lbl, cnt = max(d.items(), key=lambda x: x[1])
                        return lbl, cnt / (sum(d.values()) or 1)

                    lbl_o, conf_o = top_conf(orig_dist)
                    lbl_v, conf_v = top_conf(var_dist)

                    rows.append(
                        {
                            "strategy": ex.transformation_strategy,
                            "orig_category": (
                                ex.expected_categories[tax.value][0]
                                if ex.expected_categories
                                else None
                            ),
                            "preserved": lbl_o == lbl_v,
                            "confidence_drop": conf_o - conf_v,
                        }
                    )

            # 4) summarize & plot
            df = pl.DataFrame(rows)
            overall, bycat, violin = analyze_augmentation_effects(df)
            dash = plot_aug_dashboard(overall, bycat, violin, model_name, tax.value)
            fname = f"{sanitize_model_name(model_name)}_augmentation_dashboard_{tax.value}.html"
            dash.save(fname)
            print(f"Augmentation dashboard saved to {fname}")
