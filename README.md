# modality-llm

Modal-verb classification and grammar checking using local LLMs.

## What this repository does

- Modal verb classification (Palmer and Quirk taxonomies):
  - Optionally augment examples (substitution, entailment, contradiction, acceptability).
  - Batch LLM generation (supports constrained/regex generation via outlines.Generator).
  - Bayesian uncertainty analysis (Dirichlet-Multinomial using PyMC) and Altair dashboards.
- Grammar checking:
  - Sampling-based yes/no generation, next-token yes/no probability, and in-template log-probability.
  - Beta-binomial and PyMC analyses; hierarchical models produce PNG plots when grouping keys exist.
- Utilities:
  - Generate CSV/XLSX for human annotation from the modal-verb dataset.
  - Convert annotated CSV/XLSX back to modal_verbs.jsonl format.
  - On-disk caching of LLM outputs to speed repeated runs.

## Quick notes

Run the CLI with uv (no installation required):

```bash
uv run src/modality_llm/cli.py --help
```

Example output:

```
Using spaCy 3.8.7 (gpu: False)
usage: cli.py [-h] {generate,modal,grammar,run} ...

modality-LLM task runner

positional arguments:
  {generate,modal,grammar,run}
                        Which task to run
    generate            Generate CSV for annotation or convert annotated CSV back to JSONL
    modal               Compute modal-verb classification results
    grammar             Run grammar checks & Bayes analyses
    run                 Run full pipeline: modal classification (+ augmentation analysis) and grammar checks

options:
  -h, --help            show this help message and exit
```

Notes:

- Use "uv run" to execute the CLI module from source as shown above.
- All project dependencies are managed with uv via the pyproject.toml file.

- Entry point: the CLI can be accessed using `uv run modality-llm` or by invoking the `src/modality_llm/cli.py` module directly.
- Default model: configured in modality_llm.settings (`DEFAULT_MODEL`). If you run the subcommand that exposes a `--model` flag you can override it on the command line. Some subcommands use the configured default model name.
- This project is tested with Python 3.13 as target but will most likely work with older versions.

## Recommended environment

Recommended way of running this package is with `uv` and CUDA or equivalent hardware-accelerated PyTorch packages.
Note that you want to run with the `--extra cuda` flag to uv if running with a CUDA accelerator.

```bash
uv run --extra cuda modality-llm
```

(ROCM support could be added by adding the appropriate `--with ...` parameters to uv)

## Main subcommands

1) generate
  - Purpose: produce a CSV/XLSX for human annotation from the modal-verb dataset, or convert an annotated CSV/XLSX back to JSONL.
  - Example: create CSV for annotation
      `uv run modality_llm generate csv modal_verbs.jsonl output.csv --gen-include-alternatives`
  - Example: convert annotated CSV/XLSX to JSONL
      `uv run modality_llm generate jsonl annotated.csv out.jsonl --completed-only`

2) modal
  - Purpose: classify modal verbs across examples (Palmer and/or Quirk), optionally generate augmentations, perform human-comparison and Bayesian uncertainty analyses, and emit dashboards.
  - Important flags:
    --taxonomy palmer|quirk|both
    --num-samples N         (LLM samples per prompt)
    --batch-size N
    --model MODEL_NAME      (model name; e.g. meta-llama/Llama-3.2-3B.Instruct)
    --quantization bf16|int8|int4
    --no-augment-substitution, --no-augment-entailment, --no-augment-contradiction
    --no-augmentation-analysis
  - Example:
      `uv run modality_llm modal --taxonomy palmer --model Qwen/Qwen3-0.6B --num-samples 5 --batch-size 50 --quantization bf16`

3) grammar
  - Purpose: run grammar checks and related Bayesian analyses.
  - Methods: sampling (yes/no generations), yesno_prob (next-token probability), intemplate_lp (in-template log probability), or all.
  - Important flags:
    --grammar-source modal|file
    --examples-file path (when --grammar-source file)
    --grammar-method sampling|yesno_prob|intemplate_lp|all
    --num-samples N
    --num-examples N
    --use-advi / --no-advi  (ADVI vs MCMC for PyMC; ADVI is the default)
    --mcmc-samples, --mcmc-chains, --mcmc-cores
  - Example:
      `uv run modality_llm  grammar --grammar-source modal --grammar-method sampling --num-samples 3 --num-examples 100`

4) run
  - Purpose: convenience to run modal classification (incl. augmentations and analysis) and grammar checks end-to-end.
  - Accepts many of the same flags as modal and grammar.

## Outputs produced

Files are written to the current working directory unless you change code or move them:

- Modal task:
  - <sanitized_model_name>_palmer_results.json
  - <sanitized_model_name>_quirk_results.json
  - Per-taxonomy dashboards and artifacts:
    - <sanitized_model_name>_<taxonomy>_modal_full_dashboard.html
    - <sanitized_model_name>_<taxonomy>_modal_enhanced_dashboard.html
    - <sanitized_model_name>_<taxonomy>_modal_base_rates.html
    - <sanitized_model_name>_<taxonomy>_modal_uncertainty_distribution.html
    - <sanitized_model_name>_confidence_by_modal_violin.html
    - <sanitized_model_name>_modal_uncertainty_report.csv

- Grammar task:
  - <sanitized_model_name>_grammar_results.json
  - <sanitized_model_name>_grammar_all_results.jsonl
  - Hierarchical analysis (when grouping info is present) emits PNG plots:
    - <sanitized_model_name>_hierarchical_English_sentence_effects.png
    - <sanitized_model_name>_hierarchical_modal_effects.png
    - <sanitized_model_name>_hierarchical_variance_components.png

- Augmentations:
  - Entailment tests file: <sanitized_model_name>_<taxonomy>_entailment_tests.jsonl
  - If you generate CSV/XLSX for annotation, that file is whatever path you passed to the generate subcommand.

- Cache:
  - LLM outputs are cached in cache/ as files named <sanitized_model_name>_<md5hash>.json.

## Dataset evolution workflow (merge/diff and curated JSONL)

1) Generate an augmented CSV and include alternatives:
   uv run --extra cuda modality-llm generate csv modal_verbs.jsonl consensus_quirk_palmer_v1.csv --require-consensus both --gen-include-alternatives --format csv

2) After annotating (Palmer_Expected/Quirk_Expected, Completed=1), regenerate with a new code version and merge against the prior CSV to carry labels and flag text changes:
   uv run --extra cuda modality-llm generate csv modal_verbs.jsonl consensus_quirk_palmer_v2.csv --require-consensus both --gen-include-alternatives --format csv --existing-csv consensus_quirk_palmer_v1.csv

   Flags:
   - --existing-csv: carry forward Completed and expected labels by (EID|Tested_Modal|Transformation_Strategy)
   - --no-freeze-completed: if set, adopt new text even for Completed=1 rows (default keeps old text)
   - --no-mark-diff: if set, do not add a Marked_Diff column

3) Convert to curated JSONL for compute tasks. Choose minimal (pipeline-compatible) or extended (preserve CSV metadata):
   uv run modality-llm generate jsonl consensus_quirk_palmer_v2.csv curated_modal_verbs.jsonl --completed-only --jsonl-format minimal

   The minimal JSONL has fields: mv, utt, annotations, res (matches ModalExample).
   The extended JSONL adds: transformation_strategy, source_eid, completed, grammatical, grammatical_binary.

4) Use curated JSONL everywhere:
   uv run modality-llm modal --data-path curated_modal_verbs.jsonl --taxonomy both
   uv run modality-llm grammar --grammar-source modal --data-path curated_modal_verbs.jsonl

## API-backed modality removal (optional)

You can generate the "(removed)" alternative using an API model via an OpenAI-compatible endpoint.

Environment:
- `OPENAI_API_BASE`: base URL (e.g., http://localhost:30000/v1). Required for API backend.
- `OPENAI_API_KEY`: API key; defaults to "EMPTY" if not set.

CLI:
- --removal-backend spacy|api (default spacy)
- --removal-model openai/gpt-oss-20b (default)

Example:

```bash
uv run --extra cuda modality-llm generate csv modal_verbs.jsonl out.csv --gen-include-alternatives --format csv --removal-backend api --removal-model openai/gpt-oss-20b
```

The prompt used for the API distills the spaCy rules and test cases and is in general better than the spaCy backend.

## Behaviour and caveats

- Quantization: bitsandbytes support is optional. If bitsandbytes is missing, the code falls back to bf16.
- ADVI is used by default for PyMC models because it is faster. Use --no-advi to run MCMC; MCMC can be much slower.
- The grammar subcommand uses the configured default model name unless you run a top-level command that exposes --model. See the CLI help for details.

## Testing and development

Run the unit tests from the repo root with:

```bash
uv run pytest
```
