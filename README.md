## Output

* **Console:** ...
* **Plots:** Several plots are generated and saved as PNG files in the script's directory. Filenames include a sanitized version of the model name (e.g., `meta-llama_Llama-3_2-3B-Instruct_llm_human_agreement_palmer.png`). Plot titles also include the model name.
  * *Modal Task:* `<sanitized_model_name>_llm_human_agreement_<taxonomy>.png`, `<sanitized_model_name>_confusion_matrix_<taxonomy>.png`, `<sanitized_model_name>_modal_posterior_alpha.png`.
  * *Grammar Task (sampling method):* `<sanitized_model_name>_language_bias_posterior.png`, `<sanitized_model_name>_language_comparison.png`, `<sanitized_model_name>_trace_plot.png`.
* **Cache:** Computation results are cached in the `cache/` directory as JSON files. Cache filenames are MD5 hashes of the configuration, prefixed with the sanitized model name (e.g., `meta-llama_Llama-3_2-3B-Instruct_<hash>.json`).
* **Dataset:** ...
* **Bayesian Trace:** For the grammar task (sampling method) and modal task (Palmer), PyMC trace data is saved as a NetCDF file, prefixed with the sanitized model name (e.g., `meta-llama_Llama-3_2-3B-Instruct_bayesian_trace.nc`).

## Usage

This repository provides two main tasks—grammar checking and modal‐verb classification—via a single CLI script (`main.py`). Below are the most common invocations and explanations of key flags.

### 1. Install dependencies

```bash
pip install -r requirements.txt
# or, if using Poetry:
poetry install
```

### 2. Grammar check

Use `--task grammar` (or `--task all` to run grammar + modal).

```bash
python main.py \
  --task grammar \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --quantization bf16 \
  --grammar-method sampling \
  --num-samples 5
  --examples-file your_examples.json
```

#### Custom Examples File

To run the grammar check on your own data, create a JSON file containing a list of triples:
```json
[
  [
    "It is twelve, John *must* have been washing the car.",
    "車がピカピカだ．太郎が洗った*に違いない*．",
    "yes"
  ],
  [
    "It is twelve, John *should* have been washing the car.",
    "車がピカピカだ．太郎が洗った*はずだ*．",
    "no"
  ]
]
```

- Each item is `[english_sentence, japanese_sentence, expected_answer]`.  
- Surround the target word in each sentence with `*asterisks*`.  
- `expected_answer` must be `"yes"` or `"no"`.  

Then invoke:
```bash
python main.py \
  --task grammar \
  --examples-file your_examples.json \
  --grammar-method sampling \
  --num-samples 5
```

• `--grammar-method`  
  • `sampling`: generate N yes/no samples per sentence, then count distribution.  
  • `yesno_prob`: compute next‐token softmax probability for "yes" vs. "no."  
  • `intemplate_lp`: embed the sentence in a template and compute its total log‐probability.  
  • `all`: run all three and compare.

• `--num-samples`  
  Number of independent generations (only for sampling method).  

After running, you will see console summaries and two Bayesian analyses:
  1. Beta‐binomial update + KL divergence  
  2. Full PyMC model (trace diagnostics, posterior, plots, and NetCDF trace file)

### 3. Modal‐verb classification

```bash
python main.py \
  --task modal \
  --taxonomy palmer \
  --batch-size 50 \
  --num-samples 5 \
  --use-cache \
  --quantization int8
```

• `--taxonomy`: choose "palmer", "quirk", or "both" to evaluate different modal taxonomies.  
• `--batch-size`: number of examples per LLM batch.  
• `--use-cache` / `--no-cache`: read/write JSON cache to speed up repeated runs.  
• `--force-refresh`: ignore existing cache and re‐compute from scratch.  

On completion you get:  
  - Console reports (agreement rates, confusion matrices, kappa).  
  - Plots:  
    - `<model>_llm_human_agreement_<taxonomy>.png`  
    - `<model>_confusion_matrix_<taxonomy>.png`  
    - `<model>_modal_posterior_alpha.png` (for Palmer).  
  - JSON files in `cache/` for fast re‐runs.

### 4. Tips & why these flags matter

 - Quantization (`bf16`, `int8`, `int4`) trades off memory vs. speed vs. numeric precision.  
 - Caching avoids repeated API/LLM calls when experimenting with downstream analysis.  
 - Batch size and sample count let you balance throughput against statistical robustness.  

For full flag reference, run:

```bash
python main.py --help
```
