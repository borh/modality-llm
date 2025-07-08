"""
Grammar‐checking functions and Bayesian analyses.
"""

from typing import List, Literal

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from modality_llm.llm_cache import LLMCache
from modality_llm.llm_utils import make_generator
from modality_llm.prompts import grammar_prompts
from modality_llm.schema import Example, LanguageResult, TaskResult, UnifiedResult
from modality_llm.utils import chunked


def compute_grammar_results(
    examples: List[Example],
    num_samples: int = 5,
    grammar_language: Literal["english", "japanese", "both"] = "both",
    model: PreTrainedModel | None = None,
    model_name: str = "model",
    augment_acceptability: bool = True,
    batch_size: int = 50,
) -> List[UnifiedResult]:
    """
    Compute grammar check results (pure, no printing or Bayesian analysis).
    """
    cache = LLMCache()
    params = {
        "method": "sampling",
        "num_samples": num_samples,
        "lang": grammar_language,
    }
    hit = cache.get(model_name, "grammar_sampling", params, model_cls=UnifiedResult)
    if hit:
        # now a List[UnifiedResult]
        return hit

    generator = make_generator(model, r"yes|no", num_samples)
    # we now require a real Example everywhere
    norm_exs = examples

    # 1) build yes/no prompts for each Example
    eng_prompts, jap_prompts = grammar_prompts(norm_exs, grammar_language)
    flat_prompts: list[str] = eng_prompts[:]
    flat_prompts += jap_prompts if grammar_language in ("japanese", "both") else []

    # 2) one batched LLM call
    flat_answers: list[list[str]] = []
    for chunk in chunked(flat_prompts, batch_size):
        with torch.inference_mode():
            flat_answers.extend(generator(chunk, max_tokens=30))
        torch.cuda.empty_cache()

    # 3) zip answers back into UnifiedResult per Example
    results: list[UnifiedResult] = []
    n = len(norm_exs)
    has_jap = grammar_language in ("japanese", "both")
    for i, ex in enumerate(norm_exs):
        # English result
        e_ans, e_prompt = flat_answers[i], eng_prompts[i]
        # build nested TaskResult for grammar outputs
        gram_task = TaskResult(english=LanguageResult(prompt=e_prompt, answers=e_ans))
        if has_jap and i < len(jap_prompts):
            j_ans, j_prompt = flat_answers[n + i], jap_prompts[i]
            gram_task.japanese = LanguageResult(prompt=j_prompt, answers=j_ans)
        result = UnifiedResult(
            eid=ex.eid,
            english=ex.english,
            japanese=ex.japanese,
            english_target=ex.english_target,
            japanese_target=None,
            grammatical=ex.grammatical,
            human_annotations=ex.human_annotations,
            expected_categories=ex.expected_categories,
            grammar=gram_task,
            classification=None,
        )
        results.append(result)

    cache.save(
        model_name,
        "grammar_sampling",
        params,
        [r.model_dump() for r in results],
    )
    return results


def run_grammar_check_yesno_prob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    examples: List[Example],
    grammar_language: Literal["english", "japanese", "both"] = "both",
) -> List[dict]:
    """
    Compute normalized probability of 'yes' vs 'no' as next token after prompt.

    Returns list of dicts with P_yes for English and Japanese prompts.
    """

    eng_prompts, jap_prompts = grammar_prompts(examples, grammar_language)
    flat_prompts: list[tuple[int, str, str]] = []
    if grammar_language in ("english", "both"):
        flat_prompts.extend([(i, "English", p) for i, p in enumerate(eng_prompts)])
    if grammar_language in ("japanese", "both"):
        flat_prompts.extend([(i, "Japanese", p) for i, p in enumerate(jap_prompts)])

    # Get token ids for "yes" and "no"
    yes_token = tokenizer.encode("yes", add_special_tokens=False)
    no_token = tokenizer.encode("no", add_special_tokens=False)
    if len(yes_token) != 1 or len(no_token) != 1:
        print(
            "Warning: 'yes' or 'no' tokenized into multiple tokens, results may be unreliable"
        )
    yes_id = yes_token[0]
    no_id = no_token[0]

    # Run all prompts in a single batch (if possible)
    results_flat: list[tuple[int, str, float]] = []
    for idx, lang, prompt in flat_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # shape (batch=1, seq_len, vocab_size)
        last_token_logits = logits[0, -1, :]  # logits for next token

        probs = F.softmax(last_token_logits, dim=-1)
        p_yes = probs[yes_id].item()
        p_no = probs[no_id].item()
        norm_p_yes = p_yes / (p_yes + p_no + 1e-9)
        results_flat.append((idx, lang, norm_p_yes))

    # Group results by example index
    results: list[dict] = []
    for i, ex in enumerate(examples):
        entry = {
            "English": ex.english,
            "Japanese": ex.japanese,
            "Expected": ex.grammatical.value,
            "EID": ex.eid,
            "Target": ex.english_target or ex.japanese_target,
        }
        for _, lang, val in filter(lambda t: t[0] == i, results_flat):
            entry[f"{lang}_P_yes"] = val
        results.append(entry)

    return results


def run_grammar_check_intemplate_lp(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    examples: List[Example],
    template_str: str = "The following sentence is grammatically acceptable.\n\n{sentence}",
    grammar_language: Literal["english", "japanese", "both"] = "both",
) -> List[dict]:
    """
    Compute total log probability of sentence embedded in a template.

    Returns list of dicts with LP scores for English and Japanese.
    """

    eng_inputs = []
    jap_inputs = []
    for ex in examples:
        eng_sent = ex.english or ""
        jap_sent = ex.japanese or ""
        clean_eng = eng_sent.replace("*", "")
        clean_jap = jap_sent.replace("*", "")
        eng_inputs.append(template_str.format(sentence=clean_eng))
        jap_inputs.append(template_str.format(sentence=clean_jap))

    flat_inputs: list[tuple[int, str, str]] = []
    n = len(examples)
    if grammar_language in ("english", "both"):
        flat_inputs.extend([(i, "English", eng_inputs[i]) for i in range(n)])
    if grammar_language in ("japanese", "both"):
        flat_inputs.extend([(i, "Japanese", jap_inputs[i]) for i in range(n)])

    results_flat: list[tuple[int, str, float]] = []
    for idx, lang, text in flat_inputs:
        if not text:
            continue
        enc = tokenizer(text, return_tensors="pt").to(model.device)
        input_ids = enc["input_ids"]
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits  # (1, seq_len, vocab_size)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        seq_token_logprobs = torch.gather(
            log_probs, 2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        total_lp = seq_token_logprobs.sum().item()
        results_flat.append((idx, lang, total_lp))

    results: list[dict] = []
    for i, ex in enumerate(examples):
        eng_sent = ex.english or ""
        jap_sent = ex.japanese or ""
        entry = {
            "English": eng_sent,
            "Japanese": jap_sent,
            "Expected": ex.grammatical.value,
            "EID": ex.eid,
            "Target": ex.english_target or ex.japanese_target,
            "English_Input": eng_inputs[i],
            "Japanese_Input": jap_inputs[i],
        }
        for _, lang, val in filter(lambda t: t[0] == i, results_flat):
            entry[f"{lang}_LP"] = val
        results.append(entry)

    return results
