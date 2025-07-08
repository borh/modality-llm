import importlib
from typing import Any, Dict, Optional, Tuple

import torch
import transformers
from outlines import models
from transformers import AutoTokenizer

from modality_llm.settings import DEFAULT_QUANTIZATION_MODE

# module-level cache
model: Optional[Any] = None
tokenizer: Optional[Any] = None
quantization_mode: str = DEFAULT_QUANTIZATION_MODE
use_flash_attn: bool = True


def build_quant_config(mode: str) -> Tuple[Optional[Any], torch.dtype]:
    """Build quantization config and dtype based on mode."""
    if mode in ("int8", "int4"):
        try:
            from transformers import BitsAndBytesConfig

            print(f"Using BitsAndBytesConfig for {mode}")
            if mode == "int8":
                # only 8-bit
                bnb = BitsAndBytesConfig(load_in_8bit=True)
            else:
                # 4-bit needs the extra kwargs
                bnb = BitsAndBytesConfig(
                    load_in_8bit=False,
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            return bnb, torch.bfloat16
        except ImportError as e:
            print("bnb unavailable, falling back to bf16:", e)
            return None, torch.bfloat16
    else:
        return None, torch.bfloat16


def initialize_model(model_name: str) -> Any:
    """
    Initialize (and memoize) the LLM.

    Args:
        model_name: pretrained model identifier
    Returns:
        the loaded model object
    """
    global model, tokenizer, quantization_mode

    if model is not None:
        return model

    print(f"Initializing model: {model_name} (quant={quantization_mode})")
    config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.init_device = "meta"

    model_kwargs: Dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "device_map": {"": 0},
    }

    quant_config, dtype = build_quant_config(quantization_mode)
    if quant_config:
        model_kwargs.update(
            {
                "quantization_config": quant_config,
                "torch_dtype": dtype,
            }
        )
    else:
        model_kwargs["torch_dtype"] = dtype

    # optional flash attention
    if use_flash_attn:
        if importlib.util.find_spec("flash_attn"):
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using flash attention 2")
        else:
            print("Flash attention not found, using PyTorch defaults")
    else:
        print("Flash attention disabled by user; using PyTorch defaults")

    model = models.transformers(
        model_name=model_name,
        device="cuda",
        model_kwargs=model_kwargs,
    )

    # Initialize tokenizer (suppress failures so tests won’t crash)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Warning: Failed to load tokenizer for {model_name}: {e}")
        tokenizer = None
    return model


def get_tokenizer(model_name: str) -> Any:
    """
    Get the tokenizer for the given model name.
    """
    global tokenizer
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer for {model_name}: {e}")
            tokenizer = None
    return tokenizer
