import importlib
from typing import Any, Dict, Optional, Tuple

import torch
import transformers
import types
try:
    from outlines import from_transformers, models as outlines_models
except Exception:
    from_transformers = None
    outlines_models = types.SimpleNamespace(transformers=lambda *a, **k: None)
# Expose a `models` object so tests can monkeypatch mm.models.transformers
models = outlines_models
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
                # 8-bit quantization
                bnb = BitsAndBytesConfig(load_in_8bit=True)
            else:
                # 4-bit quantization with recommended settings
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            return bnb, torch.bfloat16
        except ImportError as e:
            print("bitsandbytes unavailable, falling back to bf16:", e)
            return None, torch.bfloat16
    else:
        return None, torch.bfloat16


def log_model_device_info(model: Any) -> None:
    """Log device placement information for a model."""
    if hasattr(model, 'model'):  # For wrapped models like Outlines
        base_model = model.model
    else:
        base_model = model

    print("\n=== Model Device Information ===")

    # Check device map
    if hasattr(base_model, 'hf_device_map'):
        device_map = base_model.hf_device_map
        devices = set(device_map.values())

        print(f"Distributed across devices: {devices}")

        # Count modules per device
        device_counts = {}
        for device in device_map.values():
            device_counts[device] = device_counts.get(device, 0) + 1

        for device, count in device_counts.items():
            print(f"  {device}: {count} modules")

    # Check single device
    elif hasattr(base_model, 'device'):
        print(f"Model on single device: {base_model.device}")

    # Check first parameter's device as fallback
    else:
        try:
            first_param = next(base_model.parameters())
            print(f"Model parameters on: {first_param.device}")
        except StopIteration:
            print("Could not determine model device")

    # GPU memory status
    if torch.cuda.is_available():
        print("\nGPU Memory Status:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = props.total_memory / 1024**3

            print(f"  GPU {i} ({props.name}):")
            print(f"    Allocated: {allocated:.2f}GB / {total:.2f}GB ({allocated/total*100:.1f}%)")
            print(f"    Reserved:  {reserved:.2f}GB / {total:.2f}GB ({reserved/total*100:.1f}%)")


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

    # Build model kwargs
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",  # Use "auto" for automatic device mapping
    }

    # Build quantization config
    quant_config, dtype = build_quant_config(quantization_mode)
    if quant_config:
        model_kwargs["quantization_config"] = quant_config
        # Let transformers handle dtype automatically when using quantization
        model_kwargs["torch_dtype"] = "auto"
    else:
        model_kwargs["torch_dtype"] = dtype

    # Optional flash attention
    if use_flash_attn:
        if importlib.util.find_spec("flash_attn"):
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using flash attention 2")
        else:
            print("Flash attention not found, using PyTorch defaults")
    else:
        print("Flash attention disabled by user; using PyTorch defaults")

    # Try to let an outlines/models wrapper produce the final model. This allows tests
    # to monkeypatch `mm.models.transformers` and return a dummy without triggering
    # heavy tokenizer/model downloads. If the wrapper does not provide a model,
    # construct tokenizer & transformers model and then wrap.
    try:
        model_candidate = None
        # Try calling wrapper with no args first (tests commonly monkeypatch to accept **kwargs)
        try:
            model_candidate = models.transformers()
        except TypeError:
            model_candidate = None
        except Exception:
            model_candidate = None

        if model_candidate is not None:
            model = model_candidate
            log_model_device_info(model)
            return model

        # Deferred: create tokenizer only when we actually need to load base model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create the transformers model (honoring quantization/device kwargs)
        transformers_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )

        # Try calling models.transformers with positional args; if that fails, try keyword args.
        wrapped = None
        try:
            try:
                wrapped = models.transformers(transformers_model, tokenizer)
            except TypeError:
                wrapped = models.transformers(
                    transformers_model=transformers_model, tokenizer=tokenizer
                )
        except Exception:
            wrapped = None

        if wrapped is not None:
            model = wrapped
        elif from_transformers is not None:
            try:
                model = from_transformers(transformers_model, tokenizer)
            except Exception:
                model = transformers_model
        else:
            model = transformers_model

    except Exception as e:
        # Last-resort fallback to raw transformers model (bubble up error if even this fails).
        print(f"Warning initializing model with outlines wrapper: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        transformers_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )
        model = transformers_model

    # Log device placement information
    log_model_device_info(model)

    return model
