import importlib
import types

import modality_llm.model_manager as mm


def test_initialize_model_flash_attention(monkeypatch):
    # Patch importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: True)
    # Patch transformers.AutoConfig.from_pretrained
    monkeypatch.setattr(
        mm.transformers.AutoConfig,
        "from_pretrained",
        lambda *a, **k: types.SimpleNamespace(init_device="meta"),
    )
    # Patch outlines.models.transformers
    monkeypatch.setattr(mm.models, "transformers", lambda **kwargs: "MODEL")
    mm.model = None
    mm.quantization_mode = "bf16"
    model = mm.initialize_model("foo")
    assert model == "MODEL"


def test_initialize_model_flash_attention_not_found(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: False)
    monkeypatch.setattr(
        mm.transformers.AutoConfig,
        "from_pretrained",
        lambda *a, **k: types.SimpleNamespace(init_device="meta"),
    )
    monkeypatch.setattr(mm.models, "transformers", lambda **kwargs: "MODEL2")
    mm.model = None
    mm.quantization_mode = "bf16"
    model = mm.initialize_model("foo2")
    assert model == "MODEL2"


def test_initialize_model_int8_int4(monkeypatch):
    # Patch importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: False)
    monkeypatch.setattr(
        mm.transformers.AutoConfig,
        "from_pretrained",
        lambda *a, **k: types.SimpleNamespace(init_device="meta"),
    )
    monkeypatch.setattr(mm.models, "transformers", lambda **kwargs: "MODEL3")
    mm.model = None

    # Patch BitsAndBytesConfig
    class DummyBnb:
        def __init__(self, **kwargs):
            pass

    import sys

    sys.modules["transformers"] = types.SimpleNamespace(BitsAndBytesConfig=DummyBnb)
    mm.quantization_mode = "int8"
    model = mm.initialize_model("foo3")
    assert model == "MODEL3"
    mm.model = None
    mm.quantization_mode = "int4"
    model = mm.initialize_model("foo4")
    assert model == "MODEL3"
    del sys.modules["transformers"]
