import json
import os

from modality_llm.llm_cache import LLMCache


def test_save_and_get_cache(tmp_path):
    cache = LLMCache(str(tmp_path))
    data = {"foo": "bar"}
    path = cache.save("model", "task", {}, [data])
    assert os.path.exists(path)
    loaded = cache.get("model", "task", {})
    # without model_cls we get the raw list we saved
    assert isinstance(loaded, list)
    assert loaded == [data]


def test_get_invalid_json(tmp_path, capsys):
    cache = LLMCache(str(tmp_path))
    path = os.path.join(str(tmp_path), "bad.json")
    with open(path, "w") as f:
        f.write("{not valid json}")
    # Patch _get_cache_path to return our file
    cache._get_cache_path = lambda model_name, cache_key: path
    cache._generate_cache_key = lambda model_name, task_type, params: "bad"
    cache.get("model", "task", {})
    out = capsys.readouterr().out
    assert "Error reading cache file" in out


def test_get_missing_results(tmp_path, capsys):
    cache = LLMCache(str(tmp_path))
    path = os.path.join(str(tmp_path), "noresults.json")
    with open(path, "w") as f:
        json.dump({"metadata": {}}, f)
    cache._get_cache_path = lambda model_name, cache_key: path
    cache._generate_cache_key = lambda model_name, task_type, params: "noresults"
    cache.get("model", "task", {})
    out = capsys.readouterr().out
    # now we print the JSON‐read error
    assert "Error reading cache file" in out


def test_modal_classification_metadata_update(tmp_path, capsys):
    cache = LLMCache(str(tmp_path))
    path = os.path.join(str(tmp_path), "modal.json")
    # missing metadata.categories
    with open(path, "w") as f:
        json.dump({"metadata": {}, "results": [{"foo": "bar"}]}, f)
    cache._get_cache_path = lambda model_name, cache_key: path
    cache._generate_cache_key = lambda model_name, task_type, params: "modal"
    params = {"categories": ["a", "b"]}
    cache.get("model", "modal_classification", params)
    out = capsys.readouterr().out
    assert "Updated cache metadata" in out
