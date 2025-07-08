import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def sample_modal_verbs(tmp_path_factory):
    src = Path(__file__).parent / "fixtures" / "modal_verbs_sample.jsonl"
    destdir = tmp_path_factory.mktemp("data")
    dst = destdir / "modal_verbs.jsonl"
    shutil.copy(src, dst)
    return str(dst)
