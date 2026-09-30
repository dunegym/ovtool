"""download endpoint/dest helpers."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ovtool.download import ENDPOINTS, _default_dest, _endpoint_url, _entry_size


def test_endpoint_url_known_and_full_urls():
    assert _endpoint_url("hf-mirror.com") == "https://hf-mirror.com"
    assert _endpoint_url("huggingface.co") == "https://huggingface.co"
    assert _endpoint_url("https://custom.example") == "https://custom.example"


def test_endpoint_url_unknown_rejected():
    with pytest.raises(SystemExit, match="unknown endpoint"):
        _endpoint_url("evil.com")


def test_endpoints_constant_matches():
    assert ENDPOINTS == ("huggingface.co", "hf-mirror.com")


def test_default_dest_from_repo_or_subfolder():
    assert _default_dest("Qwen/Qwen3-0.6B", None) == "./downloads/Qwen3-0.6B"
    assert _default_dest("dunegym/openvino-models",
                         "llm/Qwen3-0.6B/int4-sym-g128") == "./downloads/int4-sym-g128"
    assert _default_dest("r", "/llm/M/int8/") == "./downloads/int8"


@pytest.mark.parametrize("entry,want", [
    (SimpleNamespace(lfs=None, size=5), 5),                      # plain file
    (SimpleNamespace(lfs=SimpleNamespace(size=1000), size=10), 1000),  # LFS blob
    (SimpleNamespace(lfs=None, size=None), None),                # directory
])
def test_entry_size(entry, want):
    assert _entry_size(entry) == want
