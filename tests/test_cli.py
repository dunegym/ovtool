"""CLI entry-point smoke tests (argument wiring + the registry gate)."""
from __future__ import annotations

import pytest

from ovtool.cli import main


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as ei:
        main(["--version"])
    assert ei.value.code == 0
    assert "ovtool" in capsys.readouterr().out


def test_models_remote_lists_registry(capsys):
    main(["models"])
    out = capsys.readouterr().out
    assert "Qwen/Qwen3-0.6B" in out


def test_models_treats_bare_word_as_query(capsys):
    main(["models", "qwen3-0.6b"])     # backward-compat: not remote/local
    out = capsys.readouterr().out
    assert "Qwen/Qwen3-0.6B" in out
    assert "Kokoro" not in out          # filtered


def test_registry_gate_blocks_bad_combo_before_load(capsys):
    """int4-asym LLM on NPU is rejected by the registry with exit code 2 —
    before any model directory is even opened."""
    with pytest.raises(SystemExit) as ei:
        main(["chat", "-m", "models/llm/Qwen3-0.6B/int4", "-d", "NPU"])
    assert ei.value.code == 2
    out = capsys.readouterr().out
    assert "symmetric" in out
    assert "Blocked by the compatibility registry" in out
