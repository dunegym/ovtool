"""llm option/generation-config plumbing and the OpenAI-compatible server's
request mapping (no pipeline, no socket)."""
from __future__ import annotations

import argparse

import pytest

import openvino_genai as ovgenai

from ovtool.llm import (_npu_shape_from_args, build_generation_config,
                        compile_options, make_streamer)
from ovtool.server import _content_to_text, _finish_reason, _usage, build_config, render_chat


def ns_opt(*pairs):
    return argparse.Namespace(opt=list(pairs) or None)


# ---------------- compile_options ---------------- #

def test_compile_options_perf_mode_aliases():
    assert compile_options(ns_opt("perf_mode=low_latency")) == {"PERFORMANCE_HINT": "LATENCY"}
    assert compile_options(ns_opt("perf_mode=HIGH_THROUGHPUT")) == {"PERFORMANCE_HINT": "THROUGHPUT"}


def test_compile_options_rejects_bad_perf_mode():
    with pytest.raises(SystemExit):
        compile_options(ns_opt("perf_mode=TURBO"))


def test_compile_options_typed_keys():
    opts = compile_options(ns_opt("num_streams=4", "inference_num_threads=8",
                                  "enable_hyper_threading=true",
                                  "cache_dir=/c"))
    assert opts["NUM_STREAMS"] == 4 and isinstance(opts["NUM_STREAMS"], int)
    assert opts["CACHE_DIR"] == "/c"
    assert opts["INFERENCE_NUM_THREADS"] == 8
    assert opts["ENABLE_HYPER_THREADING"] is True


def test_compile_options_num_streams_auto_is_string():
    assert compile_options(ns_opt("num_streams=auto"))["NUM_STREAMS"] == "auto"


def test_compile_options_unknown_keys_pass_through():
    opts = compile_options(ns_opt("MAX_PROMPT_LEN=2048", "flag=yes"))
    assert opts["MAX_PROMPT_LEN"] == 2048 and isinstance(opts["MAX_PROMPT_LEN"], int)
    assert opts["flag"] == "yes"


def test_compile_options_malformed_pair():
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        compile_options(ns_opt("novalue"))


# ---------------- NPU shape args ---------------- #

@pytest.mark.parametrize("prompt,resp,want", [
    (None, None, None),
    (1024, None, (1024, 256)),      # document the raised defaults
    (None, 64, (16384, 64)),
    (2048, 128, (2048, 128)),
])
def test_npu_shape_from_args(prompt, resp, want):
    assert _npu_shape_from_args(argparse.Namespace(
        max_prompt_len=prompt, min_response_len=resp)) == want


# ---------------- CLI generation config ---------------- #

def test_build_generation_config_defaults():
    cfg = build_generation_config(argparse.Namespace(
        max_new_tokens=99, temperature=None, top_p=None, top_k=None,
        repetition_penalty=None, rng_seed=None, stop_tokens=None))
    assert cfg.max_new_tokens == 99
    assert not cfg.do_sample


def test_build_generation_config_maps_sampling():
    cfg = build_generation_config(argparse.Namespace(
        max_new_tokens=10, temperature=0.7, top_p=0.9, top_k=40,
        repetition_penalty=1.1, rng_seed=7, stop_tokens=["END", "\n"]))
    assert cfg.do_sample and cfg.temperature == pytest.approx(0.7)
    assert cfg.top_p == pytest.approx(0.9) and cfg.top_k == 40
    assert cfg.repetition_penalty == pytest.approx(1.1) and cfg.rng_seed == 7
    assert set(cfg.stop_strings) == {"END", "\n"}


def test_temperature_zero_disables_sampling():
    cfg = build_generation_config(argparse.Namespace(
        max_new_tokens=8, temperature=0.0, top_p=None, top_k=None,
        repetition_penalty=None, rng_seed=None, stop_tokens=None))
    assert not cfg.do_sample


def test_make_streamer(capsys):
    assert make_streamer(False) is None
    make_streamer(True)("tok")
    assert capsys.readouterr().out == "tok"


# ---------------- server request mapping ---------------- #

def test_build_config_defaults_and_max_tokens_fallbacks():
    cfg = build_config({})
    assert cfg.max_new_tokens == 512 and not cfg.do_sample
    assert build_config({"max_tokens": 0}).max_new_tokens == 512
    assert build_config({"max_completion_tokens": 33}).max_new_tokens == 33


def test_build_config_maps_openai_params():
    cfg = build_config({"temperature": 0.7, "top_p": 0.9, "top_k": 5,
                        "repetition_penalty": 1.1, "frequency_penalty": 0.3,
                        "presence_penalty": 0.2, "seed": 11,
                        "stop": ["END", "\n"], "max_tokens": 7})
    assert cfg.do_sample and cfg.temperature == pytest.approx(0.7)
    assert cfg.top_p == pytest.approx(0.9) and cfg.top_k == 5
    assert cfg.repetition_penalty == pytest.approx(1.1) and cfg.rng_seed == 11
    assert set(cfg.stop_strings) == {"END", "\n"}
    assert cfg.max_new_tokens == 7


def test_build_config_temperature_zero_is_greedy():
    assert not build_config({"temperature": 0}).do_sample


def test_content_to_text_variants():
    assert _content_to_text("plain") == "plain"
    assert _content_to_text([{"type": "text", "text": "a"},
                             {"type": "image_url"}]) == "a"
    assert _content_to_text([]) == "" and _content_to_text(None) == ""


class _Perf:
    def __init__(self, n_in=0, n_out=0):
        self._in, self._out = n_in, n_out

    def get_num_input_tokens(self):
        return self._in

    def get_num_generated_tokens(self):
        return self._out


def test_usage():
    assert _usage(_Perf(3, 5)) == {"prompt_tokens": 3, "completion_tokens": 5,
                                   "total_tokens": 8}
    assert _usage(object()) == {"prompt_tokens": 0, "completion_tokens": 0,
                                "total_tokens": 0}


class _Result:
    def __init__(self, meta=None, n_out=0):
        self.meta = meta
        self.perf_metrics = _Perf(n_out=n_out)


def _cfg(max_new=8):
    cfg = ovgenai.GenerationConfig()
    cfg.max_new_tokens = max_new
    return cfg


def test_finish_reason():
    assert _finish_reason(_Result(meta=["LENGTH"]), _cfg()) == "length"
    assert _finish_reason(_Result(meta=[], n_out=8), _cfg(8)) == "length"
    assert _finish_reason(_Result(meta=[], n_out=3), _cfg(8)) == "stop"
    assert _finish_reason(_Result(n_out=3), _cfg(8)) == "stop"


def test_render_chat_flattens_parts_and_passes_context():
    class Rec:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, history, add_generation_prompt, extra_context=None):
            self.calls.append((history, add_generation_prompt, extra_context))
            return "RENDERED"

    tok = Rec()
    msgs = [{"role": "user",
             "content": [{"type": "text", "text": "hi"}, {"type": "image_url"}]}]
    assert render_chat(tok, msgs) == "RENDERED"
    history, add_gen, extra = tok.calls[0]
    assert history[0]["content"] == "hi" and add_gen is True and extra is None
    assert render_chat(tok, msgs, {"enable_thinking": False}) == "RENDERED"
    assert tok.calls[1][2] == {"enable_thinking": False}
