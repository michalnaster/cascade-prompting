from __future__ import annotations

from unittest.mock import MagicMock

import pytest

opik = pytest.importorskip("opik")  # optional dependency — see pyproject.toml's "observability" extra

from cascade_prompting.observability import traced_run  # noqa: E402
from cascade_prompting.schemas import SimpleAnswer  # noqa: E402
from cascade_prompting.strategies.baseline import BaselineStrategy  # noqa: E402

from .helpers import ai_message  # noqa: E402


def _mock_llm(answer: str = "Paris") -> MagicMock:
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = {
        "raw": ai_message("", input_tokens=5, output_tokens=2),
        "parsed": SimpleAnswer(answer=answer),
    }
    return llm


def test_traced_run_returns_the_real_result_untouched():
    llm = _mock_llm("Paris")

    out = traced_run(
        BaselineStrategy(), llm, "capital of France?", "ctx", example_idx=0
    )

    assert out.final_text == "Paris"
    assert (out.input_tokens, out.output_tokens) == (5, 2)


def test_traced_run_attaches_a_trace_id_and_url():
    llm = _mock_llm("Paris")

    out = traced_run(
        BaselineStrategy(), llm, "capital of France?", "ctx", example_idx=0
    )

    assert out.opik_trace_id
    assert out.opik_trace_url
    assert out.opik_trace_id in out.opik_trace_url


def test_plain_strategy_run_leaves_trace_fields_none():
    # The untraced path (opik_tracing=False, the default) — no opik.track
    # involved at all — must not silently populate these.
    out = BaselineStrategy().run(_mock_llm("Paris"), "capital of France?", "ctx")

    assert out.opik_trace_id is None
    assert out.opik_trace_url is None
