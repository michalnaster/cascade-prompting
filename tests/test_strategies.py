from __future__ import annotations

from unittest.mock import MagicMock

from cascade_prompting.schemas import (
    DraftLayer,
    Layer,
    RAGAnswerCascade,
    ReasonedAnswer,
    SimpleAnswer,
)
from cascade_prompting.strategies import (
    STRATEGIES,
    BaselineStrategy,
    CascadeStrategy,
    MultiCallFullContextStrategy,
    MultiCallStrategy,
)

from .helpers import ai_message


def test_strategy_registry_has_all_four_conditions():
    assert list(STRATEGIES.keys()) == [
        "baseline",
        "multi_call",
        "multi_call_full_context",
        "cascade",
    ]


def test_baseline_strategy_returns_single_layer():
    llm = MagicMock()
    raw = ai_message("", input_tokens=10, output_tokens=5)
    llm.with_structured_output.return_value.invoke.return_value = {
        "raw": raw,
        "parsed": SimpleAnswer(answer="Paris"),
    }

    out = BaselineStrategy().run(llm, "Q?", "ctx")

    assert out.final_text == "Paris"
    assert out.layers == ["Paris"]
    assert out.reasonings is None
    assert (out.input_tokens, out.output_tokens) == (10, 5)
    assert out.total_tokens == 15


def _structured_side_effect(*, input_tokens: int = 1, output_tokens: int = 1):
    """Three structured-output responses, one per multi_call invoke() —
    each a `ReasonedAnswer(reasoning=..., answer=...)`, matching what
    `with_structured_output(ReasonedAnswer, include_raw=True).invoke(...)`
    returns."""
    return [
        {
            "raw": ai_message("", input_tokens=input_tokens, output_tokens=output_tokens),
            "parsed": ReasonedAnswer(reasoning=f"r{i}", answer=text),
        }
        for i, text in enumerate(("draft", "checked", "final"), start=1)
    ]


def test_multi_call_strategy_chains_three_structured_calls():
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.side_effect = _structured_side_effect()

    out = MultiCallStrategy().run(llm, "Q?", "ctx")

    assert llm.with_structured_output.return_value.invoke.call_count == 3
    assert out.final_text == "final"
    assert out.layers == ["draft", "checked", "final"]
    assert out.reasonings == ["r1", "r2", "r3"]
    assert out.total_tokens == 6  # 3 calls * (1 in + 1 out)


def _refine_call_human_message_content(llm: MagicMock) -> str:
    # Third with_structured_output().invoke() call, positional messages
    # list, second message (the HumanMessage — the first is the refine
    # SystemMessage).
    call_args_list = llm.with_structured_output.return_value.invoke.call_args_list
    messages = call_args_list[2][0][0]
    return messages[1].content


def test_multi_call_strategy_refine_call_has_no_documents():
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.side_effect = _structured_side_effect()

    MultiCallStrategy().run(llm, "Q?", "the source documents go here")

    assert "the source documents go here" not in _refine_call_human_message_content(llm)


def test_multi_call_full_context_strategy_refine_call_includes_documents():
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.side_effect = _structured_side_effect()

    MultiCallFullContextStrategy().run(llm, "Q?", "the source documents go here")

    assert "the source documents go here" in _refine_call_human_message_content(llm)


def test_cascade_strategy_carries_reasoning_per_layer():
    llm = MagicMock()
    parsed = RAGAnswerCascade(
        draft_answer=DraftLayer(reasoning="r1", data="d1"),
        fact_checked_answer=Layer(reasoning="r2", data="d2"),
        final_answer=Layer(reasoning="r3", data="d3"),
    )
    llm.with_structured_output.return_value.invoke.return_value = {
        "raw": ai_message("", input_tokens=20, output_tokens=30),
        "parsed": parsed,
    }

    out = CascadeStrategy().run(llm, "Q?", "ctx")

    assert out.final_text == "d3"
    assert out.layers == ["d1", "d2", "d3"]
    assert out.reasonings == ["r1", "r2", "r3"]
