from __future__ import annotations

import time

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..prompts import (
    DRAFT_INSTRUCTION,
    FACT_CHECK_INSTRUCTION,
    REFINE_INSTRUCTION,
    build_documents_message,
)
from ..schemas import DraftReasonedAnswer, GenerationOutput, ReasonedAnswer
from .base import GenerationStrategy, extract_usage, invoke_structured


def _run_multi_call(
    llm: BaseChatModel,
    question: str,
    context: str,
    *,
    refine_has_documents: bool,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> GenerationOutput:
    """Three sequential LLM calls (draft -> fact-check -> refine), each call
    re-sending its own system prompt / context and each producing a
    structured `ReasonedAnswer` (`reasoning` then `answer`) — the same
    reasoning-then-answer discipline `cascade` uses per layer, via the same
    shared `REASONING_INSTRUCTION` / `LAYER_DATA_INSTRUCTION` text, so
    a quality gap against `cascade` isn't explainable by cascade being the
    only condition asked to justify a change before making it.

    `refine_has_documents` is the one thing that varies between
    `MultiCallStrategy` and `MultiCallFullContextStrategy` — see their
    docstrings for why both exist rather than just fixing the one and
    discarding the other.
    """
    t0 = time.perf_counter()
    in_tok = out_tok = 0

    # Call 1: draft — same DRAFT_INSTRUCTION as baseline and cascade, and
    # DraftReasonedAnswer's reasoning instruction matches cascade's
    # DraftLayer: no previous layer to point to, so ask about candidates
    # instead of a change that doesn't exist yet.
    raw1, draft_result = invoke_structured(
        llm,
        DraftReasonedAnswer,
        [
            SystemMessage(content=DRAFT_INSTRUCTION),
            HumanMessage(content=build_documents_message(context, question)),
        ],
        callbacks=callbacks,
        run_name="draft",
    )
    i, o = extract_usage(raw1)
    in_tok += i
    out_tok += o
    draft = draft_result.answer

    # Call 2: fact-check — same FACT_CHECK_INSTRUCTION as cascade's fact_checked_answer
    raw2, fact_checked_result = invoke_structured(
        llm,
        ReasonedAnswer,
        [
            SystemMessage(content=FACT_CHECK_INSTRUCTION),
            HumanMessage(content=f"Documents:\n{context}\n\nAnswer:\n{draft}"),
        ],
        callbacks=callbacks,
        run_name="fact_check",
    )
    i, o = extract_usage(raw2)
    in_tok += i
    out_tok += o
    fact_checked = fact_checked_result.answer

    # Call 3: refine — same REFINE_INSTRUCTION as cascade's final_answer.
    refine_content = (
        f"Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:\n{fact_checked}"
        if refine_has_documents
        else f"Question: {question}\n\nAnswer:\n{fact_checked}"
    )
    raw3, final_result = invoke_structured(
        llm,
        ReasonedAnswer,
        [
            SystemMessage(content=REFINE_INSTRUCTION),
            HumanMessage(content=refine_content),
        ],
        callbacks=callbacks,
        run_name="refine",
    )
    i, o = extract_usage(raw3)
    in_tok += i
    out_tok += o
    final = final_result.answer

    latency = time.perf_counter() - t0
    return GenerationOutput(
        final_text=final,
        latency_s=latency,
        input_tokens=in_tok,
        output_tokens=out_tok,
        layers=[draft, fact_checked, final],
        reasonings=[draft_result.reasoning, fact_checked_result.reasoning, final_result.reasoning],
    )


class MultiCallStrategy(GenerationStrategy):
    """Three sequential LLM calls (draft -> fact-check -> refine), each one
    a structured-output call producing a `reasoning` + `answer` pair — the
    same reasoning-then-answer discipline `cascade` uses per layer, just as
    three separate calls instead of three fields in one call. The refine
    call is prompted with only the question and the fact-checked answer —
    no documents — because its instruction is "tighten the wording," which
    looks, on its face, like it shouldn't need them.

    This is the realistic default shape of a hand-rolled multi-call
    pipeline: later calls get scoped down to just what their instruction
    seems to require, to save tokens. It's also, as the worked example in
    the article built from this benchmark shows, how that scoping can
    silently cost you: a refine call with no documents in its context has
    no way to catch itself if "tightening" the wording also changes what
    the answer claims. See `MultiCallFullContextStrategy` for the
    same pipeline without that gap.
    """

    name = "multi_call"

    def run(
        self,
        llm: BaseChatModel,
        question: str,
        context: str,
        callbacks: list[BaseCallbackHandler] | None = None,
    ) -> GenerationOutput:
        return _run_multi_call(
            llm, question, context, refine_has_documents=False, callbacks=callbacks
        )


class MultiCallFullContextStrategy(GenerationStrategy):
    """Identical to `MultiCallStrategy`, except the refine call also
    receives the source documents, matching the grounding every layer of
    `cascade` always has by virtue of never leaving the original context
    window.

    Kept as a separate, named condition rather than replacing
    `MultiCallStrategy` outright: the two isolate different things. Compare
    `cascade` against `multi_call` to measure what splitting into three
    calls costs when each call is scoped the way a hand-rolled pipeline
    typically would be; compare `cascade` against `multi_call_full_context`
    to measure the same thing with the documents-in-every-call confound
    removed, so a gap that shows up there is about call count and context
    continuity, not about one pipeline being written more carefully than
    the other.
    """

    name = "multi_call_full_context"

    def run(
        self,
        llm: BaseChatModel,
        question: str,
        context: str,
        callbacks: list[BaseCallbackHandler] | None = None,
    ) -> GenerationOutput:
        return _run_multi_call(
            llm, question, context, refine_has_documents=True, callbacks=callbacks
        )
