from __future__ import annotations

import time

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..prompts import DRAFT_INSTRUCTION, build_documents_message
from ..schemas import GenerationOutput, SimpleAnswer
from .base import GenerationStrategy, extract_usage, invoke_structured


class BaselineStrategy(GenerationStrategy):
    """One call, no structure ("answer the question")."""

    name = "baseline"

    def run(
        self,
        llm: BaseChatModel,
        question: str,
        context: str,
        callbacks: list[BaseCallbackHandler] | None = None,
    ) -> GenerationOutput:
        t0 = time.perf_counter()
        raw, answer = invoke_structured(
            llm,
            SimpleAnswer,
            [
                SystemMessage(content=DRAFT_INSTRUCTION),
                HumanMessage(content=build_documents_message(context, question)),
            ],
            callbacks=callbacks,
            run_name="baseline",
        )
        latency = time.perf_counter() - t0
        in_tok, out_tok = extract_usage(raw)
        return GenerationOutput(
            final_text=answer.answer,
            latency_s=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            layers=[answer.answer],
        )
