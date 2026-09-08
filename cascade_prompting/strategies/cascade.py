from __future__ import annotations

import time

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..prompts import CASCADE_SYSTEM_PROMPT, build_documents_message
from ..schemas import GenerationOutput, RAGAnswerCascade
from .base import GenerationStrategy, extract_usage, invoke_structured


class CascadeStrategy(GenerationStrategy):
    """One call, structured output with three ordered layers (draft ->
    fact_checked -> final), each layer conditioning on the previous layer's
    content via schema field order, with a `reasoning` field before `data`
    in every layer."""

    name = "cascade"

    def run(
        self,
        llm: BaseChatModel,
        question: str,
        context: str,
        callbacks: list[BaseCallbackHandler] | None = None,
    ) -> GenerationOutput:
        t0 = time.perf_counter()
        raw, cascade = invoke_structured(
            llm,
            RAGAnswerCascade,
            [
                SystemMessage(content=CASCADE_SYSTEM_PROMPT),
                HumanMessage(content=build_documents_message(context, question)),
            ],
            callbacks=callbacks,
            run_name="cascade",
        )
        latency = time.perf_counter() - t0
        in_tok, out_tok = extract_usage(raw)
        return GenerationOutput(
            final_text=cascade.final_answer.data,
            latency_s=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            layers=[
                cascade.draft_answer.data,
                cascade.fact_checked_answer.data,
                cascade.final_answer.data,
            ],
            reasonings=[
                cascade.draft_answer.reasoning,
                cascade.fact_checked_answer.reasoning,
                cascade.final_answer.reasoning,
            ],
        )
