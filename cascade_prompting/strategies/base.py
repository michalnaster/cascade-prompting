"""GenerationStrategy: the Strategy-pattern abstraction over the four
conditions (baseline / multi_call / multi_call_full_context / cascade).

`cascade_prompting.benchmark.run_generation_phase` iterates
`cascade_prompting.strategies.registry.STRATEGIES` and calls `.run()`
polymorphically — it never branches on condition name. Adding a fifth
condition is: write a new `GenerationStrategy` subclass, register it in
`registry.py`. Nothing in `benchmark.py` changes (Open/Closed Principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar, cast

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from ..schemas import GenerationOutput

T = TypeVar("T", bound=BaseModel)


class GenerationStrategy(ABC):
    #: Registry key / CSV "condition" column value, e.g. "baseline".
    name: str

    @abstractmethod
    def run(
        self,
        llm: BaseChatModel,
        question: str,
        context: str,
        callbacks: list[BaseCallbackHandler] | None = None,
    ) -> GenerationOutput:
        """`callbacks` is optional and unused unless observability is
        enabled (see `cascade_prompting.observability`) — passed straight
        through to every underlying `invoke_structured` call so each
        individual LLM call (one for baseline/cascade, three for
        multi_call/multi_call_full_context) can be traced separately
        instead of only as one summed latency/token figure per example."""
        raise NotImplementedError


def extract_usage(msg: AIMessage) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from a LangChain AIMessage.
    Populated from Ollama's prompt_eval_count / eval_count fields for
    ChatOllama, and from the API's `usage` block for ChatAnthropic. If they
    come back as 0, check your provider's LangChain integration version —
    usage_metadata support was added in later releases."""
    meta = getattr(msg, "usage_metadata", None) or {}
    return meta.get("input_tokens", 0), meta.get("output_tokens", 0)


def invoke_structured(
    llm: BaseChatModel,
    schema: type[T],
    messages: list[BaseMessage],
    callbacks: list[BaseCallbackHandler] | None = None,
    run_name: str | None = None,
) -> tuple[AIMessage, T]:
    """Shared call shape for every strategy — every condition now uses
    structured output, `multi_call`/`multi_call_full_context` per call and
    `baseline`/`cascade` per call/layer: call
    `with_structured_output(schema, include_raw=True).invoke(messages)` and
    return (raw_message, parsed) with concrete types instead of the loosely
    typed `dict[str, Any] | BaseModel` LangChain's overloads resolve to.

    `callbacks`, when given, is forwarded as this one call's LangChain run
    config — this is the hook `cascade_prompting.observability` uses to
    attach an Opik tracer per LLM call rather than per example. `run_name`
    (e.g. "draft" / "fact_check" / "refine") is forwarded alongside it so
    that tracer can label each of a multi-call strategy's three calls
    distinctly instead of three identically-named spans.
    """
    structured = llm.with_structured_output(schema, include_raw=True)
    config: dict[str, Any] | None = None
    if callbacks:
        config = {"callbacks": callbacks}
        if run_name:
            config["run_name"] = run_name
    result = cast(dict[str, Any], structured.invoke(messages, config=config))
    return cast(AIMessage, result["raw"]), cast(T, result["parsed"])
