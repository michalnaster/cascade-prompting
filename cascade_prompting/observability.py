"""Optional Opik tracing for the generation phase.

Without this, `run_generation_phase` records exactly one summed
latency/token figure per (example, condition) row — which is what made the
`multi_call` vs. `multi_call_full_context` latency anomaly (one of
`multi_call`'s three calls costing several times its `multi_call_full_context`
counterpart on an *identical* document set, on a small fraction of examples)
take manual CSV surgery to even notice, and impossible to say from the CSV
alone which of the three calls was responsible.

With `--opik`, every `GenerationStrategy.run()` call becomes one Opik trace
(tagged with the condition name and example index), and every individual LLM
call inside it — one for `baseline`/`cascade`, three (`draft` / `fact_check`
/ `refine`) for `multi_call`/`multi_call_full_context` — becomes its own
nested span with latency and token usage captured automatically by Opik's
LangChain callback integration. That's the granularity needed to catch the
next version of the same anomaly directly in the Opik UI, filtered by
condition and run name, instead of by hand.

Gated behind `--opik` (default off) and imported lazily, never at module
top: Opik needs its own backend configured (a Comet-hosted project or a
self-hosted server, via `opik configure` / the `OPIK_API_KEY` /
`OPIK_URL_OVERRIDE` env vars) that most runs of this benchmark won't have,
so nothing in this module should be imported, and no `opik` dependency
should be required, unless tracing is actually requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

    from .schemas import GenerationOutput
    from .strategies.base import GenerationStrategy


def traced_run(
    strategy: GenerationStrategy,
    llm: BaseChatModel,
    question: str,
    context: str,
    *,
    example_idx: int,
    project_name: str = "cascade-prompting",
) -> GenerationOutput:
    """Same result as `strategy.run(llm, question, context)`, wrapped in an
    Opik trace per (example, condition) pair with a fresh `OpikTracer`
    passed down as a LangChain callback, so every underlying LLM call
    `strategy.run` makes (via `invoke_structured`'s `callbacks`/`run_name`
    params) is captured as its own nested, named span rather than only
    contributing to one summed number on the trace.

    The returned `GenerationOutput` carries that trace's id and a direct
    UI link (`opik_trace_id` / `opik_trace_url`) — `benchmark.py` puts both
    in every CSV row, so a row that looks anomalous (like the multi_call
    latency spike this was built to catch) can be opened straight in Opik
    instead of reconstructed by hand from the summed CSV columns.
    """
    import dataclasses

    import opik
    from opik import opik_context
    from opik.config import OpikConfig
    from opik.integrations.langchain import OpikTracer
    from opik.url_helpers import get_project_url_by_trace_id

    @opik.track(name=strategy.name, project_name=project_name)
    def _traced() -> GenerationOutput:
        opik_context.update_current_trace(
            tags=[strategy.name],
            metadata={"example_idx": example_idx, "question": question},
        )
        tracer = OpikTracer(tags=[strategy.name], project_name=project_name)
        out = strategy.run(llm, question, context, callbacks=[tracer])

        trace_id = opik_context.get_current_trace_data().id
        trace_url = get_project_url_by_trace_id(trace_id, OpikConfig().url_override)
        return dataclasses.replace(out, opik_trace_id=trace_id, opik_trace_url=trace_url)

    return _traced()
