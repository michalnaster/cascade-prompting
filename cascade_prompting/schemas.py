"""Pydantic structured-output schemas and the internal generation-result type.

Keeping these in one module (rather than scattering schema definitions across
each strategy file) is what lets `cascade_prompting.metrics.diagnostics` and
`cascade_prompting.benchmark` depend on a single stable shape
(`GenerationOutput`) regardless of which condition produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .prompts import (
    DRAFT_DATA_INSTRUCTION,
    DRAFT_INSTRUCTION,
    FACT_CHECK_INSTRUCTION,
    LAYER_DATA_INSTRUCTION,
    REASONING_INSTRUCTION,
    REFINE_INSTRUCTION,
)


class Layer(BaseModel):
    """One field-pair of `cascade`'s single schema — `data` names it
    correctly there, since it's literally one field threading through a
    bigger structure alongside sibling layers. `multi_call`'s three calls
    aren't layers of one schema, each is its own standalone call, so they
    use `ReasonedAnswer` (below) instead — same reasoning-then-answer
    discipline, field named for what it actually is in that context.

    Used for `fact_checked_answer` and `final_answer`. `draft_answer` uses
    `DraftLayer` instead — same fields, kept as a separate class only so
    the draft layer/call (which has no previous layer) isn't structurally
    conflated with the two that do."""

    reasoning: str = Field(description=REASONING_INSTRUCTION)
    data: str = Field(description=LAYER_DATA_INSTRUCTION)


class DraftLayer(BaseModel):
    """Same field-pair as `Layer`, for the one layer/call that has no
    previous step: `cascade`'s `draft_answer` and `multi_call`'s draft
    call. `REASONING_INSTRUCTION` is generic ("explain your reasoning
    before giving the answer") precisely so it doesn't need a
    draft-specific variant — what counts as good reasoning for this step is
    already set by `DRAFT_INSTRUCTION`/`FACT_CHECK_INSTRUCTION`/
    `REFINE_INSTRUCTION`, not by the reasoning field's own instruction.
    `data` does need a draft-specific variant, `DRAFT_DATA_INSTRUCTION`:
    `LAYER_DATA_INSTRUCTION`'s "copy the previous layer's answer" only
    makes sense once a previous layer exists."""

    reasoning: str = Field(description=REASONING_INSTRUCTION)
    data: str = Field(description=DRAFT_DATA_INSTRUCTION)


class RAGAnswerCascade(BaseModel):
    """Single structured-output call producing a grounded, refined answer."""

    draft_answer: DraftLayer = Field(description=DRAFT_INSTRUCTION)
    fact_checked_answer: Layer = Field(description=FACT_CHECK_INSTRUCTION)
    final_answer: Layer = Field(description=REFINE_INSTRUCTION)


class SimpleAnswer(BaseModel):
    """Structured output for the naive single-call baseline."""

    answer: str = Field(description=DRAFT_INSTRUCTION)


class ReasonedAnswer(BaseModel):
    """Structured output for `multi_call`'s fact-check and refine calls:
    the same reasoning-then-answer discipline as `cascade`'s `Layer`, using
    the same shared instruction text, but as its own schema rather than a
    reuse of `Layer` — each `multi_call` call is a complete, independent
    response, not one field in a larger structure, so `answer` (not `data`)
    is what that field actually is here."""

    reasoning: str = Field(description=REASONING_INSTRUCTION)
    answer: str = Field(description=LAYER_DATA_INSTRUCTION)


class DraftReasonedAnswer(BaseModel):
    """`multi_call`'s draft call: same shape as `ReasonedAnswer`, but for
    the one call with no previous call to point to — mirrors `DraftLayer`
    for the same reason, including using `DRAFT_DATA_INSTRUCTION` instead
    of `LAYER_DATA_INSTRUCTION` for `answer`. Keeps `multi_call`'s draft
    call asking the same question `cascade`'s draft layer asks, so a
    difference in candidate handling between the two conditions isn't just
    an artifact of one of them getting an instruction that doesn't apply to
    it."""

    reasoning: str = Field(description=REASONING_INSTRUCTION)
    answer: str = Field(description=DRAFT_DATA_INSTRUCTION)


@dataclass
class GenerationOutput:
    """Normalized result of running any GenerationStrategy on one example.

    Every strategy (baseline / multi_call / multi_call_full_context /
    cascade) returns this same shape, so `cascade_prompting.benchmark` and
    the metrics modules can treat conditions polymorphically instead of
    branching on condition name.
    """

    final_text: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    layers: list[str]
    reasonings: list[str] | None = field(default=None)
    #: Populated by `cascade_prompting.observability.traced_run` when
    #: `--opik` is passed; None otherwise. Carried through to the CSV row
    #: so every example/condition can be jumped to directly in the Opik UI
    #: instead of only having a summed latency/token number to go on.
    opik_trace_id: str | None = field(default=None)
    opik_trace_url: str | None = field(default=None)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
