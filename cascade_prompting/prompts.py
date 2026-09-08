"""Shared instruction text.

These are the SAME instructions used across baseline / multi_call /
multi_call_full_context / cascade. The point of the benchmark is to isolate
call *structure* (one naive call vs. three sequential calls vs. one
structured call) — and, between the two multi_call variants, how much
context each call gets — as the variables being tested; if each condition
also got differently worded instructions, a quality gap could just be an
artifact of prompt wording rather than the structural difference the
benchmark is meant to isolate. So every strategy in
`cascade_prompting.strategies` pulls from these same strings.
"""

from __future__ import annotations

DRAFT_INSTRUCTION = "Answer the question using the provided documents."

FACT_CHECK_INSTRUCTION = (
    "Check the answer against the documents, sentence by sentence. Remove or "
    "correct any claim not directly supported by the documents."
)

REFINE_INSTRUCTION = (
    "Tighten the answer to the shortest phrase that directly answers the "
    "question — a name, date, place, number, or yes/no, not a full "
    "sentence. Drop any restatement of the question or supporting "
    "explanation (that belongs in `reasoning`, not the answer), and don't "
    "add new claims."
)

# General on purpose: a "what changed" framing would break for the draft
# layer, which has no previous layer to point to.
REASONING_INSTRUCTION = "Explain your reasoning before giving the answer."

# For the draft layer/call only — no previous layer to carry forward or
# check against, so it gets its own instruction rather than reusing
# LAYER_DATA_INSTRUCTION's "copy the previous layer" language. Points back
# at this layer's own instruction above (not at `reasoning`) since that
# instruction, not the reasoning field, is what actually defines the task.
DRAFT_DATA_INSTRUCTION = "The answer itself, produced by this layer's instruction."

LAYER_DATA_INSTRUCTION = (
    "The answer produced by this layer's instruction, checked "
    "against this layer's own goal, not an earlier layer's. If nothing "
    "needs to change, copy the previous layer's answer as-is."
)

CASCADE_SYSTEM_PROMPT = (
    "Produce a grounded, fact-checked answer to a question from retrieved "
    "documents, in three layers, each conditioned on the layer before it, "
    "each judged only against its own goal below — not an earlier layer's:\n"
    f"1. {DRAFT_INSTRUCTION}\n"
    f"2. {FACT_CHECK_INSTRUCTION}\n"
    f"3. {REFINE_INSTRUCTION}"
)


def build_documents_message(context: str, question: str) -> str:
    """The same question+documents framing used to open every condition."""
    return f"Documents:\n{context}\n\nQuestion: {question}"
