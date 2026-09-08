"""HotpotQA (distractor split) loading.

Each example is a dict: {"question": str, "answer": str,
"context_sentences": list[str]} — the shape every generation strategy and
metric in this package consumes.
"""

from __future__ import annotations


def load_hotpotqa_sample(n: int) -> list[dict]:
    """Load n examples from HotpotQA (distractor, validation split).

    Each example has: question, answer, context_sentences (list[str]).
    """
    from datasets import load_dataset

    # The legacy script-based "hotpot_qa" dataset id was removed; newer
    # `datasets` versions require the Hub repo id "hotpotqa/hotpot_qa".
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=f"validation[:{n}]")
    examples = []
    for row in ds:
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        flat_sentences = [
            f"[{t}] {s}"
            for t, sents in zip(titles, sentences, strict=True)
            for s in sents
        ]
        examples.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "context_sentences": flat_sentences,
            }
        )
    return examples


def render_context(context_sentences: list[str]) -> str:
    return "\n".join(context_sentences)
