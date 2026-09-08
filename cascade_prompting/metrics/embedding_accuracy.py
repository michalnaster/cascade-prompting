"""`answer_accuracy`: a fast, no-LLM-call proxy for correctness — cosine
similarity between a local sentence-embedding of the predicted answer and
the gold answer.

This is the always-on counterpart to RAGAS's `nv_answer_accuracy` (see
`ragas_scoring.py`), which is an LLM-judge metric — two extra model calls
per example, gated behind `--ragas` because of that cost. This one has no
such cost: a small sentence-transformers model runs on CPU in milliseconds,
so it's cheap enough to compute for every example in every run,
unconditionally, rather than only on whatever subsample an LLM judge is
affordable for. The two are named differently on purpose (`answer_accuracy`
here vs. `nv_answer_accuracy` for RAGAS's) so that merging RAGAS's columns
into the results DataFrame in `cli.py` can't silently collide with or
overwrite this one.

It's also a cruder signal than an LLM judge. Cosine similarity between two
short answer embeddings can't reliably distinguish "the right answer,
differently phrased" from "a wrong but topically-similar answer" the way a
judge reading both against the question can — two named entities from the
same domain (two cities, two people, two dates) often sit close together in
embedding space regardless of which one is actually correct. Treat this as a
fast, always-on sanity check across the full sample, not a substitute for
`nv_answer_accuracy` on however much of the sample an LLM judge is
affordable for.
"""

from __future__ import annotations

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model_cache: dict[str, object] = {}


def _get_model(model_name: str):
    if model_name not in _model_cache:
        # Imported here, not at module top: sentence-transformers pulls in
        # torch, which is slow to import and unnecessary for anything that
        # doesn't actually call this function (argparse/config-only paths,
        # most unit tests, any run that never reaches the generation phase).
        from sentence_transformers import SentenceTransformer

        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def answer_accuracy(prediction: str, gold: str, model_name: str = DEFAULT_MODEL) -> float:
    """Cosine similarity between `prediction` and `gold`'s sentence
    embeddings — mathematically in [-1, 1], in practice close to [0, 1] for
    short factual answers embedded by a model trained on semantic
    similarity."""
    model = _get_model(model_name)
    embeddings = model.encode([prediction, gold], normalize_embeddings=True)
    # Both vectors are unit-normalized, so the dot product is the cosine
    # similarity directly.
    return float(embeddings[0] @ embeddings[1])
