"""RAGAS-based quality metrics: faithfulness, answer_relevancy, nv_answer_accuracy.

hotpot_em/f1 (see `cascade_prompting.metrics.hotpot`) remain the string-match
metrics used to compare against published HotpotQA results; the metrics here
cover what a string comparison can't, since they only look at the answer
text: faithfulness/answer_relevancy score groundedness and on-topic-ness,
and nv_answer_accuracy is an LLM-judge comparison against the gold answer
(RAGAS's AnswerAccuracy / nv_accuracy metric, named `nv_answer_accuracy`
here to avoid colliding with the always-on local-embedding
`answer_accuracy` column from `metrics.embedding_accuracy` — see that
module for why the two are named differently rather than one overwriting
the other) that can credit a correct answer phrased differently from the
gold string.
"""

from __future__ import annotations

import sys
import types

import pandas as pd

# --- Work around a known ragas bug ------------------------------------------
# ragas/llms/base.py unconditionally imports ChatVertexAI from
# langchain_community.chat_models.vertexai. That path was removed from
# recent langchain-community releases (ChatVertexAI moved to the separate
# langchain-google-vertexai package), so `import ragas` raises:
#   ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
# This happens purely at import time, regardless of which LLM provider you
# actually use. See: https://github.com/vibrantlabsai/ragas/issues/2745
#
# Since this project never uses Vertex AI, we stub the module out rather than
# pin exact ragas/langchain-community versions against each other (which is
# fragile and breaks again on the next `uv pip install --upgrade`). This must
# run before the first `import ragas` anywhere in the process, which is why
# it lives at the top of this module rather than in cli.py.
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class _ChatVertexAIStub:  # never instantiated; import target only
        pass

    _stub.ChatVertexAI = _ChatVertexAIStub  # type: ignore[attr-defined]  # deliberate monkeypatch
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
# -----------------------------------------------------------------------------

from datasets import Dataset
from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.metrics import AnswerAccuracy, answer_relevancy, faithfulness
from ragas.run_config import RunConfig

# RAGAS's nv_accuracy LLM-judge metric. Named nv_answer_accuracy, not
# answer_accuracy, so it can't collide with the always-on local-embedding
# proxy of the same conceptual metric (metrics.embedding_accuracy) when the
# two get merged into the same DataFrame in cli.py.
nv_answer_accuracy = AnswerAccuracy(name="nv_answer_accuracy")

RAGAS_METRICS = [faithfulness, answer_relevancy, nv_answer_accuracy]
RAGAS_METRIC_NAMES = [m.name for m in RAGAS_METRICS]


def score_with_ragas(
    df: pd.DataFrame,
    ragas_llm,
    ragas_embeddings,
    run_config: RunConfig | None = None,
) -> pd.DataFrame:
    """Run RAGAS (faithfulness, answer_relevancy, nv_answer_accuracy) once per
    condition, batched, then merge the scores back onto df by example_idx.

    `run_config` controls RAGAS's internal concurrency/timeout. RAGAS's default
    (max_workers=16, timeout=180s) assumes a provider that can genuinely serve
    16 requests in parallel. A single local Ollama server can't — it queues
    requests roughly serially — so the default settings produce a wall of
    'Exception raised in Job[N]: TimeoutError()' as most of those 16 'parallel'
    requests sit in Ollama's queue past the 180s timeout. If not provided here,
    a conservative default suited to a single local Ollama instance is used.
    """
    if df.empty or "condition" not in df.columns:
        raise ValueError(
            "score_with_ragas received an empty or malformed DataFrame — this means "
            "run_generation_phase produced no rows. Check the generation-phase output "
            "above for tracebacks before this point."
        )
    if run_config is None:
        run_config = RunConfig(max_workers=1, timeout=300)

    scored_frames = []

    for cond_name, cond_df in df.groupby("condition"):
        ragas_dataset = Dataset.from_dict(
            {
                "question": cond_df["question"].tolist(),
                "answer": cond_df["predicted_answer"].tolist(),
                "contexts": cond_df["context_sentences"].tolist(),
                "ground_truth": cond_df["gold_answer"].tolist(),
            }
        )
        result = evaluate(
            ragas_dataset,
            metrics=RAGAS_METRICS,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_config,
            raise_exceptions=False,  # log NaN per-sample instead of aborting the whole run
        )
        # evaluate() returns Executor instead when return_executor=True, which
        # we never pass — this narrows the type and guards that assumption.
        assert isinstance(result, EvaluationResult)
        scores_df = result.to_pandas()
        scores_df["example_idx"] = cond_df["example_idx"].tolist()
        scores_df["condition"] = cond_name
        scored_frames.append(
            scores_df[["example_idx", "condition", *RAGAS_METRIC_NAMES]]
        )

    ragas_scores = pd.concat(scored_frames, ignore_index=True)
    return df.merge(ragas_scores, on=["example_idx", "condition"], how="left")
