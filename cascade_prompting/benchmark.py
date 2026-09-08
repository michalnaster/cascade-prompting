"""Benchmark orchestration: run every strategy over every example, and
summarize the resulting per-example metrics into per-condition means.

This module depends only on the `GenerationStrategy` interface (via the
`STRATEGIES` registry) and the metrics modules — it has no knowledge of any
concrete condition or LLM provider, which is what keeps adding a strategy or
a provider from requiring changes here.
"""

from __future__ import annotations

import os
from ast import literal_eval

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from tqdm import tqdm

from .data import render_context
from .logging_setup import logger
from .metrics.diagnostics import edit_distance_layers, reasoning_consistency
from .metrics.embedding_accuracy import answer_accuracy
from .metrics.hotpot import hotpot_answer_metrics
from .metrics.significance import compare_all_conditions
from .strategies import STRATEGIES

#: Superset of layer names any strategy can produce, in pipeline order.
#: `GenerationOutput.layers`/`.reasonings` for a given strategy is a prefix
#: of this (e.g. baseline only ever produces "draft").
_LAYER_NAMES = ("draft", "fact_checked", "final")


def run_generation_phase(
    llm: BaseChatModel,
    examples: list[dict],
    out_path: str | None = None,
    significance_out: str | None = None,
    summary_out: str | None = None,
    resume: bool = True,
    opik_tracing: bool = False,
    opik_project: str = "cascade-prompting",
) -> pd.DataFrame:
    """Run every registered strategy on every example, collecting generation
    outputs, the official HotpotQA em/f1 metrics, and efficiency/diagnostic
    metrics. RAGAS scoring happens afterward in a separate batch pass per
    condition (see cascade_prompting.metrics.ragas_scoring.score_with_ragas),
    which is why `out_path`'s rows won't have RAGAS columns yet — the caller
    overwrites `out_path` with the RAGAS-merged frame once that phase completes.

    If `out_path` is given, each row is appended to it as soon as it's
    computed (instead of only ever hitting disk once the whole run finishes),
    so a crash or interrupt mid-run still leaves every example/condition
    result completed so far on disk.

    If `significance_out` is given, the paired significance tests (see
    `cascade_prompting.metrics.significance.compare_all_conditions`) are
    recomputed over every row collected so far, and the file overwritten
    with the result, after each example finishes — not once at the end in a
    dedicated phase. Unlike `out_path`'s rows, this can't be append-only:
    each write is a full recompute over the accumulated data, since a
    p-value isn't a per-row fact the way a generated answer is. It's cheap
    enough to redo every example regardless — the statistical tests
    themselves run in milliseconds; the LLM calls dominate the cost by
    orders of magnitude — and it doesn't need RAGAS to have run first,
    since none of `hotpot_em`/`latency_s`/`total_tokens` are RAGAS columns.

    If `summary_out` is given, the same thing happens with `summarize(...)`
    (per-condition means): recomputed over every row collected so far and
    rewritten after each example, for the same reason and at the same
    negligible cost. If `--ragas` scoring runs afterward, the caller
    overwrites `summary_out` once more with the RAGAS-merged means — until
    then, this is a live, always-current summary of the run in progress.

    `resume` defaults to True: if `out_path` already holds rows from a
    previous, interrupted run at that exact path, those (example_idx,
    question, condition) triples are loaded from disk and skipped — no LLM
    call, no re-append — instead of being generated again from scratch.
    This only makes the calls it still needs to make; everything already on
    disk is trusted as complete. The question text, not just the numeric
    index, is part of the match: `example_idx` alone is a *position* in
    `examples`, not a stable identity, so a row is only treated as already
    done if the question at that index in the current `examples` list is
    the exact same text as what's recorded on disk for that index. If it
    isn't — a different `--n`, a different dataset slice, anything that
    shifts what ends up at index N — that pair is regenerated rather than
    silently skipped as if it were the same question. Pass `resume=False`
    to force a fresh run that overwrites `out_path` from scratch instead.
    Either way, `out_path=None` or a not-yet-existing `out_path` behaves
    exactly like a fresh run — there's nothing to resume from.

    `opik_tracing`, when True, wraps every `strategy.run(...)` call below in
    an Opik trace (see `cascade_prompting.observability.traced_run`) so each
    individual LLM call inside it shows up as its own latency/token-tagged
    span instead of only contributing to this row's summed `latency_s` /
    `total_tokens`. Off by default and imported lazily, not at module top —
    it needs its own configured backend that most runs won't have.
    """
    rows: list[dict] = []
    completed: set[tuple[int, str, str]] = set()
    header_written = False

    if opik_tracing:
        from .observability import traced_run

    if resume and out_path is not None and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        existing = pd.read_csv(out_path)
        if not existing.empty:
            if "context_sentences" in existing.columns:
                # Stored as the stringified repr of a Python list (see the
                # row-building comment below); read back as a plain string
                # by pd.read_csv, so it has to be parsed back into a real
                # list to match what a freshly-generated row's value is —
                # RAGAS scoring (score_with_ragas) expects an actual list.
                existing["context_sentences"] = existing["context_sentences"].apply(literal_eval)
            rows = existing.to_dict("records")
            # Keyed by (example_idx, question, condition), not just
            # (example_idx, condition) — see the docstring above for why
            # the question text has to be part of the match.
            completed = {(int(r["example_idx"]), r["question"], r["condition"]) for r in rows}
            header_written = True
            logger.info(
                f"[resume] {len(completed)} (example, condition) pairs "
                f"already completed in {out_path!r} — skipping them."
            )

    if out_path is not None and not header_written:
        open(out_path, "w").close()  # fresh run starts with a clean file

    n_attempted = 0
    n_resumed = len(completed)
    for idx, ex in enumerate(tqdm(examples)):
        context = render_context(ex["context_sentences"])
        logger.debug(f"[{idx:03d}] question: {ex['question']!r}")
        logger.debug(f"[{idx:03d}] gold_answer: {ex['answer']!r}")
        for cond_name, strategy in STRATEGIES.items():
            if (idx, ex["question"], cond_name) in completed:
                continue  # already have this exact question/condition from a previous run
            n_attempted += 1
            try:
                if opik_tracing:
                    out = traced_run(
                        strategy,
                        llm,
                        ex["question"],
                        context,
                        example_idx=idx,
                        project_name=opik_project,
                    )
                else:
                    out = strategy.run(llm, ex["question"], context)
            except Exception:  # noqa: BLE001
                logger.exception(f"[{cond_name}] example {idx} FAILED")
                continue

            # Log every intermediate layer (and reasoning, where present) at
            # DEBUG so the file capture has the full trace, not just the
            # final answer and metrics that reach stdout below.
            layer_names = _LAYER_NAMES[: len(out.layers)]
            layer_data = dict(zip(layer_names, out.layers, strict=True))
            # Not strict: reasonings is [] (shorter than layer_names) for
            # conditions without a reasoning field.
            layer_reasoning = dict(zip(layer_names, out.reasonings or [], strict=False))
            for layer_name, layer_text in layer_data.items():
                logger.debug(
                    f"[{idx:03d}] {cond_name:<10} layer={layer_name:<12} "
                    f"data={layer_text!r}"
                )
            for layer_name, reasoning_text in layer_reasoning.items():
                logger.debug(
                    f"[{idx:03d}] {cond_name:<10} layer={layer_name:<12} "
                    f"reasoning={reasoning_text!r}"
                )

            row = {
                "example_idx": idx,
                "condition": cond_name,
                "question": ex["question"],
                "gold_answer": ex["answer"],
                "context_sentences": ex["context_sentences"],
                "predicted_answer": out.final_text,
                "latency_s": out.latency_s,
                "input_tokens": out.input_tokens,
                "output_tokens": out.output_tokens,
                "total_tokens": out.total_tokens,
                "edit_distance_layers": edit_distance_layers(out.layers),
                "answer_accuracy": answer_accuracy(out.final_text, ex["answer"]),
                **hotpot_answer_metrics(out.final_text, ex["answer"]),
                # Every intermediate layer's data/reasoning, interleaved per
                # layer (draft_data, draft_reasoning, fact_checked_data, ...),
                # one pair of columns per possible layer name — blank for
                # layers/fields a condition doesn't produce — so the full
                # trace is in the CSV itself and not only recoverable from
                # the DEBUG log.
                **{
                    key: value
                    for name in _LAYER_NAMES
                    for key, value in (
                        (f"layer_{name}_data", layer_data.get(name)),
                        (f"layer_{name}_reasoning", layer_reasoning.get(name)),
                    )
                },
                # Always present (None when n/a), never conditional: every
                # row from a single run must carry the same set of columns,
                # since out_path rows are appended to disk one at a time as
                # they're computed (see run_generation_phase's docstring) —
                # a key present on some rows but not others would misalign
                # the CSV the moment the two kinds of rows interleave.
                "reasoning_consistency": (
                    reasoning_consistency(out.reasonings, out.layers)
                    if out.reasonings
                    else None
                ),
                # None unless --opik was passed (see traced_run) — always
                # present regardless, for the same column-alignment reason
                # as reasoning_consistency above.
                "opik_trace_id": out.opik_trace_id,
                "opik_trace_url": out.opik_trace_url,
            }
            rows.append(row)
            if out_path is not None:
                pd.DataFrame([row]).to_csv(
                    out_path, mode="a", header=not header_written, index=False
                )
                header_written = True
            logger.info(
                f"[{idx:03d}] {cond_name:<10} EM={row['hotpot_em']:.0f} "
                f"F1={row['hotpot_f1']:.2f} lat={row['latency_s']:.2f}s "
                f"tok={row['total_tokens']}"
            )

        if rows:
            if significance_out is not None:
                compare_all_conditions(pd.DataFrame(rows)).to_csv(significance_out, index=False)
            if summary_out is not None:
                summarize(pd.DataFrame(rows)).to_csv(summary_out)

    n_ok = len(rows)
    if n_resumed:
        logger.info(
            f"\n[generation] {n_ok} total rows ({n_resumed} resumed from a "
            f"previous run, {n_ok - n_resumed}/{n_attempted} newly attempted "
            "calls succeeded this run)."
        )
    else:
        logger.info(f"\n[generation] {n_ok}/{n_attempted} calls succeeded.")
    if n_ok == 0:
        raise RuntimeError(
            "All generation calls failed — see the tracebacks logged above for the "
            "actual cause (e.g. the model not supporting structured/tool-calling "
            "output, an Ollama model not being pulled, the Ollama server being "
            "unreachable, or a missing/invalid Anthropic API key). Fix that first; "
            "downstream errors (like a groupby KeyError) are just a symptom of this "
            "DataFrame being empty."
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg_spec = {
        "hotpot_em": "mean",
        "hotpot_f1": "mean",
        "hotpot_precision": "mean",
        "hotpot_recall": "mean",
        "faithfulness": "mean",
        "answer_relevancy": "mean",
        "answer_accuracy": "mean",
        "nv_answer_accuracy": "mean",
        "latency_s": "mean",
        "total_tokens": "mean",
        "edit_distance_layers": "mean",
    }
    agg_spec = {col: agg for col, agg in agg_spec.items() if col in df.columns}
    if "reasoning_consistency" in df.columns:
        agg_spec["reasoning_consistency"] = "mean"
    return df.groupby("condition").agg(agg_spec).round(3)
