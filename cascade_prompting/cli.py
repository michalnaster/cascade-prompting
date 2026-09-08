"""Command-line entry point.

`build_arg_parser` and `main` are split out from module-level `if __name__`
code so the CLI can be invoked programmatically (tests, other scripts)
without going through argv/subprocess.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import BenchmarkConfig
from .data import load_hotpotqa_sample
from .llm.factory import build_llm
from .logging_setup import logger, setup_logging

_DESCRIPTION = """\
Cascade Prompting Benchmark: RAG Answer Generation with Fact-Checking

Compares four ways of producing a grounded, fact-checked answer from
retrieved documents: baseline (one naive call, no reasoning field),
multi_call (three sequential reasoning+data calls, refine call scoped to
just the question + prior answer), multi_call_full_context (same three
calls, but every call — including refine — also gets the source
documents), and cascade (one structured call, three reasoning+data layers).
See README.md for the full write-up of conditions and metrics.
"""

_EPILOG = """\
Examples:
  # Ollama (default provider)
  ollama pull qwen3.6
  python -m cascade_prompting --n 50 --model qwen3.6 --out results.csv

  # Anthropic (Claude) — requires ANTHROPIC_API_KEY in the environment
  python -m cascade_prompting --n 50 --provider anthropic --model claude-opus-5 --out results.csv

--judge-provider / --judge-model let the RAGAS scoring LLM use a different
provider/model than generation (defaults to --provider/--model).

Logging: progress and per-example results go to stdout and a log file
(--log-file, default: <out> with its extension replaced by .log). The file
additionally captures DEBUG-level detail not shown on stdout: every
intermediate layer (draft / fact-checked / final answer text, and reasoning
where present) produced by each condition for each example.
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=20, help="Number of examples to evaluate")
    parser.add_argument(
        "--provider", type=str, default="ollama", choices=["ollama", "anthropic"],
        help="LLM provider used for the four generation conditions",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model tag/id used for all four generation conditions "
             "(Ollama tag, e.g. `qwen3.6`, must already be pulled; "
             "Anthropic model id, e.g. `claude-opus-5`). "
             "Defaults to qwen3.6 (ollama) / claude-opus-5 (anthropic).",
    )
    parser.add_argument(
        "--judge-provider", type=str, default=None, choices=["ollama", "anthropic"],
        help="LLM provider used as the RAGAS evaluator (defaults to --provider)",
    )
    parser.add_argument(
        "--judge-model", type=str, default=None,
        help="Model tag/id used as the RAGAS evaluator LLM "
             "(defaults to --model if --judge-provider matches --provider, "
             "otherwise that provider's default model)",
    )
    parser.add_argument(
        "--anthropic-api-key", type=str, default=None,
        help="Anthropic API key (defaults to the ANTHROPIC_API_KEY env var); "
             "only used when --provider/--judge-provider is 'anthropic'",
    )
    parser.add_argument(
        "--base-url", type=str, default="http://localhost:11434",
        help="Ollama server base URL (ignored for the anthropic provider)",
    )
    parser.add_argument(
        "--ragas", action="store_true",
        help="Enable RAGAS scoring (faithfulness / answer_relevancy / "
             "nv_answer_accuracy). Disabled by default: it's an extra "
             "LLM-judge pass per example plus a HuggingFace embedding "
             "model download, on top of generation. hotpot_em/f1, the "
             "diagnostics, and answer_accuracy (a fast local-embedding "
             "stand-in for RAGAS's nv_answer_accuracy) always run regardless.",
    )
    parser.add_argument(
        "--embedding-model", type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="HF embedding model used by RAGAS for answer_relevancy. Only "
             "loaded/used when --ragas is passed.",
    )
    parser.add_argument(
        "--ragas-max-workers", type=int, default=2,
        help="RAGAS concurrent request limit. Keep this low (1-2) for a "
             "single local Ollama server — it can't actually serve RAGAS's "
             "default of 16 concurrent requests, which causes a wall of "
             "'Exception raised in Job[N]: TimeoutError()'.",
    )
    parser.add_argument(
        "--ragas-timeout", type=int, default=300,
        help="RAGAS per-request timeout in seconds. Local/CPU-bound "
             "inference is often slower than a hosted API, so this is set "
             "higher than RAGAS's own default (180s).",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Path to write per-example results to. Defaults to "
             "cascade_benchmark_results_<date>_<model>.csv so runs against "
             "different models/days don't overwrite each other.",
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Path to write full DEBUG-level logs (every intermediate layer "
             "per example/condition), in addition to stdout. Defaults to "
             "<out> with its extension replaced by .log",
    )
    parser.add_argument(
        "--opik", action="store_true",
        help="Enable Opik tracing: one trace per (example, condition) call, "
             "with each individual LLM call inside it (one for baseline/"
             "cascade, three for multi_call/multi_call_full_context) "
             "captured as its own latency/token-tagged span. Disabled by "
             "default — needs its own backend configured (Comet-hosted "
             "project or self-hosted server; see `opik configure` / the "
             "OPIK_API_KEY / OPIK_URL_OVERRIDE env vars) and the `opik` "
             "package installed (`pip install opik`), neither of which a "
             "plain benchmark run requires.",
    )
    parser.add_argument(
        "--opik-project", type=str, default="cascade-prompting",
        help="Opik project name traces are logged under. Only used when "
             "--opik is passed.",
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume from --out by default: skip any (example, condition) "
             "pair already present in it instead of generating it again. "
             "Pass --no-resume to force a fresh run that overwrites --out "
             "from scratch instead. Resuming requires the same --out (and "
             "the same --n / dataset) as the run being picked up — --out's "
             "default filename is date-stamped, so pass --out explicitly if "
             "you might want to resume this run later. Either way, has no "
             "effect if --out doesn't exist yet — that's always a fresh run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = BenchmarkConfig.from_args(args)

    setup_logging(config.log_file)
    logger.info(f"[log] Full intermediate-step trace being written to {config.log_file}")

    # Imported here, not at module top: this module's argparse/config path
    # should stay fast and dependency-light for --help and for
    # BenchmarkConfig-only unit tests.
    import pandas as pd

    from .benchmark import run_generation_phase, summarize

    generation_llm = build_llm(config.provider, config.model, **config.llm_kwargs())
    examples = load_hotpotqa_sample(config.n)

    logger.info(
        "=== Phase 1: generation (baseline / multi_call / "
        "multi_call_full_context / cascade) — summary and significance "
        "tests recomputed and written after each example, not in a "
        "separate phase ==="
    )
    raw_df = run_generation_phase(
        generation_llm,
        examples,
        out_path=config.out,
        significance_out=config.significance_out,
        summary_out=config.summary_out,
        resume=config.resume,
        opik_tracing=config.opik_tracing,
        opik_project=config.opik_project,
    )
    logger.info(f"\nSummary written to {config.summary_out}")
    logger.info(f"Significance tests written to {config.significance_out}")

    if config.use_ragas:
        # ragas is slow to import (and to run), which is exactly why it's
        # gated behind --ragas rather than imported unconditionally above.
        #
        # metrics.ragas_scoring must be imported before any other `ragas.*`
        # import — it applies a sys.modules stub that ragas's own import
        # chain needs (see the comment at the top of that module).
        # Importing ragas.embeddings/ragas.llms directly first breaks that,
        # so this block is order-sensitive — keep isort/ruff from
        # reshuffling it.
        # isort: off
        from .metrics.ragas_scoring import score_with_ragas

        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.run_config import RunConfig
        # isort: on

        logger.info("\n=== Phase 2: RAGAS scoring (batched per condition) ===")
        ragas_llm = LangchainLLMWrapper(
            build_llm(config.judge_provider, config.judge_model, **config.llm_kwargs())
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=config.embedding_model)
        )
        ragas_run_config = RunConfig(
            max_workers=config.ragas_max_workers, timeout=config.ragas_timeout
        )
        results_df = score_with_ragas(raw_df, ragas_llm, ragas_embeddings, ragas_run_config)

        results_df.to_csv(config.out, index=False)
        logger.info(f"\nRaw results written to {config.out}")

        # The summary written incrementally during Phase 1 doesn't have
        # faithfulness/answer_relevancy/nv_answer_accuracy yet — those only
        # exist after this RAGAS merge — so it's recomputed and rewritten
        # once more here, over the complete, RAGAS-merged data.
        summary_df = summarize(results_df)
        summary_df.to_csv(config.summary_out)
        logger.info(f"\nSummary (with RAGAS columns) written to {config.summary_out}")
    else:
        logger.info(
            "\n[skip] RAGAS scoring disabled (pass --ragas to enable "
            "faithfulness / answer_relevancy / nv_answer_accuracy)"
        )
        results_df = raw_df

    # The summary and significance files are already complete and current
    # at this point — both were computed and written incrementally during
    # Phase 1 (and the summary once more above, if --ragas ran). This just
    # logs the final tables for visibility; it doesn't write anything new.
    logger.info("\n=== Summary (mean per condition) ===")
    logger.info(pd.read_csv(config.summary_out, index_col=0).to_string())
    logger.info("\n=== Significance tests (final, all condition pairs) ===")
    logger.info(pd.read_csv(config.significance_out).to_string(index=False))
    return 0
