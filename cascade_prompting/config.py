"""Typed run configuration, resolved from parsed CLI args.

Separating this from `cli.py` means the model/judge-model default-resolution
logic (which provider's default model to fall back to, whether the judge
reuses the generation model) is unit-testable without touching argparse or
the network.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date

from .llm.factory import get_default_model


def _slugify_model(model: str) -> str:
    """Turn a provider model tag into something safe for a filename, e.g.
    "gemma3n:e4b" -> "gemma3n-e4b", "claude-opus-5" unchanged."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")


def _default_out(model: str, today: date) -> str:
    """Default --out: today's date + the generation model, so results from
    different runs/models never silently overwrite each other."""
    return f"cascade_benchmark_results_{today.isoformat()}_{_slugify_model(model)}.csv"


@dataclass(frozen=True)
class BenchmarkConfig:
    n: int
    provider: str
    model: str
    judge_provider: str
    judge_model: str
    anthropic_api_key: str | None
    base_url: str
    use_ragas: bool
    embedding_model: str
    ragas_max_workers: int
    ragas_timeout: int
    out: str
    log_file: str
    resume: bool
    opik_tracing: bool
    opik_project: str

    @classmethod
    def from_args(cls, args: argparse.Namespace, today: date | None = None) -> BenchmarkConfig:
        model = args.model or get_default_model(args.provider)
        judge_provider = args.judge_provider or args.provider
        if args.judge_model:
            judge_model = args.judge_model
        elif judge_provider == args.provider:
            judge_model = model
        else:
            judge_model = get_default_model(judge_provider)

        out = args.out or _default_out(model, today or date.today())
        log_file = args.log_file or re.sub(r"\.[^.]+$", ".log", out)

        return cls(
            n=args.n,
            provider=args.provider,
            model=model,
            judge_provider=judge_provider,
            judge_model=judge_model,
            anthropic_api_key=args.anthropic_api_key,
            base_url=args.base_url,
            use_ragas=args.ragas,
            embedding_model=args.embedding_model,
            ragas_max_workers=args.ragas_max_workers,
            ragas_timeout=args.ragas_timeout,
            out=out,
            log_file=log_file,
            resume=args.resume,
            opik_tracing=args.opik,
            opik_project=args.opik_project,
        )

    @property
    def summary_out(self) -> str:
        return self.out.replace(".csv", "_summary.csv")

    @property
    def significance_out(self) -> str:
        return self.out.replace(".csv", "_significance.csv")

    def llm_kwargs(self) -> dict:
        """Provider-agnostic kwargs forwarded to llm.factory.build_llm.
        Each provider reads only the config keys it needs (see
        cascade_prompting.llm.base.LLMProvider)."""
        return {"base_url": self.base_url, "api_key": self.anthropic_api_key}
