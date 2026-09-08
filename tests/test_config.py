from __future__ import annotations

import argparse
from datetime import date

from cascade_prompting.config import BenchmarkConfig


def _namespace(**overrides) -> argparse.Namespace:
    defaults = dict(
        n=20,
        provider="ollama",
        model=None,
        judge_provider=None,
        judge_model=None,
        anthropic_api_key=None,
        base_url="http://localhost:11434",
        ragas=False,
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        ragas_max_workers=2,
        ragas_timeout=300,
        out="results.csv",
        log_file=None,
        resume=True,
        opik=False,
        opik_project="cascade-prompting",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_model_defaults_to_provider_default():
    config = BenchmarkConfig.from_args(_namespace(provider="anthropic"))
    assert config.model == "claude-opus-5"


def test_explicit_model_is_kept():
    config = BenchmarkConfig.from_args(_namespace(model="qwen3.6-custom"))
    assert config.model == "qwen3.6-custom"


def test_judge_defaults_to_generation_model_when_same_provider():
    config = BenchmarkConfig.from_args(_namespace(provider="ollama", model="qwen3.6"))
    assert config.judge_provider == "ollama"
    assert config.judge_model == "qwen3.6"


def test_judge_falls_back_to_its_own_provider_default_when_provider_differs():
    config = BenchmarkConfig.from_args(
        _namespace(provider="ollama", model="qwen3.6", judge_provider="anthropic")
    )
    assert config.judge_provider == "anthropic"
    assert config.judge_model == "claude-opus-5"


def test_explicit_judge_model_always_wins():
    config = BenchmarkConfig.from_args(
        _namespace(judge_provider="anthropic", judge_model="claude-haiku-4-5")
    )
    assert config.judge_model == "claude-haiku-4-5"


def test_log_file_derived_from_out_by_default():
    config = BenchmarkConfig.from_args(_namespace(out="my_run.csv"))
    assert config.log_file == "my_run.log"


def test_explicit_log_file_is_kept():
    config = BenchmarkConfig.from_args(_namespace(out="my_run.csv", log_file="custom.log"))
    assert config.log_file == "custom.log"


def test_summary_out_replaces_csv_suffix():
    config = BenchmarkConfig.from_args(_namespace(out="my_run.csv"))
    assert config.summary_out == "my_run_summary.csv"


def test_ragas_disabled_by_default():
    config = BenchmarkConfig.from_args(_namespace())
    assert config.use_ragas is False


def test_ragas_enabled_when_flag_passed():
    config = BenchmarkConfig.from_args(_namespace(ragas=True))
    assert config.use_ragas is True


def test_out_defaults_to_date_and_model():
    config = BenchmarkConfig.from_args(
        _namespace(out=None, model="qwen3.6"), today=date(2026, 8, 19)
    )
    assert config.out == "cascade_benchmark_results_2026-08-19_qwen3.6.csv"


def test_out_default_slugifies_model_tag():
    config = BenchmarkConfig.from_args(
        _namespace(out=None, provider="ollama", model="gemma3n:e4b"),
        today=date(2026, 8, 19),
    )
    assert config.out == "cascade_benchmark_results_2026-08-19_gemma3n-e4b.csv"


def test_explicit_out_is_kept_over_date_model_default():
    config = BenchmarkConfig.from_args(_namespace(out="my_run.csv", model="qwen3.6"))
    assert config.out == "my_run.csv"


def test_resume_enabled_by_default():
    config = BenchmarkConfig.from_args(_namespace())
    assert config.resume is True


def test_resume_disabled_via_no_resume():
    # argparse.BooleanOptionalAction sets args.resume=False for --no-resume.
    config = BenchmarkConfig.from_args(_namespace(resume=False))
    assert config.resume is False


def test_opik_tracing_disabled_by_default():
    config = BenchmarkConfig.from_args(_namespace())
    assert config.opik_tracing is False
    assert config.opik_project == "cascade-prompting"


def test_opik_tracing_enabled_when_flag_passed():
    config = BenchmarkConfig.from_args(_namespace(opik=True, opik_project="my-project"))
    assert config.opik_tracing is True
    assert config.opik_project == "my-project"
