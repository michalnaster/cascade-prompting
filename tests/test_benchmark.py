from __future__ import annotations

import os

import pandas as pd

import cascade_prompting.benchmark as benchmark
from cascade_prompting.schemas import GenerationOutput


class _FakeStrategy:
    def __init__(self, name: str, answer: str):
        self.name = name
        self._answer = answer

    def run(self, llm, question, context):
        return GenerationOutput(
            final_text=self._answer,
            latency_s=1.0,
            input_tokens=10,
            output_tokens=5,
            layers=[self._answer],
        )


def _examples(n: int) -> list[dict]:
    return [
        {"question": f"q{i}", "answer": "gold", "context_sentences": ["ctx"]} for i in range(n)
    ]


def test_significance_recomputed_after_every_example_not_once_at_the_end(monkeypatch, tmp_path):
    fake_strategies = {
        "a": _FakeStrategy("a", "gold"),
        "b": _FakeStrategy("b", "wrong"),
    }
    monkeypatch.setattr(benchmark, "STRATEGIES", fake_strategies)

    call_sizes: list[int] = []
    real_compare_all_conditions = benchmark.compare_all_conditions

    def _spy(df: pd.DataFrame) -> pd.DataFrame:
        call_sizes.append(len(df))
        return real_compare_all_conditions(df)

    monkeypatch.setattr(benchmark, "compare_all_conditions", _spy)

    significance_out = tmp_path / "significance.csv"
    n_examples = 4

    benchmark.run_generation_phase(
        llm=None,
        examples=_examples(n_examples),
        significance_out=str(significance_out),
    )

    # Recomputed once per example (2 conditions/example), growing each time
    # — not once at the end over the full accumulated set.
    assert call_sizes == [2, 4, 6, 8]

    # And the file on disk reflects that incremental recompute happening,
    # not just a single write after the loop.
    assert significance_out.exists()
    final = pd.read_csv(significance_out)
    assert (final["n_pairs"] == n_examples).all()


def test_significance_out_is_optional(monkeypatch, tmp_path):
    fake_strategies = {"a": _FakeStrategy("a", "gold")}
    monkeypatch.setattr(benchmark, "STRATEGIES", fake_strategies)

    df = benchmark.run_generation_phase(llm=None, examples=_examples(2))

    assert len(df) == 2
    assert not os.path.exists(tmp_path / "significance.csv")


def test_summary_recomputed_after_every_example_not_once_at_the_end(monkeypatch, tmp_path):
    fake_strategies = {
        "a": _FakeStrategy("a", "gold"),
        "b": _FakeStrategy("b", "wrong"),
    }
    monkeypatch.setattr(benchmark, "STRATEGIES", fake_strategies)

    call_sizes: list[int] = []
    real_summarize = benchmark.summarize

    def _spy(df: pd.DataFrame) -> pd.DataFrame:
        call_sizes.append(len(df))
        return real_summarize(df)

    monkeypatch.setattr(benchmark, "summarize", _spy)

    summary_out = tmp_path / "summary.csv"
    n_examples = 4

    benchmark.run_generation_phase(
        llm=None,
        examples=_examples(n_examples),
        summary_out=str(summary_out),
    )

    # Recomputed once per example (2 conditions/example), growing each time
    # — not once at the end over the full accumulated set.
    assert call_sizes == [2, 4, 6, 8]

    # And the file on disk reflects that incremental recompute, with one row
    # per condition, indexed by condition (matching summarize()'s own shape).
    assert summary_out.exists()
    final = pd.read_csv(summary_out, index_col=0)
    assert sorted(final.index) == ["a", "b"]
    assert final.loc["a", "hotpot_em"] == 1.0  # "gold" exactly matches gold_answer
    assert final.loc["b", "hotpot_em"] == 0.0  # "wrong" doesn't


def test_summary_out_is_optional(monkeypatch, tmp_path):
    fake_strategies = {"a": _FakeStrategy("a", "gold")}
    monkeypatch.setattr(benchmark, "STRATEGIES", fake_strategies)

    df = benchmark.run_generation_phase(llm=None, examples=_examples(2))

    assert len(df) == 2
    assert not os.path.exists(tmp_path / "summary.csv")


class _CountingStrategy:
    def __init__(self, name: str, answer: str):
        self.name = name
        self._answer = answer
        self.calls: list[str] = []

    def run(self, llm, question, context):
        self.calls.append(question)
        return GenerationOutput(
            final_text=self._answer,
            latency_s=1.0,
            input_tokens=10,
            output_tokens=5,
            layers=[self._answer],
        )


def test_resume_skips_pairs_already_on_disk(monkeypatch, tmp_path):
    strategy_a = _CountingStrategy("a", "gold")
    monkeypatch.setattr(benchmark, "STRATEGIES", {"a": strategy_a})
    out_path = tmp_path / "results.csv"

    # First "run": only 2 of an eventual 4 examples complete before it's
    # interrupted.
    benchmark.run_generation_phase(llm=None, examples=_examples(2), out_path=str(out_path))
    assert strategy_a.calls == ["q0", "q1"]

    # Resumed run: same out_path, full 4-example set. Only the two new
    # examples should trigger a call; 0 and 1 must come back from disk.
    strategy_a.calls.clear()
    df = benchmark.run_generation_phase(
        llm=None, examples=_examples(4), out_path=str(out_path), resume=True
    )

    assert strategy_a.calls == ["q2", "q3"]
    assert len(df) == 4
    assert sorted(df["example_idx"].tolist()) == [0, 1, 2, 3]
    # context_sentences must come back as a real list, not the CSV's
    # stringified repr — score_with_ragas expects an actual list.
    assert all(isinstance(cs, list) for cs in df["context_sentences"])

    on_disk = pd.read_csv(out_path)
    assert len(on_disk) == 4


def test_resume_regenerates_when_question_at_index_has_changed(monkeypatch, tmp_path):
    strategy_a = _CountingStrategy("a", "gold")
    monkeypatch.setattr(benchmark, "STRATEGIES", {"a": strategy_a})
    out_path = tmp_path / "results.csv"

    # First "run": example_idx 0 is "q0".
    benchmark.run_generation_phase(llm=None, examples=_examples(1), out_path=str(out_path))
    assert strategy_a.calls == ["q0"]

    # Resumed run, but the example at index 0 is now a *different* question
    # — e.g. a different --n or a reshuffled dataset put something else at
    # position 0. example_idx alone would wrongly treat this as "already
    # done"; matching on the question text too must catch the mismatch and
    # regenerate it instead of silently reusing the stale row.
    strategy_a.calls.clear()
    shifted = [{"question": "a completely different question", "answer": "gold", "context_sentences": ["ctx"]}]
    df = benchmark.run_generation_phase(
        llm=None, examples=shifted, out_path=str(out_path), resume=True
    )

    assert strategy_a.calls == ["a completely different question"]
    assert len(df) == 2  # the stale q0 row plus the newly (re)generated one
    assert set(df["question"]) == {"q0", "a completely different question"}


def test_opik_tracing_disabled_by_default_calls_strategy_run_directly(monkeypatch):
    strategy_a = _CountingStrategy("a", "gold")
    monkeypatch.setattr(benchmark, "STRATEGIES", {"a": strategy_a})

    import cascade_prompting.observability as observability

    def _unexpected_traced_run(*args, **kwargs):
        raise AssertionError("traced_run should not be called when opik_tracing=False")

    monkeypatch.setattr(observability, "traced_run", _unexpected_traced_run)

    df = benchmark.run_generation_phase(llm=None, examples=_examples(2))

    assert strategy_a.calls == ["q0", "q1"]
    assert len(df) == 2


def test_opik_tracing_enabled_routes_every_call_through_traced_run(monkeypatch):
    strategy_a = _CountingStrategy("a", "gold")
    monkeypatch.setattr(benchmark, "STRATEGIES", {"a": strategy_a})

    import cascade_prompting.observability as observability

    calls: list[tuple[str, int, str]] = []

    def _fake_traced_run(strategy, llm, question, context, *, example_idx, project_name):
        calls.append((strategy.name, example_idx, project_name))
        return strategy.run(llm, question, context)

    monkeypatch.setattr(observability, "traced_run", _fake_traced_run)

    df = benchmark.run_generation_phase(
        llm=None,
        examples=_examples(2),
        opik_tracing=True,
        opik_project="test-project",
    )

    assert calls == [("a", 0, "test-project"), ("a", 1, "test-project")]
    assert strategy_a.calls == ["q0", "q1"]  # traced_run still delegated to the real strategy
    assert len(df) == 2


def test_resume_with_no_existing_file_behaves_like_a_fresh_run(monkeypatch, tmp_path):
    strategy_a = _CountingStrategy("a", "gold")
    monkeypatch.setattr(benchmark, "STRATEGIES", {"a": strategy_a})
    out_path = tmp_path / "does_not_exist_yet.csv"

    df = benchmark.run_generation_phase(
        llm=None, examples=_examples(2), out_path=str(out_path), resume=True
    )

    assert strategy_a.calls == ["q0", "q1"]
    assert len(df) == 2
