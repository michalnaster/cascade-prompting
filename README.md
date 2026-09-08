# Cascade Prompting Benchmark

Accompanying code for the article [Cascade Prompting: The Stupidly Simple
Alternative to Your Multi-Agent
Pipelines](https://medium.com/@michalnasternak/cascade-prompting-the-stupidly-simple-alternative-to-your-multi-agent-pipelines-0cb01f2e55dc).

Benchmarks four ways of producing a grounded, fact-checked answer from
retrieved documents (RAG), to isolate what a structured "cascade" prompt
actually buys you over a naive single call or a hand-rolled multi-call
pipeline:

| Condition | Calls | Structure |
|---|---|---|
| `baseline` | 1 | No structure, no reasoning field — "answer the question." |
| `multi_call` | 3 | Three sequential calls (draft → fact-check → refine), each its own structured-output call producing a `reasoning` + `answer` pair (`ReasonedAnswer`) and re-sending its own system prompt / context. The refine call is scoped to just the question and the prior answer — no source documents — since its instruction is only "tighten the wording." |
| `multi_call_full_context` | 3 | Identical to `multi_call`, except the refine call also receives the source documents, matching the grounding `cascade`'s every layer always has. Isolates whether a quality gap between `cascade` and `multi_call` is about call count itself or about `multi_call`'s refine call losing its grounding. |
| `cascade` | 1 | One structured-output call, three ordered layers (`draft_answer` → `fact_checked_answer` → `final_answer`), each layer conditioning on the previous one via schema field order, with a `reasoning` field before `data` in every layer (`Layer`). |

`multi_call`/`multi_call_full_context`/`cascade` all follow the same
reasoning-then-answer discipline per step, via the same shared instruction
text — `cascade`'s three layers are fields of one `Layer` schema threaded
through a single call, `multi_call`'s three steps are each their own
standalone `ReasonedAnswer` call, but both pairs are "state what's
changing and why, then produce the result." A quality gap between any of
them isn't explainable by only one condition being asked to justify a
change before making it — `baseline` is the only condition without a
reasoning field, since it's a single naive call with no steps to justify.

All four conditions share the exact same instruction text (`DRAFT_INSTRUCTION`,
`FACT_CHECK_INSTRUCTION`, `REFINE_INSTRUCTION` in
[`cascade_prompting/prompts.py`](cascade_prompting/prompts.py)) so a quality
gap between conditions reflects call *structure* (and, for the two
`multi_call` variants, how much context each call gets), not prompt wording.

Dataset: [HotpotQA](https://hotpotqa.github.io/) (distractor setting), loaded
from the Hugging Face Hub via the `datasets` library.

## Metrics

**Answer quality — string match** (vendored from the
[official HotpotQA eval script](https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py)):
`hotpot_em`, `hotpot_f1`, `hotpot_precision`, `hotpot_recall`.

**Answer quality — fast local proxy** (always computed, no LLM call, no
`--ragas` needed):
- `answer_accuracy` — cosine similarity between a local sentence-embedding
  of the predicted answer and the gold answer (see
  `cascade_prompting/metrics/embedding_accuracy.py`); a small
  sentence-transformers model runs on CPU in milliseconds, so this is cheap
  enough to compute for every example in every run. Cruder than an LLM
  judge at telling "correct, differently phrased" apart from "wrong, but
  topically similar" — two named entities from the same domain often sit
  close together in embedding space regardless of which is actually right.

**Answer quality — LLM judge** (via [RAGAS](https://docs.ragas.io), batched
once per condition; only runs when `--ragas` is passed):
- `faithfulness` — fraction of claims in the answer supported by the retrieved documents.
- `answer_relevancy` — how well the answer addresses the actual question.
- `nv_answer_accuracy` — RAGAS's `AnswerAccuracy` (`nv_accuracy`) metric: an
  LLM-judge comparison of the predicted answer against the gold answer,
  averaged over two differently-ordered judge prompts. Complements
  `hotpot_em`/`hotpot_f1` with a semantic judgment — e.g. it can credit a
  correct answer phrased differently from the gold string. Named
  `nv_answer_accuracy` (not `answer_accuracy`) so it can't collide with the
  always-on local-embedding proxy above when the two are merged.

**Efficiency + diagnostics** (computed per example):
`latency_s`, `input_tokens`, `output_tokens`, and:
- `edit_distance_layers` — average normalized edit distance between
  consecutive layers; near 0 means later layers barely changed anything
  (possible "theater" editing).
- `reasoning_consistency` — any condition with a `reasoning` field
  (`multi_call`, `multi_call_full_context`, `cascade` — not `baseline`);
  fraction of layers where a stated "no change" claim in `reasoning`
  matches whether `data` actually changed.

**Significance tests** (per pair of conditions, on the shared questions
both conditions answered): a summary-CSV mean difference isn't evidence of
a real difference on its own, so every pair of conditions is also compared
with paired tests that account for the fact that both conditions answered
the *same* questions:
- McNemar's exact test on `hotpot_em` — whether the specific questions each
  condition gets right/wrong disagree asymmetrically enough to rule out
  chance, rather than just comparing overall accuracy.
- A paired t-test and a Wilcoxon signed-rank test on `latency_s` and
  `total_tokens` — whether the per-question cost difference is consistently
  one-directional. Both tests are reported (see
  `cascade_prompting/metrics/significance.py` for why neither is picked as
  the default) rather than asserting significance from the summary means
  alone.

## Install

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"     # package + pytest/ruff
uv pip install -e ".[notebook]" # + jupyter/matplotlib, only needed for notebooks/
# or: uv pip install -r requirements.txt
```

## Usage

```bash
# Ollama (default provider) — make sure the model is pulled first
ollama pull qwen3.6
python -m cascade_prompting --n 50 --model qwen3.6 --out results.csv

# Anthropic (Claude) — requires ANTHROPIC_API_KEY in the environment
export ANTHROPIC_API_KEY=sk-ant-...
python -m cascade_prompting --n 50 --provider anthropic --model claude-opus-5 --out results.csv

# With RAGAS scoring (faithfulness / answer_relevancy / nv_answer_accuracy)
# — off by default; adds an LLM-judge pass and an embedding-model download
python -m cascade_prompting --n 50 --model qwen3.6 --ragas --out results.csv

# Re-running against the same --out after an interrupted run resumes it
# automatically — already-completed (example, condition) pairs are loaded
# from disk and skipped, not regenerated. --no-resume forces a fresh run
# that overwrites --out from scratch instead.
python -m cascade_prompting --n 50 --model qwen3.6 --out results.csv
python -m cascade_prompting --n 50 --model qwen3.6 --out results.csv --no-resume
```

`--judge-provider` / `--judge-model` let the RAGAS scoring LLM use a
different provider/model than generation (defaults to `--provider`/`--model`);
both are ignored unless `--ragas` is also passed.

Resuming is the default (`--resume`/`--no-resume`, see `--help`) and
requires the *same* `--out` path as the run being resumed — `--out`'s
default filename is date-stamped, so an interrupted run re-invoked without
an explicit `--out` on a later day just starts fresh under a new filename
rather than resuming anything. Pass `--out` explicitly for any run you
might want to resume.

Run `python -m cascade_prompting --help` for the full flag reference.

Three equivalent entry points, all wired to the same `cascade_prompting.cli.main`:
`python -m cascade_prompting`, the installed `cascade-benchmark` console
script, and the legacy `python in_context.py` (kept for backward compatibility).

### Output

- `<out>` (default `cascade_benchmark_results_<date>_<model>.csv`, e.g.
  `cascade_benchmark_results_2026-08-19_gemma3n-e4b.csv`, so runs against
  different models/days don't overwrite each other) — one row per
  (example, condition) with every metric above. Rows are appended as soon as
  each (example, condition) finishes generating, not only once the whole run
  completes. When `--ragas` is passed, this file is overwritten with the
  RAGAS-merged version once that phase finishes, so treat it as a live
  progress file during a run and the final output after; without `--ragas`,
  what's on disk after Phase 1 already is the final output. This file is
  also what resuming reads by default: an interrupted run's rows are
  exactly the (example, condition) pairs the next invocation, against the
  same `--out`, won't regenerate — pass `--no-resume` to overwrite instead.
- `<out>` with `_summary` inserted — mean of each metric per condition.
  Recomputed over the full accumulated data and rewritten after every
  example during Phase 1, same as the significance file below — a live,
  always-current per-condition summary during a run, not just at the end.
  When `--ragas` is passed, it's rewritten once more after Phase 2 to pick
  up `faithfulness`/`answer_relevancy`/`nv_answer_accuracy`, which aren't
  available until RAGAS has scored the run.
- `<out>` with `_significance` inserted — one row per (condition pair,
  metric, test): McNemar on `hotpot_em`, paired t-test and Wilcoxon on
  `latency_s`/`total_tokens`, for every pair of conditions with at least one
  shared question so far. Recomputed over the full accumulated data and
  rewritten after every example during Phase 1, not once at the end in a
  separate phase — like the raw results file, it's a live progress file
  during a run (readable mid-run, filling in as more condition pairs share
  at least one completed question) and the final output after.
- `<out>` with its extension replaced by `.log` (override with `--log-file`)
  — a full run log. stdout only shows INFO-level progress; the log file also
  captures DEBUG-level detail: every intermediate layer (draft /
  fact-checked / final answer text, and reasoning where present) produced by
  each condition for each example, so a run can be inspected after the fact
  without re-running the benchmark.

## Notebooks

[`notebooks/compare_results.ipynb`](notebooks/compare_results.ipynb) — loads
a results CSV, pairs up two conditions (`cascade` vs. `multi_call` by
default, configurable) by shared question, and surfaces the questions where
they disagree the most: exact-match-discordant questions ranked by how large
the `hotpot_f1` gap also is, the full draft/fact-checked/final trace
(data *and* reasoning) for the biggest gaps in each direction, a
continuous-metric ranking beyond exact match, and gap-distribution plots
across the whole run. Requires `uv pip install -e ".[notebook]"`.

## Architecture

```
cascade_prompting/
├── cli.py              # argparse + orchestration (the only place that wires everything together)
├── config.py            # BenchmarkConfig: typed, testable default-resolution for provider/model
├── benchmark.py          # run_generation_phase / summarize — iterates the strategy registry
├── prompts.py            # shared instruction text (single source of truth across conditions)
├── schemas.py             # pydantic structured-output schemas + GenerationOutput
├── logging_setup.py       # stdout (INFO) + log-file (DEBUG) logging
├── llm/                   # provider abstraction (Strategy/Factory pattern)
│   ├── base.py             # LLMProvider ABC
│   ├── ollama_provider.py
│   ├── anthropic_provider.py
│   └── factory.py          # registry + build_llm — never branches on provider name
├── strategies/             # one GenerationStrategy per condition (Strategy pattern)
│   ├── base.py              # GenerationStrategy ABC + extract_usage
│   ├── baseline.py / multi_call.py (MultiCallStrategy + MultiCallFullContextStrategy) / cascade.py
│   └── registry.py          # STRATEGIES — iterated polymorphically by benchmark.py
├── metrics/
│   ├── hotpot.py               # vendored official HotpotQA em/f1
│   ├── diagnostics.py          # edit_distance_layers, reasoning_consistency
│   ├── embedding_accuracy.py   # answer_accuracy (local-embedding proxy, always on)
│   ├── ragas_scoring.py        # score_with_ragas (faithfulness/answer_relevancy/nv_answer_accuracy, --ragas only)
│   └── significance.py         # compare_all_conditions (McNemar / paired t-test / Wilcoxon)
└── data/
    └── hotpotqa.py          # load_hotpotqa_sample
```

Two extension points are deliberately open/closed:

- **New LLM provider** (e.g. OpenAI, Bedrock): add an `LLMProvider` subclass
  in `llm/`, register it in `llm/factory.py`. Nothing in `strategies/` or
  `benchmark.py` changes — they only depend on the LangChain `BaseChatModel`
  interface.
- **New condition**: add a `GenerationStrategy` subclass in `strategies/`,
  register it in `strategies/registry.py`. `benchmark.py` iterates the
  registry polymorphically and never branches on condition name.

## Development

```bash
pytest                              # unit tests (no network / API key required)
pytest --cov=cascade_prompting      # with coverage
ruff check .                        # lint
mypy cascade_prompting              # type check
```

CI (`.github/workflows/ci.yml`) runs all three on every push/PR against
Python 3.10 and 3.13.

## License

[MIT](LICENSE)
