"""Paired statistical significance tests between conditions.

Every condition in the benchmark answers the same set of questions, so
comparing two conditions is a paired-samples problem, not an unpaired one —
each question is its own natural pair, matched across conditions rather than
treated as two independent samples. Testing it as paired is both more
correct and considerably more powerful than comparing unpaired means at
benchmark-sized n, which is why summary-CSV means alone (`benchmark.py`'s
`summarize`) shouldn't be read as evidence of a real difference on their
own.

Two tests are run per pair of conditions:

- `mcnemar_test`, on `hotpot_em` (binary correct/incorrect per question):
  are the *specific* questions each condition gets right and wrong
  different enough to rule out chance, rather than just comparing overall
  accuracy? McNemar's test only looks at the discordant pairs — questions
  where the two conditions disagree — since concordant pairs (both right or
  both wrong) carry no information about which condition is better. This is
  the exact (binomial) form of the test, appropriate regardless of how many
  discordant pairs there are, unlike the chi-square approximation which
  needs a reasonably large discordant count to be trustworthy.
- `paired_metric_test`, on any continuous per-question metric (`latency_s`,
  `total_tokens`): is the per-question difference consistently
  one-directional, not just different in the mean? Both a paired t-test and
  a Wilcoxon signed-rank test are reported, rather than picking one. The
  t-test assumes the paired differences are roughly normal; per-question
  latency and token-count differences are often right-skewed, not normal,
  so Wilcoxon (which only assumes symmetry around the median difference, not
  normality) is the more defensible default — but reporting both lets a
  reader who trusts the t-test's assumptions see that result too, instead of
  the choice being made silently on their behalf.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class McNemarResult:
    condition_a: str
    condition_b: str
    n_pairs: int
    #: Correct under `condition_a`, wrong under `condition_b`.
    a_correct_b_wrong: int
    #: Wrong under `condition_a`, correct under `condition_b`.
    a_wrong_b_correct: int
    p_value: float


@dataclass(frozen=True)
class PairedTestResult:
    condition_a: str
    condition_b: str
    metric: str
    n_pairs: int
    mean_a: float
    mean_b: float
    t_statistic: float
    t_p_value: float
    wilcoxon_statistic: float
    wilcoxon_p_value: float


def mcnemar_test(
    correct_a: pd.Series | np.ndarray,
    correct_b: pd.Series | np.ndarray,
    condition_a: str = "A",
    condition_b: str = "B",
) -> McNemarResult:
    """Exact (binomial) McNemar's test on paired binary correctness.

    `correct_a`/`correct_b` must be the same length and in matching question
    order — index alignment is the caller's responsibility (see
    `compare_conditions`, which handles it via `example_idx`).
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    if len(correct_a) != len(correct_b):
        raise ValueError(
            "mcnemar_test requires paired, equal-length arrays (same "
            "questions, same order) — got "
            f"{len(correct_a)} vs {len(correct_b)}"
        )

    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    n_discordant = a_only + b_only

    # No disagreements at all: nothing to test, and a p-value of 1.0 (rather
    # than NaN or an error) correctly reflects "no evidence of a difference".
    p_value = (
        1.0
        if n_discordant == 0
        else stats.binomtest(min(a_only, b_only), n_discordant, p=0.5, alternative="two-sided").pvalue
    )

    return McNemarResult(
        condition_a=condition_a,
        condition_b=condition_b,
        n_pairs=len(correct_a),
        a_correct_b_wrong=a_only,
        a_wrong_b_correct=b_only,
        p_value=p_value,
    )


def paired_metric_test(
    values_a: pd.Series | np.ndarray,
    values_b: pd.Series | np.ndarray,
    metric: str,
    condition_a: str = "A",
    condition_b: str = "B",
) -> PairedTestResult:
    """Paired t-test and Wilcoxon signed-rank test on a continuous
    per-question metric (e.g. `latency_s`, `total_tokens`).

    `values_a`/`values_b` must be the same length and in matching question
    order — index alignment is the caller's responsibility (see
    `compare_conditions`).
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if len(values_a) != len(values_b):
        raise ValueError(
            "paired_metric_test requires paired, equal-length arrays (same "
            "questions, same order) — got "
            f"{len(values_a)} vs {len(values_b)}"
        )

    if np.all(values_a == values_b):
        # Every paired difference is exactly zero: both tests are undefined
        # (Wilcoxon raises on an all-zero difference vector) and there is,
        # correctly, no evidence of a difference.
        t_statistic, t_p_value = 0.0, 1.0
        wilcoxon_statistic, wilcoxon_p_value = 0.0, 1.0
    else:
        t_statistic, t_p_value = stats.ttest_rel(values_a, values_b)
        wilcoxon_statistic, wilcoxon_p_value = stats.wilcoxon(values_a, values_b)

    return PairedTestResult(
        condition_a=condition_a,
        condition_b=condition_b,
        metric=metric,
        n_pairs=len(values_a),
        mean_a=float(np.mean(values_a)),
        mean_b=float(np.mean(values_b)),
        t_statistic=float(t_statistic),
        t_p_value=float(t_p_value),
        wilcoxon_statistic=float(wilcoxon_statistic),
        wilcoxon_p_value=float(wilcoxon_p_value),
    )


def compare_conditions(
    df: pd.DataFrame, condition_a: str, condition_b: str
) -> tuple[McNemarResult, PairedTestResult, PairedTestResult]:
    """Align two conditions' rows by `example_idx` (only questions both
    conditions actually answered) and run all three tests: McNemar on
    `hotpot_em`, paired t-test + Wilcoxon on `latency_s`, and the same pair
    on `total_tokens`.
    """
    a = df[df["condition"] == condition_a].set_index("example_idx").sort_index()
    b = df[df["condition"] == condition_b].set_index("example_idx").sort_index()
    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise ValueError(
            f"No shared example_idx between {condition_a!r} and {condition_b!r} "
            "— nothing to pair."
        )
    a = a.loc[common]
    b = b.loc[common]

    em_result = mcnemar_test(a["hotpot_em"] == 1, b["hotpot_em"] == 1, condition_a, condition_b)
    latency_result = paired_metric_test(a["latency_s"], b["latency_s"], "latency_s", condition_a, condition_b)
    tokens_result = paired_metric_test(
        a["total_tokens"], b["total_tokens"], "total_tokens", condition_a, condition_b
    )
    return em_result, latency_result, tokens_result


def compare_all_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Run `compare_conditions` on every pair of conditions present in `df`,
    flattened into one row per (condition pair, metric, test) — suitable for
    writing straight to CSV or logging as a table.

    Safe to call on a partial, still-accumulating `df` (see
    `benchmark.run_generation_phase`'s `significance_out`, which recomputes
    this after every example instead of once at the end): a pair with no
    shared `example_idx` yet — the normal state early in a run, before both
    conditions have completed the same question at least once — is skipped
    for this call rather than raising and losing every other pair's results
    along with it. It reappears in the output once they share one.
    """
    conditions = sorted(df["condition"].unique())
    rows: list[dict] = []

    for condition_a, condition_b in itertools.combinations(conditions, 2):
        try:
            em_result, latency_result, tokens_result = compare_conditions(df, condition_a, condition_b)
        except ValueError:
            continue

        rows.append(
            {
                "condition_a": em_result.condition_a,
                "condition_b": em_result.condition_b,
                "metric": "hotpot_em",
                "test": "mcnemar_exact",
                "n_pairs": em_result.n_pairs,
                "statistic": None,
                "p_value": em_result.p_value,
                # a_correct_b_wrong / a_wrong_b_correct double as the two
                # conditions' win counts on the discordant pairs; there's no
                # single scalar "mean" for a binary paired test, so these
                # take the place of mean_a/mean_b for this row.
                "mean_a": em_result.a_correct_b_wrong,
                "mean_b": em_result.a_wrong_b_correct,
            }
        )
        for result in (latency_result, tokens_result):
            rows.append(
                {
                    "condition_a": result.condition_a,
                    "condition_b": result.condition_b,
                    "metric": result.metric,
                    "test": "paired_t",
                    "n_pairs": result.n_pairs,
                    "statistic": result.t_statistic,
                    "p_value": result.t_p_value,
                    "mean_a": result.mean_a,
                    "mean_b": result.mean_b,
                }
            )
            rows.append(
                {
                    "condition_a": result.condition_a,
                    "condition_b": result.condition_b,
                    "metric": result.metric,
                    "test": "wilcoxon",
                    "n_pairs": result.n_pairs,
                    "statistic": result.wilcoxon_statistic,
                    "p_value": result.wilcoxon_p_value,
                    "mean_a": result.mean_a,
                    "mean_b": result.mean_b,
                }
            )

    return pd.DataFrame(rows)
