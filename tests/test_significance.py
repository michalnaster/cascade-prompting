from __future__ import annotations

import pandas as pd
import pytest

from cascade_prompting.metrics.significance import (
    compare_all_conditions,
    compare_conditions,
    mcnemar_test,
    paired_metric_test,
)


def test_mcnemar_no_discordant_pairs_is_not_significant():
    result = mcnemar_test([True, True, False, False], [True, True, False, False], "a", "b")
    assert result.a_correct_b_wrong == 0
    assert result.a_wrong_b_correct == 0
    assert result.p_value == 1.0


def test_mcnemar_strongly_asymmetric_disagreement_is_significant():
    # a is right and b is wrong on 18 of 20 questions; b is never uniquely right.
    correct_a = [True] * 18 + [False] * 2
    correct_b = [False] * 18 + [False] * 2
    result = mcnemar_test(correct_a, correct_b, "a", "b")
    assert result.a_correct_b_wrong == 18
    assert result.a_wrong_b_correct == 0
    assert result.n_pairs == 20
    assert result.p_value < 0.001


def test_mcnemar_evenly_split_disagreement_is_not_significant():
    # 5 questions where only a is right, 5 where only b is right: pure noise.
    correct_a = [True] * 5 + [False] * 5
    correct_b = [False] * 5 + [True] * 5
    result = mcnemar_test(correct_a, correct_b, "a", "b")
    assert result.a_correct_b_wrong == 5
    assert result.a_wrong_b_correct == 5
    assert result.p_value == 1.0


def test_mcnemar_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="paired, equal-length"):
        mcnemar_test([True, False], [True, False, True], "a", "b")


def test_paired_metric_test_identical_arrays_is_not_significant():
    values = [1.0, 2.0, 3.0, 4.0]
    result = paired_metric_test(values, values, "latency_s", "a", "b")
    assert result.t_p_value == 1.0
    assert result.wilcoxon_p_value == 1.0
    assert result.mean_a == result.mean_b


def test_paired_metric_test_consistent_difference_is_significant():
    # b is reliably ~2x a, every single pair, with a bit of noise.
    values_a = [10.0, 12.0, 9.0, 11.0, 10.5, 9.5, 10.2, 11.5, 9.8, 10.1]
    values_b = [20.5, 23.5, 18.0, 22.0, 21.0, 19.0, 20.5, 23.0, 19.5, 20.0]
    result = paired_metric_test(values_a, values_b, "total_tokens", "a", "b")
    assert result.mean_a < result.mean_b
    assert result.t_p_value < 0.01
    assert result.wilcoxon_p_value < 0.01


def test_paired_metric_test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="paired, equal-length"):
        paired_metric_test([1.0, 2.0], [1.0, 2.0, 3.0], "latency_s", "a", "b")


def _results_df() -> pd.DataFrame:
    # Two conditions, 5 shared questions, deliberately out of order and with
    # one example_idx (99) that only "cascade" has, to check alignment drops
    # non-shared rows rather than erroring or misaligning.
    return pd.DataFrame(
        [
            {"example_idx": 2, "condition": "cascade", "hotpot_em": 1.0, "latency_s": 5.0, "total_tokens": 100},
            {"example_idx": 0, "condition": "cascade", "hotpot_em": 1.0, "latency_s": 4.0, "total_tokens": 90},
            {"example_idx": 1, "condition": "cascade", "hotpot_em": 0.0, "latency_s": 6.0, "total_tokens": 110},
            {"example_idx": 99, "condition": "cascade", "hotpot_em": 1.0, "latency_s": 4.5, "total_tokens": 95},
            {"example_idx": 0, "condition": "multi_call", "hotpot_em": 0.0, "latency_s": 12.0, "total_tokens": 250},
            {"example_idx": 1, "condition": "multi_call", "hotpot_em": 0.0, "latency_s": 15.0, "total_tokens": 260},
            {"example_idx": 2, "condition": "multi_call", "hotpot_em": 1.0, "latency_s": 13.0, "total_tokens": 240},
        ]
    )


def test_compare_conditions_aligns_by_example_idx_and_drops_unshared_rows():
    em_result, latency_result, tokens_result = compare_conditions(_results_df(), "cascade", "multi_call")

    # Only example_idx 0/1/2 are shared; 99 (cascade-only) must be excluded.
    assert em_result.n_pairs == 3
    assert latency_result.n_pairs == 3
    assert tokens_result.n_pairs == 3

    # cascade correct on 0,2 (multi_call wrong there); multi_call correct
    # only on 2 (where cascade is also correct — concordant, not discordant).
    assert em_result.a_correct_b_wrong == 1  # example_idx 0
    assert em_result.a_wrong_b_correct == 0

    assert latency_result.mean_a < latency_result.mean_b
    assert tokens_result.mean_a < tokens_result.mean_b


def test_compare_conditions_raises_on_no_shared_questions():
    df = pd.DataFrame(
        [
            {"example_idx": 0, "condition": "a", "hotpot_em": 1.0, "latency_s": 1.0, "total_tokens": 10},
            {"example_idx": 1, "condition": "b", "hotpot_em": 1.0, "latency_s": 1.0, "total_tokens": 10},
        ]
    )
    with pytest.raises(ValueError, match="No shared example_idx"):
        compare_conditions(df, "a", "b")


def test_compare_all_conditions_covers_every_pair_and_metric():
    df = _results_df()
    result = compare_all_conditions(df)

    # C(2, 2) = 1 pair * (1 mcnemar row + 2 metrics * 2 tests) = 5 rows.
    assert len(result) == 5
    assert set(result["test"]) == {"mcnemar_exact", "paired_t", "wilcoxon"}
    assert set(result["metric"]) == {"hotpot_em", "latency_s", "total_tokens"}
    assert set(result["condition_a"]) == {"cascade"}
    assert set(result["condition_b"]) == {"multi_call"}
    assert (result["n_pairs"] == 3).all()


def test_compare_all_conditions_skips_pairs_with_no_overlap_instead_of_raising():
    # "a" and "b" share example_idx 0; "c" has only ever answered example_idx
    # 1 so far — the normal state mid-run, before "c" and "a"/"b" have both
    # completed the same question at least once. This must not blow up the
    # pairs that *do* have data.
    df = pd.DataFrame(
        [
            {"example_idx": 0, "condition": "a", "hotpot_em": 1.0, "latency_s": 1.0, "total_tokens": 10},
            {"example_idx": 0, "condition": "b", "hotpot_em": 0.0, "latency_s": 2.0, "total_tokens": 20},
            {"example_idx": 1, "condition": "c", "hotpot_em": 1.0, "latency_s": 3.0, "total_tokens": 30},
        ]
    )

    result = compare_all_conditions(df)

    pairs = set(zip(result["condition_a"], result["condition_b"], strict=True))
    assert pairs == {("a", "b")}
