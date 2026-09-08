from __future__ import annotations

from cascade_prompting.metrics.hotpot import (
    hotpot_answer_metrics,
    hotpot_exact_match_score,
    hotpot_f1_score,
    hotpot_normalize_answer,
)


def test_normalize_answer_strips_articles_punctuation_case():
    assert hotpot_normalize_answer("The Eiffel Tower!") == "eiffel tower"


def test_exact_match_ignores_articles_and_case():
    assert hotpot_exact_match_score("the Paris", "Paris") is True
    assert hotpot_exact_match_score("Paris", "London") is False


def test_f1_score_partial_overlap():
    f1, precision, recall = hotpot_f1_score("New York City", "New York")
    assert precision == 2 / 3
    assert recall == 1.0
    assert round(f1, 4) == round(2 * (2 / 3) / (2 / 3 + 1), 4)


def test_f1_score_no_overlap_is_zero():
    assert hotpot_f1_score("apple", "orange") == (0.0, 0.0, 0.0)


def test_f1_score_yes_no_mismatch_is_zeroed_not_partial():
    # Official HotpotQA rule: yes/no/noanswer predictions that don't exactly
    # match the gold answer get zero credit, not partial token overlap.
    assert hotpot_f1_score("no", "yes") == (0.0, 0.0, 0.0)


def test_hotpot_answer_metrics_shape():
    metrics = hotpot_answer_metrics("Paris", "Paris")
    assert metrics == {
        "hotpot_em": 1.0,
        "hotpot_f1": 1.0,
        "hotpot_precision": 1.0,
        "hotpot_recall": 1.0,
    }
