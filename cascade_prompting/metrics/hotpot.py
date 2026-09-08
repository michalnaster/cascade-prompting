"""Official HotpotQA answer metrics.

Vendored (with light adaptation, answer-only) from:
https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py
Original functions: normalize_answer, f1_score, exact_match_score. The
original script also scores supporting-fact (sp) predictions and a joint
em/f1; those are omitted here since none of the conditions in this
benchmark produce supporting-fact predictions.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def hotpot_normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def hotpot_f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Returns (f1, precision, recall)."""
    normalized_prediction = hotpot_normalize_answer(prediction)
    normalized_ground_truth = hotpot_normalize_answer(ground_truth)

    zero_metric = (0.0, 0.0, 0.0)

    # Official rule: yes/no/noanswer predictions that don't exactly match are
    # zeroed out rather than given partial token-overlap credit.
    if normalized_prediction in ("yes", "no", "noanswer") and \
            normalized_prediction != normalized_ground_truth:
        return zero_metric
    if normalized_ground_truth in ("yes", "no", "noanswer") and \
            normalized_prediction != normalized_ground_truth:
        return zero_metric

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero_metric
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def hotpot_exact_match_score(prediction: str, ground_truth: str) -> bool:
    return hotpot_normalize_answer(prediction) == hotpot_normalize_answer(ground_truth)


def hotpot_answer_metrics(prediction: str, ground_truth: str) -> dict:
    em = hotpot_exact_match_score(prediction, ground_truth)
    f1, precision, recall = hotpot_f1_score(prediction, ground_truth)
    return {
        "hotpot_em": float(em),
        "hotpot_f1": f1,
        "hotpot_precision": precision,
        "hotpot_recall": recall,
    }
