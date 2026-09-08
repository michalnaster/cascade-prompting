"""Efficiency + diagnostic metrics (not from ragas or the HotpotQA script)."""

from __future__ import annotations

import difflib


def edit_distance_layers(layers: list[str]) -> float:
    """Average normalized edit distance between consecutive layers.
    Near 0 means later layers barely changed anything (possible 'theater').
    """
    if len(layers) < 2:
        return 0.0
    scores = []
    for a, b in zip(layers, layers[1:], strict=False):  # deliberately offset by one
        ratio = 1 - difflib.SequenceMatcher(None, a, b).ratio()
        scores.append(ratio)
    return sum(scores) / len(scores)


def reasoning_consistency(reasonings: list[str], layers: list[str]) -> float:
    """Fraction of layers (from the 2nd on) where the stated 'no change'
    claim in reasoning matches whether the data actually changed.
    Returns 1.0 = fully consistent, 0.0 = fully inconsistent. NaN if n/a.
    """
    if not reasonings or len(layers) < 2:
        return float("nan")
    no_change_phrases = ("no change", "unchanged", "copy", "already satisf")
    consistent = 0
    checked = 0
    for i in range(1, len(layers)):
        reasoning = reasonings[i].lower()
        claims_no_change = any(p in reasoning for p in no_change_phrases)
        actually_unchanged = layers[i].strip() == layers[i - 1].strip()
        consistent += int(claims_no_change == actually_unchanged)
        checked += 1
    return consistent / checked if checked else float("nan")
