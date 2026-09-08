from __future__ import annotations

import math

from cascade_prompting.metrics.diagnostics import edit_distance_layers, reasoning_consistency


def test_edit_distance_zero_when_layers_identical():
    assert edit_distance_layers(["same text", "same text", "same text"]) == 0.0


def test_edit_distance_single_layer_is_zero():
    assert edit_distance_layers(["only one layer"]) == 0.0


def test_edit_distance_nonzero_when_layers_differ():
    score = edit_distance_layers(["hello world", "goodbye world entirely"])
    assert 0.0 < score <= 1.0


def test_reasoning_consistency_nan_when_no_reasonings():
    assert math.isnan(reasoning_consistency([], ["a", "b"]))


def test_reasoning_consistency_fully_consistent():
    # Layer 1 claims no change and data is unchanged; layer 2 claims a
    # change and data is different -> both consistent.
    reasonings = ["draft", "no change needed here", "changed the wording"]
    layers = ["a", "a", "b"]
    assert reasoning_consistency(reasonings, layers) == 1.0


def test_reasoning_consistency_fully_inconsistent():
    # Claims "no change" but data actually changed.
    reasonings = ["draft", "no change needed here"]
    layers = ["a", "b"]
    assert reasoning_consistency(reasonings, layers) == 0.0
