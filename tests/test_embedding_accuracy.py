from __future__ import annotations

import numpy as np
import pytest

from cascade_prompting.metrics import embedding_accuracy


class _FakeModel:
    """Deterministic stand-in for SentenceTransformer, keyed by exact text,
    so this test exercises `answer_accuracy`'s cosine-similarity math
    without importing torch or touching the network — the actual embedding
    model is exactly what's mocked out here."""

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def encode(self, texts, normalize_embeddings=True):
        vectors = np.array([self._vectors[t] for t in texts], dtype=float)
        if normalize_embeddings:
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


def test_answer_accuracy_is_one_for_same_direction_vectors(monkeypatch):
    fake_model = _FakeModel({"Paris": [3.0, 0.0], "Paris, France": [1.0, 0.0]})
    monkeypatch.setattr(embedding_accuracy, "_get_model", lambda model_name: fake_model)

    score = embedding_accuracy.answer_accuracy("Paris", "Paris, France")
    assert score == pytest.approx(1.0)


def test_answer_accuracy_is_zero_for_orthogonal_vectors(monkeypatch):
    fake_model = _FakeModel({"Paris": [1.0, 0.0], "Tokyo": [0.0, 1.0]})
    monkeypatch.setattr(embedding_accuracy, "_get_model", lambda model_name: fake_model)

    score = embedding_accuracy.answer_accuracy("Paris", "Tokyo")
    assert score == pytest.approx(0.0, abs=1e-9)


def test_answer_accuracy_is_negative_one_for_opposite_vectors(monkeypatch):
    fake_model = _FakeModel({"Paris": [1.0, 0.0], "not Paris": [-1.0, 0.0]})
    monkeypatch.setattr(embedding_accuracy, "_get_model", lambda model_name: fake_model)

    score = embedding_accuracy.answer_accuracy("Paris", "not Paris")
    assert score == pytest.approx(-1.0)
