from __future__ import annotations

from cascade_prompting.data import render_context


def test_render_context_joins_with_newlines():
    assert render_context(["a", "b", "c"]) == "a\nb\nc"
