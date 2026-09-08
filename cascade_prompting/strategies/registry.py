"""Ordered registry of generation strategies.

Insertion order is preserved (dict) and matches the order conditions are run
and reported in — baseline, multi_call, multi_call_full_context, cascade.
"""

from __future__ import annotations

from .base import GenerationStrategy
from .baseline import BaselineStrategy
from .cascade import CascadeStrategy
from .multi_call import MultiCallFullContextStrategy, MultiCallStrategy

STRATEGIES: dict[str, GenerationStrategy] = {
    strategy.name: strategy
    for strategy in (
        BaselineStrategy(),
        MultiCallStrategy(),
        MultiCallFullContextStrategy(),
        CascadeStrategy(),
    )
}
