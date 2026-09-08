"""Logging configuration for the benchmark.

A single module-level logger (`cascade_benchmark`) is shared across the
package. `setup_logging` wires it to two sinks: stdout at INFO (progress and
per-example results — what used to be bare `print()` calls) and a log file at
DEBUG, which additionally captures every intermediate layer (draft /
fact-checked / final answer text, and reasoning where present) produced by
each condition for each example. That DEBUG detail is never printed to
stdout — it's there so a run can be inspected after the fact without
re-running the benchmark.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("cascade_benchmark")


def setup_logging(log_file: str) -> None:
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)
