"""Canonical R-SPD randomized-truncation API.

The implementation remains importable from :mod:`spd.compression` so pickles
and callers created by the first prototype continue to work.  New code should
import randomized-truncation symbols from this module or the top-level
``spd`` package.
"""

from .compression import (
    RandomizedTruncationStats,
    optimal_inclusion_probabilities,
    pivotal_select,
    pivotal_truncate,
)


__all__ = [
    "RandomizedTruncationStats",
    "optimal_inclusion_probabilities",
    "pivotal_select",
    "pivotal_truncate",
]
