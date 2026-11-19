"""Optimized MAGMa implementation with deferred fragment scoring.

- Generate fragments without computing scores.
- Score only matched fragments.
- Reduce the number of score_fragment calls.
"""

from .fragmentation_op_v8 import FragmentEngine

__all__ = ['FragmentEngine']
