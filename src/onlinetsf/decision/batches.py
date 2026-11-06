"""Keep every selected unit's feedback until its next batch update."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .adapters import PreparedFeedback


class FeedbackBatches:
    """Retain shared immutable feedback records, with one queue per unit."""

    def __init__(self) -> None:
        self.pending: dict[str, list[PreparedFeedback]] = {}

    def append(self, item: PreparedFeedback, units: Iterable[str]) -> None:
        for name in units:
            self.pending.setdefault(name, []).append(item)

    def take(self, units: Iterable[str]) -> list[tuple[frozenset[str], tuple[PreparedFeedback, ...]]]:
        """Share one update when units have exactly the same pending samples."""
        grouped: dict[tuple[int, ...], tuple[set[str], tuple[PreparedFeedback, ...]]] = {}
        for name in units:
            items = self.pending.pop(name, [])
            if items:
                key = tuple(id(item) for item in items)
                if key not in grouped:
                    grouped[key] = (set(), tuple(items))
                grouped[key][0].add(name)
        return [(frozenset(names), items) for names, items in grouped.values()]
